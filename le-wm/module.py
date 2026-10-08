import os
import contextlib
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time


class StateLipschitzReg(nn.Module):
    """Bound latent change by true-state change (obs_state-Lipschitz, FD form).

    The le-wm encoder is pixel-only, so d embed / d state is not directly available.
    Using the state present in the batch, bound the embedding distance between frames
    by their state distance with a hinge:
        relu(||emb_i - emb_j||^2 - L^2 ||state_i - state_j||^2).
    This makes the pixel->embed map Lipschitz in the *underlying state*, so a latent
    CBF margin converts to a state-space margin. The bounding quantity is the state
    change -- the correct one, since a small action does not imply a small state
    transition.
    """

    def __init__(self, target_L=1.0):
        super().__init__()
        self.target_L = target_L

    def forward(self, emb, state):
        # emb: (B, T, D); state: (B, T, S). Temporally-adjacent frame pairs.
        de2 = (emb[:, 1:] - emb[:, :-1]).pow(2).sum(-1)       # (B, T-1)
        ds2 = (state[:, 1:] - state[:, :-1]).float().pow(2).sum(-1)
        return F.relu(de2 - (self.target_L ** 2) * ds2).mean()


class PixelLipschitzReg(nn.Module):
    """Regress the encoder's FINITE-perturbation sensitivity toward target_L.

    Apply delta = sigma * u, u ~ N(0, I), and regress the per-unit latent response
        ||enc(x + delta) - enc(x)|| / sigma     (its expected square is ||J||_F^2)
    toward target_L (TWO-SIDED, like `JacobianNormReg`). delta is applied to the SAME
    frame, so it is a controlled perturbation (not adjacent frames, which differ by
    uncontrolled dynamics/drift). Differs from `JacobianNormReg(mode="fd")` only in
    using a FINITE perturbation (sigma, default 0.1) -- a real perturbation ball rather
    than the infinitesimal derivative. The two-sided regress (vs the earlier one-sided
    hinge) pins the norm to target_L regardless of dataset, avoiding the hinge's
    dataset-dependent "constraint already satisfied -> reg goes inactive" failure.
    """

    def __init__(self, target_L=1.0, sigma=0.1, n_probes=1):
        super().__init__()
        self.target_L = target_L
        self.sigma = sigma
        self.n_probes = n_probes

    def forward(self, encode_fn, pixels):
        # pixels: (B, T, C, H, W) or (B, C, H, W) -> use one frame per sequence
        if pixels.dim() == 5:
            pixels = pixels[:, 0]
        x = pixels.float()
        z0 = encode_fn(x)
        fro2 = torch.zeros(z0.size(0), device=z0.device)
        for _ in range(self.n_probes):
            delta = torch.randn_like(x) * self.sigma
            z1 = encode_fn(x + delta)
            # ||Δemb||/sigma ~= ||J u||; E||J u||^2 = ||J||_F^2 for u ~ N(0, I)
            fro2 = fro2 + (z1 - z0).flatten(1).pow(2).sum(1) / (self.sigma ** 2)
        fro_norm = (fro2 / self.n_probes + 1e-12).sqrt()
        if os.environ.get("DEBUG_PIXLIP"):
            print(f"[PIXLIP] fro_norm median={fro_norm.median().item():.4g} target_L={self.target_L}")
        return (fro_norm - self.target_L).pow(2).mean()


class InvarianceReg(nn.Module):
    """Minimize latent change under a controlled perturbation (positive-only invariance).

        minimize ||enc(x + delta) - enc(x)||^2,   delta = sigma * N(0, I).

    SimSiam/VICReg-style invariance term that relies on SIGReg to prevent collapse
    (no negatives, no target). The loss weight -- balanced against sigreg + the
    prediction loss -- implicitly sets the equilibrium encoder Lipschitz constant.
    Pushes the upper-Lipschitz / robustness leg; does NOT enforce lower-bound
    separability (sigreg only guards the *global* marginal, not local aliasing), so
    on a CBF this may over-smooth. delta perturbs the SAME frame (controlled), not
    adjacent frames (uncontrolled drift).
    """

    def __init__(self, sigma=0.1, n_probes=1):
        super().__init__()
        self.sigma = sigma
        self.n_probes = n_probes

    def forward(self, encode_fn, pixels):
        if pixels.dim() == 5:
            pixels = pixels[:, 0]
        x = pixels.float()
        z0 = encode_fn(x)
        loss = 0.0
        for _ in range(self.n_probes):
            delta = torch.randn_like(x) * self.sigma
            z1 = encode_fn(x + delta)
            loss = loss + (z1 - z0).flatten(1).pow(2).sum(1).mean()  # ||delta z||^2
        return loss / self.n_probes


class AugInvarianceReg(nn.Module):
    """Encoder invariance to nuisance augmentations: ||enc(aug(x)) - enc(x)||^2.

    Generalizes `InvarianceReg` (Gaussian pixel noise) to a real nuisance
    pipeline: color jitter (brightness/contrast/saturation/hue), random
    grayscale, Gaussian blur, random convolution (network-randomization
    style), and small cutout. NO geometric augs (crop/shift/rotate): in
    dubins/pushT position IS the state, so translation invariance would
    alias exactly the information the planner/CBF needs.

    Pixels arrive ImageNet-normalized, so the pipeline denormalizes to
    [0, 1] RGB, augments with per-sample random parameters (batched tensor
    ops, no kornia), and renormalizes. Both views run in ONE 2B-sized
    encoder forward (shared BatchNorm stats in the projector). Positive-only
    invariance; SIGReg guards collapse.
    """

    def __init__(self, p_jitter=0.8, brightness=0.3, contrast=0.3, saturation=0.3,
                 hue=0.1, p_gray=0.2, p_blur=0.5, blur_sigma=(0.1, 1.5),
                 p_randconv=0.3, p_cutout=0.5, cutout_frac=0.2, noise_sigma=0.0):
        super().__init__()
        # noise_sigma: additive Gaussian noise in NORMALIZED pixel space (applied
        # after renorm), matching InvarianceReg's sigma so results are comparable.
        # CAUTION (dubins eval, 2026-07-04): color IS semantic in color-coded envs
        # (green goal / blue agent / red obstacles) — hue/grayscale/randconv
        # invariance aliases the goal marker and collapsed planner success
        # 62.5% -> 15.5%. Use color-preserving settings there: hue=0, saturation
        # low/0, p_gray=0, p_randconv=0, p_cutout=0.
        self.noise_sigma = noise_sigma
        self.p_jitter = p_jitter
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p_gray = p_gray
        self.p_blur = p_blur
        self.blur_sigma = tuple(blur_sigma)
        self.p_randconv = p_randconv
        self.p_cutout = p_cutout
        self.cutout_frac = cutout_frac
        self.register_buffer("im_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("im_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @staticmethod
    def _gray(x):
        w = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (x * w).sum(1, keepdim=True)

    def _augment(self, x):
        """x: (B, 3, H, W) in [0, 1]."""
        B, _, H, W = x.shape
        dev = x.device

        def rand(lo, hi):
            return torch.empty(B, 1, 1, 1, device=dev).uniform_(lo, hi)

        def bern(p):
            return (torch.rand(B, 1, 1, 1, device=dev) < p).float()

        # color jitter: brightness / contrast / saturation / hue-rotation
        if self.p_jitter > 0:
            m = bern(self.p_jitter)
            y = x * rand(1 - self.brightness, 1 + self.brightness)
            y = (y - y.mean((1, 2, 3), keepdim=True)) * rand(1 - self.contrast, 1 + self.contrast) \
                + y.mean((1, 2, 3), keepdim=True)
            g = self._gray(y)
            y = g + (y - g) * rand(1 - self.saturation, 1 + self.saturation)
            if self.hue > 0:
                # rotate RGB about the gray axis (batched hue approximation)
                theta = torch.empty(B, 1, 1, device=dev).uniform_(-1, 1) * self.hue * 2 * torch.pi
                k = y.new_tensor([1.0, 1.0, 1.0]).div(3 ** 0.5)
                K = y.new_tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
                eye = torch.eye(3, device=dev)
                R = eye * theta.cos() + K * theta.sin() + torch.outer(k, k) * (1 - theta.cos())
                y = torch.einsum("bij,bjhw->bihw", R, y)
            x = m * y.clamp(0, 1) + (1 - m) * x

        # random grayscale
        if self.p_gray > 0:
            m = bern(self.p_gray)
            x = m * self._gray(x).expand_as(x) + (1 - m) * x

        # Gaussian blur (one sigma per call, per-sample Bernoulli apply)
        if self.p_blur > 0:
            sigma = float(torch.empty(1).uniform_(*self.blur_sigma))
            ksize = 9
            t = torch.arange(ksize, device=dev, dtype=x.dtype) - ksize // 2
            k1d = torch.exp(-t.pow(2) / (2 * sigma ** 2))
            k1d = (k1d / k1d.sum()).view(1, 1, 1, ksize).expand(3, 1, 1, ksize)
            y = F.conv2d(x, k1d, padding=(0, ksize // 2), groups=3)
            y = F.conv2d(y, k1d.transpose(2, 3), padding=(ksize // 2, 0), groups=3)
            m = bern(self.p_blur)
            x = m * y + (1 - m) * x

        # random convolution (one He-init 3x3 kernel per call), min-max renormalized
        if self.p_randconv > 0:
            w = torch.randn(3, 3, 3, 3, device=dev, dtype=x.dtype) * (2.0 / 27) ** 0.5
            y = F.conv2d(x, w, padding=1)
            lo = y.amin((1, 2, 3), keepdim=True)
            hi = y.amax((1, 2, 3), keepdim=True)
            y = (y - lo) / (hi - lo + 1e-6)
            m = bern(self.p_randconv)
            x = m * y + (1 - m) * x

        # small cutout filled with per-image mean color
        if self.p_cutout > 0:
            fh = torch.empty(B, device=dev).uniform_(0.05, self.cutout_frac)
            fw = torch.empty(B, device=dev).uniform_(0.05, self.cutout_frac)
            ch = (torch.rand(B, device=dev) * H).long()
            cw = (torch.rand(B, device=dev) * W).long()
            rows = torch.arange(H, device=dev).view(1, H)
            cols = torch.arange(W, device=dev).view(1, W)
            in_h = (rows >= (ch - fh * H / 2).view(B, 1)) & (rows < (ch + fh * H / 2).view(B, 1))
            in_w = (cols >= (cw - fw * W / 2).view(B, 1)) & (cols < (cw + fw * W / 2).view(B, 1))
            hole = (in_h.view(B, 1, H, 1) & in_w.view(B, 1, 1, W)).float()
            hole = hole * bern(self.p_cutout)
            x = hole * x.mean((2, 3), keepdim=True) + (1 - hole) * x

        return x.clamp(0, 1)

    def forward(self, encode_fn, pixels):
        if pixels.dim() == 5:
            pixels = pixels[:, 0]
        x = pixels.float()
        with torch.no_grad():
            x01 = (x * self.im_std + self.im_mean).clamp(0, 1)
            x_aug = (self._augment(x01) - self.im_mean) / self.im_std
            if self.noise_sigma > 0:
                x_aug = x_aug + torch.randn_like(x_aug) * self.noise_sigma
        z = encode_fn(torch.cat([x, x_aug], dim=0))
        z0, z1 = z.chunk(2, dim=0)
        return (z1 - z0).pow(2).sum(-1).mean()


class AdvColorInvarianceReg(nn.Module):
    """Adversarial (worst-case) hazard-color invariance = the Ilyas min-max robust-feature
    objective specialized to the appearance threat model. Each step, recolor the (blue) sg
    hazard to each palette color, pick per-sample the WORST color (max encoder-feature
    deviation), and penalize ||enc(worst) - enc(orig).detach()||^2 — so min over encoder of
    max over color. Purple held out of the palette (stays OOD). Blue-hazard isolation mask
    a = relu(min(B-R, B-G))/255; remap out = x + a*(C - BLUE) (mirrors utils.ObstacleRecolor).
    Pixels arrive ImageNet-normalized; denorm->recolor(0..255)->renorm. Cost: (K+1) single-frame
    encoder forwards/step (K probe colors no_grad + 1 grad). SIGReg guards collapse."""
    BLUE = (0.0, 0.0, 255.0)
    PALETTE = [(255, 153, 0), (0, 153, 153), (255, 255, 0), (140, 69, 18), (0, 178, 0)]  # non-blue, non-purple

    def __init__(self, palette=None):
        super().__init__()
        self.palette = palette or self.PALETTE
        self.register_buffer("im_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("im_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _recolor(self, x255, C):
        R, G, B = x255[:, 0], x255[:, 1], x255[:, 2]
        a = torch.clamp(torch.minimum(B - R, B - G), min=0.0) / 255.0  # (B,H,W) blue-hazard mask
        out = x255.clone()
        for ci in range(3):
            out[:, ci] = torch.clamp(x255[:, ci] + a * (C[ci] - self.BLUE[ci]), 0.0, 255.0)
        return out

    def _norm(self, x255): return ((x255 / 255.0) - self.im_mean) / self.im_std

    def forward(self, encode_fn, pixels):
        if pixels.dim() == 5:
            pixels = pixels[:, 0]
        x = pixels.float()
        with torch.no_grad():
            x255 = ((x * self.im_std + self.im_mean).clamp(0, 1) * 255.0)
            z0 = encode_fn(x)                                  # target (orig), detached below
            recolored, devs = [], []
            for C in self.palette:
                xr = self._norm(self._recolor(x255, C))
                zc = encode_fn(xr)
                recolored.append(xr); devs.append((zc - z0).pow(2).sum(-1))   # (B,)
            devs = torch.stack(devs, 0)                        # (K,B)
            worst = devs.argmax(0)                             # (B,) per-sample worst color
            xw = torch.stack(recolored, 0)[worst, torch.arange(x.shape[0])]   # (B,3,H,W)
        zw = encode_fn(xw)                                     # WITH grad
        return (zw - z0.detach()).pow(2).sum(-1).mean()


class SafetyAdvInvarianceReg(nn.Module):
    """Safety-projected adversarial appearance invariance, label-only, no OOD data.

    A margin head h (margin_gp: zs + hinge@0 + WGAN-GP) is trained ON THE FLY on
    detached embeddings + the dataset's binary `failures` labels, with its OWN
    optimizer (the main optimizer only owns `model`). The encoder/predictor are then
    penalized for letting a generated appearance perturbation move the SAFETY READOUT:

        L = mean_ctx ((h(f(g x)) - h(f(x))) / gap)^2  +  ((h(pred(f(g x))) - h(pred(f(x)))) / gap)^2

    i.e. invariance only along the direction h reads (nonlinear version of projecting
    onto w = grad_z h), normalized by the EMA safe-unsafe head gap so the encoder can't
    win by shrinking the safety axis. h's params are frozen inside the penalty (it
    cannot cheat by rotating its readout away); clean targets are detached.

    Perturbation g = spatially varying HUE ROTATION + SATURATION scale about the gray
    axis (coarse GxG param grid, bilinear-upsampled, one field per sequence shared
    across context frames). Preserves luminance-mean, leaves white/gray background
    untouched and cannot erase an obstacle (chroma only scaled within [1/s, s]); found
    per-batch by PGD maximizing the penalty itself (random start). Generated, not data:
    it's the prior "colors can change", no recolor masks or palettes. NOTE red->purple
    is reachable in this family (hue -60deg, sat x0.5), so held-out purple tests the
    prior; rotation/shape remain outside it.

    Schedule: the head always trains; the penalty is OFF until step >= min_steps AND
    the head's EMA in-batch AUC >= auc_gate, then ramps linearly 0->1 over ramp_steps
    (latched). While off, PGD is skipped (no cost). The config `weight` is lambda_max.
    """

    def __init__(self, n_sub=32, grid=4, eps_hue=1.5708, eps_logsat=0.6931, attack="hue", eps_pix=8 / 255, target="head", pgd_steps=2,
                 pgd_step_frac=0.5, pred_weight=1.0, enc_weight=1.0, rand_init="uniform", scale_grad=True, head_units=512, head_lr=3e-4,
                 zs_weight=0.1, relu_weight=1.0, gp_weight=10.0, gp_thresh=0.1,
                 min_steps=2000, auc_gate=0.95, ramp_steps=10000, ema=0.99, embed_dim=192, lambda_mode="batch"):
        super().__init__()
        self.n_sub, self.grid = n_sub, grid
        # attack="hue": spatial hue/saturation field (the color prior). attack="linf": per-pixel
        # L_inf noise |delta|<=eps_pix in [0,1] RGB, one delta per sequence shared across ctx frames
        # (threat-agnostic ablation: same safety-projected penalty, no color assumption).
        assert attack in ("hue", "linf"), attack
        self.attack, self.eps_pix = attack, float(eps_pix)
        # target="head": penalize the change of the safety readout h (safety-projected, default).
        # target="latent": ablation without h — penalize the whole latent change ||f(x+d)-f(x)||^2
        # (and of the prediction), normalized by the live batch latent spread instead of the gap.
        # The head still trains (gate + logging); only the penalty stops reading through it.
        # target="lambda": penalize only the component of the latent change along the safety
        # direction lambda = mean(z_safe) - mean(z_unsafe): ((dz . lambda) / ||lambda||^2)^2, i.e. the
        # change projected on lambda in units of the safe/unsafe distance (0 iff dz is orthogonal).
        # lambda_mode="batch": direction + length from the batch centroids, with grad.
        # lambda_mode="global": direction from EMA safe/unsafe means over all batches (detached);
        # length ||lambda|| still from the batch, with grad. No head involved (use auc_gate=0).
        assert target in ("head", "latent", "lambda"), target
        assert lambda_mode in ("batch", "global"), lambda_mode
        self.target, self.lambda_mode = target, lambda_mode
        self._u = None
        self.eps = (float(eps_hue), float(eps_logsat))
        self.pgd_steps, self.pgd_step_frac = pgd_steps, pgd_step_frac
        self.pred_weight, self.enc_weight = pred_weight, enc_weight
        # ablation switches (defaults = original behaviour):
        #   rand_init="sign": random start at the box corners (+-eps per element) instead of
        #     uniform; with pgd_steps=0 this is a NON-adversarial perturbation of the same size
        #     as PGD's (sign-step) iterates.
        #   scale_grad=False: detach the live normalizer (head gap / centroid distance^2), so
        #     the penalty keeps its scale but gives no incentive to separate the classes.
        #   enc_weight / pred_weight: weight of the encoder (ctx-frame) / predictor term.
        assert rand_init in ("uniform", "sign"), rand_init
        self.rand_init, self.scale_grad = rand_init, bool(scale_grad)
        self.zs_weight, self.relu_weight = zs_weight, relu_weight
        self.gp_weight, self.gp_thresh = gp_weight, gp_thresh
        self.min_steps, self.auc_gate, self.ramp_steps, self.ema = min_steps, auc_gate, ramp_steps, ema
        layers, d = [], embed_dim
        for _ in range(2):
            layers += [nn.Linear(d, head_units, bias=False), nn.LayerNorm(head_units, eps=1e-3), nn.SiLU()]
            d = head_units
        self.head = nn.Sequential(*layers, nn.Linear(d, 1))   # same arch as latent_cbf MarginHead
        self.head_opt = torch.optim.AdamW(self.head.parameters(), lr=head_lr)
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self.register_buffer("open_step", torch.full((), -1, dtype=torch.long))
        self.register_buffer("auc_ema", torch.zeros(()))
        self.register_buffer("gap_ema", torch.zeros(()))
        self.register_buffer("mu_safe_ema", torch.zeros(embed_dim))
        self.register_buffer("mu_unsafe_ema", torch.zeros(embed_dim))
        self.register_buffer("n_ema", torch.zeros((), dtype=torch.long))   # for EMA bias correction
        self.register_buffer("im_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("im_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.stats = {}

    # ---------------- margin head (own optimizer, fp32, detached features) -------------
    def _head_step(self, z, fail):
        safe, unsafe = z[fail == 0], z[fail == 1]
        if safe.shape[0] < 2 or unsafe.shape[0] < 2:
            return
        with torch.enable_grad():
            pos, neg = self.head(safe), self.head(unsafe)
            N = max(pos.shape[0], neg.shape[0])
            rs = lambda x: x[torch.randint(0, x.shape[0], (N,), device=x.device)]
            a = torch.rand(N, 1, device=z.device)
            interp = (a * rs(safe) + (1 - a) * rs(unsafe)).requires_grad_(True)
            g = torch.autograd.grad(self.head(interp).sum(), interp, create_graph=True)[0]
            gp = ((g.pow(2).sum(1) + 1e-12).sqrt() - self.gp_thresh).pow(2).mean()
            loss = (self.zs_weight * (neg.mean() - pos.mean())
                    + self.relu_weight * (F.relu(neg).mean() + F.relu(-pos).mean())
                    + self.gp_weight * gp)
            grads = torch.autograd.grad(loss, list(self.head.parameters()))
        for p, gr in zip(self.head.parameters(), grads):
            p.grad = gr
        self.head_opt.step()
        self.head_opt.zero_grad(set_to_none=True)
        with torch.no_grad():   # in-batch AUC (safe should score higher) + head gap, EMA'd
            ps, ns = pos.detach().flatten(), neg.detach().flatten()
            auc = (ps[:, None] > ns[None, :]).float().mean()
            self.auc_ema.mul_(self.ema).add_((1 - self.ema) * auc)
            self.gap_ema.mul_(self.ema).add_((1 - self.ema) * (ps.mean() - ns.mean()).clamp_min(1e-3))
            self.n_ema += 1
        self.stats["head_loss"] = loss.detach()

    def _debias(self, v):
        return v / max(1e-8, 1.0 - self.ema ** int(self.n_ema)) if int(self.n_ema) > 0 else v

    @property
    def auc(self): return self._debias(self.auc_ema)

    @property
    def gap(self): return self._debias(self.gap_ema).clamp_min(1e-3)

    def _h(self, z):
        """Head with FROZEN params: gradients reach z (encoder/predictor) only."""
        params = {k: v.detach() for k, v in self.head.named_parameters()}
        return torch.func.functional_call(self.head, params, (z.float(),)).squeeze(-1)

    # ---------------- generated perturbation: spatial hue rotation + saturation --------
    def _perturb(self, x01, P):
        # x01: (n, T, 3, H, W) in [0,1]; P: (n, 2, G, G) = (hue angle, log-sat)
        n, T, _, H, W = x01.shape
        f = F.interpolate(P, size=(H, W), mode="bilinear", align_corners=False)   # (n,2,H,W)
        th, s = f[:, 0:1].unsqueeze(1), f[:, 1:2].exp().unsqueeze(1)             # (n,1,1,H,W)
        gray = x01.mean(2, keepdim=True)                                          # u(u.v) component
        c = x01 - gray                                                            # chroma (sum 0)
        r, g, b = c[:, :, 0:1], c[:, :, 1:2], c[:, :, 2:3]
        uxc = torch.cat([b - g, r - b, g - r], 2) / 3 ** 0.5                     # u x c, u=(1,1,1)/sqrt3
        out = gray + s * (c * th.cos() + uxc * th.sin())
        return out.clamp(0, 1)

    def _norm(self, x01):
        return ((x01.flatten(0, 1) - self.im_mean) / self.im_std).view_as(x01)

    def _readout(self, z):
        if self.target == "lambda":
            return z.float()          # projected in _dev, on the DIFFERENCE dz (so grad wrt lambda is via dz)
        return self._h(z) if self.target == "head" else z.float()

    def _dev(self, r, r_c, scale):
        """Squared deviation of a readout from its clean value, in units of `scale`."""
        if self.target == "lambda":
            return (((r - r_c) @ self._u) / scale).pow(2)
        if self.target == "head":
            return ((r - r_c) / scale).pow(2)
        return (r - r_c).pow(2).sum(-1) / scale       # latent: squared distance / spread

    def _objective(self, model, encode_fn, xp_n, act, hz_c, hp_c, gap):
        """Per-sample safety-readout deviation (ctx frames + 1-step pred) in units of `gap`.
        Must run inside `_CleanStatBN(...).apply()`: perturbed frames are normalized with the
        CLEAN batch's BN statistics, so they cannot shape normalization (see _CleanStatBN)."""
        n, T = xp_n.shape[:2]
        z = encode_fn(xp_n.flatten(0, 1)).view(n, T, -1)
        pred = model.predict(z, act)[:, -1]
        hz = self._readout(z.flatten(0, 1)).view((n, T) + hz_c.shape[2:])
        hp = self._readout(pred)
        return self.enc_weight * self._dev(hz, hz_c, gap).mean(1) + self.pred_weight * self._dev(hp, hp_c, gap)

    def forward(self, model, encode_fn, emb, ctx_act, pixels, failures):
        dev = emb.device
        fail = failures.flatten().round().long()
        if self.training:
            self.step += 1
            with torch.autocast(device_type=dev.type, enabled=False):
                self._head_step(emb.detach().flatten(0, 1).float(), fail)
            if self.open_step < 0 and self.step >= self.min_steps and self.auc >= self.auc_gate:
                self.open_step.fill_(int(self.step))
        ramp = 0.0 if self.open_step < 0 else min(1.0, float(self.step - self.open_step) / max(1, self.ramp_steps))
        self.stats.update(auc=self.auc.clone(), gap=self.gap.clone(), ramp=torch.tensor(ramp, device=dev))
        if ramp == 0.0 or not self.training:   # val may run under inference_mode (no PGD grads)
            return torch.zeros((), device=dev)

        # LIVE in-batch safe-unsafe head gap, WITH grad into the encoder: shrinking the
        # safety separation raises the penalty immediately (a detached EMA gap only reacts
        # with lag, so flattening h everywhere was the cheapest descent direction).
        # Floored at 10% of the EMA gap for stability; PGD uses the detached value.
        h_all = self._h(emb.flatten(0, 1))
        if (fail == 0).any() and (fail == 1).any():
            gap_live = (h_all[fail == 0].mean() - h_all[fail == 1].mean()).clamp_min(0.1 * float(self.gap))
        else:
            gap_live = self.gap.detach()
        if self.target == "lambda":
            e = emb.flatten(0, 1).float()
            if not ((fail == 0).any() and (fail == 1).any()):
                return torch.zeros((), device=dev)
            lam = e[fail == 0].mean(0) - e[fail == 1].mean(0)            # batch safety direction, with grad
            with torch.no_grad():
                self.mu_safe_ema.mul_(self.ema).add_((1 - self.ema) * e[fail == 0].mean(0).detach())
                self.mu_unsafe_ema.mul_(self.ema).add_((1 - self.ema) * e[fail == 1].mean(0).detach())
            gap_live = lam.norm().clamp_min(1e-3)                          # ||lambda||, with grad
            if self.lambda_mode == "batch":
                self._u = lam / gap_live
            else:
                g = self.mu_safe_ema - self.mu_unsafe_ema                  # debiasing cancels in the direction
                self._u = (g / g.norm().clamp_min(1e-8)).detach()
                self.stats["cos_batch_global"] = F.cosine_similarity(lam.detach(), g, dim=0)
        elif self.target == "latent":
            # the identity-readout analog of the gap: squared distance between the safe and
            # unsafe latent centroids of the batch, with grad (squashing the classes together
            # raises the penalty instead of lowering it)
            e = emb.flatten(0, 1).float()
            if (fail == 0).any() and (fail == 1).any():
                gap_live = (e[fail == 0].mean(0) - e[fail == 1].mean(0)).pow(2).sum().clamp_min(1e-3)
            else:
                gap_live = (e - e.mean(0)).pow(2).sum(-1).mean().detach().clamp_min(1e-3)
        if not self.scale_grad:
            gap_live = gap_live.detach()
        self.stats["gap_live"] = gap_live.detach()

        T = ctx_act.size(1)
        n = min(self.n_sub, pixels.size(0))
        x = pixels[:n, :T].float()
        act = ctx_act[:n].detach()
        x01 = (x * self.im_std.unsqueeze(0) + self.im_mean.unsqueeze(0)).clamp(0, 1)
        if self.attack == "hue":
            eps = torch.tensor(self.eps, device=dev).view(1, 2, 1, 1)
            pshape, perturb = (n, 2, self.grid, self.grid), self._perturb
        else:
            eps = torch.tensor(self.eps_pix, device=dev)
            pshape, perturb = (n, 1) + tuple(x01.shape[2:]), (lambda x, P: (x + P).clamp(0, 1))
        bn = _CleanStatBN(model)
        u_live = self._u
        if u_live is not None:
            self._u = u_live.detach()                 # clean targets + PGD use the detached direction
        with bn.capture(), torch.no_grad():          # clean targets + clean BN stats (detached)
            zc = encode_fn(x.flatten(0, 1)).view(n, T, -1)
            hz_c = self._readout(zc.flatten(0, 1)).view((n, T) + ((-1,) if self.target in ("latent", "lambda") else ()))
            hp_c = self._readout(model.predict(zc, act)[:, -1])
        gap_pgd = gap_live.detach()
        with bn.apply():
            obj = lambda P, g: self._objective(model, encode_fn, self._norm(perturb(x01, P)), act, hz_c, hp_c, g)
            if self.rand_init == "sign":
                P = (torch.randint(0, 2, pshape, device=dev) * 2 - 1).float() * eps    # random corner
            else:
                P = (torch.rand(pshape, device=dev) * 2 - 1) * eps                    # random start
            best_P, best_d, d_start = P, None, None
            for k in range(self.pgd_steps + 1):     # PGD (L_inf box), keep per-sample best iterate
                if k < self.pgd_steps:
                    P = P.detach().requires_grad_(True)
                    with torch.enable_grad():
                        dk = obj(P, gap_pgd)
                        gP = torch.autograd.grad(dk.sum(), P)[0]
                else:
                    with torch.no_grad():
                        dk = obj(P, gap_pgd)
                dk = dk.detach()
                if best_d is None:
                    best_d, d_start = dk, dk
                else:
                    better = dk > best_d
                    best_P = torch.where(better.view((-1,) + (1,) * (P.dim() - 1)), P.detach(), best_P)
                    best_d = torch.maximum(dk, best_d)
                if k < self.pgd_steps:
                    P = torch.max(torch.min(P.detach() + self.pgd_step_frac * eps * gP.sign(), eps), -eps)
            self._u = u_live                          # final graded pass: direction with grad (batch mode)
            d = obj(best_P.detach(), gap_live)
        self.stats["adv_gain"] = best_d.mean() / (d_start.mean() + 1e-8)
        return ramp * d.mean()


class _CleanStatBN:
    """Normalize the regularizer's forwards with CLEAN-batch BatchNorm statistics.

    Why (measured on run A, 2026-10-02): with clean+perturbed in one train-mode batch, the
    encoder learned to make perturbed frames "loud" so the shared BN variance blew up and
    squashed BOTH halves' head readouts together (clean gap .97 -> .07) — the penalty looked
    tiny with no real invariance — and the loud stats leaked into running_var (x6.7), breaking
    every eval-mode use. capture(): normalize with this batch's own stats (= train-mode BN
    output) and record them, detached. apply(): normalize with the recorded clean stats.
    Running stats are never updated in either mode."""

    def __init__(self, model):
        self.bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        self.stats, self.mode = {}, None

    def _fwd(self, bn):
        def fwd(x):
            if self.mode == "capture":
                dims = [0] + list(range(2, x.dim()))
                self.stats[id(bn)] = (x.mean(dims).detach(), x.var(dims, unbiased=False).detach())
            mean, var = self.stats[id(bn)]
            return F.batch_norm(x, mean, var, bn.weight, bn.bias, False, 0.0, bn.eps)
        return fwd

    @contextlib.contextmanager
    def _patched(self, mode):
        self.mode = mode
        for bn in self.bns:
            bn.forward = self._fwd(bn)
        try:
            yield self
        finally:
            for bn in self.bns:
                del bn.forward          # restore the class method
            self.mode = None

    def capture(self): return self._patched("capture")
    def apply(self): return self._patched("apply")


class ActionSeparationReg(nn.Module):
    """Lower-bound predicted-latent separation by action separation (predictor leg).

    Counterfactual pair: same context, same past actions, only the LAST context
    action swapped with another batch sample's. Require the predicted next
    latents to differ at least in proportion to the action difference:
        relu(L^2 ||a - a'||^2 - ||pred(z, a) - pred(z, a')||^2).
    Context embeddings are detached, so the gradient shapes the predictor and
    action encoder only (the "separate" leg), while invariance/Lipschitz regs
    shape the encoder (the "invariant" leg). Directly penalizes the
    control-authority collapse where pred_loss stays low but actions stop
    mattering (the fs=5 dubins failure). Both branches run in one 2B-sized
    predictor forward.

    One-way by design: the upper side ("not too much" separation) is already
    guarded by pred_loss, which anchors pred(z, a) to the TRUE next latent
    (scale pinned by sigreg), so over-separation shows up as prediction error.
    Only under-separation is invisible to pred_loss. `target_U` adds an
    optional upper hinge relu(||dz||^2 - U^2 ||da||^2) as a safety valve;
    disabled (None) by default. `last_ratio` exposes the observed median
    ||dz||/||da|| for logging.
    """

    def __init__(self, target_L=1.0, target_U=None, pull_target=None):
        super().__init__()
        # pull_target: if set, replaces the hinges with an always-on two-sided
        # regression (||dz|| - pull_target*||da||)^2 -- the predictor-side
        # analogue of JacobianNormReg's (||J|| - L)^2. Difference form (not a
        # ratio) so near-identical permuted actions don't blow up the loss.
        self.target_L = target_L
        self.target_U = target_U
        self.pull_target = pull_target
        self.last_ratio = 0.0

    def forward(self, model, ctx_emb, ctx_act):
        # ctx_emb: (B, T, D); ctx_act: (B, T, A) normalized raw actions
        z = ctx_emb.detach()
        perm = torch.randperm(ctx_act.size(0), device=ctx_act.device)
        alt_act = ctx_act.clone()
        alt_act[:, -1] = ctx_act[perm, -1]
        acts = torch.cat([ctx_act, alt_act], dim=0)
        preds = model.predict(z.repeat(2, 1, 1), model.action_encoder(acts))
        pred, alt_pred = preds.chunk(2, dim=0)
        dz2 = (pred[:, -1] - alt_pred[:, -1]).pow(2).sum(-1)
        da2 = (ctx_act[:, -1] - alt_act[:, -1]).float().pow(2).sum(-1)
        with torch.no_grad():
            valid = da2 > 1e-8  # perm can map a sample to itself
            self.last_ratio = (
                (dz2[valid] / da2[valid]).sqrt().median().item() if valid.any() else 0.0
            )
        if self.pull_target is not None:
            dz = (dz2 + 1e-12).sqrt()
            da = da2.sqrt()
            return (dz - self.pull_target * da).pow(2).mean()
        loss = F.relu((self.target_L ** 2) * da2 - dz2).mean()
        if self.target_U is not None:
            loss = loss + F.relu(dz2 - (self.target_U ** 2) * da2).mean()
        return loss


class PredAnchorReg(nn.Module):
    """Suppress encoder sensitivity the PREDICTOR cannot carry forward (label-free).

    Perturb the last context frame (sigma * N(0,I) in pixel space), get the
    encoder's response dz = enc(x+δ) − enc(x). Roll BOTH the clean and perturbed
    last-latent through the predictor and measure how much of dz reaches the
    next-step prediction, dpred. A nuisance perturbation moves the latent (dz>0)
    but the predictor can't propagate it (dpred≈0, since unpredictable = regressed
    to mean); a signal perturbation propagates (dpred≈dz). Penalize the excess:
        relu(||dz||^2 − λ||dpred||^2).
    So the dynamics (the predictor) DEFINE signal vs nuisance — no privileged
    state, no per-env slice, no hand-picked augmentations. Unlike a uniform ||J||
    cap it is DIRECTIONAL: it suppresses only predictor-invisible sensitivity, so
    it should reject nuisance WITHOUT blurring the fine signal resolution a CBF
    needs. `last_ratio` logs median ||dpred||/||dz|| (how predictor-aligned the
    encoder's sensitivity is).

    Early-training note: with AdaLN-zero the predictor is ~identity at init, so
    dpred≈dz and the penalty is ~0; it becomes discriminative (and directional)
    only as the predictor learns to damp the unpredictable part — graceful, acts
    like a mild jacobian early then specializes.
    """

    def __init__(self, sigma=0.1, lam=1.0, n_probes=1, detach_pred=True):
        super().__init__()
        self.sigma = sigma
        self.lam = lam
        self.n_probes = n_probes
        # detach_pred=True (v2): the predictable budget lam*||dpred||^2 is a DETACHED
        # per-sample threshold, so the reg penalizes the unpredictable excess of
        # ||dz|| ABSOLUTELY and can only push ||dz|| down — it cannot be gamed by
        # amplifying predictor-visible directions (the v1 failure, which raised ||J||
        # and worsened conditioning). detach_pred=False reproduces the v1 form.
        self.detach_pred = detach_pred
        self.last_ratio = 0.0

    def forward(self, model, encode_fn, ctx_emb, ctx_act, last_frame):
        # ctx_emb: (B,T,D) latents; ctx_act: (B,T,A_emb) encoded actions;
        # last_frame: (B,C,H,W) pixels of the final context step.
        ref = ctx_emb.detach()
        z0 = ref[:, -1]
        pred0 = model.predict(ref, ctx_act).detach()[:, -1]
        x = last_frame.float()
        loss = 0.0
        ratios = []
        for _ in range(self.n_probes):
            zp = encode_fn(x + self.sigma * torch.randn_like(x))       # encoder response (grad)
            dz2 = (zp - z0).pow(2).sum(-1)
            ctxp = torch.cat([ref[:, :-1], zp.unsqueeze(1)], dim=1)
            predp = model.predict(ctxp, ctx_act)[:, -1]
            dpred2 = (predp - pred0).pow(2).sum(-1)
            budget = self.lam * (dpred2.detach() if self.detach_pred else dpred2)
            loss = loss + F.relu(dz2 - budget).mean()
            with torch.no_grad():
                ratios.append((dpred2 / (dz2 + 1e-8)).sqrt().median())
        with torch.no_grad():
            self.last_ratio = float(torch.stack(ratios).mean())
        return loss / self.n_probes


class JacobianNormReg(nn.Module):
    """Stochastic Jacobian-norm penalty on the encoder.

    Estimates ||J_enc|| via Hutchinson-style VJPs and pulls it toward
    `target_L`. Directly controls the obs->latent Lipschitz constant
    that a latent-space CBF margin depends on.
    """

    def __init__(self, target_L=1.0, n_probes=1, mode="exact", fd_eps=0.01):
        super().__init__()
        self.target_L = target_L
        self.n_probes = n_probes
        self.mode = mode          # "exact" (double-backward VJP) | "fd" (finite-difference JVP)
        self.fd_eps = fd_eps

    def forward(self, encode_fn, pixels):
        # pixels: (B, C, H, W) or (B, T, C, H, W)
        if pixels.dim() == 5:
            pixels = pixels[:, 0]
        if self.mode == "fd":
            # Finite-difference JVP: for u ~ N(0, I), E||J u||^2 = ||J||_F^2, with
            # J u ~= (enc(x + eps*u) - enc(x)) / eps. Two forwards, no double-backward
            # (cheap), at the cost of an O(eps) curvature bias. Grads flow into encoder.
            x = pixels.float()
            z0 = encode_fn(x)
            fro2 = torch.zeros(z0.size(0), device=z0.device)
            for _ in range(self.n_probes):
                u = torch.randn_like(x)
                z1 = encode_fn(x + self.fd_eps * u)
                fro2 = fro2 + (z1 - z0).flatten(1).pow(2).sum(1) / (self.fd_eps ** 2)
            fro_norm = (fro2 / self.n_probes + 1e-12).sqrt()
            return (fro_norm - self.target_L).pow(2).mean()
        # Flash/efficient SDPA kernels lack double-backward; force math kernel
        # so autograd.grad(..., create_graph=True) works through ViT attention.
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
            pixels = pixels.float().detach().requires_grad_(True)
            z = encode_fn(pixels)
            # Hutchinson Frobenius estimate: for v ~ N(0, I), E||J^T v||^2 = ||J||_F^2,
            # so target_L pins the true Frobenius norm (unit-normalized v would instead
            # pin a sqrt(D)-scaled quantity, making target_L dimension-dependent).
            fro2 = torch.zeros(z.size(0), device=z.device)
            for _ in range(self.n_probes):
                v = torch.randn_like(z)
                Jtv = torch.autograd.grad(
                    z, pixels, grad_outputs=v,
                    create_graph=self.training, retain_graph=True,
                )[0]
                fro2 = fro2 + Jtv.flatten(1).pow(2).sum(1)
            fro_norm = (fro2 / self.n_probes + 1e-12).sqrt()
        return (fro_norm - self.target_L).pow(2).mean()
    
class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):
        x = self.input_proj(x)
        if c is not None:
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        return self.output_proj(x)

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


class GRUPredictor(nn.Module):
    """Dreamer-style recurrent predictor for LE-WM (the JEPA+GRU ablation).

    Carries a GRU hidden state (the "deter" memory) across the sequence and
    predicts the next embedding from it. Same forward(x, c) -> (B, T, D) contract
    as ARPredictor (position i predicts x[i+1] from x[:i+1] + actions), so the
    training loop and JEPA.predict wrapper are unchanged. Additionally exposes
    `step` and `unroll_h` so the adapter can carry the hidden state across imagined
    steps and expose it in get_feat (= concat(embed, deter_h)) — the predictive
    analog of Dreamer's [stoch, deter], with NO reconstruction. `deter_dim` is the
    recurrent memory width. `pos_embedding` is kept (frozen) only so the adapter's
    shape-inference (`_infer_history_size`) keeps working.
    """

    def __init__(self, *, num_frames, input_dim, hidden_dim, action_dim,
                 deter_dim=512, output_dim=None, dropout=0.0, **_ignored):
        super().__init__()
        output_dim = output_dim or input_dim
        self.input_dim = input_dim
        self.deter_dim = deter_dim
        self.cell = nn.GRUCell(input_dim + action_dim, deter_dim)
        self.out = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, num_frames, input_dim), requires_grad=False)

    def step(self, z, a_emb, h):
        """One recurrent step: (z_t, a_emb_t, h_t) -> (pred z_{t+1}, h_{t+1})."""
        h = self.cell(torch.cat([z, a_emb], dim=-1), h)
        return self.out(h), h

    def unroll_h(self, x, c, h0=None):
        """x: (B,T,D), c: (B,T,A_emb). Returns preds (B,T,D) [pos i predicts
        x[i+1]] and hidden states hs (B,T,deter_dim) [h after consuming x[:i+1]]."""
        B, T = x.shape[0], x.shape[1]
        h = x.new_zeros(B, self.deter_dim) if h0 is None else h0
        preds, hs = [], []
        for t in range(T):
            pred, h = self.step(x[:, t], c[:, t], h)
            preds.append(pred)
            hs.append(h)
        return torch.stack(preds, 1), torch.stack(hs, 1)

    def forward(self, x, c):
        preds, _ = self.unroll_h(x, c)
        return preds
