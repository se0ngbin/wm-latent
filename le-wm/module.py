import os
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


class ActionNCEReg(nn.Module):
    """AC-MTM anti-collapse (port of action-contrastive-jepa, arXiv 2608.17542).

    Training-only inverse-dynamics head (z_t, z_{t+1}) -> a_t (the whole
    frameskip*action_dim block). Its prediction is a query classified among the
    N = B(T-1) in-batch true action blocks by -||pred - tgt||^2 / (tau * d_a);
    a collapsed encoder gives identical queries, so the loss floors at log N.
    Meant to REPLACE sigreg. The head lives here, so the optimizer regex must
    include `regularizers.action_nce` (train.py handles it).
    """

    def __init__(self, latent_dim, action_dim, hidden_dim=512, depth=2, temperature=0.1):
        super().__init__()
        layers = [nn.Linear(2 * latent_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.net = nn.Sequential(*layers)
        self.temperature = temperature
        self.last_acc = 0.0

    def forward(self, emb, action):
        """emb (B,T,D); action (B,T,A) normalized; action[:, t] carries t -> t+1."""
        pred = self.net(torch.cat([emb[:, :-1], emb[:, 1:]], -1)).flatten(0, 1).float()
        tgt = action[:, : emb.size(1) - 1].flatten(0, 1).float()
        logits = -(pred[:, None] - tgt[None]).pow(2).mean(-1) / max(self.temperature, 1e-8)
        labels = torch.arange(pred.size(0), device=pred.device)
        with torch.no_grad():
            self.last_acc = (logits.argmax(1) == labels).float().mean().item()
        return F.cross_entropy(logits, labels)


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
