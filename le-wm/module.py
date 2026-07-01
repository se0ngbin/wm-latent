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
