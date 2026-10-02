"""Safe-by-construction action maps for latent-space CEM planning.

Instead of filtering or projecting actions after planning, the CEM samples an
unconstrained coordinate ``u`` and a state-dependent map transforms it onto the
safe action set

    S(z) = { a : h(f(z, a)) >= (1 - alpha) * max(h(z), 0) + h_offset }

where ``h`` is a safety value (learned margin head on latents, or ground-truth
distance-to-obstacle on true states in debug mode) and ``f`` is the one-step
latent dynamics. Every candidate the CEM evaluates and the final emitted action
is the image of this map — there is no rejection or projection stage.

Backends
--------
GridGaugeMap   1-D actions (Dubins): evaluate the constraint on a K-point action
               grid, take the largest contiguous safe interval, and gauge-map
               ``u`` affinely onto it. Exact up to grid resolution; the chosen
               action's next latent is reused from the grid rollout (snap-to-grid).
(HardNetPPMap, any action dim, lands later: iterative damped linearization of
the constraint per HardNet++, arXiv 2604.19669.)

Safety values
-------------
LearnedMarginH  h(z) = wm.heads["margin_gp"|"margin_nogp"](z) on LE-WM latents.
GroundTruthH    h(x) = min_i(||p - c_i|| - r_i) on a parallel true-state tensor
                advanced with the same batched RK4 step as DubinsEnv. Debug path
                separating "the mechanism works" from "the learned h is bad".
"""
from __future__ import annotations

import torch


class GridGaugeMap:
    """Gauge map from u in [nlo, nhi] onto the largest safe action interval.

    The caller evaluates the safety value of the K grid actions' successors
    (``h_next_grid``) and hands them in; this class only owns the grid, the
    CBF threshold, the interval extraction, and the gauge map.
    """

    def __init__(
        self,
        alpha: float,
        nlo: float,
        nhi: float,
        grid_size: int = 21,
        h_offset: float = 0.0,
        h_shift: float = 0.0,
        recovery: str = "instant",  # "instant" | "geometric"
        device: str = "cuda:0",
    ):
        self.alpha = float(alpha)
        self.nlo, self.nhi = float(nlo), float(nhi)
        self.h_offset = float(h_offset)
        # h_shift shrinks the safe set to {h >= h_shift} by shifting the VALUE
        # (h~ = h - shift on both sides of the condition). Unlike h_offset —
        # which inflates the per-step increment and turns infeasible when h is
        # small — the shifted condition stays feasible and exerts recovery
        # pressure toward the shrunk set. Set it to the h-error scale near the
        # boundary (e.g. ~0.1 for learned margin heads).
        self.h_shift = float(h_shift)
        # recovery: what the condition demands when h~_cur < 0 (outside the
        # (shrunk) safe set). "instant" clamps the threshold at 0 — requires
        # re-entering the set in ONE step, which is dynamically infeasible
        # beyond ~max|dh| and triggers fallback noise-following. "geometric"
        # uses the unclamped (1-alpha)*h~_cur — requires a reachable 30%/step
        # improvement instead (the actually-graceful variant).
        assert recovery in ("instant", "geometric"), recovery
        self.recovery = recovery
        self.K = int(grid_size)
        self.a_grid = torch.linspace(self.nlo, self.nhi, self.K, device=device)  # (K,)
        # fallback bookkeeping (fraction of choices with an empty safe set)
        self.n_choices = 0
        self.n_fallback = 0

    def reset_stats(self):
        self.n_choices = 0
        self.n_fallback = 0

    @property
    def fallback_rate(self) -> float:
        return self.n_fallback / max(self.n_choices, 1)

    def choose(
        self, u: torch.Tensor, h_cur: torch.Tensor, h_next_grid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        u:           (S, 1) unconstrained CEM coordinate in [nlo, nhi]
        h_cur:       (S,)   safety value at the current state
        h_next_grid: (S, K) safety value at each grid action's successor
        Returns (idx (S,) long, a_safe (S, 1)).
        """
        h_cur = h_cur - self.h_shift
        h_next_grid = h_next_grid - self.h_shift
        base = h_cur if self.recovery == "geometric" else h_cur.clamp(min=0.0)
        thresh = (1.0 - self.alpha) * base + self.h_offset  # (S,)
        safe = h_next_grid >= thresh.unsqueeze(1)  # (S, K) bool
        lo, hi, any_safe = self._largest_interval(safe)

        t = ((u.squeeze(1) - self.nlo) / (self.nhi - self.nlo)).clamp(0.0, 1.0)
        idx = torch.round(lo.float() + t * (hi - lo).float()).long()
        idx = torch.where(any_safe, idx, h_next_grid.argmax(dim=1))

        self.n_choices += int(u.shape[0])
        self.n_fallback += int((~any_safe).sum().item())
        return idx, self.a_grid[idx].unsqueeze(1)

    @staticmethod
    def _largest_interval(safe: torch.Tensor):
        """Largest contiguous run of True per row. safe: (S, K) bool.
        Returns (lo, hi, any_safe), each (S,). O(K) python loop, batched over S.
        """
        S, K = safe.shape
        dev = safe.device
        run = torch.zeros(S, dtype=torch.long, device=dev)
        best_len = torch.zeros(S, dtype=torch.long, device=dev)
        best_hi = torch.zeros(S, dtype=torch.long, device=dev)
        for k in range(K):
            run = torch.where(safe[:, k], run + 1, torch.zeros_like(run))
            better = run > best_len
            best_len = torch.where(better, run, best_len)
            best_hi = torch.where(better, torch.full_like(best_hi, k), best_hi)
        best_lo = best_hi - best_len + 1
        return best_lo.clamp(min=0), best_hi, best_len > 0


class HardNetPPMap:
    """Dimension-agnostic safe map via HardNet++-style iterative damped
    linearization (arXiv 2604.19669): treat the sampled action as the initial
    output and repeatedly correct it along the constraint gradient

        g(a) = h_next(a) - [(1 - alpha) * max(h_cur, 0) + h_offset]
        a   <- clamp(a + damping * relu(-g) * grad_a g / (||grad_a g||^2 + eps))

    Each iteration is differentiable; here it is used inside a no-grad planner,
    so gradients are taken locally per step w.r.t. the action only. Unlike the
    1-D grid map this generalizes to any action dimension (pushT, real robots).
    """

    def __init__(
        self,
        alpha: float,
        nlo: float,
        nhi: float,
        iters: int = 3,
        damping: float = 1.0,
        h_offset: float = 0.0,
        eps: float = 1e-8,
    ):
        self.alpha = float(alpha)
        self.nlo, self.nhi = float(nlo), float(nhi)
        self.iters = int(iters)
        self.damping = float(damping)
        self.h_offset = float(h_offset)
        self.eps = float(eps)
        # residual bookkeeping (post-correction constraint violation)
        self.n_choices = 0
        self.n_violated = 0
        self.sum_residual = 0.0

    def reset_stats(self):
        self.n_choices = 0
        self.n_violated = 0
        self.sum_residual = 0.0

    @property
    def violation_rate(self) -> float:
        return self.n_violated / max(self.n_choices, 1)

    @property
    def mean_residual(self) -> float:
        return self.sum_residual / max(self.n_choices, 1)

    def choose(self, u: torch.Tensor, h_cur: torch.Tensor, h_next_fn):
        """
        u:         (S, A) initial (unconstrained) actions
        h_cur:     (S,)   safety value at the current state
        h_next_fn: callable a (S, A) -> h_next (S,), differentiable in a
        Returns (a_safe (S, A), g_final (S,)).
        """
        thresh = (1.0 - self.alpha) * h_cur.clamp(min=0.0) + self.h_offset
        a = u.clone()
        with torch.enable_grad():
            for _ in range(self.iters):
                a_v = a.detach().requires_grad_(True)
                g = h_next_fn(a_v) - thresh  # (S,)
                (grad,) = torch.autograd.grad(g.sum(), a_v)
                denom = (grad * grad).sum(-1, keepdim=True) + self.eps
                step = self.damping * torch.relu(-g).unsqueeze(-1) * grad / denom
                a = (a_v + step).detach().clamp(self.nlo, self.nhi)
        with torch.no_grad():
            g_final = h_next_fn(a) - thresh
        res = torch.relu(-g_final)
        self.n_choices += int(u.shape[0])
        self.n_violated += int((res > 0).sum().item())
        self.sum_residual += float(res.sum().item())
        return a, g_final


class LearnedMarginH:
    """h(z) from a trained LE-WM margin head; works on (..., D) latents."""

    def __init__(self, wm, head: str = "margin_gp"):
        self.head = wm.heads[head]

    def value(self, feat: torch.Tensor) -> torch.Tensor:
        out = self.head(feat)
        return out.squeeze(-1) if out.shape[-1] == 1 else out


class GroundTruthH:
    """Ground-truth safety value + parallel true-state rollout for Dubins.

    Two h sources:
      h_source="hj"    (default) trilinear interpolation of the HJ avoid value
                       V*(x, y, theta) from compute_hj_ground_truth.py. This is
                       the *valid* CBF for Dubins: its zero-superlevel set is
                       control-invariant, so the per-step condition is
                       recursively feasible.
      h_source="dist"  min_i(||p - c_i|| - r_i). NOT a valid CBF for Dubins
                       (omega affects h only at O(dt^2)); kept for ablation —
                       expect fallbacks/collisions near inevitable-collision
                       states.

    States advance with the same RK4 step as DubinsEnv._update_state, batched
    in torch. Actions arrive in the planner's normalized space and are
    denormalized with the planner's stats.
    """

    def __init__(
        self,
        obstacles: list[tuple[float, float, float]],
        speed: float,
        dt: float,
        act_mean: float,
        act_std: float,
        act_low: float,
        act_high: float,
        device: str = "cuda:0",
        h_source: str = "hj",
        hj_path: str = "/home/seongbin/latent/latent_cbf/results/hj_truth.npz",
    ):
        obs = torch.tensor(obstacles, dtype=torch.float32, device=device)  # (M, 3)
        self.centers = obs[:, :2]  # (M, 2)
        self.radii = obs[:, 2]  # (M,)
        self.speed = float(speed)
        self.dt = float(dt)
        self.amean, self.astd = float(act_mean), float(act_std)
        self.alow, self.ahigh = float(act_low), float(act_high)
        self.h_source = h_source
        if h_source == "hj":
            import numpy as np

            d = np.load(hj_path)
            self.V = torch.tensor(d["V"], device=device)  # (nx, ny, nth)
            xs, ys, ths = d["grid_xs"], d["grid_ys"], d["grid_thetas"]
            self.x0v, self.dxv = float(xs[0]), float(xs[1] - xs[0])
            self.y0v, self.dyv = float(ys[0]), float(ys[1] - ys[0])
            self.th0v, self.dthv = float(ths[0]), float(ths[1] - ths[0])
        elif h_source != "dist":
            raise ValueError(f"h_source must be 'hj' or 'dist', got {h_source!r}")

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., 3) true states -> h (...,)."""
        if self.h_source == "hj":
            return self._hj_value(x)
        d = torch.linalg.norm(
            x[..., None, :2] - self.centers, dim=-1
        )  # (..., M)
        return (d - self.radii).min(dim=-1).values

    def _hj_value(self, x: torch.Tensor) -> torch.Tensor:
        """Trilinear interpolation of V on the (x, y, theta) grid; theta wraps."""
        nx, ny, nth = self.V.shape
        fx = ((x[..., 0] - self.x0v) / self.dxv).clamp(0, nx - 1 - 1e-6)
        fy = ((x[..., 1] - self.y0v) / self.dyv).clamp(0, ny - 1 - 1e-6)
        th = torch.atan2(torch.sin(x[..., 2]), torch.cos(x[..., 2]))
        ft = (th - self.th0v) / self.dthv  # theta grid is periodic (endpoint=False)
        i0, j0 = fx.long(), fy.long()
        k0 = ft.floor().long() % nth
        i1, j1 = (i0 + 1).clamp(max=nx - 1), (j0 + 1).clamp(max=ny - 1)
        k1 = (k0 + 1) % nth
        wx, wy = fx - i0.float(), fy - j0.float()
        wt = (ft - ft.floor()).clamp(0, 1)
        V = self.V

        def g(i, j, k):
            return V[i, j, k]

        v00 = g(i0, j0, k0) * (1 - wt) + g(i0, j0, k1) * wt
        v01 = g(i0, j1, k0) * (1 - wt) + g(i0, j1, k1) * wt
        v10 = g(i1, j0, k0) * (1 - wt) + g(i1, j0, k1) * wt
        v11 = g(i1, j1, k0) * (1 - wt) + g(i1, j1, k1) * wt
        v0 = v00 * (1 - wy) + v01 * wy
        v1 = v10 * (1 - wy) + v11 * wy
        return v0 * (1 - wx) + v1 * wx

    def _deriv(self, x: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                self.speed * torch.cos(x[..., 2]),
                self.speed * torch.sin(x[..., 2]),
                omega,
            ],
            dim=-1,
        )

    def step(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        """One RK4 step. x: (..., 3); a_norm: (...,) normalized actions.
        Broadcasting: to expand (S,3) states over a (K,) grid, pass
        x (S,1,3) and a_norm (1,K) or (S,K)."""
        omega = (a_norm * self.astd + self.amean).clamp(self.alow, self.ahigh)
        batch = torch.broadcast_shapes(x.shape[:-1], omega.shape)
        omega = omega.expand(batch)
        x = x.expand(*batch, 3)
        dt = self.dt
        k1 = self._deriv(x, omega)
        k2 = self._deriv(x + 0.5 * dt * k1, omega)
        k3 = self._deriv(x + 0.5 * dt * k2, omega)
        k4 = self._deriv(x + dt * k3, omega)
        xn = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        theta = torch.atan2(torch.sin(xn[..., 2]), torch.cos(xn[..., 2]))
        return torch.cat([xn[..., :2], theta.unsqueeze(-1)], dim=-1)
