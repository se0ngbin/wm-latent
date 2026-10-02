"""Safe-by-construction CEM solver for TwoRoom (gt / 'contact = unsafe').

TwoRoom analog of safe_solver.py (Dubins). Differences:
  * Actions are 2-D (velocity), and the planner uses frameskip block=5, so each
    CEM "horizon" step is a 10-D vector = 5 stacked 2-D env-frame actions. The
    safe map unpacks blocks to 25 env-frames, rolls the true agent position, and
    2-D-gauge-maps each frame onto the safe action set, then repacks.
  * Safety value is the holonomic wall-clearance CBF from tworoom_safety.py
    (no HJ PDE needed): h(x) = dist(agent_center, wall_material) - agent_r >= 0.
  * State comes from info['proprio'] (agent x,y in pixels), z-scored by the
    dataset proprio scaler; actions z-scored by the action scaler.

gt only (per current scope): the safe set is exact on the true state, so no
predictor compute is added (cost eval is delegated to the wrapped JEPA).
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, "/home/seongbin/latent/le-wm")
from stable_worldmodel.solver.cem import CEMSolver

from tworoom_safety import TwoRoomH, DOOR_Y0, DOOR_Y1


class MarginHead(torch.nn.Module):
    """192->512->512->1 clearance-margin head (Linear(bias=False)+LN+SiLU x2 then
    Linear to 1). Same arch as the Dubins _MarginHead; trained by
    train_margin_clearance_tworoom.py, loaded here for safe_mode='learned'."""

    def __init__(self, inp_dim=192, units=512, name="Margin GP"):
        super().__init__()
        self.layers = torch.nn.Sequential()
        d = inp_dim
        for i in range(2):
            self.layers.add_module(f"{name}_linear{i}", torch.nn.Linear(d, units, bias=False))
            self.layers.add_module(f"{name}_norm{i}", torch.nn.LayerNorm(units, eps=1e-03))
            self.layers.add_module(f"{name}_act{i}", torch.nn.SiLU())
            d = units
        self.mean_layer = torch.nn.Linear(units, 1)

    def forward(self, x):
        return self.mean_layer(self.layers(x)).squeeze(-1)


class _SafeTwoRoom:
    """Model proxy: 2-D gauge-maps candidate env-frame actions on the true-state
    holonomic rollout (gt clearance h) before delegating cost to the JEPA."""

    def __init__(self, jepa, alpha, grid_size, device, speed, agent_r,
                 door_y, block, safe_mode="gt", margin_ckpt=None, qp_iters=8):
        self.jepa = jepa
        self.device = device
        self.safe_mode = safe_mode
        self._alpha = float(alpha)
        self._grid_size = int(grid_size)      # per-dim (K); total K*K grid
        self._speed = float(speed)
        self._agent_r = float(agent_r)
        self._door_y = door_y
        self.block = int(block)               # frameskip (env frames per WM step)
        self._qp_iters = int(qp_iters)        # HardNet++ correction steps (learned_qp)
        self.h_fn: TwoRoomH | None = None
        self.a_grid_norm = None               # (KK, 2) normalized grid actions
        self.a_grid_raw = None                # (KK, 2) raw [-1,1] for stepping
        self._pmean = self._pstd = None       # proprio scaler
        self.n_choices = 0
        self.n_fallback = 0
        self.head = None
        if safe_mode in ("learned", "learned_qp"):
            assert margin_ckpt, f"safe_mode={safe_mode!r} requires margin_ckpt"
            sd = torch.load(margin_ckpt, map_location=device)
            self.head = MarginHead().to(device)
            self.head.load_state_dict(sd["margin_gp"])
            self.head.eval()
        elif safe_mode != "gt":
            raise ValueError(f"safe_mode must be gt|learned|learned_qp, got {safe_mode!r}")

    def parameters(self):
        return self.jepa.parameters()

    @property
    def fallback_rate(self):
        return self.n_fallback / max(self.n_choices, 1)

    def set_scalers(self, action_scaler, proprio_scaler):
        amean = torch.tensor(action_scaler.mean_, dtype=torch.float32, device=self.device)
        astd = torch.tensor(action_scaler.scale_, dtype=torch.float32, device=self.device)
        self._amean, self._astd = amean, astd  # (2,)
        # per-dim normalized bounds for the raw action range [-1, 1]
        nlo = (-1.0 - amean) / astd
        nhi = (1.0 - amean) / astd
        K = self._grid_size
        gx = torch.linspace(nlo[0].item(), nhi[0].item(), K, device=self.device)
        gy = torch.linspace(nlo[1].item(), nhi[1].item(), K, device=self.device)
        gxx, gyy = torch.meshgrid(gx, gy, indexing="ij")
        self.a_grid_norm = torch.stack([gxx.reshape(-1), gyy.reshape(-1)], 1)  # (KK,2)
        self.a_grid_raw = (self.a_grid_norm * astd + amean).clamp(-1.0, 1.0)   # (KK,2)
        self.h_fn = TwoRoomH(agent_r=self._agent_r, speed=self._speed,
                             door_y=self._door_y, device=self.device)
        self._pmean = torch.tensor(proprio_scaler.mean_, dtype=torch.float32, device=self.device)
        self._pstd = torch.tensor(proprio_scaler.scale_, dtype=torch.float32, device=self.device)

    def _x0_from_info(self, info_dict, B, S):
        """Denormalized agent px position (B*S, 2) from info['proprio']."""
        pr = info_dict["proprio"]
        if not torch.is_tensor(pr):
            pr = torch.as_tensor(pr)
        pr = pr.to(self.device, dtype=torch.float32)
        if pr.dim() == 4:      # (B, S, H, 2): last history frame
            x = pr[:, :, -1].reshape(-1, 2)
        elif pr.dim() == 3:    # (B, H, 2)
            x = pr[:, -1]
        else:                  # (B, 2)
            x = pr
        x = x * self._pstd + self._pmean
        if x.shape[0] == B and S > 1:
            x = x.repeat_interleave(S, 0)
        assert x.shape[0] == B * S, f"proprio batch {x.shape[0]} != {B * S}"
        return x

    def _choose(self, u_t, x):
        """Pick, per row, the safe 2-D grid action nearest to nominal u_t.
        u_t: (BS, 2) normalized nominal; x: (BS, 2) px state.
        Returns (a_norm (BS,2), x_next (BS,2))."""
        BS = x.shape[0]
        KK = self.a_grid_raw.shape[0]
        h_cur = self.h_fn.value(x)                                   # (BS,)
        xg = x.unsqueeze(1).expand(BS, KK, 2)
        ag = self.a_grid_raw.unsqueeze(0).expand(BS, KK, 2)
        x_next = self.h_fn.step(xg, ag)                              # (BS, KK, 2)
        h_next = self.h_fn.value(x_next)                             # (BS, KK)
        thresh = (1.0 - self._alpha) * h_cur.clamp(min=0.0)          # (BS,)
        safe = h_next >= thresh.unsqueeze(1)                         # (BS, KK)
        # distance of each grid action to the nominal (normalized space)
        d = ((self.a_grid_norm.unsqueeze(0) - u_t.unsqueeze(1)) ** 2).sum(-1)  # (BS,KK)
        BIG = torch.finfo(d.dtype).max / 4
        d_safe = torch.where(safe, d, torch.full_like(d, BIG))
        idx = d_safe.argmin(1)                                       # (BS,)
        any_safe = safe.any(1)
        idx = torch.where(any_safe, idx, h_next.argmax(1))          # fallback: max clearance
        self.n_choices += BS
        self.n_fallback += int((~any_safe).sum().item())
        ar = torch.arange(BS, device=self.device)
        return self.a_grid_norm[idx], x_next[ar, idx]

    def _hist_emb(self, info_dict, B, S):
        """Encoded pixel history (B*S, H, D); samples share frames, tile."""
        px = info_dict["pixels"]
        if px.dim() == 6:  # (B, S, H, C, h, w) expanded
            px = px[:, 0]
        emb = self.jepa.encode({"pixels": px.to(self.device).float()})["emb"]
        return emb.repeat_interleave(S, 0)

    def _choose_learned(self, u2d, h_cur, h_next):
        """Pick per row the safe 2-D grid action nearest to nominal, using the
        margin-head values (no true state). Returns idx (BS,)."""
        BS = u2d.shape[0]
        thresh = (1.0 - self._alpha) * h_cur.clamp(min=0.0)
        safe = h_next >= thresh.unsqueeze(1)                        # (BS,K)
        d = ((self.a_grid_norm.unsqueeze(0) - u2d.unsqueeze(1)) ** 2).sum(-1)  # (BS,K)
        BIG = torch.finfo(d.dtype).max / 4
        idx = torch.where(safe, d, torch.full_like(d, BIG)).argmin(1)
        any_safe = safe.any(1)
        idx = torch.where(any_safe, idx, h_next.argmax(1))
        self.n_choices += BS
        self.n_fallback += int((~any_safe).sum().item())
        return idx

    def map_actions_learned(self, info_dict, u):
        """Per-block gauge map on imagined latents: the margin head reads the
        predictor's imagined latent, which advances one WM-step per fs=5 block.
        Grid actions are constant-velocity blocks (one 2-D action x block), so
        the 2-D grid matches the gt map. Mirrors the Dubins map_actions_learned."""
        B, S, Hb, Ablk = u.shape
        blk = self.block
        BS = B * S
        K = self.a_grid_norm.shape[0]
        uf = u.reshape(BS, Hb, blk, 2).to(self.device, dtype=torch.float32)
        u_nom = uf.mean(2)                                          # (BS,Hb,2) nominal per block
        hist = self._hist_emb(info_dict, B, S)                     # (BS,H,D)
        Hh = hist.shape[1]
        # constant-velocity grid blocks in NORMALIZED action space (predictor was
        # trained on z-scored actions), K-major within each env block.
        grid_blk = self.a_grid_norm.repeat(1, blk).view(K, 1, blk * 2)  # (K,1,10)
        grid_emb = self.jepa.action_encoder(grid_blk).repeat(BS, 1, 1)  # (BS*K,1,A_emb)
        hist_act = self.jepa.action_encoder(
            torch.zeros(BS, Hh, blk * 2, device=self.device))         # (BS,H,A_emb)
        ar = torch.arange(BS, device=self.device)
        out = torch.empty(BS, Hb, Ablk, device=self.device)
        for t in range(Hb):
            he_g = hist.repeat_interleave(K, 0)                       # (BS*K,H,D)
            ha_g = torch.cat([hist_act.repeat_interleave(K, 0)[:, 1:], grid_emb], 1)
            pred = self.jepa.predict(he_g[:, -Hh:], ha_g[:, -Hh:])[:, -1].view(BS, K, -1)
            h_cur = self.head(hist[:, -1])                           # (BS,)
            h_next = self.head(pred)                                 # (BS,K)
            idx = self._choose_learned(u_nom[:, t], h_cur, h_next)
            a_norm = self.a_grid_norm[idx]                           # (BS,2)
            out[:, t] = a_norm.repeat(1, blk)                        # const-velocity block
            hist_act = ha_g.view(BS, K, Hh, -1)[ar, idx]             # chosen branch
            hist = torch.cat([hist[:, 1:], pred[ar, idx].unsqueeze(1)], 1)
        return out.view(B, S, Hb, Ablk).to(u.dtype)

    def _qp_filter_block(self, hist, u2d):
        """Min-intervention CBF filter on ONE constant-velocity 2-D block, against
        the REAL current latent (1-step, re-grounded). Replaces grid-project with a
        HardNet++-style damped-projection: minimally perturb the nominal velocity to
        satisfy the discrete decrease bound h(z') >= (1-alpha)*max(h(z),0), using the
        differentiable margin(predict(z, enc(a))). hist: (BS,H,D) real encoded
        history; u2d: (BS,2) nominal normalized velocity. Returns a_safe (BS,2)."""
        BS, Hh = u2d.shape[0], hist.shape[1]
        blk = self.block
        # solve() runs under inference_mode, whose tensors cannot feed autograd;
        # rebuild the QP inputs as normal tensors and take gradients explicitly.
        with torch.inference_mode(False), torch.enable_grad():
            hist = hist.clone()                                        # (BS,H,D) normal
            h_cur = self.head(hist[:, -1]).detach()                    # (BS,)
            thresh = (1.0 - self._alpha) * h_cur.clamp(min=0.0)        # (BS,)
            hist_act = self.jepa.action_encoder(
                torch.zeros(BS, Hh, blk * 2, device=self.device))     # (BS,H,A_emb)
            nlo = (-1.0 - self._amean) / self._astd                   # (2,)
            nhi = (1.0 - self._amean) / self._astd                    # (2,)

            def h_next_fn(a):  # a: (BS,2) -> h_next (BS,)
                blk_a = a.repeat(1, blk).view(BS, 1, blk * 2)         # (BS,1,10) const-vel
                a_emb = self.jepa.action_encoder(blk_a)               # (BS,1,A_emb)
                ha = torch.cat([hist_act[:, 1:], a_emb], 1)           # (BS,H,A_emb)
                pred = self.jepa.predict(hist, ha)[:, -1]             # (BS,D)
                return self.head(pred)

            a = u2d.clone()
            for _ in range(self._qp_iters):
                a_v = a.detach().requires_grad_(True)
                g = h_next_fn(a_v) - thresh                           # (BS,)
                (grad,) = torch.autograd.grad(g.sum(), a_v)          # (BS,2)
                denom = (grad * grad).sum(-1, keepdim=True) + 1e-8
                step = torch.relu(-g).unsqueeze(-1) * grad / denom    # damping=1
                a = (a_v + step).detach()
                a = torch.max(torch.min(a, nhi), nlo)
            g_final = (h_next_fn(a) - thresh).detach()
        self.n_choices += BS
        self.n_fallback += int((g_final < 0).sum().item())           # residual violation
        return a.detach()

    def map_actions_qp(self, info_dict, u):
        """learned_qp: min-intervention CBF filter on the EXECUTED block only, against
        the real current latent (1-step, re-grounded). CEM optimized freely (get_cost
        does not map candidates); here we correct block 0 and leave the rest nominal.
        Pair with receding_horizon=1 so only the filtered block is executed."""
        B, S, Hb, Ablk = u.shape
        blk = self.block
        BS = B * S
        uf = u.reshape(BS, Hb, blk, 2).to(self.device, dtype=torch.float32)
        u_nom0 = uf[:, 0].mean(1)                                      # (BS,2) block-0 nominal
        hist = self._hist_emb(info_dict, B, S)                        # (BS,H,D)
        a0 = self._qp_filter_block(hist, u_nom0)                      # (BS,2)
        out = uf.clone()
        out[:, 0] = a0.unsqueeze(1).expand(BS, blk, 2)                # const-vel block
        return out.reshape(B, S, Hb, Ablk).to(u.dtype)

    def map_actions(self, info_dict, u):
        """u: (B, S, Hb, 2*block) normalized block actions -> safe. gt: per-frame
        2-D gauge map on the true holonomic rollout. learned: per-block gauge map
        on imagined latents (margin head). learned_qp: min-intervention CBF filter on
        the executed block, re-grounded."""
        assert self.h_fn is not None, "call set_scalers() first"
        if self.safe_mode == "learned_qp":
            return self.map_actions_qp(info_dict, u)
        if self.safe_mode == "learned":
            return self.map_actions_learned(info_dict, u)
        B, S, Hb, Ablk = u.shape
        blk = self.block
        assert Ablk == 2 * blk, f"expected {2*blk}-D blocks, got {Ablk}"
        BS = B * S
        uf = u.reshape(BS, Hb, blk, 2).reshape(BS, Hb * blk, 2)     # (BS, 25, 2)
        uf = uf.to(self.device, dtype=torch.float32)
        x = self._x0_from_info(info_dict, B, S)                     # (BS, 2)
        T = Hb * blk
        out = torch.empty_like(uf)
        for t in range(T):
            a_norm, x = self._choose(uf[:, t], x)
            out[:, t] = a_norm
        out = out.reshape(BS, Hb, blk * 2).reshape(B, S, Hb, Ablk)
        return out.to(u.dtype)

    def get_cost(self, info_dict, action_candidates):
        # learned_qp: CEM optimizes freely; the CBF filter is applied only to the
        # executed action in solve() (min-intervention, 1-step, re-grounded).
        if self.safe_mode == "learned_qp":
            return self.jepa.get_cost(info_dict, action_candidates)
        safe = self.map_actions(info_dict, action_candidates)
        return self.jepa.get_cost(info_dict, safe)


class SafeTwoRoomCEMSolver(CEMSolver):
    """CEMSolver whose TwoRoom candidates + final plan are safe by construction."""

    def __init__(self, model, safe_alpha: float = 0.3, safe_grid: int = 11,
                 env_speed: float = 5.0, agent_radius: float = 7.0,
                 door_y0: float = float(DOOR_Y0), door_y1: float = float(DOOR_Y1),
                 block: int = 5, receding_horizon: int | None = None,
                 safe_mode: str = "gt", margin_ckpt: str | None = None,
                 qp_iters: int = 8, **kwargs):
        device = kwargs.get("device", "cuda")
        proxy = _SafeTwoRoom(model, safe_alpha, safe_grid, device, env_speed,
                             agent_radius, (door_y0, door_y1), block,
                             safe_mode=safe_mode, margin_ckpt=margin_ckpt,
                             qp_iters=qp_iters)
        self.receding_horizon = receding_horizon
        super().__init__(proxy, **kwargs)

    def set_process(self, process: dict):
        self.model.set_scalers(process["action"], process["proprio"])

    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        out = super().solve(info_dict, init_action)
        u = out["actions"].to(self.device).unsqueeze(1)      # (B, 1, Hb, Ablk)
        safe = self.model.map_actions(info_dict, u).squeeze(1).detach().cpu()
        Hb = out["actions"].shape[1]
        rec = Hb if self.receding_horizon is None else min(int(self.receding_horizon), Hb)
        actions = out["actions"].clone()
        actions[:, :rec] = safe[:, :rec]
        out["actions"] = actions
        print(f"[safe_tworoom] fallback_rate={self.model.fallback_rate:.4f}")
        return out
