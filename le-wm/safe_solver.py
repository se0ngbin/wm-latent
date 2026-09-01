"""Safe-by-construction CEM solver for the swm eval protocol (le-wm/eval.py).

Wraps the loaded JEPA model so that every action sequence the CEM evaluates —
and the final executed plan — is first passed through a state-dependent safe
action map (GridGaugeMap over the discrete CBF condition with the HJ avoid
value as h). The CEM optimizes an unconstrained coordinate u; the map
transforms u onto the safe action set step by step along the trajectory. No
filtering or projection stage exists: unsafe actions are never candidates.

gt mode is pure state-space (parallel RK4 rollout + HJ interpolation), so the
latent rollout/cost path (JEPA.get_cost) is reused unchanged and the safety
map adds no predictor compute.

Usage: config/eval/solver/safe_cem.yaml points hydra at SafeCEMSolver; eval.py
injects the dataset scalers via set_process() (needed to denormalize the
z-scored 'state' and 'action' entries of the info dict).

Caveat: warm-start tails (`outputs['actions'][receding_horizon:]`) are safe-
mapped actions, not u — fine for receding_horizon == horizon (the
sg25clean200_fs1h25rec25 protocol), where the tail is empty.
"""
from __future__ import annotations

import sys

import torch

from stable_worldmodel.solver.cem import CEMSolver

# Reuse the safe-action-map module from latent_cbf (self-contained: torch+numpy).
_SAM_DIR = "/home/seongbin/latent/latent_cbf/src/latent_cbf/controllers"
if _SAM_DIR not in sys.path:
    sys.path.append(_SAM_DIR)
from safe_action_map import GridGaugeMap, GroundTruthH  # noqa: E402

OBSTACLES = [(0.25, 0.65, 0.5), (0.25, -0.65, 0.5)]


class _MarginHead(torch.nn.Module):
    """Exact replica of the dreamerv3_torch MLP margin head (192->512->512->1,
    Linear(bias=False)+LayerNorm(eps=1e-3)+SiLU x2), rebuilt here so the
    le-wm venv doesn't need latent_cbf's dreamer deps. Loads the state dicts
    saved by train_margin_hj_lewm.py verbatim (module names must match)."""

    def __init__(self, inp_dim=192, units=512, name="Margin GP"):
        super().__init__()
        self.layers = torch.nn.Sequential()
        d = inp_dim
        for i in range(2):
            self.layers.add_module(f"{name}_linear{i}",
                                   torch.nn.Linear(d, units, bias=False))
            self.layers.add_module(f"{name}_norm{i}",
                                   torch.nn.LayerNorm(units, eps=1e-03))
            self.layers.add_module(f"{name}_act{i}", torch.nn.SiLU())
            d = units
        self.mean_layer = torch.nn.Linear(units, 1)

    def forward(self, x):
        return self.mean_layer(self.layers(x)).squeeze(-1)


class _SafeJEPA:
    """Model proxy: safe-maps candidate actions (state-space, gt h) before
    delegating cost evaluation to the wrapped JEPA unchanged."""

    def __init__(self, jepa, alpha, grid_size, h_offset, hj_path, speed, dt,
                 act_low, act_high, device, safe_mode="gt", margin_ckpt=None,
                 h_shift=0.0, recovery="instant"):
        self.jepa = jepa
        self.device = device
        self.safe_mode = safe_mode
        self._alpha = alpha
        self._grid_size = grid_size
        self._h_offset = h_offset
        self._h_shift = h_shift
        self._recovery = recovery
        self._hj_path = hj_path
        self._speed, self._dt = speed, dt
        self._alow, self._ahigh = act_low, act_high
        # built lazily in set_scalers (needs the dataset action scaler)
        self.safe_map: GridGaugeMap | None = None
        self.h_fn: GroundTruthH | None = None
        self._smean = self._sstd = None  # state scaler params
        self.head = None
        if safe_mode == "learned":
            assert margin_ckpt, "safe_mode='learned' requires margin_ckpt"
            sd = torch.load(margin_ckpt, map_location=device)
            self.head = _MarginHead().to(device)
            self.head.load_state_dict(sd["margin_gp"])
            self.head.eval()
        elif safe_mode != "gt":
            raise ValueError(f"safe_mode must be gt|learned, got {safe_mode!r}")

    def parameters(self):
        return self.jepa.parameters()

    def set_scalers(self, action_scaler, state_scaler):
        amean = float(action_scaler.mean_[0])
        astd = float(action_scaler.scale_[0])
        nlo = (self._alow - amean) / astd
        nhi = (self._ahigh - amean) / astd
        self.safe_map = GridGaugeMap(
            alpha=self._alpha, nlo=nlo, nhi=nhi, grid_size=self._grid_size,
            h_offset=self._h_offset, h_shift=self._h_shift,
            recovery=self._recovery, device=self.device,
        )
        if self.safe_mode == "gt":
            self.h_fn = GroundTruthH(
                obstacles=OBSTACLES, speed=self._speed, dt=self._dt,
                act_mean=amean, act_std=astd, act_low=self._alow,
                act_high=self._ahigh, device=self.device,
                h_source="hj", hj_path=self._hj_path,
            )
        self._smean = torch.tensor(
            state_scaler.mean_, dtype=torch.float32, device=self.device)
        self._sstd = torch.tensor(
            state_scaler.scale_, dtype=torch.float32, device=self.device)

    def _x0_from_info(self, info_dict, B, S):
        """Denormalized true state (B*S, 3) from the (possibly expanded) info.
        Shapes seen: (B, S, H, 3) inside CEMSolver.solve's expanded infos,
        (B, H, 3) at the final map_actions call, (B, 3) defensively."""
        st = info_dict["state"]
        if not torch.is_tensor(st):
            st = torch.as_tensor(st)
        st = st.to(self.device, dtype=torch.float32)
        if st.dim() == 4:      # (B, S, H, 3): last history frame, flatten samples
            x = st[:, :, -1].reshape(-1, 3)
        elif st.dim() == 3:    # (B, H, 3)
            x = st[:, -1]
        else:                  # (B, 3)
            x = st
        x = x * self._sstd + self._smean  # invert StandardScaler
        if x.shape[0] == B and S > 1:
            x = x.repeat_interleave(S, 0)
        assert x.shape[0] == B * S, f"state batch {x.shape[0]} != {B * S}"
        return x

    def _hist_emb(self, info_dict, B, S):
        """Encoded pixel history (B*S, H, D); samples share the same frames,
        so encode once per env and tile."""
        px = info_dict["pixels"]
        if px.dim() == 6:  # (B, S, H, C, h, w) expanded
            px = px[:, 0]
        emb = self.jepa.encode({"pixels": px.to(self.device).float()})["emb"]
        return emb.repeat_interleave(S, 0)

    def map_actions_learned(self, info_dict, u):
        """Latent-chain variant: h = margin head on imagined latents. Mirrors
        LeWMPlannerController._imagine_safe (grid-expand + snap-to-grid)."""
        B, S, T, A = u.shape
        BS, K = B * S, self.safe_map.K
        uf = u.reshape(BS, T, A).to(self.device, dtype=torch.float32)
        hist = self._hist_emb(info_dict, B, S)              # (BS, H, D)
        H = hist.shape[1]
        a_grid = self.safe_map.a_grid
        # (K,1,A_emb) tiled to (BS*K,1,A_emb), K-major within each env block
        grid_emb = self.jepa.action_encoder(a_grid.view(K, 1, 1)).repeat(BS, 1, 1)
        hist_act = self.jepa.action_encoder(
            torch.zeros(BS, H, 1, device=self.device))       # (BS, H, A_emb)
        ar = torch.arange(BS, device=self.device)
        out = torch.empty_like(uf)
        for t in range(T):
            he_g = hist.repeat_interleave(K, 0)
            ha_g = torch.cat([hist_act.repeat_interleave(K, 0)[:, 1:], grid_emb], 1)
            pred = self.jepa.predict(he_g, ha_g)[:, -1].view(BS, K, -1)
            h_cur = self.head(hist[:, -1])                   # (BS,)
            h_next = self.head(pred)                         # (BS, K)
            idx, a_t = self.safe_map.choose(uf[:, t], h_cur, h_next)
            hist_act = ha_g.view(BS, K, H, -1)[ar, idx]      # chosen branch
            hist = torch.cat([hist[:, 1:], pred[ar, idx].unsqueeze(1)], 1)
            out[:, t] = a_t
        return out.view(B, S, T, A).to(u.dtype)

    def map_actions(self, info_dict, u):
        """u: (B, S, T, A) unconstrained z-scored coords -> safe actions,
        via per-step gauge-mapping on the true-state rollout (gt) or the
        imagined-latent chain (learned)."""
        assert self.safe_map is not None, "call set_scalers() first"
        if self.safe_mode == "learned":
            return self.map_actions_learned(info_dict, u)
        B, S, T, A = u.shape
        assert A == 1, "state-space safe map assumes 1-D dubins actions"
        BS = B * S
        uf = u.reshape(BS, T, A).to(self.device, dtype=torch.float32)
        x = self._x0_from_info(info_dict, B, S)  # (BS, 3)
        a_grid = self.safe_map.a_grid  # (K,)
        ar = torch.arange(BS, device=self.device)
        out = torch.empty_like(uf)
        for t in range(T):
            h_cur = self.h_fn.value(x)                                    # (BS,)
            x_g = self.h_fn.step(x.unsqueeze(1), a_grid.view(1, -1))      # (BS, K, 3)
            h_next = self.h_fn.value(x_g)                                 # (BS, K)
            idx, a_t = self.safe_map.choose(uf[:, t], h_cur, h_next)
            x = x_g[ar, idx]
            out[:, t] = a_t
        return out.view(B, S, T, A).to(u.dtype)

    def get_cost(self, info_dict, action_candidates):
        safe = self.map_actions(info_dict, action_candidates)
        return self.jepa.get_cost(info_dict, safe)


class SafeCEMSolver(CEMSolver):
    """CEMSolver whose candidates and final plan are safe by construction."""

    def __init__(self, model, safe_alpha: float = 0.3, safe_grid: int = 21,
                 h_offset: float = 0.0,
                 hj_path: str = "/home/seongbin/latent/latent_cbf/results/hj_truth.npz",
                 env_speed: float = 1.0, env_dt: float = 0.05,
                 act_low: float = -2.0, act_high: float = 2.0,
                 safe_mode: str = "gt", margin_ckpt: str | None = None,
                 h_shift: float = 0.0, recovery: str = "instant",
                 receding_horizon: int | None = None,
                 **kwargs):
        device = kwargs.get("device", "cuda")
        proxy = _SafeJEPA(model, safe_alpha, safe_grid, h_offset, hj_path,
                          env_speed, env_dt, act_low, act_high, device,
                          safe_mode=safe_mode, margin_ckpt=margin_ckpt,
                          h_shift=h_shift, recovery=recovery)
        # Only the first `receding_horizon` actions are executed; the tail is a
        # warm-start seed. Keep the tail in u-space so the base CEM re-optimizes it
        # in the right space next round (None => safe-map the whole plan, the old
        # receding==horizon behavior where the tail is empty).
        self.receding_horizon = receding_horizon
        super().__init__(proxy, **kwargs)

    def set_process(self, process: dict):
        self.model.set_scalers(process["action"], process.get("state"))

    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        out = super().solve(info_dict, init_action)
        # The stored mean is u-space; execution must receive its safe image.
        u = out["actions"].to(self.device).unsqueeze(1)      # (B, 1, T, A)
        safe = self.model.map_actions(info_dict, u).squeeze(1).detach().cpu()
        T = out["actions"].shape[1]
        rec = T if self.receding_horizon is None else min(int(self.receding_horizon), T)
        # Executed prefix -> safe-mapped; warm-start tail -> keep u-space raw mean.
        actions = out["actions"].clone()
        actions[:, :rec] = safe[:, :rec]
        out["actions"] = actions
        if self.model.safe_map is not None:
            print(f"[safe_cem] fallback_rate={self.model.safe_map.fallback_rate:.4f}")
        return out
