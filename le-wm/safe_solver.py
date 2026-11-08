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

import os
import sys

import numpy as np
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


class _LinMarginHead(torch.nn.Module):
    """Linear margin function h(z) = w.z + b (saved as {"margin_lin": ...} by train_lin_margin_dubins.py). Under recolor
    the MLP margin function flattens in the shifted region (gap/noise 1.3 at step 1 on purple rollouts) while a linear
    readout of the same latents keeps 4x more (5.0)."""

    def __init__(self, inp_dim=192):
        super().__init__()
        self.lin = torch.nn.Linear(inp_dim, 1)

    def forward(self, x):
        return self.lin(x).squeeze(-1)


def _load_margin_head(path, device):
    ck = torch.load(path, map_location=device)
    head = _LinMarginHead() if "margin_lin" in ck else _MarginHead()
    head.load_state_dict(ck["margin_lin"] if "margin_lin" in ck else ck["margin_gp"])
    return head.to(device)


class _SafeJEPA:
    """Model proxy: safe-maps candidate actions (state-space, gt h) before
    delegating cost evaluation to the wrapped JEPA unchanged."""

    def __init__(self, jepa, alpha, grid_size, h_offset, hj_path, speed, dt,
                 act_low, act_high, device, safe_mode="gt", margin_ckpt=None,
                 h_shift=0.0, recovery="instant", map_candidates=True):
        self.jepa = jepa
        self.device = device
        self.safe_mode = safe_mode
        # If False, CEM optimizes the goal cost over RAW (unmapped) candidates and
        # safety is applied only to the executed prefix in solve(). This stops the
        # CEM optimizer from selecting candidates that exploit the learned margin's
        # false-negatives (the reason safe-by-construction CEM is net-harmful with a
        # learned h, while the reactive DDPG-actor + one-shot filter is not).
        self.map_candidates = map_candidates
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
        # map_candidates=False => optimize goal cost over raw candidates (safety is
        # applied post-hoc to the executed prefix only, in SafeCEMSolver.solve).
        acts = (self.map_actions(info_dict, action_candidates)
                if self.map_candidates else action_candidates)
        return self.jepa.get_cost(info_dict, acts)


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
                 map_candidates: bool = True,
                 **kwargs):
        device = kwargs.get("device", "cuda")
        proxy = _SafeJEPA(model, safe_alpha, safe_grid, h_offset, hj_path,
                          env_speed, env_dt, act_low, act_high, device,
                          safe_mode=safe_mode, margin_ckpt=margin_ckpt,
                          h_shift=h_shift, recovery=recovery,
                          map_candidates=map_candidates)
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


class _SoftCostJEPA:
    """Model proxy: goal cost + w * sum_t relu(m - h(z_t)) over the imagined
    rollout. Unsafe-looking candidates just rank badly; no action map."""

    def __init__(self, jepa, head, weight, margin, const, goal_cost="final", h_quantile=None, h_steps=None, h_rel=False,
                 ccem=False, ccem_cost_limit=2.0, ccem_fallback="count", h_offsets=None):
        # h_offsets=[c_0, ..., c_K]: per-position threshold recalibration for CCEM (calib_pred_dubins.py): the rule reads
        # h(z_k) - c_k, k = 0 the current real frame, k >= 1 the k-th imagined latent (positions past K reuse c_K).
        self.h_offsets = None if h_offsets is None else [float(c) for c in h_offsets]
        # ccem: SafeDreamer's constrained-CEM elite rule (OSRP CCEPlanner, SafeDreamer/behaviors.py:316-391) on top of
        # LeWM's CEM: per-candidate cost = (number of states on the plan, current one included, with h < 0) * 1000/H,
        # feasible iff cost < ccem_cost_limit (SafeDreamer default 2.0). If >= topk candidates are feasible, the elites
        # are the feasible ones with the best goal cost; otherwise the topk lowest-cost candidates (ties: lower index
        # first, as jax.lax.top_k). Encoded as the score CEMSolver takes the topk smallest of. weight/const unused.
        # ccem_fallback="minh": feasible = h > 0 on every state of the plan (current included); if fewer than topk are
        # feasible, all feasible plans are kept and the rest of the elites are the infeasible plans with the highest
        # worst-case margin min_k h(z_k). ("count" = SafeDreamer: fewest states with h < 0.)
        # threshold-free rules (only the ranking of the safety score min_k h(z_k) is used, never its sign):
        # "safe_task": keep the ccem_pre_k safest candidates, elites = the topk with the best goal cost among them;
        # "task_safe": keep the ccem_pre_k with the best goal cost, elites = the topk safest among them.
        # "safe_task_norm": safe_task with h z-scored per imagined step across the candidates before the min over steps
        # (cancels a depth-dependent or OOD offset of h shared by all candidates; the current state is shared, so excluded)
        assert ccem_fallback in ("count", "minh", "safe_task", "task_safe", "safe_task_norm"), ccem_fallback
        self.ccem_pre_k = 50
        self.ccem, self.ccem_cost_limit, self.ccem_fallback, self.topk = ccem, ccem_cost_limit, ccem_fallback, None
        self.jepa, self.head, self.weight, self.margin = jepa, head, weight, margin
        self.const = const  # flat cost for ANY violation: safe beats unsafe lexicographically
        # "final": JEPA.criterion, goal distance of the last imagined step only (receding-horizon replanning
        # then keeps pushing arrival later). "mean": goal distance averaged over every imagined step.
        assert goal_cost in ("final", "mean"), goal_cost
        self.goal_cost = goal_cost
        # h_quantile=p: rank-normalize h per replan over all candidates x imagined steps, h' = (h - Q_p) / (Q_.5 - Q_p),
        # so the bottom p of imagined latents read unsafe. Label-free fix for OOD h that keeps its ranking but collapses
        # its scale (purple: safe/unsafe medians 1.56/-0.53 -> 0.38/0.32 at AUC 0.95).
        self.h_quantile = h_quantile
        # h_steps=n: safety cost on the first n imagined steps only (under purple, h ranks imagined latents for ~5
        # steps, then the predictor drifts to chance; with replanning every 5 only those steps are executed)
        self.h_steps = h_steps
        # h_rel: penalize relu(h(z_0) - h(z_t)), z_0 = current real frame: plans that go lower on h than where the agent
        # is now. Needs only h's ranking along the plan, not its sign (purple keeps the ranking, loses the sign).
        self.h_rel = h_rel

    def parameters(self):
        return self.jepa.parameters()

    def get_cost(self, info_dict, action_candidates):
        # h_jepa (optional, set by the caller): a second world model whose predicted latents h reads, e.g. the unadapted model
        # while the goal cost uses a BatchNorm-adapted one. Its rollout gets a clean copy of the inputs (get_cost mutates them).
        h_jepa = getattr(self, "h_jepa", None)
        info_h = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in info_dict.items()} if h_jepa is not None else None
        cost = self.jepa.get_cost(info_dict, action_candidates)   # (B, S)
        if h_jepa is not None:
            dev = next(h_jepa.parameters()).device
            info_h = h_jepa.rollout({k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in info_h.items()}, action_candidates.to(dev))
            info_dict = {**info_dict, "predicted_emb": info_h["predicted_emb"]}   # h reads the second model's rollout
        H = info_dict["pixels"].size(2)
        z = info_dict["predicted_emb"][:, :, H:]                  # imagined only
        if self.goal_cost == "mean":
            g = info_dict["goal_emb"]
            if g.dim() == z.dim() - 1:
                g = g.unsqueeze(1)
            g = g[..., -1:, :]                                    # same goal latent JEPA.criterion uses
            cost = (z - g.detach()).pow(2).sum(-1).mean(-1).to(cost.dtype)
        if self.ccem:
            zc = info_dict["predicted_emb"][:, :, H - 1:]          # current (real) state + imagined, as SafeDreamer's traj
            hc = self.head(zc.float())                             # (B, S, 1 + T')
            if self.h_offsets is not None:
                n = hc.size(-1)
                c = self.h_offsets[:n] + [self.h_offsets[-1]] * max(0, n - len(self.h_offsets))
                hc = hc - torch.tensor(c, device=hc.device, dtype=hc.dtype)
            if self.ccem_fallback == "safe_task_norm":
                hi = hc[..., 1:]                                   # imagined steps only (B, S, T')
                hn = (hi - hi.mean(1, keepdim=True)) / hi.std(1, keepdim=True).clamp_min(1e-6)
                worst = hn.min(-1).values                          # worst per-step-normalized margin
                J = cost.float()
                pre = min(self.ccem_pre_k, J.size(1))
                keep = torch.zeros_like(J, dtype=torch.bool).scatter_(1, torch.topk(worst, pre, dim=1).indices, True)
                self.last_h = hc
                self.last = (cost.mean().item(), worst.mean().item(), (hc.min(-1).values <= 0).float().mean().item())
                return torch.where(keep, J, J + 1e9).to(cost.dtype)
            if self.ccem_fallback in ("safe_task", "task_safe"):
                worst = hc.min(-1).values                          # safety score: worst-case margin over the plan
                J = cost.float()
                pre = min(self.ccem_pre_k, J.size(1))
                if self.ccem_fallback == "safe_task":
                    keep = torch.zeros_like(J, dtype=torch.bool).scatter_(1, torch.topk(worst, pre, dim=1).indices, True)
                    score = torch.where(keep, J, J + 1e9)          # topk smallest = best goal cost among the pre safest
                else:
                    keep = torch.zeros_like(J, dtype=torch.bool).scatter_(1, torch.topk(J, pre, dim=1, largest=False).indices, True)
                    score = torch.where(keep, -worst, torch.full_like(J, 1e9))   # topk smallest = safest among the pre best-goal
                self.last_h = hc
                self.last = (cost.mean().item(), worst.mean().item(), (worst <= 0).float().mean().item())
                return score.to(cost.dtype)
            if self.ccem_fallback == "minh":
                worst = hc.min(-1).values                          # worst-case margin over the plan
                feas = worst > 0
                enough = feas.sum(1, keepdim=True) >= self.topk
                # enough: feasible ranked by J, infeasible excluded; else: all feasible first, then highest worst-case h
                score = torch.where(feas, cost.float() - torch.where(enough, 0.0, 1e9), torch.where(enough, cost.float() + 1e9, -worst))
                self.last_h = hc
                self.last = (cost.mean().item(), worst.mean().item(), (~feas).float().mean().item())
                return score.to(cost.dtype)
            c = (hc < 0).float().sum(-1) * (1000.0 / z.size(2))    # SafeDreamer: cost.sum(0) * (1000 / horizon)
            feas = c < self.ccem_cost_limit
            enough = feas.sum(1, keepdim=True) >= self.topk
            S = c.size(1)
            tie = torch.arange(S, device=c.device, dtype=c.dtype) * 1e-6     # lower index first among equal costs
            score = torch.where(enough, torch.where(feas, cost.float(), cost.float() + 1e9), c + tie)
            self.last_h = hc
            self.last = (cost.mean().item(), c.mean().item(), (~feas).float().mean().item())
            return score.to(cost.dtype)
        h = self.head(z.float())                                  # (B, S, T')
        if self.h_steps is not None:
            h = h[..., :self.h_steps]
        # ponytail: S == 1 is _plan_h re-scoring the chosen plan; no candidate set to rank against, so traces keep raw h
        if self.h_quantile is not None and h.size(1) > 1:
            q = torch.quantile(h.flatten(1), torch.tensor([self.h_quantile, 0.5], device=h.device), dim=1)  # (2, B)
            h = (h - q[0, :, None, None]) / (q[1] - q[0]).clamp_min(1e-6)[:, None, None]
        # margin: scalar, or one value per imagined step (h on imagined latents grows more optimistic further ahead)
        m = torch.as_tensor(list(self.margin) if hasattr(self.margin, "__len__") else self.margin,
                            device=h.device, dtype=h.dtype)
        m = m if m.dim() == 0 else m[:h.size(-1)]
        if self.h_rel:   # threshold-free: margin = h of the current (real, encoded) frame; a constant OOD shift of h cancels
            m = m + self.head(info_dict["predicted_emb"][:, :, H - 1].float())[..., None]
        pen = torch.relu(m - h).sum(-1)
        viol = (h < m).any(-1)
        self.last_h = h
        self.last =(cost.mean().item(), pen.mean().item(), viol.float().mean().item())
        return cost + (self.weight * pen + self.const * viol).to(cost.dtype)


class _EliteSpread:
    """CEM callback (SOFTCEM_SPREAD=<path>): how converged are the elites when CEM stops?
    Every iteration: mean elite std. Last iteration, per env: elite std per horizon step, RMS distance from the best
    elite to the returned mean, elite cost range, all elites predicted safe (cost < const), and the share of elite
    variance on the top principal direction (near 1 with a large std = elites spread along one line, e.g. two modes)."""

    def __init__(self, n_steps, const):
        self.n_steps, self.const, self.rows, self.curve = n_steps, const, [], []
        self.output_key, self.history = "elite_spread", []              # read by CEMSolver.solve

    def reset(self): pass
    def start_batch(self): pass
    def end_solve(self): pass

    def __call__(self, step, topk_candidates, topk_vals, mean, **_):
        e = topk_candidates.float()                                      # (B, K, H, D), K sorted best first
        sd = e.std(1)                                                    # (B, H, D)
        self.curve.append((step, sd.mean((1, 2)).cpu()))
        if step != self.n_steps - 1:
            return
        c = (e - e.mean(1, keepdim=True)).flatten(2)                     # (B, K, H*D)
        s = torch.linalg.svdvals(c) ** 2
        v = topk_vals.float()
        self.rows.append(dict(sd_h=sd.mean(-1).cpu(), best_to_mean=(e[:, 0] - mean.float()).flatten(1).pow(2).mean(1).sqrt().cpu(),
                              cost_range=(v[:, -1] - v[:, 0]).cpu(), all_safe=(v[:, -1] < self.const).cpu() if self.const else None,
                              pc1=(s[:, 0] / s.sum(1).clamp_min(1e-12)).cpu()))


class SoftSafeCEMSolver(CEMSolver):
    """Plain CEM with the learned margin's unsafe region as a soft cost."""

    def __init__(self, model, margin_ckpt: str, unsafe_weight: float = 1.0,
                 unsafe_margin: float = 0.0, unsafe_const: float = 0.0, goal_cost: str = "final",
                 h_quantile: float | None = None, h_steps: int | None = None, h_rel: bool = False,
                 ccem: bool = False, ccem_cost_limit: float = 2.0, ccem_fallback: str = "count",
                 h_offsets: list | None = None, **kwargs):
        device = kwargs.get("device", "cuda")
        head = _load_margin_head(margin_ckpt, device)
        super().__init__(_SoftCostJEPA(model, head.eval(), unsafe_weight, unsafe_margin, unsafe_const, goal_cost,
                                       h_quantile, h_steps, h_rel, ccem, ccem_cost_limit, ccem_fallback,
                                       None if h_offsets is None else list(h_offsets)),
                         **kwargs)
        self.model.topk = self.topk   # the CCEM rule needs the elite count
        self._trace = []
        if os.environ.get("SOFTCEM_SPREAD"):
            self._spread = _EliteSpread(self.n_steps, unsafe_const)
            self.callbacks.append(self._spread)

    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        idx = info_dict.pop("_env_idx", None)  # set by eval.py only when SOFTCEM_TRACE is on
        out = super().solve(info_dict, init_action)
        g, p, f = self.model.last
        print(f"[soft_cem] goal={g:.3f} pen={p:.3f} frac_cand_unsafe={f:.3f}")
        if idx is not None:  # h along the returned plan's imagined rollout, per episode, one entry per replan
            self._trace.append((np.asarray(idx), self._plan_h(info_dict, out["actions"])))
            torch.save(self._trace, os.environ["SOFTCEM_TRACE"] + "_plan_h.pt")
        if os.environ.get("SOFTCEM_SPREAD"):
            torch.save({"rows": self._spread.rows, "curve": self._spread.curve}, os.environ["SOFTCEM_SPREAD"])
        return out

    def _plan_h(self, info_dict, actions):
        hs = []
        for s in range(0, len(actions), self.batch_size):   # same per-batch expansion as CEMSolver.solve, 1 sample
            exp = {}
            for k, v in info_dict.items():
                v = v[s:s + self.batch_size]
                if torch.is_tensor(v):
                    v = v.to(self.device, self.dtype if v.is_floating_point() else None).unsqueeze(1)
                elif isinstance(v, np.ndarray):
                    v = v[:, None]
                exp[k] = v
            self.model.get_cost(exp, actions[s:s + self.batch_size].to(self.device, self.dtype).unsqueeze(1))
            hs.append(self.model.last_h[:, 0].float().cpu())
        return torch.cat(hs).numpy()                          # (n_replanned, horizon)
