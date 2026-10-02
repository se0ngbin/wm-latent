"""LE-WM planner controller for the online Dubins task.

Runs a CEM planner in the latent space of a frozen LE-WM (JEPA) world model,
matching the online protocol of the diffusion/MPPI controllers in
run_experiment.py: encode the recent frame history, sample action sequences,
roll them out through the world model, pick the first action of the sequence
whose predicted latent is closest to a goal-image latent. No CBF filter, no
obstacle term — the only signal is latent distance to the goal image.

Frameskip-aware ("macro clock"): the model consumes action *blocks* of
``A = frameskip * action_dim`` values, one prediction = one ``frameskip``-env-step
jump, and history frames are spaced ``frameskip`` env-steps apart. The controller
plans a sequence of macro-actions at each macro boundary and executes the block's
``frameskip`` raw actions one env-step at a time before replanning. For frameskip=1
(A=1) this reduces to per-step planning.

Actions are planned in the model's *normalized* action space (LE-WM trains on
z-scored actions) and denormalized before being returned to the env.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

# le-wm (jepa.py / module.py) must be importable for load_pretrained to
# instantiate the JEPA checkpoint.
for _p in ("/home/seongbin/latent", "/home/seongbin/latent/le-wm"):
    if _p not in sys.path:
        sys.path.append(_p)


class LeWMPlannerController:
    def __init__(
        self,
        lewm_ckpt: str,
        cache_dir: str | None = None,
        device: str = "cuda:0",
        horizon: int = 8,
        num_samples: int = 300,
        n_iters: int = 5,
        topk: int = 30,
        var_scale: float = 1.0,
        action_dim: int = 1,
        action_low: float = -2.0,
        action_high: float = 2.0,
        act_mean: float = -0.02112957,
        act_std: float = 1.1531051,
        cost_mode: str = "terminal",
        seed: int = 0,
        # --- safe-by-construction action map (all off by default) ---
        safe_mode: str = "off",        # "off" | "learned" | "gt"
        safe_backend: str = "grid",    # "grid" | "hardnet"
        margin_ckpt: str | None = None,
        margin_head: str = "margin_gp",
        cbf_alpha: float = 0.3,
        safe_grid: int = 21,
        gt_h: str = "hj",  # "hj" | "dist" (gt mode only)
        hardnet_iters: int = 3,
        hardnet_damping: float = 1.0,
        h_offset: float = 0.0,
        penalty_weight: float = 0.0,
    ):
        from configs import DreamerConfig
        from latent_cbf.adapters import LEWMWorldModel

        cfg = DreamerConfig()
        cfg.device = device
        cfg.lewm_cache_dir = cache_dir
        self.wm = LEWMWorldModel(cfg, lewm_ckpt).to(device).eval()
        self.device = device
        self.H = int(self.wm.history_size)
        self.D = int(self.wm.embed_dim)

        # Block dim A = action-encoder input channels = frameskip * action_dim.
        self.A = int(self.wm.jepa.action_encoder.patch_embed.in_channels)
        self.action_dim = action_dim
        self.block_steps = self.A // self.action_dim  # env-steps per macro (=frameskip)

        self.horizon = horizon
        self.num_samples = num_samples
        self.n_iters = n_iters
        self.topk = topk
        self.var_scale = var_scale
        self.alow, self.ahigh = action_low, action_high
        self.amean, self.astd = act_mean, act_std
        self.nlo = (action_low - act_mean) / act_std
        self.nhi = (action_high - act_mean) / act_std
        self.cost_mode = cost_mode
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.goal_emb = None

        # --- safety wiring ---
        from latent_cbf.controllers.safe_action_map import (
            GridGaugeMap,
            GroundTruthH,
            HardNetPPMap,
            LearnedMarginH,
        )
        from configs import get_default_config

        self.safe_mode = safe_mode
        self.safe_backend = safe_backend
        self.penalty_weight = float(penalty_weight)
        self.h_fn = None
        self.safe_map = None
        if margin_ckpt:  # needed for safe_mode=="learned" and the penalty baseline
            sd = torch.load(margin_ckpt, map_location=device)
            for name in ("margin_gp", "margin_nogp"):
                if name in sd:
                    self.wm.heads[name].load_state_dict(sd[name])
        if safe_mode == "learned":
            assert margin_ckpt, "safe_mode='learned' requires margin_ckpt"
            self.h_fn = LearnedMarginH(self.wm, head=margin_head)
        elif safe_mode == "gt":
            env_cfg = get_default_config().environment
            self.h_fn = GroundTruthH(
                obstacles=env_cfg.get_obstacles_list(),
                speed=env_cfg.speed,
                dt=env_cfg.dt,
                act_mean=act_mean,
                act_std=act_std,
                act_low=action_low,
                act_high=action_high,
                device=device,
                h_source=gt_h,
            )
        elif safe_mode != "off":
            raise ValueError(f"safe_mode must be off|learned|gt, got {safe_mode!r}")
        if self.safe_on:
            if safe_backend == "grid":
                assert self.A == 1, "GridGaugeMap requires 1-D action blocks (frameskip=1)"
                self.safe_map = GridGaugeMap(
                    alpha=cbf_alpha, nlo=self.nlo, nhi=self.nhi,
                    grid_size=safe_grid, h_offset=h_offset, device=device,
                )
            elif safe_backend == "hardnet":
                self.safe_map = HardNetPPMap(
                    alpha=cbf_alpha, nlo=self.nlo, nhi=self.nhi,
                    iters=hardnet_iters, damping=hardnet_damping,
                    h_offset=h_offset,
                )
            else:
                raise ValueError(f"safe_backend must be grid|hardnet, got {safe_backend!r}")
        if self.penalty_weight > 0:
            assert margin_ckpt, "penalty_weight>0 requires margin_ckpt"
            self.penalty_h = LearnedMarginH(self.wm, head=margin_head)
        self._x0 = None  # true state (gt mode), set in compute_action
        self.reset()

    @property
    def safe_on(self) -> bool:
        return self.safe_mode in ("learned", "gt")

    def reset(self):
        self.key_frames: list[np.ndarray] = []        # frames at macro boundaries
        self.macro_actions: list[np.ndarray] = []      # executed normalized blocks (len A)
        self.action_queue: list[float] = []            # remaining raw actions of current macro
        self._mean = None

    # ---- encoding ----
    @torch.inference_mode()
    def _encode(self, key_frames: list[np.ndarray]) -> torch.Tensor:
        fr = key_frames[-self.H:]
        while len(fr) < self.H:  # left-pad with earliest keyframe at episode start
            fr = [fr[0]] + fr
        img = np.stack(fr, 0)[None]  # (1, H, Hh, Ww, 3)
        data = self.wm.preprocess({"image": torch.from_numpy(img).float()})
        return self.wm.encoder(data)  # (1, H, D)

    @torch.inference_mode()
    def set_goal(self, goal_img: np.ndarray):
        img = np.asarray(goal_img)[None, None]
        data = self.wm.preprocess({"image": torch.from_numpy(img).float()})
        self.goal_emb = self.wm.encoder(data)[:, -1]  # (1, D)

    def _state_from_history(self, hist_emb: torch.Tensor) -> dict:
        # action blocks aligned to the H history keyframes; last block is a
        # placeholder (imagine_with_action overwrites it with the candidate).
        blocks = list(self.macro_actions[-(self.H - 1):]) if self.H > 1 else []
        blocks = blocks + [np.zeros(self.A, dtype=np.float32)]
        while len(blocks) < self.H:
            blocks = [np.zeros(self.A, dtype=np.float32)] + blocks
        a = torch.tensor(np.stack(blocks), device=self.device, dtype=torch.float32).view(1, self.H, self.A)
        hist_act = self.wm.action_encoder(a)  # (1, H, A_emb)
        return {
            "hist_emb": hist_emb,
            "hist_act": hist_act,
            "deter": hist_emb[:, -1],
            "stoch": torch.zeros(1, 1, device=self.device),
        }

    def _step_next(self, cur_he, cur_ha, a):
        """One latent dynamics step for actions a (S, A). Differentiable in a.
        Returns (next_emb (S, D), new_hist_act (S, H, A_emb))."""
        a_emb = self.wm.action_encoder(a.unsqueeze(1))  # (S, 1, A_emb)
        ha = torch.cat([cur_ha[:, 1:], a_emb], 1)
        pred = self.wm.predict(cur_he, ha)
        return pred[:, -1], ha

    @torch.no_grad()
    def _imagine_safe_hardnet(self, u, state_S, x0=None):
        """HardNet++ backend: per step, iteratively correct the raw action
        along the local constraint gradient (dimension-agnostic; no grid)."""
        S, T, A = u.shape
        # clone: history may be an inference tensor (from _encode); backward
        # through the predictor needs ordinary tensors.
        cur_he = state_S["hist_emb"].clone()
        cur_ha = state_S["hist_act"].clone()
        x = x0
        deters, a_seq = [], []
        for t in range(T):
            if self.safe_mode == "gt":
                h_cur = self.h_fn.value(x)

                def h_next_fn(a, _x=x):
                    return self.h_fn.value(self.h_fn.step(_x, a.squeeze(-1)))
            else:
                h_cur = self.h_fn.value(cur_he[:, -1])

                def h_next_fn(a, _he=cur_he, _ha=cur_ha):
                    return self.h_fn.value(self._step_next(_he, _ha, a)[0])

            a_t, _ = self.safe_map.choose(u[:, t], h_cur, h_next_fn)  # (S, A)
            next_emb, cur_ha = self._step_next(cur_he, cur_ha, a_t)
            cur_he = torch.cat([cur_he[:, 1:], next_emb.unsqueeze(1)], 1)
            if x is not None:
                x = self.h_fn.step(x, a_t.squeeze(-1))
            deters.append(next_emb)
            a_seq.append(a_t)
        return torch.stack(deters, 1), torch.stack(a_seq, 1)

    @torch.no_grad()
    def _imagine_safe(self, u, state_S, x0=None):
        """Safe-by-construction rollout: at each step the raw CEM coordinate
        u_t is gauge-mapped onto the safe action set before stepping dynamics.

        u:       (S, T, 1) unconstrained coordinates in [nlo, nhi]
        state_S: root state dict tiled to S
        x0:      (S, 3) true states (gt mode only)
        Returns (deters (S, T, D), a_safe (S, T, 1)).
        """
        if self.safe_backend == "hardnet":
            return self._imagine_safe_hardnet(u, state_S, x0)
        S, T, _ = u.shape
        K = self.safe_map.K
        a_grid = self.safe_map.a_grid  # (K,)
        # grid action embeddings, tiled to (S*K, 1, A_emb)
        grid_emb = self.wm.action_encoder(a_grid.view(K, 1, 1)).repeat(S, 1, 1)

        cur_he = state_S["hist_emb"]   # (S, H, D)
        cur_ha = state_S["hist_act"]   # (S, H, A_emb)
        x = x0                          # (S, 3) or None
        ar = torch.arange(S, device=u.device)
        deters, a_seq = [], []
        for t in range(T):
            # one predictor forward for all K grid actions per sample
            he_g = cur_he.repeat_interleave(K, 0)                    # (S*K, H, D)
            ha_g = torch.cat([cur_ha.repeat_interleave(K, 0)[:, 1:], grid_emb], 1)
            pred = self.wm.predict(he_g, ha_g)                       # (S*K, H, D)
            next_g = pred[:, -1].view(S, K, -1)                      # (S, K, D)

            if self.safe_mode == "gt":
                x_g = self.h_fn.step(x.unsqueeze(1), a_grid.view(1, K))  # (S, K, 3)
                h_cur = self.h_fn.value(x)                                # (S,)
                h_next = self.h_fn.value(x_g)                             # (S, K)
            else:  # learned
                h_cur = self.h_fn.value(cur_he[:, -1])
                h_next = self.h_fn.value(next_g)

            idx, a_t = self.safe_map.choose(u[:, t], h_cur, h_next)  # (S,), (S,1)

            next_emb = next_g[ar, idx]                                # (S, D)
            cur_ha = ha_g.view(S, K, self.H, -1)[ar, idx]             # chosen branch
            cur_he = torch.cat([cur_he[:, 1:], next_emb.unsqueeze(1)], 1)
            if x is not None:
                x = x_g[ar, idx]
            deters.append(next_emb)
            a_seq.append(a_t)
        return torch.stack(deters, 1), torch.stack(a_seq, 1)

    @torch.no_grad()
    def _plan(self, hist_emb: torch.Tensor) -> torch.Tensor:
        state = self._state_from_history(hist_emb)
        S, T, A = self.num_samples, self.horizon, self.A
        mean = self._mean if self._mean is not None else torch.zeros(T, A, device=self.device)
        var = torch.full((T, A), float(self.var_scale), device=self.device)
        goal = self.goal_emb  # (1, D)

        x0 = None
        if self.safe_mode == "gt":
            x0 = torch.tensor(self._x0, device=self.device, dtype=torch.float32).view(1, 3)

        for _ in range(self.n_iters):
            noise = torch.randn(S, T, A, generator=self.gen, device=self.device)
            cand = (mean.unsqueeze(0) + noise * var.sqrt().unsqueeze(0)).clamp(self.nlo, self.nhi)
            state_S = {k: v.repeat(S, *([1] * (v.dim() - 1))) for k, v in state.items()}
            if self.safe_on:
                deters, _ = self._imagine_safe(
                    cand, state_S, x0.repeat(S, 1) if x0 is not None else None
                )
            else:
                roll = self.wm.dynamics.imagine_with_action(cand, state_S)
                deters = roll["deter"]  # (S, T, D)
            if self.cost_mode == "min":
                cost = ((deters - goal.unsqueeze(1)) ** 2).sum(-1).min(dim=1).values
            else:  # terminal
                cost = ((deters[:, -1] - goal) ** 2).sum(-1)
            if self.penalty_weight > 0 and not self.safe_on:
                cost = cost + self.penalty_weight * torch.relu(
                    -self.penalty_h.value(deters)
                ).sum(1)
            elite = cand[cost.topk(self.topk, largest=False).indices]  # (topk, T, A)
            mean = elite.mean(0)
            var = elite.var(0).clamp_min(1e-6)

        # warm-start: shift the plan one macro-step forward
        self._mean = torch.cat([mean[1:], torch.zeros(1, A, device=self.device)], 0)
        if self.safe_on:
            # emit the safe-mapped image of the refit mean: the executed action
            # is in S(z_root) by construction.
            _, a_safe = self._imagine_safe(mean.unsqueeze(0), state, x0)
            return a_safe[0]  # (T, A) normalized, safe
        return mean  # (T, A) normalized

    def _pop(self) -> np.ndarray:
        norm_a = self.action_queue.pop(0)
        raw = float(np.clip(norm_a * self.astd + self.amean, self.alow, self.ahigh))
        return np.array([raw], dtype=np.float32)

    def compute_action(self, info, obs):
        if self.safe_mode == "gt":
            self._x0 = [*np.asarray(info["agent_position"], dtype=float),
                        float(info["agent_orientation"])]
        if self.action_queue:  # still executing the current macro block
            return self._pop()
        # macro boundary: current obs is a keyframe -> replan
        self.key_frames.append(np.asarray(obs))
        hist_emb = self._encode(self.key_frames)
        plan = self._plan(hist_emb)                 # (T, A) normalized
        first = plan[0].detach().cpu().numpy().astype(np.float32)  # (A,)
        self.macro_actions.append(first)
        self.action_queue = list(first)             # A raw-normalized steering values
        return self._pop()
