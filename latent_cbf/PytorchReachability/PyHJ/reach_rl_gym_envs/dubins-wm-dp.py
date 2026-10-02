from os import path
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import io
from PIL import Image
import matplotlib.patches as patches
import torch
import math
class Dubins_WM_DP_Env(gym.Env):
    # TODO: 1. baseline over approximation; 2. our critic loss drop faster 
    def __init__(self, params):
        
        if len(params) == 1:
            config = params[0]
        else:
            wm = params[0]
            past_data = params[1]
            config = params[2]
            dp = params[3]
            self.set_wm(wm, past_data, config, dp)

        self.render_mode = None
        self.device = 'cuda:0'
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(544,), dtype=np.float32)
        image_size = config.size[0] #128
        img_obs_space = gym.spaces.Box(
                low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
            )
        obs_space = gym.spaces.Box(
                low=-1., high=1., shape=(2,), dtype=np.float32
            )
        bool_space = gym.spaces.Box(
                low=0., high=1., shape=(1,)
            )
        self.observation_space_full = gym.spaces.Dict({
            'image': img_obs_space,
            'obs_state': obs_space,
            'is_first': bool_space,
            'is_last': bool_space,
            'is_terminal': bool_space,
        })
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32) # joint action space
        self.image_size=config.size[0]
        self.turnRate = config.turnRate
        self.no_gp = config.no_gp

    def set_wm(self, wm, past_data, config, dp):
        self.device = config.device
        self.encoder = wm.encoder.to(self.device) if hasattr(wm.encoder, "to") else wm.encoder
        self.wm = wm.to(self.device)
        self.data = past_data
        self.dp = dp
        # Feat size is set by the WM backend: LE-WM exposes `embed_dim` (e.g.
        # 192 for ViT-tiny); Dreamer V3 uses dyn_stoch (+ dyn_discrete) + dyn_deter.
        if hasattr(wm, "embed_dim") and wm.embed_dim:
            self.feat_size = int(wm.embed_dim)
        elif config.dyn_discrete:
            self.feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            self.feat_size = config.dyn_stoch + config.dyn_deter
        # Resize observation_space so DDPG critic/actor get the right input dim.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.feat_size,), dtype=np.float32
        )
        # Reset-distribution mode: 'buffer' (original) or 'uniform' (sample
        # (x,y,θ) uniformly over the safe region). Uniform gives the DDPG
        # critic broad state-space coverage.
        self.uniform_reset = bool(getattr(config, "uniform_reset", False))
        if self.uniform_reset:
            self._ensure_render_env()
        # Reward shaping: multiply the (tanh'd) margin signal so the critic
        # gets a stronger Bellman target. Default 1.0 keeps original behavior.
        self.reward_scale = float(getattr(config, "reward_scale", 1.0))
        # OOD appearance override for the RETRAINED value-function column: re-render
        # the reset-seed images under a shifted obstacle appearance so the DDPG
        # critic is trained on the shifted latent distribution (with a shifted
        # margin). Env vars: CRITIC_OBST_COLOR (e.g. purple), CRITIC_OBST_SHAPE
        # (diamond), CRITIC_ROT_DEG (0/90/...). Matches critic_ood_eval's render.
        import os as _os
        self._ood_color = (_os.environ.get("CRITIC_OBST_COLOR", "").strip() or None)
        _shp = (_os.environ.get("CRITIC_OBST_SHAPE", "").strip() or None)
        self._ood_diamond = (_shp in ("diamond", "square"))
        self._ood_rot = int(_os.environ.get("CRITIC_ROT_DEG", "0") or 0)
        self._ood_on = bool(self._ood_color or self._ood_diamond or self._ood_rot)
        if self._ood_on:
            self._ensure_render_env()
            print(f"[dubins-wm] OOD reset appearance: color={self._ood_color} "
                  f"diamond={self._ood_diamond} rot={self._ood_rot}")
        # GT-margin ablation: instead of reading the LEARNED margin head off the
        # imagined latent, track the TRUE (x,y,θ) alongside the latent rollout
        # (frameskip=1, one imagine = one RK4 step at omega = action*turnRate) and
        # use the ground-truth signed distance as the reward. Isolates the
        # PREDICTOR's role: the critic still reads the learned latent feature and
        # bootstraps over the learned dynamics, only the reward is oracle-clean.
        self._gt_margin = bool(int(_os.environ.get("CRITIC_GT_MARGIN", "0") or 0))
        if self._gt_margin:
            self._ensure_render_env()
            self._gt_state = None
            print("[dubins-wm] GT-margin reward ON (predictor-isolation ablation)")

    def _gt_reward(self):
        """tanh(signed distance to the nearest obstacle boundary) at self._gt_state;
        sign>=0 safe, matching the learned margin's tanh'd reward convention."""
        x, y, _ = self._gt_state
        g = min(float(np.hypot(x - cx, y - cy) - r) for (cx, cy, r) in self._obstacles)
        return float(np.tanh(g))

    def _gt_integrate(self, omega):
        """One RK4 dubins step (dt, speed from the render env) with turn rate omega."""
        env = self._render_env
        dt, spd = env.dt, env.speed
        s = self._gt_state
        def f(st):
            return np.array([spd * np.cos(st[2]), spd * np.sin(st[2]), omega], np.float32)
        k1 = f(s); k2 = f(s + 0.5 * dt * k1); k3 = f(s + 0.5 * dt * k2); k4 = f(s + dt * k3)
        self._gt_state = (s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)).astype(np.float32)
    
    def step(self, action):
        ac = self.action_buffer.pop(0)
        if ac is not None:
            action = np.array([ac])/self.turnRate

        assert action <= 1, f"raw {ac}, new {action}"
        assert action >= -1, f"raw {ac}, new {action}"

        init = {k: v[:, -1] for k, v in self.latent.items()}
        ac_torch = torch.tensor([[action]], dtype=torch.float32).to(self.device)*self.turnRate
        self.latent = self.wm.dynamics.imagine_with_action(ac_torch, init)
        rew, cont = self.safety_margin(self.latent) # rew is negative if unsafe
        if self._gt_margin:
            self._gt_integrate(float(action) * self.turnRate)
            rew = self._gt_reward()

        self.feat = self.wm.dynamics.get_feat(self.latent).detach().cpu().numpy()

        if len(self.action_buffer) == 0:
            truncated = True
        else:
            truncated = False
        terminated = False


        ac_unnorm = action
        info = {"is_first":False, "is_terminal":terminated, 'action': ac_unnorm}
        return np.copy(self.feat), self.reward_scale * rew, terminated, truncated, info
    
    def _ensure_render_env(self):
        """Lazily instantiate a real DubinsEnv used only for rendering single
        (x, y, θ) states during uniform-reset sampling. Not used in step()."""
        if getattr(self, "_render_env", None) is not None:
            return self._render_env
        # Import here so the dubins-wm env stays importable without the
        # latent_cbf source tree on PYTHONPATH for callers that only need step().
        import sys as _sys
        from pathlib import Path as _Path
        # latent_cbf/src on path so `latent_cbf.dubins` resolves.
        _src = _Path(__file__).resolve().parents[3] / "src"
        if str(_src) not in _sys.path:
            _sys.path.append(str(_src))
        from latent_cbf.dubins.dubins_env import DubinsEnv as _DubinsEnv  # noqa: E402
        from latent_cbf.configs import Config as _EnvConfig  # noqa: E402
        ec = _EnvConfig().environment
        self._render_env = _DubinsEnv(
            image_size=tuple(ec.image_size),
            world_bounds=tuple(ec.world_bounds),
            max_angular_velocity=ec.max_angular_velocity,
            speed=ec.speed, dt=ec.dt,
            obstacles=ec.get_obstacles_list(),
            goal_radius=ec.goal_radius,
            collision_radius=ec.collision_radius,
            render_mode="rgb_array",
        )
        self._obstacles = ec.get_obstacles_list()
        self._world_bounds = tuple(ec.world_bounds)
        return self._render_env

    def _render_diamond(self, env):
        """Render the obstacle as an L1 diamond (matches critic_ood_eval + the
        diamond failure label). Same geometry as env._render_image otherwise."""
        scale = 4; h = (env.image_size[0] * scale, env.image_size[1] * scale)
        img = Image.new("RGB", h, env.colors["background"]); from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        def w2p(c):
            x, y = c
            return (int((x - env.x_min) / (env.x_max - env.x_min) * h[0]),
                    int((env.y_max - y) / (env.y_max - env.y_min) * h[1]))
        for ox, oy, r in env.obstacles:
            cx, cy = w2p((ox, oy)); rp = r / (env.x_max - env.x_min) * h[0]
            draw.polygon([(cx, cy - rp), (cx + rp, cy), (cx, cy + rp), (cx - rp, cy)],
                         fill=env.colors["obstacle"])
        gc = w2p(env.goal_position); gr = env.goal_radius / (env.x_max - env.x_min) * h[0]
        draw.ellipse([(gc[0] - gr, gc[1] - gr), (gc[0] + gr, gc[1] + gr)], fill=env.colors["goal"])
        env._draw_agent(draw, w2p(env.state[:2]), float(env.state[2]), scale)
        return np.array(img.resize(env.image_size, Image.Resampling.LANCZOS))

    def _reappearance(self, init_traj):
        """Re-render init_traj['image'] from init_traj['state'] under the OOD
        obstacle appearance so the seeded latent lives in the shifted region."""
        if not self._ood_on:
            return init_traj
        env = self._ensure_render_env()
        env.colors = {"background": "white", "agent": "blue", "goal": "green",
                      "obstacle": self._ood_color or "red"}
        # Full (x, y, theta) lives under privileged_state; "state" is theta only.
        imgs = init_traj["image"]; states = init_traj["privileged_state"]
        T = imgs.shape[1]; H, W = imgs.shape[2], imgs.shape[3]
        new = np.empty_like(imgs)
        for t in range(T):
            env.state = np.asarray(states[0, t], dtype=np.float32)
            im = self._render_diamond(env) if self._ood_diamond else env.render()
            if im.shape[0] != H or im.shape[1] != W:
                im = np.array(Image.fromarray(im).resize((W, H)))
            if self._ood_rot:
                im = np.ascontiguousarray(np.rot90(im, k=self._ood_rot // 90))
            new[0, t] = im
        init_traj = dict(init_traj); init_traj["image"] = new
        return init_traj

    def _sample_uniform_state(self, rng=None):
        """Draw (x, y, θ) uniformly within world bounds, rejecting points
        inside any obstacle. Hot path during DDPG training — keep it tight."""
        rng = rng or np.random
        x_min, x_max, y_min, y_max = self._world_bounds
        for _ in range(64):
            x = rng.uniform(x_min, x_max)
            y = rng.uniform(y_min, y_max)
            in_obs = False
            for (cx, cy, r) in self._obstacles:
                if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                    in_obs = True; break
            if not in_obs:
                theta = rng.uniform(-np.pi, np.pi)
                return np.array([x, y, theta], dtype=np.float32)
        # rare: rejection failed, return a fallback corner state
        return np.array([x_min + 0.1, y_min + 0.1, 0.0], dtype=np.float32)

    def _build_init_traj_from_state(self, state, hist=3):
        """Construct an init_traj dict (same shape as a buffer chunk) where the
        agent has been stationary at `state` for `hist` frames. Used by the
        uniform-reset path to seed the latent without buffer trajectories."""
        env = self._ensure_render_env()
        env.state = state.astype(np.float32)
        img = env.render()  # (H, W, 3) uint8
        images = np.broadcast_to(img, (1, hist, *img.shape)).copy()
        x, y, th = float(state[0]), float(state[1]), float(state[2])
        states = np.broadcast_to(
            np.array([x, y, th], dtype=np.float32), (1, hist, 3)
        ).copy()
        obs_state = np.broadcast_to(
            np.array([np.cos(th), np.sin(th)], dtype=np.float32), (1, hist, 2)
        ).copy()
        actions = np.zeros((1, hist, 1), dtype=np.float32)
        is_first = np.zeros((1, hist), dtype=np.float32); is_first[0, 0] = 1.0
        return {
            "image": images, "state": states, "obs_state": obs_state,
            "action": actions, "is_first": is_first,
            "is_terminal": np.zeros((1, hist), dtype=np.float32),
            "reward": np.zeros((1, hist), dtype=np.float32),
        }

    def reset(self, initial_state=None, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Uniform reset: sample (x,y,θ) across the full state space (excluding
        # obstacle interiors) to give the DDPG critic broad coverage. Falls
        # back to the buffer path if it's not enabled.
        if getattr(self, "uniform_reset", False):
            state = self._sample_uniform_state()
            init_traj = self._build_init_traj_from_state(state)
            self.action_buffer = [None] * 8
        else:
            init_traj = next(self.data)
            if self.dp is not None and np.random.rand() < 0.5:
                dp_img = init_traj['image'][0, -1]
                info = {'agent_orientation': init_traj['state'][0, -1]}
                ac = self.dp.compute_action(info, dp_img)
                self.action_buffer = list(np.concatenate([[ac]] + self.dp.action_buffer))
            else:
                self.action_buffer = [None] * 8

        if getattr(self, "_gt_margin", False):
            # Full (x,y,θ) lives under privileged_state in the buffer path; the
            # uniform-reset path stores it under "state" (theta-only in the buffer).
            _sk = "privileged_state" if "privileged_state" in init_traj else "state"
            self._gt_state = np.asarray(init_traj[_sk][0, -1], np.float32).reshape(-1)[:3].copy()

        init_traj = self._reappearance(init_traj)
        data = self.wm.preprocess(init_traj)
        embed = self.encoder(data)
        self.latent, _ = self.wm.dynamics.observe(
            embed, data["action"], data["is_first"]
        )

        for k, v in self.latent.items():
            self.latent[k] = v[:, [-1]]
        self.feat = self.wm.dynamics.get_feat(self.latent).detach().cpu().numpy()
        return np.copy(self.feat), {"is_first": True, "is_terminal": False}
      

    def safety_margin(self, state):
        g_xList = []
        
        feat = self.wm.dynamics.get_feat(state).detach()
        cont = self.wm.heads["cont"](feat)

        if self.no_gp:
            with torch.no_grad():
                    outputs = torch.tanh(self.wm.heads["margin_nogp"](feat))
                    g_xList.append(outputs.detach().cpu().numpy())
        else:
            with torch.no_grad():
                outputs = torch.tanh(self.wm.heads["margin_gp"](feat))
                g_xList.append(outputs.detach().cpu().numpy())
        safety_margin = np.array(g_xList).squeeze()

        return safety_margin, cont.mean.squeeze().detach().cpu().numpy()
    