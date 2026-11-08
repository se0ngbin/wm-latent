"""Minimal swm-compatible Dubins-car env for planning with LE-WM.

Reproduces (pixel-for-pixel) the renderer + RK4 dynamics that generated
``dubins_expert.h5`` (latent_cbf's ``DubinsEnv``), but exposes the narrow
interface le-wm's dataset-driven ``World.evaluate`` needs:

- ``reset`` / ``step`` return ``obs = {"proprio", "state"}`` plus ``info["goal"]``.
- ``_set_state([x, y, theta])`` teleports the agent (called from the eval callable).
- ``_set_goal_state([x, y, theta])`` stores the goal config and places the green
  goal marker there, so rendered frames match the dataset goal image (which draws
  the goal at the episode's true target). ``step`` terminates when the agent is
  within ``success_tol`` of that goal position — that boolean is what
  ``World.evaluate`` scores as success.

Registered as ``swm/Dubins-v0`` on import.
"""
from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image, ImageDraw

# Fixed world matching dubins_expert.h5 collection (latent_cbf EnvironmentConfig).
WORLD_BOUNDS = (-1.5, 1.5, -1.5, 1.5)  # x_min, x_max, y_min, y_max
OBSTACLES = [(0.25, 0.65, 0.5), (0.25, -0.65, 0.5)]  # (x, y, radius)
IMAGE_SIZE = (128, 128)


class DubinsSwmEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: str | None = None,
        speed: float = 1.0,
        dt: float = 0.05,
        max_angular_velocity: float = 2.0,
        goal_radius: float = 0.05,
        success_tol: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.image_size = IMAGE_SIZE
        self.x_min, self.x_max, self.y_min, self.y_max = WORLD_BOUNDS
        self.obstacles = OBSTACLES
        self.speed = speed
        self.dt = dt
        self.max_angular_velocity = max_angular_velocity
        self.goal_radius = goal_radius
        self.success_tol = success_tol
        # OOD appearance overrides (env vars) — closed-loop C3. COLOR shifts only the
        # rendered obstacle (task/collision identical). SHAPE=square is a GENUINE
        # geometry change: rendered as a 45-deg diamond AND collided against the L1
        # ball |dx|+|dy|<=r (matches the margin study's diamond label). ROT rotates
        # the obstacle layout (used by both _collides and the renderer).
        import os as _os
        _obst_color = _os.environ.get("DUBINS_OBST_COLOR", "red")
        self._obst_shape = _os.environ.get("DUBINS_OBST_SHAPE", "circle")
        self.colors = {"background": "white", "agent": "blue", "goal": "green", "obstacle": _obst_color}
        # Rigid 90° rotation of the OBSTACLE LAYOUT to an OOD position (closed-loop
        # rotation axis). Rotates obstacles only (used by both _collides and the
        # renderer); the start/goal states — and thus the task and the canonical
        # dataset goal image — are unchanged, matching the color/shape framing
        # (canonical goal + shifted deployment world). A fully faithful full-scene
        # rotation (states+goal-image too) would need the harness to re-render the
        # goal, which it doesn't expose.
        self._rot_k = (int(_os.environ.get("DUBINS_ROT_DEG", "0")) // 90) % 4
        if self._rot_k:
            self.obstacles = [(*self._rot_xy(ox, oy), r) for ox, oy, r in self.obstacles]

        self.state = np.zeros(3, dtype=np.float32)
        self.goal_state = np.zeros(3, dtype=np.float32)
        self.goal_position = np.array([self.x_max - 0.2, 0.0], dtype=np.float32)
        self._goal = None
        self._collided = False  # episode-level: did the trajectory ever hit an obstacle
        self._reached = False   # episode-level: did the trajectory ever reach the goal

        self.action_space = spaces.Box(
            low=-max_angular_velocity, high=max_angular_velocity, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Dict(
            {
                "proprio": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                "state": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            }
        )

    # ---- obs helpers ----
    def _obs(self):
        th = float(self.state[2])
        return {
            "proprio": np.array([math.cos(th), math.sin(th)], dtype=np.float32),
            "state": self.state.astype(np.float32),
        }

    def _info(self, extra=None):
        info = {"goal": self._goal, "goal_state": self.goal_state.copy()}
        if extra:
            info.update(extra)
        return info

    # ---- gym API ----
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and "initial_state" in options:
            self.state = np.asarray(options["initial_state"], dtype=np.float32).copy()
        self._collided = False
        self._reached = False  # episode-level: did the trajectory ever reach the goal
        self._goal = self.render()
        return self._obs(), self._info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        omega = float(np.clip(action[0], -self.max_angular_velocity, self.max_angular_velocity))
        self._integrate(omega)
        dist = float(np.linalg.norm(self.state[:2] - self.goal_position))
        terminated = dist <= self.success_tol
        collision = bool(self._collides())
        self._collided = self._collided or collision
        self._reached = self._reached or terminated
        info = self._info({"goal_distance": dist, "collision": collision})
        return self._obs(), -dist, terminated, False, info

    def render(self):
        return self._render_image()

    def close(self):
        pass

    # ---- eval callables ----
    def _rot_xy(self, x, y):
        """Apply self._rot_k quarter-turns CCW about the world origin: (x,y)->(-y,x)."""
        for _ in range(getattr(self, "_rot_k", 0)):
            x, y = -y, x
        return x, y

    def _rot_state(self, s):
        """Rigidly rotate a (x,y,theta) state by self._rot_k*90deg (position + heading)."""
        s = np.asarray(s, dtype=np.float32).copy()
        if getattr(self, "_rot_k", 0):
            s[0], s[1] = self._rot_xy(s[0], s[1])
            s[2] = float(np.arctan2(np.sin(s[2] + self._rot_k * np.pi / 2),
                                    np.cos(s[2] + self._rot_k * np.pi / 2)))
        return s

    def _set_state(self, state):
        # Full-world rotation: rotate the start state so the whole problem (obstacles
        # already rotated) is a rigid rotation — traffic/obstacle geometry preserved,
        # only the encoder's view of the configuration is OOD.
        self.state = self._rot_state(state)

    def _set_goal_state(self, goal_state):
        self.goal_state = self._rot_state(goal_state)
        self.goal_position = self.goal_state[:2].copy()

    # ---- dynamics (RK4, matches DubinsEnv._update_state) ----
    def _integrate(self, omega):
        def f(s):
            _, _, th = s
            return np.array([self.speed * np.cos(th), self.speed * np.sin(th), omega])

        s = self.state.astype(np.float64)
        k1 = f(s)
        k2 = f(s + 0.5 * self.dt * k1)
        k3 = f(s + 0.5 * self.dt * k2)
        k4 = f(s + self.dt * k3)
        s = s + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        s[2] = np.arctan2(np.sin(s[2]), np.cos(s[2]))
        self.state = s.astype(np.float32)

    def _collides(self):
        p = self.state[:2]
        if self._obst_shape == "square":  # genuine 45-deg diamond: L1 ball |dx|+|dy|<=r
            return any(abs(p[0] - ox) + abs(p[1] - oy) <= r for ox, oy, r in self.obstacles)
        return any(np.linalg.norm(p - np.array([ox, oy])) <= r for ox, oy, r in self.obstacles)

    # ---- rendering (matches DubinsEnv._render_image / _draw_agent) ----
    def _render_image(self):
        scale = 4
        h = (self.image_size[0] * scale, self.image_size[1] * scale)
        img = Image.new("RGB", h, self.colors["background"])
        draw = ImageDraw.Draw(img)

        def w2p(coord):
            x, y = coord
            px = int((x - self.x_min) / (self.x_max - self.x_min) * h[0])
            py = int((self.y_max - y) / (self.y_max - self.y_min) * h[1])
            return px, py

        for ox, oy, r in self.obstacles:
            c = w2p((ox, oy))
            rp = r / (self.x_max - self.x_min) * h[0]
            if self._obst_shape == "square":  # 45-deg diamond (label geometry stays circular)
                draw.polygon(
                    [(c[0], c[1] - rp), (c[0] + rp, c[1]), (c[0], c[1] + rp), (c[0] - rp, c[1])],
                    fill=self.colors["obstacle"],
                )
            else:
                draw.ellipse(
                    [(c[0] - rp, c[1] - rp), (c[0] + rp, c[1] + rp)],
                    fill=self.colors["obstacle"], width=2 * scale,
                )

        gc = w2p(self.goal_position)
        gr = self.goal_radius / (self.x_max - self.x_min) * h[0]
        draw.ellipse(
            [(gc[0] - gr, gc[1] - gr), (gc[0] + gr, gc[1] + gr)],
            fill=self.colors["goal"], width=2 * scale,
        )

        self._draw_agent(draw, w2p(self.state[:2]), float(self.state[2]), scale)
        img = img.resize(self.image_size, Image.Resampling.LANCZOS)
        return np.array(img)

    def _draw_agent(self, draw, center_px, angle_rad, scale):
        angle_rad = -angle_rad  # flip for correct visual orientation
        length = 20 * scale / 2
        width = 12 * scale / 2
        radius = width / 2
        tip_x = center_px[0] + length * math.cos(angle_rad)
        tip_y = center_px[1] + length * math.sin(angle_rad)
        perp = angle_rad + math.pi / 2
        p1 = (center_px[0] + radius * math.cos(perp), center_px[1] + radius * math.sin(perp))
        p2 = (center_px[0] - radius * math.cos(perp), center_px[1] - radius * math.sin(perp))
        draw.polygon([(tip_x, tip_y), p1, p2], fill=self.colors["agent"])
        bbox = [
            (center_px[0] - radius, center_px[1] - radius),
            (center_px[0] + radius, center_px[1] + radius),
        ]
        draw.pieslice(
            bbox, start=math.degrees(angle_rad) + 90, end=math.degrees(angle_rad) - 90,
            fill=self.colors["agent"],
        )


# Register on import (idempotent).
try:
    from stable_worldmodel.envs import register

    register(id="swm/Dubins-v0", entry_point="dubins_swm_env:DubinsSwmEnv")
except Exception:
    pass
