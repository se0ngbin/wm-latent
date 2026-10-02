"""Dubins-analog wrapper over SafetyCarGoal1 (safety-gymnasium). Top-down camera, a
hazard-recolor hook (the appearance-shift OOD lever, analog of DUBINS_OBST_COLOR), and
dubins-style state / signed-distance margin / failure extraction. Runs in the safety-gym
venv only (dataset generation); le-wm trains offline on the emitted h5."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import safety_gymnasium as sg
import safety_gymnasium.assets.color as _C

# non-purple training palette + a held-out OOD color (analog of red->purple).
PALETTE = {
    "blue":   [0, 0, 1, 1],   # default hazard color (in-dist)
    "orange": [1, 0.6, 0, 1], "teal": [0, 0.6, 0.6, 1], "yellow": [1, 1, 0, 1],
    "brown":  [0.55, 0.27, 0.07, 1], "green": [0, 0.7, 0, 1],
    "purple": [0.5, 0, 0.9, 1],   # HELD OUT (OOD)
}


class SGDubins:
    def __init__(self, task="SafetyCarGoal1-v0", size=128, camera="fixedfar", hazard_color="blue"):
        self.env = sg.make(task, render_mode="rgb_array", camera_name=camera, width=size, height=size)
        self.size = size
        self.hazard_color = hazard_color

    @property
    def action_space(self):
        return self.env.action_space

    def _apply_color(self, name):
        rgba = np.array(PALETTE[name], dtype=float)
        self.env.unwrapped.task.hazards.color = rgba
        _C.COLOR["hazard"] = rgba   # belt-and-suspenders for rebuild

    def reset(self, seed=None):
        self._apply_color(self.hazard_color)   # set before the model is built
        obs, info = self.env.reset(seed=seed)
        self._apply_color(self.hazard_color)    # ensure the built model uses it
        return self._image()

    def set_hazard_color(self, name):
        """Appearance-shift hook. Call before reset() to render hazards in `name`."""
        self.hazard_color = name

    def _image(self):
        return self.env.unwrapped.render().astype(np.uint8)   # (size,size,3)

    def state(self):
        t = self.env.unwrapped.task
        x, y = float(t.agent.pos[0]), float(t.agent.pos[1])
        rot = np.asarray(t.agent.mat).reshape(3, 3)
        th = float(np.arctan2(rot[1, 0], rot[0, 0]))
        return np.array([x, y, th], dtype=np.float32)

    def margin(self):
        """Signed distance to the nearest hazard boundary (>=0 safe), like dubins dist-radius."""
        t = self.env.unwrapped.task
        xy = np.asarray(t.agent.pos[:2]); r = float(t.hazards.size)
        d = [np.linalg.norm(xy - np.asarray(h[:2])) - r for h in t.hazards.pos]
        return float(min(d))

    def step(self, action):
        obs, reward, cost, term, trunc, info = self.env.step(np.asarray(action, dtype=np.float64))
        m = self.margin()
        failure = int(m < 0.0 or float(cost) > 0.0)   # inside a hazard
        return self._image(), self.state(), float(reward), failure, bool(term or trunc)

    def close(self):
        self.env.close()
