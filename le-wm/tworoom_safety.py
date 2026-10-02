"""TwoRoom safety primitives: 'contact with the dividing wall = unsafe'.

TwoRoom is holonomic (action is a 2D velocity in [-1,1]^2, pos' = pos + a*speed)
with a HARD vertical divider wall (slab x in [107,117]) broken by one door gap.
Because the dynamics are holonomic with bounded speed, the signed clearance to the
wall — h(x) = dist(agent_center, wall_material) - agent_radius — is a *valid* CBF
(from any state you can move directly away from the nearest wall point), so no HJ
PDE solve is needed, unlike Dubins. h >= 0 is safe (no contact); h <= 0 is contact.

This module is the TwoRoom analog of latent_cbf's GroundTruthH + hj_truth.npz:
  - build_wall_sdf(): discrete signed-distance field of the wall-with-door.
  - TwoRoomH: parallel holonomic step() + value() over batches of states.
Geometry constants match le-wm/scripts/tworoom_wall_dist.py and the env.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

IMG = 224
BORDER = 14
WALL_X = 112
WALL_HALF = 5          # wall_thickness//2 (thickness 10) -> slab x in [107,117]
DOOR_Y0, DOOR_Y1 = 35, 63   # door gap (center 49, half-extent 14)


def build_wall_sdf(agent_r: float = 7.0, door_y=(DOOR_Y0, DOOR_Y1)):
    """Signed clearance field h[y, x] = dist_to_wall(center) - agent_r over the
    224x224 image. Positive outside the wall (safe), negative where the agent's
    disk would overlap the wall material. Returns (h_grid[y,x], solid[y,x])."""
    dy0, dy1 = int(round(door_y[0])), int(round(door_y[1]))
    solid = np.zeros((IMG, IMG), bool)   # [y, x]
    solid[BORDER:IMG - BORDER, WALL_X - WALL_HALF:WALL_X + WALL_HALF + 1] = True
    solid[dy0:dy1, WALL_X - WALL_HALF:WALL_X + WALL_HALF + 1] = False
    # EDT gives, for each free cell, distance to the nearest solid cell.
    dist_free = distance_transform_edt(~solid)
    # inside the wall: negative penetration depth (distance into the solid).
    dist_in = distance_transform_edt(solid)
    signed = np.where(solid, -dist_in, dist_free).astype(np.float32)
    return signed - float(agent_r), solid


class TwoRoomH:
    """Parallel holonomic dynamics + wall-clearance value, batched over states.
    States are agent (x, y) in pixels; actions are 2D in [-1,1] (pre-scale)."""

    def __init__(self, agent_r=7.0, speed=5.0, door_y=(DOOR_Y0, DOOR_Y1),
                 device="cuda"):
        self.device = device
        self.speed = float(speed)
        self.agent_r = float(agent_r)
        h_grid, solid = build_wall_sdf(agent_r, door_y)
        self.h_grid = torch.tensor(h_grid, device=device)          # [y, x]
        self.solid = torch.tensor(solid, device=device)
        self._lo = BORDER + agent_r
        self._hi = IMG - BORDER - agent_r

    def value(self, x):
        """h(x) via bilinear sample of the clearance field. x: (..., 2) pixels."""
        xy = x.reshape(-1, 2)
        gx = xy[:, 0].clamp(0, IMG - 1.001)
        gy = xy[:, 1].clamp(0, IMG - 1.001)
        x0 = gx.floor().long(); y0 = gy.floor().long()
        x1 = (x0 + 1).clamp(max=IMG - 1); y1 = (y0 + 1).clamp(max=IMG - 1)
        wx = (gx - x0.float()); wy = (gy - y0.float())
        g = self.h_grid
        v = (g[y0, x0] * (1 - wx) * (1 - wy) + g[y0, x1] * wx * (1 - wy)
             + g[y1, x0] * (1 - wx) * wy + g[y1, x1] * wx * wy)
        return v.reshape(x.shape[:-1])

    def step(self, x, a):
        """Holonomic step with the env's hard-wall + border clamp.
        x: (..., 2) pixels; a: broadcastable (..., 2) in [-1,1]. Mirrors
        TwoRoomEnv._apply_collisions (started-side wall block + border clamp)."""
        a = a.clamp(-1.0, 1.0)
        x1 = x[..., 0]; y1 = x[..., 1]
        nx = (x1 + a[..., 0] * self.speed)
        ny = (y1 + a[..., 1] * self.speed)
        # border clamp (accounts for agent radius, as the env does).
        nx = nx.clamp(self._lo, self._hi)
        ny = ny.clamp(self._lo, self._hi)
        # central wall block: if crossing the effective wall from the start side
        # outside a door, clamp to the effective face.
        eff_l = (WALL_X - WALL_HALF) - self.agent_r
        eff_r = (WALL_X + WALL_HALF) + self.agent_r
        in_door = (ny >= DOOR_Y0) & (ny <= DOOR_Y1)
        started_left = x1 < WALL_X
        # left-start crossing into wall
        block_l = started_left & (nx > eff_l) & (~in_door)
        nx = torch.where(block_l, torch.full_like(nx, eff_l - 0.5), nx)
        # right-start crossing into wall
        block_r = (~started_left) & (nx < eff_r) & (~in_door)
        nx = torch.where(block_r, torch.full_like(nx, eff_r + 0.5), nx)
        return torch.stack([nx, ny], -1)


if __name__ == "__main__":
    # Validate h and step against the real TwoRoomEnv: contact (env clamps the
    # agent against the wall) should coincide with h ~ 0, and our step() should
    # match the env's next position.
    import sys
    sys.path.insert(0, "/home/seongbin/latent/le-wm")
    import numpy as np
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401  registers swm/TwoRoom-v1
    dev = "cpu"
    env = gym.make("swm/TwoRoom-v1").unwrapped
    env.reset()
    # fix radius/speed/door to our defaults for a clean check
    r = float(env.variation_space['agent']['radius'].value.item())
    sp = float(env.variation_space['agent']['speed'].value.item())
    dpos = float(env.variation_space['door']['position'].value[0])
    dsz = float(env.variation_space['door']['size'].value[0])
    print(f"env: agent_r={r} speed={sp} door_pos={dpos} door_half={dsz}")
    H = TwoRoomH(agent_r=r, speed=sp, door_y=(dpos - dsz, dpos + dsz), device=dev)
    rng = np.random.default_rng(0)
    max_step_err = 0.0
    contact_h = []
    for _ in range(400):
        p0 = rng.uniform(BORDER + r, IMG - BORDER - r, size=2).astype(np.float32)
        if abs(p0[0] - WALL_X) < WALL_HALF + r:   # start not inside wall
            continue
        a = rng.uniform(-1, 1, size=2).astype(np.float32)
        env.agent_position = torch.tensor(p0)
        env.step(a)
        p_env = env.agent_position.numpy()
        p_ours = H.step(torch.tensor(p0)[None], torch.tensor(a)[None])[0].numpy()
        err = np.abs(p_env - p_ours).max()
        max_step_err = max(max_step_err, err)
        # if the env blocked us against the wall, our h at the blocked pos ~ 0
        blocked = abs(p_env[0] - ((WALL_X - WALL_HALF) - r - 0.5)) < 1e-2 or \
                  abs(p_env[0] - ((WALL_X + WALL_HALF) + r + 0.5)) < 1e-2
        if blocked:
            contact_h.append(float(H.value(torch.tensor(p_env)[None])[0]))
    print(f"max step position error vs env: {max_step_err:.3f} px")
    if contact_h:
        ch = np.array(contact_h)
        print(f"h at wall-contact positions: n={len(ch)} "
              f"mean={ch.mean():+.3f} min={ch.min():+.3f} max={ch.max():+.3f} "
              f"(expect ~0)")
    print("done")
