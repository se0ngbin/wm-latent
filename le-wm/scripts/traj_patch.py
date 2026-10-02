"""Import BEFORE running eval.py to record dubins car trajectories.
Monkeypatches the env; dumps per-episode trajectories to $OBSDIST_OUT at exit."""
import os, atexit, numpy as np
import dubins_swm_env
from dubins_swm_env import DubinsSwmEnv

TRAJS, GOALS = {}, {}
_set, _step, _sg = DubinsSwmEnv._set_state, DubinsSwmEnv.step, DubinsSwmEnv._set_goal_state
def set_state(self, state):
    _set(self, state); TRAJS[id(self)] = [self.state[:2].copy()]
def step(self, action):
    out = _step(self, action)
    TRAJS.setdefault(id(self), []).append(self.state[:2].copy()); return out
def set_goal(self, goal_state):
    _sg(self, goal_state); GOALS[id(self)] = self.goal_position.copy()
DubinsSwmEnv._set_state = set_state
DubinsSwmEnv.step = step
DubinsSwmEnv._set_goal_state = set_goal

@atexit.register
def _dump():
    out = os.environ.get("OBSDIST_OUT")
    if not out:
        return
    keys = [k for k in TRAJS if len(TRAJS[k]) > 1]
    d = {f"t{i}": np.asarray(TRAJS[k]) for i, k in enumerate(keys)}
    d.update({f"g{i}": np.asarray(GOALS[k]) for i, k in enumerate(keys) if k in GOALS})
    np.savez(out, **d)
    print(f"[traj_patch] dumped {len(keys)} trajectories (+{sum(k in GOALS for k in keys)} goals) -> {out}")
