"""Import BEFORE running eval.py (config-name=tworoom) to record agent trajectories.
Monkeypatches TwoRoomEnv.step; dumps per-episode agent (x,y) paths to $OBSDIST_OUT at exit."""
import os, atexit, numpy as np
from stable_worldmodel.envs.two_room.env import TwoRoomEnv

TRAJS, GOALS = {}, {}
_step, _sg = TwoRoomEnv.step, TwoRoomEnv._set_goal_state
def step(self, action):
    out = _step(self, action)
    TRAJS.setdefault(id(self), []).append(np.asarray(self.agent_position.detach().cpu().numpy()))
    return out
def set_goal(self, goal_state):
    _sg(self, goal_state); GOALS[id(self)] = np.asarray(self.target_position.detach().cpu().numpy())
TwoRoomEnv.step = step
TwoRoomEnv._set_goal_state = set_goal

@atexit.register
def _dump():
    out = os.environ.get("OBSDIST_OUT")
    if not out:
        return
    keys = [k for k in TRAJS if len(TRAJS[k]) > 1]
    d = {f"t{i}": np.asarray(TRAJS[k]) for i, k in enumerate(keys)}
    d.update({f"g{i}": np.asarray(GOALS[k]) for i, k in enumerate(keys) if k in GOALS})
    np.savez(out, **d)
    print(f"[traj_patch_tworoom] dumped {len(keys)} trajectories (+{sum(k in GOALS for k in keys)} goals) -> {out}")
