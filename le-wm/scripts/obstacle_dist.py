"""Does regularization change how close the planned car gets to obstacles?

Runs eval's own CEM planner (via eval.run) under the sg25clean protocol for ONE
policy (argv), records each episode's car trajectory by monkeypatching the env,
and reports distance to the nearest obstacle EDGE (‖pos−center‖ − r; <0 = inside).

Usage:  python obstacle_dist.py <policy_ckpt> <num_eval>
"""
import os, sys
sys.path.insert(0, "/home/seongbin/latent/le-wm")
os.environ.setdefault("STABLEWM_HOME", "/data/seongbin/lewm")
os.environ.setdefault("LOCAL_DATASET_DIR", "/data/seongbin/lewm")
import numpy as np
import dubins_swm_env
from dubins_swm_env import DubinsSwmEnv, OBSTACLES

POLICY = sys.argv[1]
NUM = int(sys.argv[2]) if len(sys.argv) > 2 else 100
OBS = np.array(OBSTACLES)                 # (2,3): x,y,r
TRAJS = {}                                # id(env) -> list of car (x,y)

_set, _step = DubinsSwmEnv._set_state, DubinsSwmEnv.step
def set_state(self, state):
    _set(self, state); TRAJS[id(self)] = [self.state[:2].copy()]
def step(self, action):
    out = _step(self, action)
    TRAJS.setdefault(id(self), []).append(self.state[:2].copy()); return out
DubinsSwmEnv._set_state = set_state
DubinsSwmEnv.step = step

def edge_dist(traj):                       # (T,2) -> per-step min clearance to any obstacle edge
    d = np.stack([np.linalg.norm(traj - OBS[k, :2], axis=1) - OBS[k, 2] for k in range(len(OBS))])
    return d.min(0)

# drive eval.run() via hydra argv (its @hydra.main reads sys.argv)
sys.argv = ["eval.py", "--config-name=dubins", f"policy={POLICY}",
            f"eval.num_eval={NUM}", "eval.goal_offset_steps=25",
            "eval.goal_at_episode_end=false", "eval.eval_budget=50",
            "plan_config.horizon=25", "plan_config.action_block=1", "plan_config.receding_horizon=25",
            "+eval.goal_avoid_obstacles=[[0.25,0.65,0.5],[0.25,-0.65,0.5]]",
            "+eval.goal_avoid_margin=0.05", "+eval.goal_near_margin=0.5",
            "+eval.goal_clean_source=true", "hydra.run.dir=outputs/obsdist_tmp",
            "hydra.output_subdir=null"]
import eval as E
E.run()

trajs = [np.asarray(v) for v in TRAJS.values() if len(v) > 1]
mins = np.array([edge_dist(t).min() for t in trajs])
means = np.array([edge_dist(t).mean() for t in trajs])
tag = POLICY.split("/")[0]
print(f"OBSDIST {tag:22s} n={len(trajs):3d}  mean_min_dist={mins.mean():+.4f}  "
      f"mean_mean_dist={means.mean():+.4f}  frac_inside={np.mean(mins<0):.3f}  min_min={mins.min():+.4f}")
np.savez(f"/home/seongbin/latent/le-wm/outputs/obsdist_{tag}.npz", mins=mins, means=means)
