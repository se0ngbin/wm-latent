"""Check the eval.py goal re-render: with the default (red) appearance a re-rendered goal frame must match the dataset frame
(only the re-render path is exercised, forced here); with DUBINS_OBST_COLOR=purple it must differ only on obstacle pixels."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, h5py
import eval as E                     # installs the patch; defines _rerender_goal
import dubins_swm_env

with h5py.File("/data/seongbin/lewm/datasets/dubins_expert.h5", "r") as f:
    idx = np.sort(np.random.default_rng(0).choice(len(f["state"]), 64, replace=False))
    px, st = f["pixels"][idx], f["state"][idx]
env = dubins_swm_env.DubinsSwmEnv()
d = np.array([np.abs(E._rerender_goal(env, im, s).astype(float) - im.astype(float)).mean() for im, s in zip(px, st)])
print(f"red re-render vs dataset: mean |diff| {d.mean():.2f}, max {d.max():.2f}")
assert d.mean() < 1.0, "red re-render does not reproduce the dataset goal frames"
os.environ["DUBINS_OBST_COLOR"] = "purple"
envp = dubins_swm_env.DubinsSwmEnv()
im, s = px[0], st[0]
out = E._rerender_goal(envp, im, s)
x = im.astype(float)
red = np.minimum(x[..., 0] - x[..., 1], x[..., 0] - x[..., 2]) > 25   # obstacle incl. anti-aliased edges
diff = np.abs(out.astype(int) - im.astype(int)).sum(-1) > 30
print(f"purple: changed pixels {diff.sum()}, of which on red obstacle pixels {(diff & red).sum()}; "
      f"purple pixels in output {((out[..., 0] == 128) & (out[..., 1] == 0) & (out[..., 2] == 128)).sum()}")
assert (diff & ~red).sum() < 0.05 * diff.sum() + 20, "purple re-render changed non-obstacle pixels"
print("ok")
