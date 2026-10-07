"""Online eval of the LE-WM planner on the Dubins task, matching the
latent_cbf protocol (random start, fixed green goal, live rollout, success +
collision measured online). No CBF filter.

Usage (from latent_cbf/):
  .venv/bin/python src/latent_cbf/scripts/eval_lewm_planner.py --episodes 50 \
      --ckpt sigreg_only_dubins/weights_epoch_50.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
for p in [str(_THIS.parents[3]), str(_THIS.parents[2]), str(_THIS.parents[1])]:
    if p not in sys.path:
        sys.path.append(p)

from configs import get_default_config
from controllers.factory import create_controller_from_config
from scripts.run_experiment import create_env_from_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--ckpt", default="sigreg_only_dubins/weights_epoch_50.pt")
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--n_iters", type=int, default=4)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--safe_mode", default="off", choices=["off", "learned", "gt"])
    ap.add_argument("--safe_backend", default="grid", choices=["grid", "hardnet"])
    ap.add_argument("--hardnet_iters", type=int, default=3)
    ap.add_argument("--hardnet_damping", type=float, default=1.0)
    ap.add_argument("--margin_ckpt", default=None)
    ap.add_argument("--margin_head", default="margin_gp")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--safe_grid", type=int, default=21)
    ap.add_argument("--gt_h", default="hj", choices=["hj", "dist"])
    ap.add_argument("--h_offset", type=float, default=0.0)
    ap.add_argument("--penalty_weight", type=float, default=0.0)
    args = ap.parse_args()

    config = get_default_config()
    c = config.controller
    c.controller_type = "lewm_planner"
    c.lewm_ckpt = args.ckpt
    c.lewm_horizon = args.horizon
    c.lewm_num_samples = args.num_samples
    c.lewm_n_iters = args.n_iters
    c.lewm_topk = args.topk
    c.lewm_safe_mode = args.safe_mode
    c.lewm_safe_backend = args.safe_backend
    c.lewm_hardnet_iters = args.hardnet_iters
    c.lewm_hardnet_damping = args.hardnet_damping
    c.lewm_margin_ckpt = args.margin_ckpt
    c.lewm_margin_head = args.margin_head
    c.lewm_cbf_alpha = args.alpha
    c.lewm_safe_grid = args.safe_grid
    c.lewm_gt_h = args.gt_h
    c.lewm_h_offset = args.h_offset
    c.lewm_penalty_weight = args.penalty_weight

    env = create_env_from_config(config)
    controller = create_controller_from_config(config)

    obstacles = config.environment.get_obstacles_list()

    def true_h(pos):
        return min(
            float(np.linalg.norm(np.asarray(pos) - np.array([ox, oy])) - r)
            for ox, oy, r in obstacles
        )

    # HJ value at the start state: episodes with V*(x0) < 0 are provably
    # doomed (no controller can avoid collision) — report the collision rate
    # conditioned on feasible starts as well as the raw rate.
    hj_file = Path(__file__).resolve().parents[3] / "results/hj_truth.npz"
    v0_interp = None
    if hj_file.exists():
        from train_margin_hj_lewm import hj_interp

        _hj = np.load(hj_file)
        _Vg, _xs, _ys, _ths = _hj["V"], _hj["grid_xs"], _hj["grid_ys"], _hj["grid_thetas"]
        v0_interp = lambda s: float(hj_interp(_Vg, _xs, _ys, _ths, s[None].astype(np.float32))[0])  # noqa: E731

    successes, collisions, steps_to_goal, min_hs, v0s = [], [], [], [], []
    for ep in range(args.episodes):
        np.random.seed(args.seed0 + ep)  # goal-y draw uses global np.random
        obs, info = env.reset(seed=args.seed0 + ep)
        controller.reset()
        gx, gy = np.array(env.goal_position, dtype=float)
        saved = env.state.copy()
        env.state = np.array([gx, gy, 0.0], dtype=np.float32)
        goal_img = env.render()
        env.state = saved
        controller.set_goal(goal_img)

        success = False
        collided = False
        used = args.max_steps
        d0 = float(info["goal_distance"])
        dmin = d0
        min_h = true_h(info["agent_position"])
        v0 = v0_interp(env.state) if v0_interp is not None else float("nan")
        v0s.append(v0)
        for step in range(args.max_steps):
            action = controller.compute_action(info, obs)
            obs, _, term, trunc, info = env.step(action)
            collided = collided or bool(info["collision"])
            dmin = min(dmin, float(info["goal_distance"]))
            min_h = min(min_h, true_h(info["agent_position"]))
            if info["goal_reached"]:
                success = True
                used = step + 1
                break
            if term or trunc:
                break
        dfin = float(info["goal_distance"])
        successes.append(success)
        collisions.append(collided)
        min_hs.append(min_h)
        if success:
            steps_to_goal.append(used)
        sm = getattr(controller, "safe_map", None)
        if sm is None:
            fb = ""
        elif hasattr(sm, "fallback_rate"):
            fb = f" fallback={sm.fallback_rate:.3f}"
        else:
            fb = f" viol={sm.violation_rate:.3f} resid={sm.mean_residual:.4f}"
        print(f"ep {ep:3d}: success={success} collided={collided} steps={used} "
              f"d0={d0:.2f} dmin={dmin:.2f} dfin={dfin:.2f} min_h={min_h:.3f} "
              f"V0={v0:.3f}{fb}")

    env.close()
    n = len(successes)
    sr = 100.0 * np.mean(successes)
    cr = 100.0 * np.mean(collisions)
    print("=" * 50)
    print(f"episodes:       {n}")
    print(f"success_rate:   {sr:.1f}%")
    print(f"collision_rate: {cr:.1f}%")
    if v0_interp is not None:
        feas = np.array(v0s) >= 0
        if feas.any():
            cr_f = 100.0 * np.mean(np.array(collisions)[feas])
            print(f"collision_rate (feasible starts, V0>=0): {cr_f:.1f}% "
                  f"({int(feas.sum())}/{n} episodes)")
    print(f"min_true_h:     mean={np.mean(min_hs):.3f} worst={np.min(min_hs):.3f}")
    if steps_to_goal:
        print(f"avg steps (success): {np.mean(steps_to_goal):.1f}")
    sm = getattr(controller, "safe_map", None)
    if sm is not None and hasattr(sm, "fallback_rate"):
        print(f"fallback_rate:  {sm.fallback_rate:.4f}")
    elif sm is not None:
        print(f"violation_rate: {sm.violation_rate:.4f} mean_residual={sm.mean_residual:.5f}")


if __name__ == "__main__":
    main()
