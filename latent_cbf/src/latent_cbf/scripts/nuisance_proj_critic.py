"""Lever #3 applied to the VALUE FUNCTION: does projecting the color direction out
of the critic's input feature un-collapse its color zero-shot, the way it did for
the margin? Estimate the color subspace U from D = z(purple)-z(red) over states,
build P = I - U U^T, and re-score the frozen policy (actor+critic) on P z.

If color zero-shot AUC (vs HJ V*) jumps toward .5+ -> the critic's collapse is
extrapolation along an unregularized nuisance axis (same story as the margin).
If it stays inverted -> the value function's color failure is deeper than one axis.
Usage: nuisance_proj_critic.py --lewm_ckpt <w> --margin_ckpt <m> --policy <p> [--ks 1,2,4,8]
"""
import sys, argparse
from pathlib import Path
import numpy as np, torch
_THIS = Path(__file__).resolve()
for p in [str(_THIS.parents[3]), str(_THIS.parents[2]), str(_THIS.parents[1]), str(_THIS.parent)]:
    if p not in sys.path: sys.path.append(p)
from sklearn.metrics import roc_auc_score
from configs import DreamerConfig, Config
from latent_cbf.dubins.dubins_env import DubinsEnv
from latent_cbf.adapters import LEWMWorldModel
from critic_ood_eval import (DEV, HJ, N, hj_interp, load_ddpg, eval_V, feats,
                             render_cond, CONDS)


def nuisance_basis(D, k):
    Dc = D - D.mean(0)
    _, _, Vt = np.linalg.svd(Dc, full_matrices=False)
    return Vt[:k]                                    # k x d orthonormal rows


def project_out(Z, U):
    return Z - (Z @ U.T) @ U


def auc_vs_hj(policy, feat_np, safe):
    Vh = eval_V(policy, torch.tensor(feat_np, dtype=torch.float32, device=DEV))
    acc = float(((Vh >= 0) == (safe == 1)).mean())
    auc = float(roc_auc_score(safe, Vh)) if 0 < safe.mean() < 1 else float("nan")
    return acc, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lewm_ckpt", required=True); ap.add_argument("--margin_ckpt", required=True)
    ap.add_argument("--policy", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ks", default="1,2,4,8")
    a = ap.parse_args()
    d = np.load(HJ); V, xs, ys, ths = d["V"], d["grid_xs"], d["grid_ys"], d["grid_thetas"]
    rng = np.random.default_rng(a.seed)
    st = np.stack([rng.uniform(xs[0], xs[-1], N), rng.uniform(ys[0], ys[-1], N),
                   rng.uniform(-np.pi, np.pi, N)], 1).astype(np.float32)
    y = hj_interp(V, xs, ys, ths, st); safe = (y >= 0).astype(int)
    cfg = DreamerConfig(); cfg.lewm_ckpt_path = a.lewm_ckpt
    wm = LEWMWorldModel(cfg, a.lewm_ckpt).to(DEV)
    sd = torch.load(a.margin_ckpt, map_location=DEV)
    wm.heads["margin_gp"].load_state_dict(sd["margin_gp"])
    wm.heads["margin_nogp"].load_state_dict(sd["margin_nogp"]); wm.eval()
    policy = load_ddpg(a.policy, int(wm.embed_dim), cfg, DEV)
    ec = Config().environment
    env = DubinsEnv(image_size=tuple(ec.image_size), world_bounds=tuple(ec.world_bounds),
                    max_angular_velocity=ec.max_angular_velocity, speed=ec.speed, dt=ec.dt,
                    obstacles=ec.get_obstacles_list(), goal_radius=ec.goal_radius,
                    collision_radius=ec.collision_radius, render_mode="rgb_array")
    print(f"N={N} frac_safe(V*>=0)={safe.mean():.3f} dim={int(wm.embed_dim)}")

    Zr = feats(wm, render_cond(env, st, CONDS["red"])).cpu().numpy()
    Zp = feats(wm, render_cond(env, st, CONDS["purple"])).cpu().numpy()
    D = Zp - Zr

    ir_a, ir_u = auc_vs_hj(policy, Zr, safe)              # in-dist raw
    cp_a, cp_u = auc_vs_hj(policy, Zp, safe)              # color zero-shot raw
    print(f"CRITPROJ mode=none k=0 indist_acc={ir_a:.3f} indist_auc={ir_u:.3f} "
          f"color_acc={cp_a:.3f} color_auc={cp_u:.3f}")
    for k in [int(x) for x in a.ks.split(",")]:
        U = nuisance_basis(D, k)
        i_a, i_u = auc_vs_hj(policy, project_out(Zr, U), safe)
        c_a, c_u = auc_vs_hj(policy, project_out(Zp, U), safe)
        print(f"CRITPROJ mode=svd k={k} indist_acc={i_a:.3f} indist_auc={i_u:.3f} "
              f"color_acc={c_a:.3f} color_auc={c_u:.3f} "
              f"d_color_auc={c_u - cp_u:+.3f} d_indist_auc={i_u - ir_u:+.3f}")
    print("CRITPROJ_DONE")


if __name__ == "__main__":
    main()
