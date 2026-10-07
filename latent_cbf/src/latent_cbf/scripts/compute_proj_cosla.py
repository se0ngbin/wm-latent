"""Recompute cos_la in the PROJECTED feature space for each cell's best nuisance
projection. cos_la = |cos(mean recolor shift, safe→unsafe axis)|; after projecting
out the color subspace (P = I - UU^T) both the shift and the axis live in Pz, so
cos_la_proj = |cos(P·mu, P·la)|. Shows mean-mode projection kills the alignment
(cos_la→0) while svd-mode largely preserves it. Head-free."""
import sys
from pathlib import Path
import numpy as np, torch
_THIS = Path(__file__).resolve()
for p in [str(_THIS.parents[3]), str(_THIS.parents[2]), str(_THIS.parents[1]), str(_THIS.parent)]:
    if p not in sys.path: sys.path.append(p)
from configs import DreamerConfig, Config
from critic_ood_eval import HJ, hj_interp, DEV, render_cond as jr, feats as jf, CONDS
from latent_cbf.adapters import LEWMWorldModel
from latent_cbf.dubins.dubins_env import DubinsEnv

CK = "/data/seongbin/lewm/checkpoints"
RSSM = "/data/seongbin/dreamer/dreamer/enc_lip_sweep/baseline/rssm_ckpt.pt"
N = 2000

# (name, wm_run, backend, label {failure|vstar}, mode, k)
CELLS = [
    ("baseline-margin", "sigreg_only_dubins", "lewm", "failure", "svd", 2),
    ("jacpull-margin",  "lewm_dubins_jacpull50", "lewm", "failure", "svd", 2),
    ("dreamer-margin",  None, "dreamer", "failure", "mean", 1),
    ("jacobian-value",  "jacobian_w1_dubins", "lewm", "vstar", "svd", 4),
    ("jacpull-value",   "lewm_dubins_jacpull50", "lewm", "vstar", "svd", 4),
]


def basis(D, mode, k):
    if mode == "mean":
        mu = D.mean(0); return (mu / (np.linalg.norm(mu) + 1e-12))[None, :]
    _, _, Vt = np.linalg.svd(D - D.mean(0), full_matrices=False); return Vt[:k]


def cosla(mu, la):
    return float(abs((mu @ la) / (np.linalg.norm(mu) * np.linalg.norm(la) + 1e-12)))


def main():
    ec = Config().environment
    d = np.load(HJ); V, xs, ys, ths = d["V"], d["grid_xs"], d["grid_ys"], d["grid_thetas"]
    rng = np.random.default_rng(0)
    st = np.stack([rng.uniform(xs[0], xs[-1], N), rng.uniform(ys[0], ys[-1], N),
                   rng.uniform(-np.pi, np.pi, N)], 1).astype(np.float32)
    obst = ec.get_obstacles_list()
    vstar = (hj_interp(V, xs, ys, ths, st) >= 0).astype(int)      # 1=safe
    fail = np.zeros(N, int)
    for cx, cy, r in obst:
        fail |= (np.hypot(st[:, 0] - cx, st[:, 1] - cy) <= r).astype(int)
    safe_fail = (fail == 0).astype(int)                          # 1=safe (outside)
    env = DubinsEnv(image_size=tuple(ec.image_size), world_bounds=tuple(ec.world_bounds),
                    max_angular_velocity=ec.max_angular_velocity, speed=ec.speed, dt=ec.dt,
                    obstacles=obst, goal_radius=ec.goal_radius,
                    collision_radius=ec.collision_radius, render_mode="rgb_array")

    for name, run, backend, lab, mode, k in CELLS:
        if backend == "lewm":
            cfg = DreamerConfig(); cfg.lewm_ckpt_path = f"{CK}/{run}/weights_epoch_50.pt"
            wm = LEWMWorldModel(cfg, f"{CK}/{run}/weights_epoch_50.pt").to(DEV).eval()
            Z0 = jf(wm, jr(env, st, CONDS["red"])).cpu().numpy()
            Z1 = jf(wm, jr(env, st, CONDS["purple"])).cpu().numpy()
            del wm; torch.cuda.empty_cache()
        else:
            from compare_critic_to_hj_dreamer import build_dreamer_wm
            from critic_ood_eval_dreamer import make_env, render_cond as dr, feats as df
            cfg = DreamerConfig(); cfg.turnRate = ec.max_angular_velocity
            cfg.x_min, cfg.x_max = ec.world_bounds[0], ec.world_bounds[1]
            cfg.y_min, cfg.y_max = ec.world_bounds[2], ec.world_bounds[3]
            cfg.size = [128, 128]; cfg.filter_mode = "cbf"; cfg.no_gp = False; cfg.wm_backend = "dreamer"
            wm = build_dreamer_wm(cfg, RSSM, DEV); denv = make_env()
            Z0 = df(wm, dr(denv, st, CONDS["red"]), st[:, 2]).cpu().numpy()
            Z1 = df(wm, dr(denv, st, CONDS["purple"]), st[:, 2]).cpu().numpy()
        safe = safe_fail if lab == "failure" else vstar
        D = Z1 - Z0; mu = D.mean(0)
        la = Z0[safe == 0].mean(0) - Z0[safe == 1].mean(0)       # unsafe - safe
        U = basis(D, mode, k)
        Pmu = mu - (mu @ U.T) @ U
        Pla = la - (la @ U.T) @ U
        print(f"COSLA {name} mode={mode} k={k} cos_la_orig={cosla(mu,la):.3f} cos_la_proj={cosla(Pmu,Pla):.3f}")
    print("COSLA_DONE")


if __name__ == "__main__":
    main()
