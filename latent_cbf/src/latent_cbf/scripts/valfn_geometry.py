"""Geometry for the VALUE-FUNCTION version of the probe summary figure: per
(encoder x axis), the encoder-latent displacement and the alignment of the shift
with the V*-defined safe/unsafe axis (safe = HJ V* >= 0), to pair with the critic's
zero-shot AUC. Mirrors the margin probe but uses V* labels for the axis (the critic
is scored against V*, not the instantaneous failure label). N=2000.
"""
import sys
from pathlib import Path
import numpy as np, torch
_THIS = Path(__file__).resolve()
for p in [str(_THIS.parents[3]), str(_THIS.parents[2]), str(_THIS.parents[1]), str(_THIS.parent)]:
    if p not in sys.path: sys.path.append(p)
from configs import DreamerConfig, Config
from critic_ood_eval import HJ, hj_interp, DEV, render_cond as jepa_render, feats as jepa_feats, CONDS
from latent_cbf.adapters import LEWMWorldModel
from latent_cbf.dubins.dubins_env import DubinsEnv

AXES = {"color": "purple", "shape": "diamond", "rotate": "rot90"}
CK = "/data/seongbin/lewm/checkpoints"
WM = {"baseline": "sigreg_only_dubins", "jacobian": "jacobian_w1_dubins", "jacpull": "lewm_dubins_jacpull50"}
RSSM = "/data/seongbin/dreamer/dreamer/enc_lip_sweep/baseline/rssm_ckpt.pt"
N = 2000


def sigma(Z):
    return float(np.sqrt(2.0 * Z.var(0).sum())) + 1e-9


def geom(Z0, Z1, safe):
    D = Z1 - Z0
    displ = float(np.linalg.norm(D, axis=1).mean()) / sigma(Z0)
    mu = D.mean(0)
    la = Z0[safe == 0].mean(0) - Z0[safe == 1].mean(0)       # unsafe - safe axis (V*)
    cos_la = float(abs((mu @ la) / (np.linalg.norm(mu) * np.linalg.norm(la) + 1e-12)))
    return displ, cos_la


def main():
    d = np.load(HJ); V, xs, ys, ths = d["V"], d["grid_xs"], d["grid_ys"], d["grid_thetas"]
    rng = np.random.default_rng(0)
    st = np.stack([rng.uniform(xs[0], xs[-1], N), rng.uniform(ys[0], ys[-1], N),
                   rng.uniform(-np.pi, np.pi, N)], 1).astype(np.float32)
    safe = (hj_interp(V, xs, ys, ths, st) >= 0).astype(int)   # V* safe label
    ec = Config().environment
    env = DubinsEnv(image_size=tuple(ec.image_size), world_bounds=tuple(ec.world_bounds),
                    max_angular_velocity=ec.max_angular_velocity, speed=ec.speed, dt=ec.dt,
                    obstacles=ec.get_obstacles_list(), goal_radius=ec.goal_radius,
                    collision_radius=ec.collision_radius, render_mode="rgb_array")

    for tag, w in WM.items():
        cfg = DreamerConfig(); cfg.lewm_ckpt_path = f"{CK}/{w}/weights_epoch_50.pt"
        wm = LEWMWorldModel(cfg, f"{CK}/{w}/weights_epoch_50.pt").to(DEV).eval()
        Z0 = jepa_feats(wm, jepa_render(env, st, CONDS["red"])).cpu().numpy()
        for axis, new in AXES.items():
            Z1 = jepa_feats(wm, jepa_render(env, st, CONDS[new])).cpu().numpy()
            displ, cos_la = geom(Z0, Z1, safe)
            print(f"VGEOM {tag} {axis} displ={displ:.3f} cos_la={cos_la:.3f}")
        del wm; torch.cuda.empty_cache()

    # dreamer
    from compare_critic_to_hj_dreamer import build_dreamer_wm
    from critic_ood_eval_dreamer import make_env as make_denv, render_cond as dr_render, feats as dr_feats
    cfg = DreamerConfig()
    cfg.turnRate = ec.max_angular_velocity
    cfg.x_min, cfg.x_max = ec.world_bounds[0], ec.world_bounds[1]
    cfg.y_min, cfg.y_max = ec.world_bounds[2], ec.world_bounds[3]
    cfg.size = [128, 128]; cfg.filter_mode = "cbf"; cfg.no_gp = False; cfg.wm_backend = "dreamer"
    wm = build_dreamer_wm(cfg, RSSM, DEV)
    denv = make_denv()
    Z0 = dr_feats(wm, dr_render(denv, st, CONDS["red"]), st[:, 2]).cpu().numpy()
    for axis, new in AXES.items():
        Z1 = dr_feats(wm, dr_render(denv, st, CONDS[new]), st[:, 2]).cpu().numpy()
        displ, cos_la = geom(Z0, Z1, safe)
        print(f"VGEOM dreamer {axis} displ={displ:.3f} cos_la={cos_la:.3f}")
    print("VGEOM_DONE")


if __name__ == "__main__":
    main()
