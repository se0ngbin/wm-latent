"""Measure the DIMENSIONALITY of each perturbation's nuisance subspace (instead of
projecting it out). For each encoder x axis, the shift cloud D = z(new) - z(og):
  displ       = mean||D||/sigma            (total shift size, for context)
  offset_frac = ||mean D||^2 / mean||D||^2 (fraction that is a single global offset)
  eff_rank    = participation ratio of the CENTERED D cloud (effective # deform dims)
  k90         = # PCs to reach 90% of the centered-D variance
Low eff_rank / k90  => low-dim nuisance (projectable, like Dreamer color);
high eff_rank / k90 => high-dim deformation (not projectable, like the critic's color).
"""
import sys
from pathlib import Path
import numpy as np, torch
_THIS = Path(__file__).resolve()
for p in [str(_THIS.parents[3]), str(_THIS.parents[2]), str(_THIS.parents[1]), str(_THIS.parent)]:
    if p not in sys.path: sys.path.append(p)
from configs import DreamerConfig, Config
from critic_ood_eval import DEV, render_cond as jepa_render, feats as jepa_feats, CONDS
from latent_cbf.adapters import LEWMWorldModel
from latent_cbf.dubins.dubins_env import DubinsEnv

AXES = {"color": "purple", "shape": "diamond", "rotate": "rot90"}
CK = "/data/seongbin/lewm/checkpoints"
WM = {"baseline": "sigreg_only_dubins", "jacobian": "jacobian_w1_dubins", "jacpull": "lewm_dubins_jacpull50"}
RSSM = "/data/seongbin/dreamer/dreamer/enc_lip_sweep/baseline/rssm_ckpt.pt"
N = 4000


def stats(Z0, Z1):
    D = Z1 - Z0
    sig = float(np.sqrt(2.0 * Z0.var(0).sum())) + 1e-9
    displ = float(np.linalg.norm(D, axis=1).mean()) / sig
    mu = D.mean(0)
    tot = float((D ** 2).sum(1).mean())                  # mean ||D||^2
    offset_frac = float((mu ** 2).sum()) / (tot + 1e-12)
    Dc = D - mu                                          # centered (deformation)
    # PCA variances = singular values^2 of Dc / N
    s = np.linalg.svd(Dc, compute_uv=False)
    lam = (s ** 2)
    eff_rank = float((lam.sum() ** 2) / ((lam ** 2).sum() + 1e-12))
    cum = np.cumsum(lam) / lam.sum()
    k90 = int(np.searchsorted(cum, 0.90) + 1)
    return displ, offset_frac, eff_rank, k90, Z0.shape[1]


def make_states(seed, ec):
    rng = np.random.default_rng(seed); xs = ec.world_bounds
    return np.stack([rng.uniform(xs[0], xs[1], N), rng.uniform(xs[2], xs[3], N),
                     rng.uniform(-np.pi, np.pi, N)], 1).astype(np.float32)


def main():
    ec = Config().environment
    env = DubinsEnv(image_size=tuple(ec.image_size), world_bounds=tuple(ec.world_bounds),
                    max_angular_velocity=ec.max_angular_velocity, speed=ec.speed, dt=ec.dt,
                    obstacles=ec.get_obstacles_list(), goal_radius=ec.goal_radius,
                    collision_radius=ec.collision_radius, render_mode="rgb_array")
    st = make_states(0, ec)
    for tag, w in WM.items():
        cfg = DreamerConfig(); cfg.lewm_ckpt_path = f"{CK}/{w}/weights_epoch_50.pt"
        wm = LEWMWorldModel(cfg, f"{CK}/{w}/weights_epoch_50.pt").to(DEV).eval()
        Z0 = jepa_feats(wm, jepa_render(env, st, CONDS["red"])).cpu().numpy()
        for axis, new in AXES.items():
            Z1 = jepa_feats(wm, jepa_render(env, st, CONDS[new])).cpu().numpy()
            displ, off, er, k90, dim = stats(Z0, Z1)
            print(f"NDIM {tag} {axis} dim={dim} displ={displ:.3f} offset_frac={off:.3f} "
                  f"eff_rank={er:.1f} k90={k90}")
        del wm; torch.cuda.empty_cache()

    from compare_critic_to_hj_dreamer import build_dreamer_wm
    from critic_ood_eval_dreamer import make_env as make_denv, render_cond as dr_render, feats as dr_feats
    cfg = DreamerConfig()
    cfg.turnRate = ec.max_angular_velocity
    cfg.x_min, cfg.x_max = ec.world_bounds[0], ec.world_bounds[1]
    cfg.y_min, cfg.y_max = ec.world_bounds[2], ec.world_bounds[3]
    cfg.size = [128, 128]; cfg.filter_mode = "cbf"; cfg.no_gp = False; cfg.wm_backend = "dreamer"
    wm = build_dreamer_wm(cfg, RSSM, DEV); denv = make_denv()
    Z0 = dr_feats(wm, dr_render(denv, st, CONDS["red"]), st[:, 2]).cpu().numpy()
    for axis, new in AXES.items():
        Z1 = dr_feats(wm, dr_render(denv, st, CONDS[new]), st[:, 2]).cpu().numpy()
        displ, off, er, k90, dim = stats(Z0, Z1)
        print(f"NDIM dreamer {axis} dim={dim} displ={displ:.3f} offset_frac={off:.3f} "
              f"eff_rank={er:.1f} k90={k90}")
    print("NDIM_DONE")


if __name__ == "__main__":
    main()
