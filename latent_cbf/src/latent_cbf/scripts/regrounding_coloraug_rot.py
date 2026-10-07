"""Re-grounding probe for the color+rotation-aug LEWM WMs. Roll red vs {purple, rot90}
latents under IDENTICAL ZERO actions (no critic needed — isolates the PREDICTOR) and
track divergence div(t) = mean_b ||feat_red(t) - feat_shift(t)|| / sigma_red over steps.
Contraction (div falls with t) = the predictor pulls the OOD appearance back onto the
trained manifold (re-grounding). Tests the hypothesis that jac+pull still re-grounds under
the combined aug while baseline/jacobian diverge — explaining why only jac+pull's critic
absorbs both augs."""
import sys
from pathlib import Path
import numpy as np, torch
_THIS = Path(__file__).resolve()
for p in [str(_THIS.parents[3]), str(_THIS.parents[2]), str(_THIS.parents[1]), str(_THIS.parent)]:
    if p not in sys.path: sys.path.append(p)
from configs import DreamerConfig, Config
from critic_ood_eval import DEV, render_cond, CONDS
from model_based_value import init_latent
from latent_cbf.adapters import LEWMWorldModel
from latent_cbf.dubins.dubins_env import DubinsEnv

CK = "/data/seongbin/lewm/checkpoints"
WM = {"baseline": "coloraug_rot_baseline", "jacobian": "coloraug_rot_jacobian",
      "jac+pull": "coloraug_rot_jacpull"}
N, K = 2000, 12


def sigma(Z):
    return float(np.sqrt(2.0 * Z.var(0).sum())) + 1e-9


@torch.no_grad()
def zero_div(wm, lat_r, lat_s, K, sig):
    """div(t) rolling both latents under identical zero action."""
    fr = wm.dynamics.get_feat(lat_r); fs = wm.dynamics.get_feat(lat_s)
    out = [float(np.linalg.norm((fr - fs).cpu().numpy(), axis=1).mean()) / sig]
    a = torch.zeros(fr.shape[0], 1, 1, device=DEV)
    for _ in range(K):
        lat_r = {k: v[:, -1] for k, v in wm.dynamics.imagine_with_action(a, lat_r).items()}
        lat_s = {k: v[:, -1] for k, v in wm.dynamics.imagine_with_action(a, lat_s).items()}
        fr = wm.dynamics.get_feat(lat_r); fs = wm.dynamics.get_feat(lat_s)
        out.append(float(np.linalg.norm((fr - fs).cpu().numpy(), axis=1).mean()) / sig)
    return np.array(out)


def main():
    ec = Config().environment
    env = DubinsEnv(image_size=tuple(ec.image_size), world_bounds=tuple(ec.world_bounds),
                    max_angular_velocity=ec.max_angular_velocity, speed=ec.speed, dt=ec.dt,
                    obstacles=ec.get_obstacles_list(), goal_radius=ec.goal_radius,
                    collision_radius=ec.collision_radius, render_mode="rgb_array")
    rng = np.random.default_rng(0); xs = ec.world_bounds
    st = np.stack([rng.uniform(xs[0], xs[1], N), rng.uniform(xs[2], xs[3], N),
                   rng.uniform(-np.pi, np.pi, N)], 1).astype(np.float32)
    for tag, w in WM.items():
        ck = f"{CK}/{w}/weights_epoch_50.pt"
        cfg = DreamerConfig(); cfg.lewm_ckpt_path = ck
        wm = LEWMWorldModel(cfg, ck).to(DEV).eval()
        hist = wm.history_size
        imgs_r = render_cond(env, st, CONDS["red"])
        for shift in ("purple", "rot90"):
            imgs_s = render_cond(env, st, CONDS[shift])
            divs = []
            for i in range(0, N, 512):
                lr = init_latent(wm, imgs_r[i:i + 512], hist)
                ls = init_latent(wm, imgs_s[i:i + 512], hist)
                sig = sigma(wm.dynamics.get_feat(lr).cpu().numpy())
                divs.append(zero_div(wm, lr, ls, K, sig))
            curve = np.mean(divs, axis=0)
            print(f"REGROUND {tag} {shift} div/step " + " ".join(f"{v:.3f}" for v in curve))
        del wm; torch.cuda.empty_cache()
    print("REGROUND_DONE")


if __name__ == "__main__":
    main()
