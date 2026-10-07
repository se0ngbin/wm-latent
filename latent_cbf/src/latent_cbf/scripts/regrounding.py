"""Quantify the 're-grounding' the model-based value revealed: drive the RED and
PURPLE latents forward under the SAME action sequence and measure how their latent
divergence evolves. If the predictor pulls purple back onto the red manifold, the
divergence SHRINKS with rollout step (contractive along the nuisance direction).

div(t) = mean_b || feat_red(t) - feat_purple(t) || / sigma_red   (sigma_red = RMS
pairwise dist of the red feats at t=0, a fixed scale). Same action applied to both,
so the ONLY difference is the appearance-induced initial gap; its decay = re-grounding.
Emits per-encoder div(t) and a plot. JEPA encoders (baseline/jacobian/jacpull).
Usage: regrounding.py [--K 12]
"""
import sys, argparse
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
_THIS = Path(__file__).resolve()
for p in [str(_THIS.parents[3]), str(_THIS.parents[2]), str(_THIS.parents[1]), str(_THIS.parent)]:
    if p not in sys.path: sys.path.append(p)
from configs import DreamerConfig, Config
from latent_cbf.dubins.dubins_env import DubinsEnv
from latent_cbf.adapters import LEWMWorldModel
from PyHJ.data import Batch
from critic_ood_eval import DEV, render_cond, CONDS, load_ddpg
from model_based_value import init_latent

CK = "/data/seongbin/lewm/checkpoints"; MG = "/data/seongbin/dreamer/lewm"
VD = "/data/seongbin/dreamer/lewm/_value_ood"
WM = {"baseline": ("sigreg_only_dubins", "jepa_baseline"),
      "jacobian": ("jacobian_w1_dubins", "jepa_jacobian"),
      "jacpull":  ("lewm_dubins_jacpull50", "jepa_jacpull_headfix")}
N = 2000


def sigma(Z):
    return float(np.sqrt(2.0 * Z.var(0).sum()))


@torch.no_grad()
def paired_div(wm, policy, lat_r, lat_p, K, turn, sig):
    """Roll red+purple under the SAME (red-derived) actions; return div(t), t=0..K."""
    fr = wm.dynamics.get_feat(lat_r); fp = wm.dynamics.get_feat(lat_p)
    divs = [float(np.linalg.norm((fr - fp).cpu().numpy(), axis=1).mean()) / sig]
    for _ in range(K):
        b = Batch(obs=fr.detach().cpu().numpy(), info=Batch())
        act = policy(b, model="actor_old").act
        a = torch.as_tensor(act, dtype=torch.float32, device=DEV).view(-1, 1, 1) * turn
        lat_r = {k: v[:, -1] for k, v in wm.dynamics.imagine_with_action(a, lat_r).items()}
        lat_p = {k: v[:, -1] for k, v in wm.dynamics.imagine_with_action(a, lat_p).items()}
        fr = wm.dynamics.get_feat(lat_r); fp = wm.dynamics.get_feat(lat_p)
        divs.append(float(np.linalg.norm((fr - fp).cpu().numpy(), axis=1).mean()) / sig)
    return np.array(divs)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    ec = Config().environment
    env = DubinsEnv(image_size=tuple(ec.image_size), world_bounds=tuple(ec.world_bounds),
                    max_angular_velocity=ec.max_angular_velocity, speed=ec.speed, dt=ec.dt,
                    obstacles=ec.get_obstacles_list(), goal_radius=ec.goal_radius,
                    collision_radius=ec.collision_radius, render_mode="rgb_array")
    rng = np.random.default_rng(a.seed); xs = ec.world_bounds
    st = np.stack([rng.uniform(xs[0], xs[1], N), rng.uniform(xs[2], xs[3], N),
                   rng.uniform(-np.pi, np.pi, N)], 1).astype(np.float32)
    turn = ec.max_angular_velocity
    curves = {}
    for tag, (w, vd) in WM.items():
        ck = f"{CK}/{w}/weights_epoch_50.pt"
        cfg = DreamerConfig(); cfg.lewm_ckpt_path = ck
        wm = LEWMWorldModel(cfg, ck).to(DEV)
        sd = torch.load(f"{MG}/{w}/margin_heads.pt", map_location=DEV)
        wm.heads["margin_gp"].load_state_dict(sd["margin_gp"]); wm.eval()
        pol = sorted(Path(f"{VD}/{vd}/PyHJ/gp").glob("epoch_id_*"),
                     key=lambda p: int(p.name.split("_")[-1]))[-1] / "policy.pth"
        policy = load_ddpg(str(pol), int(wm.embed_dim), cfg, DEV)
        imgs_r = render_cond(env, st, CONDS["red"]); imgs_p = render_cond(env, st, CONDS["purple"])
        hist = wm.history_size
        divs = []
        for i in range(0, N, 512):
            lr = init_latent(wm, imgs_r[i:i + 512], hist)
            lp = init_latent(wm, imgs_p[i:i + 512], hist)
            sig = sigma(wm.dynamics.get_feat(lr).cpu().numpy())
            divs.append(paired_div(wm, policy, lr, lp, a.K, turn, sig))
        curve = np.mean(divs, axis=0)
        curves[tag] = curve
        print(f"REGROUND {tag} div/step " + " ".join(f"{v:.3f}" for v in curve))
        del wm, policy; torch.cuda.empty_cache()

    plt.figure(figsize=(6.5, 4.2))
    for tag, c in curves.items():
        plt.plot(range(len(c)), c, marker="o", ms=3, label=tag)
    plt.xlabel("predictor rollout step"); plt.ylabel("red↔purple latent divergence / σ_red")
    plt.title("Re-grounding: does the predictor pull purple onto the red manifold?")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    out = "/home/seongbin/latent/figs/regrounding.png"; plt.savefig(out, dpi=120)
    print("saved", out); print("REGROUND_DONE")


if __name__ == "__main__":
    main()
