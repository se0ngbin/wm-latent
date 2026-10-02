"""Policy-free re-grounding for the JEPA+GRU ablation. Roll RED and PURPLE latents
forward under IDENTICAL (zero) actions and measure how their divergence evolves —
no critic/policy needed. For the GRU model, decompose the feature divergence into
the encoder-embed part vs the recurrent-memory (deter_h) part: if the GRU memory
ACCUMULATES the appearance nuisance (Dreamer-like), the deter_h divergence GROWS;
if the predictive objective keeps it off, it stays small.

div is normalized per step by the RMS-pairwise spread (sigma) of the red cloud in
that subspace, so it is scale-free and comparable across models/subspaces.
Usage: regrounding_gru.py --K 10   (runs baseline/jacobian/jacpull/gru)
"""
import sys, argparse
from pathlib import Path
import numpy as np, torch
_THIS = Path(__file__).resolve()
for p in [str(_THIS.parents[3]), str(_THIS.parents[2]), str(_THIS.parents[1]), str(_THIS.parent)]:
    if p not in sys.path: sys.path.append(p)
from configs import DreamerConfig, Config
from latent_cbf.dubins.dubins_env import DubinsEnv
from latent_cbf.adapters import LEWMWorldModel
from critic_ood_eval import DEV, render_cond, CONDS

CK = "/data/seongbin/lewm/checkpoints"
MODELS = {"baseline": "sigreg_only_dubins", "jacobian": "jacobian_w1_dubins",
          "jacpull": "lewm_dubins_jacpull50", "gru": "lewm_gru_dubins",
          "gru_jacpull": "lewm_gru_jacpull"}
N = 2000


def sig(Z):
    return float(np.sqrt(2.0 * Z.var(0).sum())) + 1e-9


def init_state(wm, imgs):
    b = imgs.shape[0]
    batch = {"image": imgs[:, None], "action": np.zeros((b, 1, 1), np.float32),
             "is_first": np.ones((b, 1), np.float32), "is_terminal": np.zeros((b, 1), np.float32)}
    data = wm.preprocess(batch); embed = wm.encoder(data)
    post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
    return {k: v[:, -1] for k, v in post.items()}


@torch.no_grad()
def rollout(wm, st_r, st_p, K):
    """Roll red+purple under identical zero actions. Return per-step arrays of the
    embed parts and (if GRU) deter_h parts, for red and purple."""
    is_gru = wm._is_gru
    def parts(state):
        e = state["deter"].cpu().numpy()
        h = state["deter_h"].cpu().numpy() if is_gru else None
        return e, h
    er, hr = parts(st_r); ep, hp = parts(st_p)
    Er, Hr, Ep, Hp = [er], [hr], [ep], [hp]
    for _ in range(K):
        a = torch.zeros(st_r["deter"].shape[0], 1, 1, device=DEV)
        st_r = {k: v[:, -1] for k, v in wm.dynamics.imagine_with_action(a, st_r).items()}
        st_p = {k: v[:, -1] for k, v in wm.dynamics.imagine_with_action(a, st_p).items()}
        er, hr = parts(st_r); ep, hp = parts(st_p)
        Er.append(er); Hr.append(hr); Ep.append(ep); Hp.append(hp)
    return Er, Hr, Ep, Hp


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    ec = Config().environment
    env = DubinsEnv(image_size=tuple(ec.image_size), world_bounds=tuple(ec.world_bounds),
                    max_angular_velocity=ec.max_angular_velocity, speed=ec.speed, dt=ec.dt,
                    obstacles=ec.get_obstacles_list(), goal_radius=ec.goal_radius,
                    collision_radius=ec.collision_radius, render_mode="rgb_array")
    rng = np.random.default_rng(a.seed); xs = ec.world_bounds
    st = np.stack([rng.uniform(xs[0], xs[1], N), rng.uniform(xs[2], xs[3], N),
                   rng.uniform(-np.pi, np.pi, N)], 1).astype(np.float32)
    imgs_r = render_cond(env, st, CONDS["red"]); imgs_p = render_cond(env, st, CONDS["purple"])

    for tag, run in MODELS.items():
        ck = f"{CK}/{run}/weights_epoch_50.pt"
        cfg = DreamerConfig(); cfg.lewm_ckpt_path = ck
        wm = LEWMWorldModel(cfg, ck).to(DEV).eval()
        Er, Hr, Ep, Hp = [], [], [], []
        for i in range(0, N, 512):
            sr = init_state(wm, imgs_r[i:i + 512]); sp = init_state(wm, imgs_p[i:i + 512])
            er, hr, ep, hp = rollout(wm, sr, sp, a.K)
            (Er.append(er), Hr.append(hr), Ep.append(ep), Hp.append(hp))
        K1 = a.K + 1
        emb_div, det_div = [], []
        for t in range(K1):
            er = np.concatenate([b[t] for b in Er]); ep = np.concatenate([b[t] for b in Ep])
            emb_div.append(float(np.linalg.norm(er - ep, axis=1).mean()) / sig(er))
            if wm._is_gru:
                hr = np.concatenate([b[t] for b in Hr]); hp = np.concatenate([b[t] for b in Hp])
                det_div.append(float(np.linalg.norm(hr - hp, axis=1).mean()) / sig(hr) if t > 0 else 0.0)
        print(f"REGRU {tag} embed_div/step " + " ".join(f"{v:.3f}" for v in emb_div))
        if wm._is_gru:
            print(f"REGRU {tag} deter_h_div/step " + " ".join(f"{v:.3f}" for v in det_div))
        del wm; torch.cuda.empty_cache()
    print("REGRU_DONE")


if __name__ == "__main__":
    main()
