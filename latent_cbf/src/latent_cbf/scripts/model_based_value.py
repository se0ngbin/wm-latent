"""Model-BASED reachability value vs the model-free critic, under color shift.

The user's point: the dynamics (encoder+predictor) is SHARED across appearance, so
the value SHOULD transfer — yet the model-free critic (a memorized Q(z) tabulated on
red) collapses on purple. Test: compute the value model-based instead — roll the
SHARED predictor forward from the (red or purple) encoded latent under the avoid-actor
and take V_mb = min_t margin(z_t) over the horizon (the avoid/BRT value). Score vs HJ
V*. If purple V_mb stays as good as red across the horizon, the shared dynamics + the
(fixed, color-robust) margin transfer where the memorized critic did not.

K=0 is the pure instantaneous margin; K>0 adds rolled-out dynamics.
Usage: model_based_value.py --lewm_ckpt <w> --margin_ckpt <m> --policy <p> [--ks 0,1,3,5,10]
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
from PyHJ.data import Batch
from critic_ood_eval import DEV, HJ, hj_interp, load_ddpg, render_cond, CONDS

N = 3000


def init_latent(wm, imgs, hist):
    """imgs: (B,H,W,3) uint8 single frames -> initial latent (time-stripped) by
    observing `hist` static copies (matches the env's uniform-reset seeding)."""
    B = imgs.shape[0]
    img = np.repeat(imgs[:, None], hist, axis=1)                 # (B,hist,H,W,3)
    batch = {"image": img,
             "action": np.zeros((B, hist, 1), np.float32),
             "is_first": np.zeros((B, hist), np.float32),
             "is_terminal": np.zeros((B, hist), np.float32)}
    batch["is_first"][:, 0] = 1.0
    data = wm.preprocess(batch)
    embed = wm.encoder(data)                                     # (B,hist,D)
    latent, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
    return {k: v[:, -1] for k, v in latent.items()}             # (B, ...)


@torch.no_grad()
def rollout_min_margin(wm, policy, latent, K, turn):
    """V_mb = min_t margin(z_t), t=0..K, under the avoid-actor. Returns (B,) numpy."""
    head = wm.heads["margin_gp"]
    feat = wm.dynamics.get_feat(latent)                          # (B,D)
    mins = head(feat).view(-1)                                   # t=0
    for _ in range(K):
        b = Batch(obs=feat.detach().cpu().numpy(), info=Batch())
        act = policy(b, model="actor_old").act                  # (B,1) in [-1,1]
        a = torch.as_tensor(act, dtype=torch.float32, device=DEV).view(-1, 1, 1) * turn
        latent = wm.dynamics.imagine_with_action(a, latent)
        latent = {k: v[:, -1] for k, v in latent.items()}       # strip T_act=1
        feat = wm.dynamics.get_feat(latent)
        mins = torch.minimum(mins, head(feat).view(-1))
    return mins.cpu().numpy()


def score(v, safe):
    acc = float(((v >= 0) == (safe == 1)).mean())
    auc = float(roc_auc_score(safe, v)) if 0 < safe.mean() < 1 else float("nan")
    return acc, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lewm_ckpt", required=True); ap.add_argument("--margin_ckpt", required=True)
    ap.add_argument("--policy", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ks", default="0,1,3,5,10")
    a = ap.parse_args()
    ks = [int(x) for x in a.ks.split(",")]
    d = np.load(HJ); V, xs, ys, ths = d["V"], d["grid_xs"], d["grid_ys"], d["grid_thetas"]
    rng = np.random.default_rng(a.seed)
    st = np.stack([rng.uniform(xs[0], xs[-1], N), rng.uniform(ys[0], ys[-1], N),
                   rng.uniform(-np.pi, np.pi, N)], 1).astype(np.float32)
    safe = (hj_interp(V, xs, ys, ths, st) >= 0).astype(int)
    cfg = DreamerConfig(); cfg.lewm_ckpt_path = a.lewm_ckpt
    wm = LEWMWorldModel(cfg, a.lewm_ckpt).to(DEV)
    sd = torch.load(a.margin_ckpt, map_location=DEV)
    wm.heads["margin_gp"].load_state_dict(sd["margin_gp"])
    wm.heads["margin_nogp"].load_state_dict(sd["margin_nogp"]); wm.eval()
    policy = load_ddpg(a.policy, int(wm.embed_dim), cfg, DEV)
    hist = wm.history_size; turn = Config().environment.max_angular_velocity
    ec = Config().environment
    env = DubinsEnv(image_size=tuple(ec.image_size), world_bounds=tuple(ec.world_bounds),
                    max_angular_velocity=ec.max_angular_velocity, speed=ec.speed, dt=ec.dt,
                    obstacles=ec.get_obstacles_list(), goal_radius=ec.goal_radius,
                    collision_radius=ec.collision_radius, render_mode="rgb_array")
    print(f"N={N} frac_safe={safe.mean():.3f} hist={hist} turn={turn} dim={int(wm.embed_dim)}")

    for cname in ("red", "purple"):
        imgs = render_cond(env, st, CONDS[cname])
        for K in ks:
            # chunk to bound memory
            vs = []
            for i in range(0, N, 512):
                lat = init_latent(wm, imgs[i:i + 512], hist)
                vs.append(rollout_min_margin(wm, policy, lat, K, turn))
            v = np.concatenate(vs)
            acc, auc = score(v, safe)
            print(f"MBVAL cond={cname} K={K} acc={acc:.3f} auc={auc:.3f}")
    print("MBVAL_DONE")


if __name__ == "__main__":
    main()
