"""Model-based reachability value for DREAMER/RSSM, under color shift — the sharp
thesis test. For JEPA-jacpull, rolling the SHARED predictor re-grounds purple onto
the red manifold and the model-based value recovers color (.20 -> .78). Does the
RSSM do the same? Prediction: NO — its reconstruction latent has no red manifold to
collapse toward, so rolling won't rescue color.

Same protocol as model_based_value.py: encode (red or purple) -> roll the RSSM
forward under the avoid-actor -> V_mb = min_t margin(z_t), scored vs HJ V*.
Usage: dreamer_model_based_value.py --rssm <ck> --policy <p> [--ks 0,1,3,5,10]
"""
import sys, argparse
from pathlib import Path
import numpy as np, torch
_THIS = Path(__file__).resolve()
for p in [str(_THIS.parents[3]), str(_THIS.parents[2]), str(_THIS.parents[1]), str(_THIS.parent)]:
    if p not in sys.path: sys.path.append(p)
from sklearn.metrics import roc_auc_score
from configs import DreamerConfig, Config
from PyHJ.data import Batch
from compare_critic_to_hj_dreamer import build_dreamer_wm
from critic_ood_eval import DEV, HJ, hj_interp, load_ddpg
from critic_ood_eval_dreamer import make_env, render_cond, feats, CONDS

N = 3000


def build_wm(rssm):
    cfg = DreamerConfig(); ec = Config().environment
    cfg.turnRate = ec.max_angular_velocity
    cfg.x_min, cfg.x_max = ec.world_bounds[0], ec.world_bounds[1]
    cfg.y_min, cfg.y_max = ec.world_bounds[2], ec.world_bounds[3]
    cfg.size = [128, 128]; cfg.filter_mode = "cbf"; cfg.no_gp = False; cfg.wm_backend = "dreamer"
    return build_dreamer_wm(cfg, rssm, DEV), cfg


def init_state(wm, imgs, ths):
    """observe single frames -> RSSM posterior, time-stripped (B, ...)."""
    b = imgs.shape[0]
    obs_state = np.stack([np.cos(ths), np.sin(ths)], -1)[:, None].astype(np.float32)
    batch = {"image": imgs[:, None], "obs_state": obs_state,
             "action": np.zeros((b, 1, 1), np.float32),
             "is_first": np.ones((b, 1, 1), np.float32),
             "is_terminal": np.zeros((b, 1, 1), np.float32)}
    data = wm.preprocess(batch); embed = wm.encoder(data)
    post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
    return {k: v[:, -1] for k, v in post.items()}


@torch.no_grad()
def rollout_min_margin(wm, policy, state, K, turn):
    head = wm.heads["margin_gp"]
    feat = wm.dynamics.get_feat(state)
    mins = head(feat).view(-1)
    for _ in range(K):
        b = Batch(obs=feat.detach().cpu().numpy(), info=Batch())
        act = policy(b, model="actor_old").act
        a = torch.as_tensor(act, dtype=torch.float32, device=DEV).view(-1, 1, 1) * turn
        state = wm.dynamics.imagine_with_action(a, state)
        state = {k: v[:, -1] for k, v in state.items()}
        feat = wm.dynamics.get_feat(state)
        mins = torch.minimum(mins, head(feat).view(-1))
    return mins.cpu().numpy()


def score(v, safe):
    acc = float(((v >= 0) == (safe == 1)).mean())
    auc = float(roc_auc_score(safe, v)) if 0 < safe.mean() < 1 else float("nan")
    return acc, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rssm", required=True); ap.add_argument("--policy", required=True)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--ks", default="0,1,3,5,10")
    a = ap.parse_args()
    ks = [int(x) for x in a.ks.split(",")]
    d = np.load(HJ); V, xs, ys, ths = d["V"], d["grid_xs"], d["grid_ys"], d["grid_thetas"]
    rng = np.random.default_rng(a.seed)
    st = np.stack([rng.uniform(xs[0], xs[-1], N), rng.uniform(ys[0], ys[-1], N),
                   rng.uniform(-np.pi, np.pi, N)], 1).astype(np.float32)
    safe = (hj_interp(V, xs, ys, ths, st) >= 0).astype(int)
    wm, cfg = build_wm(a.rssm)
    env = make_env()
    f0 = feats(wm, render_cond(env, st[:2], CONDS["red"]), st[:2, 2])
    policy = load_ddpg(a.policy, int(f0.shape[1]), cfg, DEV)
    turn = Config().environment.max_angular_velocity
    print(f"N={N} frac_safe={safe.mean():.3f} feat={int(f0.shape[1])} turn={turn}")

    for cname in ("red", "purple"):
        imgs = render_cond(env, st, CONDS[cname])
        for K in ks:
            vs = []
            for i in range(0, N, 512):
                stt = init_state(wm, imgs[i:i + 512], st[i:i + 512, 2])
                vs.append(rollout_min_margin(wm, policy, stt, K, turn))
            v = np.concatenate(vs)
            acc, auc = score(v, safe)
            print(f"DMBVAL cond={cname} K={K} acc={acc:.3f} auc={auc:.3f}")
    print("DMBVAL_DONE")


if __name__ == "__main__":
    main()
