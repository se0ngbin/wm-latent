"""Dreamer analog of save_ood_margin_lewm.py: train + SAVE a per-appearance
margin (margin_gp+nogp, 544-d) by reinitializing and training the ACTUAL dreamer
wm.heads modules, then saving their state_dicts. Reloads cleanly into wm_ddpg's
dreamer WorldModel (--dreamer_margin_ckpt). Env-parity feats = observe->get_feat.
Usage (from src/latent_cbf):
  save_ood_margin_dreamer.py --rssm_ckpt <ckpt> --cond <purple|diamond|rot90|red> --out <p.pt> [--seed 0]
"""
import sys, argparse
from pathlib import Path
import numpy as np, torch
_THIS = Path(__file__).resolve()
for p in [str(_THIS.parents[3]), str(_THIS.parents[2]), str(_THIS.parents[1]), str(_THIS.parent)]:
    if p not in sys.path: sys.path.append(p)
import gym as old_gym  # noqa: F401
import gymnasium as gym  # noqa: F401
from sklearn.metrics import roc_auc_score
from configs import DreamerConfig, Config
from compare_critic_to_hj_dreamer import build_dreamer_wm
from ood_margin_gp_dreamer import (inside_obstacle, render_cond, make_env, encode_feats,
                                   GRAD_THR, ZS_W, RELU_W, GP_W, GAMMA_LX, LR, STEPS)

DEV, N, NTR = "cuda:0", 8000, 6000
# cond -> render/label spec (matches critic_ood_eval.CONDS)
CONDS = {"red":    dict(obstacle="red",    shape="circle",  rot=0),
         "purple": dict(obstacle="purple", shape="circle",  rot=0),
         "diamond":dict(obstacle="red",    shape="diamond", rot=0),
         "rot90":  dict(obstacle="red",    shape="circle",  rot=90)}


def reinit(module):
    for m in module.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()


def margin_gp_loss(head, safe, unsafe):
    pos, neg = head(safe).squeeze(-1), head(unsafe).squeeze(-1)
    Np = max(pos.shape[0], neg.shape[0])
    def rs(x, n):
        if x.shape[0] >= n: return x[:n]
        return x[torch.randint(0, x.shape[0], (n,), device=x.device)]
    sd, ud = rs(safe, Np), rs(unsafe, Np)
    alpha = torch.rand(Np, 1, device=DEV)
    interp = (alpha * sd + (1 - alpha) * ud).requires_grad_(True)
    out = head(interp)
    grads = torch.autograd.grad(out, interp, torch.ones_like(out), create_graph=True)[0]
    gnorm = torch.sqrt((grads ** 2).sum(1) + 1e-12)
    gp = ((gnorm - GRAD_THR) ** 2).mean()
    return ZS_W * (neg.mean() - pos.mean()) + RELU_W * (torch.relu(neg).mean() + torch.relu(-pos).mean()) + GP_W * gp


def margin_nogp_loss(head, safe, unsafe):
    return torch.relu(GAMMA_LX - head(safe)).mean() + torch.relu(GAMMA_LX + head(unsafe)).mean()


def train_head(head, feat, y, kind, seed):
    torch.manual_seed(seed); reinit(head); head.train()
    opt = torch.optim.AdamW(head.parameters(), lr=LR)
    safe = feat[y == 0]; unsafe = feat[y == 1]
    lossfn = margin_gp_loss if kind == "gp" else margin_nogp_loss
    for _ in range(STEPS):
        bs = min(256, safe.shape[0], unsafe.shape[0])
        s = safe[torch.randint(0, safe.shape[0], (bs,), device=DEV)]
        u = unsafe[torch.randint(0, unsafe.shape[0], (bs,), device=DEV)]
        loss = lossfn(head, s, u)
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval()


@torch.no_grad()
def sign_acc(head, feat, y):
    h = head(feat).squeeze(-1).cpu().numpy(); safe = (y == 0).astype(int)
    acc = float(((h >= 0) == (safe == 1)).mean())
    auc = float(roc_auc_score(safe, h)) if 0 < safe.mean() < 1 else float("nan")
    return acc, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rssm_ckpt", required=True); ap.add_argument("--cond", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cond = CONDS[a.cond]
    cfg = DreamerConfig(); ec = Config().environment
    cfg.turnRate = ec.max_angular_velocity
    cfg.x_min, cfg.x_max = ec.world_bounds[0], ec.world_bounds[1]
    cfg.y_min, cfg.y_max = ec.world_bounds[2], ec.world_bounds[3]
    cfg.size = [128, 128]; cfg.filter_mode = "cbf"; cfg.no_gp = False; cfg.wm_backend = "dreamer"
    wm = build_dreamer_wm(cfg, a.rssm_ckpt, DEV)
    rng = np.random.default_rng(a.seed)
    x0, x1, y0, y1 = ec.world_bounds
    st = np.stack([rng.uniform(x0, x1, N), rng.uniform(y0, y1, N),
                   rng.uniform(-np.pi, np.pi, N)], 1).astype(np.float32)
    y = inside_obstacle(st[:, :2], cond["shape"])
    env = make_env()
    feat = encode_feats(wm, render_cond(env, st, cond), st[:, 2]).to(DEV)
    d = feat.shape[1]
    ftr, fte, ytr, yte = feat[:NTR], feat[NTR:], y[:NTR], y[NTR:]
    print(f"cond={a.cond} shape={cond['shape']} N={N} feat={d} frac_unsafe={y.mean():.3f}")
    out = {"embed_dim": int(d)}
    for kind, hkey in (("gp", "margin_gp"), ("nogp", "margin_nogp")):
        head = wm.heads[hkey]
        train_head(head, ftr, ytr, kind, a.seed)
        acc, auc = sign_acc(head, fte, yte)
        print(f"KEY oodmargin_dreamer cond={a.cond} kind={kind} acc={acc:.3f} auc={auc:.3f}")
        out[hkey] = {k: v.detach().cpu() for k, v in head.state_dict().items()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, a.out)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
