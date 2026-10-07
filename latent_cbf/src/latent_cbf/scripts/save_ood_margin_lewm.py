"""Train + SAVE a per-appearance instantaneous margin (margin_gp + margin_nogp)
for a frozen LEWM encoder, in the exact `margin_heads.pt` format the reach RL env
(dubins-wm-dp.py) reloads. Used to build the RETRAINED value-function column:
the critic is retrained on a shifted appearance with a margin trained on that same
appearance.

Guaranteed key-compatibility: we take the ACTUAL wm.heads["margin_gp"/"margin_nogp"]
modules (networks.MLP, names "Margin GP"/"Margin NoGP"), reinitialize them, train,
and save their state_dicts. Features come from critic_ood_eval.feats() == the env's
get_feat pipeline, so the reward the critic sees is on the right features.

Label = binary in/out-obstacle `failure` (geometry-exact per shape). Loss/config =
original latent_cbf (dreamer_conf.py): gp_thr .1, zs .1, relu 1.0, gp 10.0, gamma_lx
.75, lr 3e-4, 5000 steps. Usage (from src/latent_cbf):
  save_ood_margin_lewm.py --lewm_ckpt <w> --cond <purple|diamond|rot90|red> --out <p.pt> [--seed 0]
"""
import sys, argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn
_THIS = Path(__file__).resolve()
for p in [str(_THIS.parents[3]), str(_THIS.parents[2]), str(_THIS.parents[1]), str(_THIS.parent)]:
    if p not in sys.path: sys.path.append(p)
for _p in ("/home/seongbin/latent", "/home/seongbin/latent/le-wm"):
    if _p not in sys.path: sys.path.append(_p)
from sklearn.metrics import roc_auc_score
from configs import DreamerConfig, Config
from latent_cbf.adapters import LEWMWorldModel
from critic_ood_eval import feats, render_cond, CONDS, DEV

# margin loss config — configs/dreamer_conf.py (exact)
GRAD_THR, ZS_W, RELU_W, GP_W, GAMMA_LX, LR, STEPS = 0.1, 0.1, 1.0, 10.0, 0.75, 3e-4, 5000
N, NTR = 8000, 6000


def inside_obstacle(xy, shape, obstacles):
    """Binary failure label: 1 if agent center is inside an obstacle.
    circle: ||p-c||<=r ; diamond: L1 ball |dx|+|dy|<=r (matches the diamond render)."""
    hit = np.zeros(len(xy), bool)
    for cx, cy, r in obstacles:
        dx, dy = np.abs(xy[:, 0] - cx), np.abs(xy[:, 1] - cy)
        d = (dx + dy) if shape == "diamond" else np.hypot(dx, dy)
        hit |= (d <= r)
    return hit.astype(np.float32)


def reinit(module):
    for m in module.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()


def margin_gp_loss(head, safe, unsafe):
    pos, neg = head(safe), head(unsafe)                      # sign>=0 = safe
    Np = max(pos.shape[0], neg.shape[0])
    def resample(x, n):
        if x.shape[0] >= n: return x[:n]
        idx = torch.randint(0, x.shape[0], (n,), device=x.device); return x[idx]
    sd, ud = resample(safe, Np), resample(unsafe, Np)
    alpha = torch.rand(Np, 1, device=DEV)
    interp = (alpha * sd + (1 - alpha) * ud).requires_grad_(True)
    out = head(interp)
    grads = torch.autograd.grad(out, interp, torch.ones_like(out), create_graph=True)[0]
    gnorm = torch.sqrt((grads ** 2).sum(1) + 1e-12)
    gp = ((gnorm - GRAD_THR) ** 2).mean()
    zs = neg.mean() - pos.mean()
    relu = torch.relu(neg).mean() + torch.relu(-pos).mean()
    return ZS_W * zs + RELU_W * relu + GP_W * gp


def margin_nogp_loss(head, safe, unsafe):
    return torch.relu(GAMMA_LX - head(safe)).mean() + torch.relu(GAMMA_LX + head(unsafe)).mean()


def train_head(head, feat, y, kind, seed):
    torch.manual_seed(seed)
    reinit(head); head.train()
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
    h = head(feat).view(-1).cpu().numpy(); safe = (y == 0).astype(int)
    acc = float(((h >= 0) == (safe == 1)).mean())
    auc = float(roc_auc_score(safe, h)) if 0 < safe.mean() < 1 else float("nan")
    return acc, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lewm_ckpt", required=True); ap.add_argument("--cond", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run_name", default="")
    a = ap.parse_args()
    cond = CONDS[a.cond]
    cfg = DreamerConfig(); cfg.lewm_ckpt_path = a.lewm_ckpt
    wm = LEWMWorldModel(cfg, a.lewm_ckpt).to(DEV).eval()
    ec = Config().environment
    obstacles = ec.get_obstacles_list()
    from latent_cbf.dubins.dubins_env import DubinsEnv
    env = DubinsEnv(image_size=tuple(ec.image_size), world_bounds=tuple(ec.world_bounds),
                    max_angular_velocity=ec.max_angular_velocity, speed=ec.speed, dt=ec.dt,
                    obstacles=obstacles, goal_radius=ec.goal_radius,
                    collision_radius=ec.collision_radius, render_mode="rgb_array")
    xs = ec.world_bounds
    rng = np.random.default_rng(a.seed)
    st = np.stack([rng.uniform(xs[0], xs[1], N), rng.uniform(xs[2], xs[3], N),
                   rng.uniform(-np.pi, np.pi, N)], 1).astype(np.float32)
    y = inside_obstacle(st[:, :2], cond["shape"], obstacles)   # 1=unsafe(inside)
    feat = feats(wm, render_cond(env, st, cond))               # env get_feat parity
    ytr = torch.tensor(y[:NTR]); yte = y[NTR:]
    ftr, fte = feat[:NTR], feat[NTR:]
    print(f"cond={a.cond} shape={cond['shape']} N={N} frac_unsafe={y.mean():.3f} feat={feat.shape[1]}")
    out = {"lewm_run_name": a.run_name, "embed_dim": int(wm.embed_dim)}
    for kind, hkey in (("gp", "margin_gp"), ("nogp", "margin_nogp")):
        head = wm.heads[hkey]
        train_head(head, ftr.to(DEV), ytr.numpy(), kind, a.seed)
        acc, auc = sign_acc(head, fte.to(DEV), yte)
        print(f"KEY oodmargin cond={a.cond} kind={kind} acc={acc:.3f} auc={auc:.3f}")
        out[hkey] = {k: v.detach().cpu() for k, v in head.state_dict().items()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, a.out)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
