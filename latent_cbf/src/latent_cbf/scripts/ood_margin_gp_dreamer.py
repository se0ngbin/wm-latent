"""OOD study on the REAL instantaneous margin (Dreamer/RSSM) — margin_gp, the
WGAN-GP signed-distance separator with a binary in/out-obstacle label. Dreamer
analog of ood_margin_gp_jepa.py; feature = observe->get_feat (544-d).
Usage: ood_margin_gp_dreamer.py --rssm_ckpt <ckpt> --axis <color|shape|rotate> [--tag t] [--seed s]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn

_THIS = Path(__file__).resolve()
for p in [str(_THIS.parents[3]), str(_THIS.parents[2]), str(_THIS.parents[1])]:
    if p not in sys.path: sys.path.append(p)
import gym as old_gym  # noqa: F401
import gymnasium as gym  # noqa: F401
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score
from configs import DreamerConfig, Config
from latent_cbf.dubins.dubins_env import DubinsEnv
from compare_critic_to_hj_dreamer import build_dreamer_wm
sys.path.insert(0, "/home/seongbin/latent/le-wm")
from tworoom_safe_solver import MarginHead   # exact replica of networks.MLP margin head

DEV, N, NTR = "cuda:0", 8000, 6000
GRAD_THR, ZS_W, RELU_W, GP_W, GAMMA_LX, LR, STEPS = 0.1, 0.1, 1.0, 10.0, 0.75, 3e-4, 5000
AXES = {
    "color":  (dict(obstacle="red", shape="circle", rot=0),
               dict(obstacle="purple", shape="circle", rot=0), "red", "purple"),
    "shape":  (dict(obstacle="red", shape="circle", rot=0),
               dict(obstacle="red", shape="diamond", rot=0), "circle", "diamond"),
    "rotate": (dict(obstacle="red", shape="circle", rot=0),
               dict(obstacle="red", shape="circle", rot=90), "rot0", "rot90"),
}
OBSTACLES = [(0.25, 0.65, 0.5), (0.25, -0.65, 0.5)]


def inside_obstacle(xy, shape):
    hit = np.zeros(len(xy), bool)
    for cx, cy, r in OBSTACLES:
        dx, dy = np.abs(xy[:, 0] - cx), np.abs(xy[:, 1] - cy)
        d = (dx + dy) if shape == "diamond" else np.hypot(dx, dy)
        hit |= (d <= r)
    return hit.astype(np.float32)


def render_diamond(env):
    scale = 4; h = (env.image_size[0]*scale, env.image_size[1]*scale)
    img = Image.new("RGB", h, env.colors["background"]); draw = ImageDraw.Draw(img)
    def w2p(c):
        x, y = c
        return (int((x-env.x_min)/(env.x_max-env.x_min)*h[0]),
                int((env.y_max-y)/(env.y_max-env.y_min)*h[1]))
    for ox, oy, r in env.obstacles:
        cx, cy = w2p((ox, oy)); rp = r/(env.x_max-env.x_min)*h[0]
        draw.polygon([(cx,cy-rp),(cx+rp,cy),(cx,cy+rp),(cx-rp,cy)], fill=env.colors["obstacle"])
    gc = w2p(env.goal_position); gr = env.goal_radius/(env.x_max-env.x_min)*h[0]
    draw.ellipse([(gc[0]-gr,gc[1]-gr),(gc[0]+gr,gc[1]+gr)], fill=env.colors["goal"])
    env._draw_agent(draw, w2p(env.state[:2]), float(env.state[2]), scale)
    return np.array(img.resize(env.image_size, Image.Resampling.LANCZOS))


def make_env():
    ec = Config().environment
    return DubinsEnv(image_size=tuple(ec.image_size), world_bounds=tuple(ec.world_bounds),
                     max_angular_velocity=ec.max_angular_velocity, speed=ec.speed, dt=ec.dt,
                     obstacles=ec.get_obstacles_list(), goal_radius=ec.goal_radius,
                     collision_radius=ec.collision_radius, render_mode="rgb_array")


def render_cond(env, st, cond):
    env.reset()
    env.colors = {"background": "white", "agent": "blue", "goal": "green", "obstacle": cond["obstacle"]}
    dia = cond["shape"] == "diamond"; rot = cond.get("rot", 0)
    imgs = []
    for s in st:
        env.state = s.astype(np.float32)
        im = render_diamond(env) if dia else env.render()
        if rot: im = np.ascontiguousarray(np.rot90(im, k=rot // 90))
        imgs.append(im)
    return np.stack(imgs, 0)


@torch.no_grad()
def encode_feats(wm, imgs, ths, bs=128):
    feats = []
    for i in range(0, imgs.shape[0], bs):
        ch, tc = imgs[i:i+bs], ths[i:i+bs]; b = ch.shape[0]
        obs_state = np.stack([np.cos(tc), np.sin(tc)], -1)[:, None].astype(np.float32)
        batch = {"image": ch[:, None], "obs_state": obs_state,
                 "action": np.zeros((b,1,1), np.float32), "is_first": np.ones((b,1,1), np.float32),
                 "is_terminal": np.zeros((b,1,1), np.float32)}
        data = wm.preprocess(batch); embed = wm.encoder(data)
        states, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
        feats.append(wm.dynamics.get_feat(states)[:, -1].cpu())
    return torch.cat(feats)


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
    zs = neg.mean() - pos.mean()
    relu = torch.relu(neg).mean() + torch.relu(-pos).mean()
    return ZS_W * zs + RELU_W * relu + GP_W * gp


def margin_nogp_loss(head, safe, unsafe):
    return torch.relu(GAMMA_LX - head(safe)).mean() + torch.relu(GAMMA_LX + head(unsafe)).mean()


def train_margin(Z, y, d, kind, seed=0):
    torch.manual_seed(seed)
    head = MarginHead(inp_dim=d).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=LR)
    safe = Z[y == 0].to(DEV); unsafe = Z[y == 1].to(DEV)
    lossfn = margin_gp_loss if kind == "gp" else margin_nogp_loss
    for _ in range(STEPS):
        bs = min(256, safe.shape[0], unsafe.shape[0])
        s = safe[torch.randint(0, safe.shape[0], (bs,), device=DEV)]
        u = unsafe[torch.randint(0, unsafe.shape[0], (bs,), device=DEV)]
        loss = lossfn(head, s, u)
        opt.zero_grad(); loss.backward(); opt.step()
    return head.eval()


def score(head, Z, y):
    with torch.no_grad():
        h = head(Z.to(DEV)).squeeze(-1).cpu().numpy()
    safe = (y == 0).astype(int)
    acc = float(((h >= 0) == (safe == 1)).mean())
    auc = float(roc_auc_score(safe, h)) if 0 < safe.mean() < 1 else float("nan")
    return acc, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rssm_ckpt", required=True); ap.add_argument("--axis", default="color", choices=list(AXES))
    ap.add_argument("--tag", default="dreamer"); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    og_cond, new_cond, og_tag, new_tag = AXES[args.axis]

    cfg = DreamerConfig(); ec = Config().environment
    cfg.turnRate = ec.max_angular_velocity
    cfg.x_min, cfg.x_max = ec.world_bounds[0], ec.world_bounds[1]
    cfg.y_min, cfg.y_max = ec.world_bounds[2], ec.world_bounds[3]
    cfg.size = [128, 128]; cfg.filter_mode = "cbf"; cfg.no_gp = False; cfg.wm_backend = "dreamer"
    wm = build_dreamer_wm(cfg, args.rssm_ckpt, DEV)

    rng = np.random.default_rng(args.seed)
    x0, x1, y0, y1 = ec.world_bounds
    st = np.stack([rng.uniform(x0, x1, N), rng.uniform(y0, y1, N),
                   rng.uniform(-np.pi, np.pi, N)], 1).astype(np.float32)
    y_og = inside_obstacle(st[:, :2], og_cond["shape"]); y_new = inside_obstacle(st[:, :2], new_cond["shape"])
    env = make_env()
    Z_og = encode_feats(wm, render_cond(env, st, og_cond), st[:, 2])
    Z_new = encode_feats(wm, render_cond(env, st, new_cond), st[:, 2])
    d = Z_og.shape[1]
    print(f"[{args.tag}] axis={args.axis} seed={args.seed} feat={d} frac_unsafe og={y_og.mean():.3f}")

    for kind in ("gp", "nogp"):
        h_og = train_margin(Z_og[:NTR], y_og[:NTR], d, kind, seed=args.seed)
        h_new = train_margin(Z_new[:NTR], y_new[:NTR], d, kind, seed=args.seed)
        oa, ou = score(h_og, Z_og[NTR:], y_og[NTR:]); za, zu = score(h_og, Z_new[NTR:], y_new[NTR:])
        na, nu = score(h_new, Z_new[NTR:], y_new[NTR:])
        print(f"=== dreamer_{kind}:{args.tag} axis={args.axis} seed={args.seed} ===")
        for lbl, a, u in (("M_og on og (in-dist)", oa, ou), ("M_og on new (zero-shot)", za, zu),
                          ("M_new on new (retrain)", na, nu)):
            print(f"{lbl:26s} {a:9.3f} {u:7.3f}")
        print(f"KEY dreamer_{kind} {args.tag} axis={args.axis} rk=0 seed={args.seed} "
              f"ogog_acc={oa:.3f} ogog_auc={ou:.3f} zs_acc={za:.3f} zs_auc={zu:.3f} "
              f"or_acc={na:.3f} or_auc={nu:.3f}")
    print("done")


if __name__ == "__main__":
    main()
