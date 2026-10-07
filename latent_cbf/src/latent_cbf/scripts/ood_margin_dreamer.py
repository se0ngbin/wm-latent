"""Dreamer/RSSM analog of the JEPA obstacle-appearance margin-OOD test.

Same question, reconstruction-based world model: does a margin (safe/unsafe CBF
classifier, here a fresh MLP on the RSSM feature) trained on the ORIGINAL obstacle
appearance survive an appearance shift, vs one retrained on the new appearance?
Thread-A hypothesis: RSSM's pixel-reconstruction objective anchors the latent to
appearance, so its margin should degrade MORE under recolor than the invariance-
regularized JEPA.

WM frozen. Feature path mirrors filter time: preprocess -> encoder ->
dynamics.observe(single frame, is_first=1) -> get_feat  (544-d). We train a fresh
HJ-regression head on that feature (Dreamer has no comparable deployed HJ margin),
so both M_og and M_new are trained identically; only the render appearance differs.

Usage (from latent_cbf/):
  .venv/bin/python src/latent_cbf/scripts/ood_margin_dreamer.py \
      --rssm_ckpt <ckpt> --axis color   [--tag baseline]
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_THIS = Path(__file__).resolve()
for p in [str(_THIS.parents[3]), str(_THIS.parents[2]), str(_THIS.parents[1])]:
    if p not in sys.path:
        sys.path.append(p)

import gym as old_gym  # noqa: F401
import gymnasium as gym  # noqa: F401
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

from configs import DreamerConfig, Config
from latent_cbf.dubins.dubins_env import DubinsEnv
from compare_critic_to_hj_dreamer import build_dreamer_wm

HJ = "/home/seongbin/latent/latent_cbf/results/hj_truth.npz"
DEV, N = "cuda:0", 8000
AXES = {
    "color": (dict(obstacle="red",    shape="circle", rot=0),
              dict(obstacle="purple", shape="circle", rot=0), "red", "purple"),
    "shape": (dict(obstacle="red", shape="circle", rot=0),
              dict(obstacle="red", shape="square", rot=0), "circle", "square"),
    "rotate": (dict(obstacle="red", shape="circle", rot=0),
               dict(obstacle="red", shape="circle", rot=90), "rot0", "rot90"),
}


def hj_interp(V, xs, ys, ths, pts):
    nx, ny, nth = V.shape
    fx = np.clip((pts[:,0]-xs[0])/(xs[1]-xs[0]), 0, nx-1-1e-6)
    fy = np.clip((pts[:,1]-ys[0])/(ys[1]-ys[0]), 0, ny-1-1e-6)
    th = np.arctan2(np.sin(pts[:,2]), np.cos(pts[:,2]))
    ft = (th-ths[0])/(ths[1]-ths[0])
    i0, j0 = fx.astype(int), fy.astype(int); k0 = np.floor(ft).astype(int) % nth
    i1, j1 = np.minimum(i0+1, nx-1), np.minimum(j0+1, ny-1); k1 = (k0+1) % nth
    wx, wy, wt = fx-i0, fy-j0, np.clip(ft-np.floor(ft), 0, 1)
    out = 0.0
    for di, wi in ((i0,1-wx),(i1,wx)):
        for dj, wj in ((j0,1-wy),(j1,wy)):
            for dk, wk in ((k0,1-wt),(k1,wt)):
                out = out + V[di,dj,dk]*wi*wj*wk
    return out.astype(np.float32)


def render_square(env):
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
    sq = cond["shape"] == "square"
    rot = cond.get("rot", 0)
    imgs = []
    for s in st:
        env.state = s.astype(np.float32)
        im = render_square(env) if sq else env.render()
        if rot:
            im = np.ascontiguousarray(np.rot90(im, k=rot // 90))
        imgs.append(im)
    return np.stack(imgs, 0)


@torch.no_grad()
def encode_feats(wm, imgs, ths, bs=128):
    feats = []
    for i in range(0, imgs.shape[0], bs):
        ch, tc = imgs[i:i+bs], ths[i:i+bs]
        b = ch.shape[0]
        obs_state = np.stack([np.cos(tc), np.sin(tc)], -1)[:, None].astype(np.float32)
        batch = {"image": ch[:, None], "obs_state": obs_state,
                 "action": np.zeros((b,1,1), np.float32),
                 "is_first": np.ones((b,1,1), np.float32),
                 "is_terminal": np.zeros((b,1,1), np.float32)}
        data = wm.preprocess(batch)
        embed = wm.encoder(data)
        states, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
        feats.append(wm.dynamics.get_feat(states)[:, -1].cpu())
    return torch.cat(feats)


def train_head(Ztr, ytr, d, steps=4000, seed=0):
    torch.manual_seed(seed)
    head = nn.Sequential(nn.Linear(d,512), nn.SiLU(), nn.Linear(512,512), nn.SiLU(),
                         nn.Linear(512,1)).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4)
    for _ in range(steps):
        idx = torch.randint(0, Ztr.shape[0], (256,), device=DEV)
        loss = nn.functional.mse_loss(head(Ztr[idx]).squeeze(-1), ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return head.eval()


def score(head, Z, y):
    with torch.no_grad():
        p = head(Z.to(DEV)).squeeze(-1).cpu().numpy()
    acc = float(((p>=0)==(y>=0)).mean())
    auc = float(roc_auc_score((y>=0).astype(int), p)) if 0<(y>=0).mean()<1 else float("nan")
    r = float(pearsonr(p, y)[0])
    return acc, auc, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rssm_ckpt", required=True)
    ap.add_argument("--axis", default="color", choices=list(AXES))
    ap.add_argument("--tag", default="dreamer")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    og_cond, new_cond, og_tag, new_tag = AXES[args.axis]

    cfg = DreamerConfig(); ec = Config().environment
    cfg.turnRate = ec.max_angular_velocity
    cfg.x_min, cfg.x_max = ec.world_bounds[0], ec.world_bounds[1]
    cfg.y_min, cfg.y_max = ec.world_bounds[2], ec.world_bounds[3]
    cfg.size = [128, 128]; cfg.filter_mode = "cbf"; cfg.no_gp = False; cfg.wm_backend = "dreamer"
    wm = build_dreamer_wm(cfg, args.rssm_ckpt, DEV)

    d = np.load(HJ); V, xs, ys, ths = d["V"], d["grid_xs"], d["grid_ys"], d["grid_thetas"]
    rng = np.random.default_rng(args.seed)
    st = np.stack([rng.uniform(xs[0], xs[-1], N), rng.uniform(ys[0], ys[-1], N),
                   rng.uniform(-np.pi, np.pi, N)], 1).astype(np.float32)
    y = hj_interp(V, xs, ys, ths, st)
    ntr = 6000
    env = make_env()

    feats = {}
    for tag, cond in ((og_tag, og_cond), (new_tag, new_cond)):
        imgs = render_cond(env, st, cond)
        feats[tag] = encode_feats(wm, imgs, st[:, 2])
        print(f"[{args.tag}] encoded {tag}: {tuple(feats[tag].shape)}")
    dfeat = feats[og_tag].shape[1]
    yt = torch.tensor(y, device=DEV)

    heads = {}
    for tag in (og_tag, new_tag):
        heads[tag] = train_head(feats[tag][:ntr].to(DEV), yt[:ntr], dfeat, seed=args.seed)

    yte = y[ntr:]
    print(f"\n=== dreamer:{args.tag}  axis={args.axis} seed={args.seed}  OG={og_tag} NEW={new_tag}  "
          f"feat={dfeat} frac_safe={(y>=0).mean():.3f} ===")
    print(f"{'margin / eval-on':28s} {'sign-acc':>9s} {'AUC':>7s} {'pearson':>8s}")
    vals = {}
    for htag in (og_tag, new_tag):
        for ztag in (og_tag, new_tag):
            acc, auc, r = score(heads[htag], feats[ztag][ntr:], yte)
            vals[(htag, ztag)] = (acc, auc)
            mark = f"  <- M_{htag} on NEW" if ztag == new_tag else ""
            print(f"M_{htag:<8s} on {ztag:<12s}      {acc:9.3f} {auc:7.3f} {r:8.3f}{mark}")
    ogog_acc, ogog_auc = vals[(og_tag, og_tag)]
    zs_acc, zs_auc = vals[(og_tag, new_tag)]
    or_acc, or_auc = vals[(new_tag, new_tag)]
    print(f"KEY dreamer {args.tag} axis={args.axis} rk=0 seed={args.seed} "
          f"ogog_acc={ogog_acc:.3f} ogog_auc={ogog_auc:.3f} "
          f"zs_acc={zs_acc:.3f} zs_auc={zs_auc:.3f} or_acc={or_acc:.3f} or_auc={or_auc:.3f}")
    print("done")


if __name__ == "__main__":
    main()
