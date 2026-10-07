"""Train an HJ-labeled margin head on LE-WM latents.

The failure-labeled margin heads (train_margin_lewm.py) learn a distance-like
separator, which is NOT a valid CBF for Dubins: angular velocity affects the
distance only at O(dt^2), so the safe-by-construction action map hits
inevitable-collision states before the constraint can act (verified online:
gt_h=dist collides, gt_h=hj does not). This script instead regresses a margin
head onto the HJ avoid value V*(x, y, theta) — whose zero-superlevel set is
control-invariant — evaluated at randomly sampled states rendered through the
env and encoded by the frozen LE-WM.

Saves {"margin_gp": state_dict} so the planner's --margin_ckpt loads it as-is.

Usage (from latent_cbf/):
  STABLEWM_HOME=/data/seongbin/lewm .venv/bin/python \
      src/latent_cbf/scripts/train_margin_hj_lewm.py \
      --lewm_ckpt sigreg_only_dubins/weights_epoch_50.pt \
      --out_path /data/seongbin/dreamer/lewm/sigreg_only_dubins/margin_hj.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_THIS = Path(__file__).resolve()
for p in [str(_THIS.parents[3]), str(_THIS.parents[2]), str(_THIS.parents[1])]:
    if p not in sys.path:
        sys.path.append(p)
# le-wm (jepa.py / module.py) must be importable for load_pretrained.
for _p in ("/home/seongbin/latent", "/home/seongbin/latent/le-wm"):
    if _p not in sys.path:
        sys.path.append(_p)

from configs import get_default_config
from scripts.run_experiment import create_env_from_config


def _render_square(env):
    """DubinsEnv._render_image with obstacles drawn as 45-deg diamonds (half-width
    r) instead of circles. Appearance/shape nuisance only — the COLLISION geometry
    (and thus the HJ V* labels) is still the circle, so training + eval stay
    label-consistent as long as both use this same renderer."""
    import math as _m
    from PIL import Image, ImageDraw
    scale = 4
    h = (env.image_size[0] * scale, env.image_size[1] * scale)
    img = Image.new("RGB", h, env.colors["background"]); draw = ImageDraw.Draw(img)
    def w2p(c):
        x, y = c
        return (int((x - env.x_min) / (env.x_max - env.x_min) * h[0]),
                int((env.y_max - y) / (env.y_max - env.y_min) * h[1]))
    for ox, oy, r in env.obstacles:
        cx, cy = w2p((ox, oy)); rp = r / (env.x_max - env.x_min) * h[0]
        draw.polygon([(cx, cy - rp), (cx + rp, cy), (cx, cy + rp), (cx - rp, cy)],
                     fill=env.colors["obstacle"])
    gc = w2p(env.goal_position); gr = env.goal_radius / (env.x_max - env.x_min) * h[0]
    draw.ellipse([(gc[0] - gr, gc[1] - gr), (gc[0] + gr, gc[1] + gr)],
                 fill=env.colors["goal"])
    env._draw_agent(draw, w2p(env.state[:2]), float(env.state[2]), scale)
    return np.array(img.resize(env.image_size, Image.Resampling.LANCZOS))


def hj_interp(V, xs, ys, ths, pts):
    """Trilinear interpolation (numpy) of V at pts (N, 3); theta wraps."""
    nx, ny, nth = V.shape
    fx = np.clip((pts[:, 0] - xs[0]) / (xs[1] - xs[0]), 0, nx - 1 - 1e-6)
    fy = np.clip((pts[:, 1] - ys[0]) / (ys[1] - ys[0]), 0, ny - 1 - 1e-6)
    th = np.arctan2(np.sin(pts[:, 2]), np.cos(pts[:, 2]))
    ft = (th - ths[0]) / (ths[1] - ths[0])
    i0, j0 = fx.astype(int), fy.astype(int)
    k0 = np.floor(ft).astype(int) % nth
    i1, j1 = np.minimum(i0 + 1, nx - 1), np.minimum(j0 + 1, ny - 1)
    k1 = (k0 + 1) % nth
    wx, wy, wt = fx - i0, fy - j0, np.clip(ft - np.floor(ft), 0, 1)
    out = 0.0
    for di, wxi in ((i0, 1 - wx), (i1, wx)):
        for dj, wyj in ((j0, 1 - wy), (j1, wy)):
            for dk, wtk in ((k0, 1 - wt), (k1, wt)):
                out = out + V[di, dj, dk] * wxi * wyj * wtk
    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lewm_ckpt", default="sigreg_only_dubins/weights_epoch_50.pt")
    ap.add_argument("--hj_path", default=str(_THIS.parents[3] / "results/hj_truth.npz"))
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--n_states", type=int, default=20000)
    ap.add_argument("--rollout_k", type=int, default=8,
                    help="also train on latents imagined 1..k steps ahead "
                         "(labeled with V* at the RK4-propagated true state); "
                         "0 disables. The planner evaluates h on imagined "
                         "latents, so training only on encoded frames leaves "
                         "the head off-distribution there.")
    ap.add_argument("--near_obstacle_frac", type=float, default=0.0,
                    help="fraction of start states drawn from an annulus around "
                         "the obstacles (radius r..r+1.0) instead of uniform, to "
                         "match the sg25clean eval distribution where the CEM "
                         "planner dwells and the imagined-margin fallback spikes.")
    ap.add_argument("--obstacle_color", default="red",
                    help="OOD appearance: obstacle fill color (PIL name/hex). "
                         "geometry/labels unchanged -> pure nuisance.")
    ap.add_argument("--obstacle_shape", default="circle", choices=["circle", "square"],
                    help="OOD appearance: 'square' draws a 45-deg diamond (label "
                         "geometry stays circular).")
    ap.add_argument("--rotate_deg", type=int, default=0, choices=[0, 90, 180, 270],
                    help="OOD viewpoint: rigidly rotate the rendered scene by this "
                         "many degrees (np.rot90). Labels are preserved exactly by "
                         "the rotation-equivariance of the Dubins+obstacle geometry.")
    ap.add_argument("--fresh_head", action="store_true",
                    help="re-initialize the margin head instead of starting from "
                         "the pretrained margin_gp weights (fair OOD oracle).")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--encode_bs", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    config = get_default_config()
    dc_mod = __import__("configs", fromlist=["DreamerConfig"])
    dcfg = dc_mod.DreamerConfig()
    dcfg.device = args.device

    from latent_cbf.adapters import LEWMWorldModel

    wm = LEWMWorldModel(dcfg, args.lewm_ckpt).to(args.device)
    wm.eval()

    env = create_env_from_config(config)
    env.reset()
    # OOD appearance overrides (WM frozen; only rendering changes).
    if args.obstacle_color != "red":
        env.colors["obstacle"] = args.obstacle_color
    if args.obstacle_shape == "square":
        import types
        env._render_image = types.MethodType(_render_square, env)
    print(f"obstacle appearance: color={args.obstacle_color} shape={args.obstacle_shape} "
          f"rotate={args.rotate_deg}")

    hj = np.load(args.hj_path)
    V, xs, ys, ths = hj["V"], hj["grid_xs"], hj["grid_ys"], hj["grid_thetas"]

    rng = np.random.default_rng(args.seed)
    pts = rng.uniform(
        [xs[0], ys[0], -np.pi], [xs[-1], ys[-1], np.pi], size=(args.n_states, 3)
    ).astype(np.float32)
    # On-distribution emphasis: redraw a fraction of (x,y) in an annulus around
    # each obstacle (the region the sg25clean planner dwells in and where the
    # imagined-margin fallback concentrates). theta stays uniform.
    if args.near_obstacle_frac > 0:
        obs = config.environment.get_obstacles_list()  # [(cx, cy, r), ...]
        n_near = int(args.n_states * args.near_obstacle_frac)
        cxy = np.array([[o[0], o[1]] for o in obs], dtype=np.float32)
        rr = np.array([o[2] for o in obs], dtype=np.float32)
        which = rng.integers(0, len(obs), size=n_near)
        rad = rr[which] + rng.uniform(0.0, 1.0, size=n_near).astype(np.float32)
        ang = rng.uniform(-np.pi, np.pi, size=n_near).astype(np.float32)
        near = cxy[which] + np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)
        near = np.clip(near, [xs[0], ys[0]], [xs[-1], ys[-1]])
        pts[:n_near, :2] = near.astype(np.float32)
        print(f"near-obstacle start states: {n_near}/{args.n_states}")
    labels = hj_interp(V, xs, ys, ths, pts)
    print(f"labels: mean={labels.mean():.3f} frac_safe={(labels >= 0).mean():.3f}")

    # Ground-truth dynamics for propagating labels along imagined rollouts.
    ec = config.environment
    from latent_cbf.controllers.safe_action_map import GroundTruthH

    gt = GroundTruthH(
        obstacles=ec.get_obstacles_list(), speed=ec.speed, dt=ec.dt,
        act_mean=0.0, act_std=1.0, act_low=-ec.max_angular_velocity,
        act_high=ec.max_angular_velocity, device=args.device,
        h_source="dist",  # dynamics only; labels come from hj_interp below
    )
    H = wm.history_size

    # Render + encode all sampled states; optionally roll each state forward
    # 1..rollout_k imagined steps and label those latents with V*(x_step).
    all_feats, all_labels = [], []
    rk = args.rollout_k
    for i in range(0, args.n_states, args.encode_bs):
        chunk_pts = pts[i : i + args.encode_bs]
        b = chunk_pts.shape[0]
        imgs = []
        for x, y, th in chunk_pts:
            # re-randomize the goal disk per frame: the head must be invariant
            # to goal position (it varies per episode at eval time).
            env.reset()
            env.state = np.array([x, y, th], dtype=np.float32)
            im = env.render()
            if args.rotate_deg:
                im = np.ascontiguousarray(np.rot90(im, k=args.rotate_deg // 90))
            imgs.append(im)
        # static history of the same frame (planner history converges to
        # imagined embeddings anyway; t=0 uses real frames).
        chunk = np.stack(imgs, 0)[:, None].repeat(H, 1)  # (b, H, Hh, Ww, 3)
        with torch.no_grad():
            data = wm.preprocess({"image": torch.from_numpy(chunk).float()})
            embed = wm.encoder(data)  # (b, H, D)
        all_feats.append(embed[:, -1].cpu().numpy())
        all_labels.append(labels[i : i + args.encode_bs])

        if rk > 0:
            a = torch.tensor(
                rng.uniform(-2.0, 2.0, size=(b, rk)).astype(np.float32),
                device=args.device,
            )
            with torch.no_grad():
                roll = wm.dynamics.imagine_with_action(
                    a.unsqueeze(-1),
                    {
                        "deter": embed[:, -1],
                        "stoch": torch.zeros(b, 1, device=args.device),
                        "hist_emb": embed,
                        "hist_act": wm.action_encoder(
                            torch.zeros(b, H, 1, device=args.device)
                        ),
                    },
                )
            x_t = torch.tensor(chunk_pts, device=args.device)
            for k in range(rk):
                x_t = gt.step(x_t, a[:, k])
                all_feats.append(roll["deter"][:, k].cpu().numpy())
                all_labels.append(
                    hj_interp(V, xs, ys, ths, x_t.cpu().numpy())
                )
        if (i // args.encode_bs) % 20 == 0:
            print(f"encoded {i}/{args.n_states}")
    env.close()

    feats_np = np.concatenate(all_feats, 0)
    labels_np = np.concatenate(all_labels, 0)
    perm = rng.permutation(feats_np.shape[0])
    feats_t = torch.tensor(feats_np[perm], device=args.device)
    labels_t = torch.tensor(labels_np[perm], device=args.device)
    print(f"dataset: {feats_t.shape[0]} latents "
          f"({args.n_states} encoded + {feats_t.shape[0] - args.n_states} imagined)")
    n_val = feats_t.shape[0] // 10
    val_f, val_y = feats_t[:n_val], labels_t[:n_val]
    tr_f, tr_y = feats_t[n_val:], labels_t[n_val:]

    head = wm.heads["margin_gp"]  # fresh MLP; we overwrite it with HJ training
    if args.fresh_head:
        # Re-initialize from scratch. The pretrained margin_gp weights are a
        # hostile init for OOD-appearance static fits (rk=0) and can leave the
        # optimizer in a bad basin (observed: purple oracle worse than the
        # transferred in-dist head). A clean init makes the oracle a fair ceiling.
        for mod in head.modules():
            if hasattr(mod, "reset_parameters"):
                mod.reset_parameters()
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)
    for step in range(args.steps):
        idx = torch.randint(0, tr_f.shape[0], (args.batch_size,), device=args.device)
        pred = head(tr_f[idx]).squeeze(-1)
        loss = torch.nn.functional.mse_loss(pred, tr_y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == args.steps - 1:
            with torch.no_grad():
                vp = head(val_f).squeeze(-1)
                vmse = torch.nn.functional.mse_loss(vp, val_y).item()
                sign_acc = ((vp >= 0) == (val_y >= 0)).float().mean().item()
            print(f"step {step}: train_mse={loss.item():.4f} val_mse={vmse:.4f} "
                  f"val_sign_acc={sign_acc:.3f}")

    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"margin_gp": head.state_dict()}, out)
    print(f"saved HJ-labeled margin head -> {out}")


if __name__ == "__main__":
    main()
