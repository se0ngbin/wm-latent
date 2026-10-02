"""Train a wall-clearance margin head on TwoRoom LE-WM (JEPA) latents.

TwoRoom analog of latent_cbf/.../train_margin_hj_lewm.py. The learned CBF reads a
margin h(z) off the JEPA latent; here the target is the holonomic wall-clearance
value (tworoom_safety.build_wall_sdf): h = dist(agent_center, wall) - agent_r,
positive = safe (no contact).

Two distribution-matching choices, both learned from the Dubins CBF work:
  * label imagined latents (rolled 1..K blocks ahead through the predictor) with
    the clearance at the true holonomically-rolled position — the filter queries h
    on imagined latents, so the head must be accurate there, not just on encoded
    frames.
  * blocks are CONSTANT-velocity (one 2D action repeated `block` times), matching
    the learned filter's 2D action grid; the true state is rolled `block` frames.

Saves {"margin_gp": head.state_dict()} loadable by tworoom_safe_solver (safe_mode
'learned'). Run from le-wm/ in the le-wm venv.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/seongbin/latent/le-wm")
import stable_worldmodel as swm
from utils import get_img_preprocessor
from tworoom_safety import (TwoRoomH, build_wall_sdf, IMG, BORDER, WALL_X,
                            WALL_HALF, DOOR_Y0, DOOR_Y1)
from tworoom_safe_solver import MarginHead


def sample_positions(n, agent_r, rng):
    """Uniform agent positions in the playable region (inside border, not inside
    the wall slab), biased 50% near the wall faces where the filter operates."""
    lo, hi = BORDER + agent_r, IMG - BORDER - agent_r
    pts = rng.uniform(lo, hi, size=(n, 2)).astype(np.float32)
    # near-wall band: x within a few px of either effective face, y anywhere
    n_near = n // 2
    side = rng.integers(0, 2, n_near)
    facex = np.where(side == 0, (WALL_X - WALL_HALF) - agent_r, (WALL_X + WALL_HALF) + agent_r)
    pts[:n_near, 0] = facex + rng.uniform(-3, 3, n_near).astype(np.float32)
    pts[:n_near, 1] = rng.uniform(lo, hi, n_near).astype(np.float32)
    # keep out of the solid slab
    inside = np.abs(pts[:, 0] - WALL_X) < (WALL_HALF + agent_r - 0.5)
    pts[inside, 0] = (WALL_X - WALL_HALF) - agent_r - 1.0
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lewm_ckpt", required=True,
                    help="e.g. lewm_tworoom_jac/weights_epoch_20.pt")
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--n_states", type=int, default=16000)
    ap.add_argument("--rollout_k", type=int, default=5,
                    help="imagine 1..k constant-velocity blocks ahead")
    ap.add_argument("--block", type=int, default=5)
    ap.add_argument("--speed", type=float, default=5.0)
    ap.add_argument("--agent_r", type=float, default=7.0)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--encode_bs", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = args.device
    m = swm.wm.utils.load_pretrained(args.lewm_ckpt).to(dev).eval()
    HS = 3  # history size
    tf = get_img_preprocessor(source="pixels", target="pixels", img_size=IMG)
    H = TwoRoomH(agent_r=args.agent_r, speed=args.speed,
                 door_y=(DOOR_Y0, DOOR_Y1), device=dev)
    hgrid, _ = build_wall_sdf(args.agent_r)
    hgrid_t = torch.tensor(hgrid, device=dev)

    def clearance(xy):  # xy (n,2) px tensor -> h (n,)
        xi = xy[:, 0].clamp(0, IMG - 1).long()
        yi = xy[:, 1].clamp(0, IMG - 1).long()
        return hgrid_t[yi, xi]

    env = swm.envs  # ensure registered
    import gymnasium as gym
    e = gym.make("swm/TwoRoom-v1").unwrapped
    e.reset()

    rng = np.random.default_rng(args.seed)
    pts = sample_positions(args.n_states, args.agent_r, rng)
    labels_np = hgrid[np.clip(pts[:, 1].astype(int), 0, IMG - 1),
                      np.clip(pts[:, 0].astype(int), 0, IMG - 1)]
    print(f"labels: mean={labels_np.mean():.2f} frac_safe={(labels_np >= 0).mean():.3f}")

    def encode_frames(frames):  # (b,C,h,w) -> (b,D)
        with torch.no_grad():
            out = m.encoder(frames.to(dev), interpolate_pos_encoding=True)
            return m.projector(out.last_hidden_state[:, 0])

    all_feats, all_labels = [], []
    blk = args.block
    for i in range(0, args.n_states, args.encode_bs):
        cp = pts[i:i + args.encode_bs]
        b = cp.shape[0]
        imgs = []
        for x, y in cp:
            # randomize the goal (green target) so the head is goal-invariant
            e.reset()
            e.agent_position = torch.tensor([x, y], dtype=torch.float32)
            e.target_position = torch.tensor(
                rng.uniform(BORDER + args.agent_r, IMG - BORDER - args.agent_r, 2),
                dtype=torch.float32)
            frame = e.render()  # HWC uint8
            imgs.append(tf({"pixels": frame[None]})["pixels"][0])
        frames = torch.stack(imgs).float()                     # (b,C,h,w)
        z0 = encode_frames(frames)                             # (b,D)
        all_feats.append(z0.cpu().numpy())
        all_labels.append(labels_np[i:i + args.encode_bs])

        if args.rollout_k > 0:
            # imagined constant-velocity blocks through the predictor
            hist = z0.unsqueeze(1).repeat(1, HS, 1)            # (b,HS,D)
            hist_act = m.action_encoder(torch.zeros(b, HS, 2 * blk, device=dev))  # (b,HS,A_emb)
            x_true = torch.tensor(cp, device=dev)             # (b,2)
            for _k in range(args.rollout_k):
                a2 = torch.tensor(rng.uniform(-1, 1, (b, 2)).astype(np.float32), device=dev)
                a_blk = a2.repeat(1, blk).unsqueeze(1)         # (b,1,10) const-velocity
                ha = torch.cat([hist_act[:, 1:], m.action_encoder(a_blk)], 1)
                pred = m.predict(hist[:, -HS:], ha[:, -HS:])[:, -1]   # (b,D)
                for _f in range(blk):
                    x_true = H.step(x_true, a2)
                all_feats.append(pred.detach().cpu().numpy())
                all_labels.append(clearance(x_true).cpu().numpy())
                hist = torch.cat([hist[:, 1:], pred.unsqueeze(1)], 1)
                hist_act = ha
        if (i // args.encode_bs) % 25 == 0:
            print(f"encoded {i}/{args.n_states}")
    e.close()

    feats = np.concatenate(all_feats, 0)
    labels = np.concatenate(all_labels, 0).astype(np.float32)
    perm = rng.permutation(feats.shape[0])
    feats_t = torch.tensor(feats[perm], device=dev)
    labels_t = torch.tensor(labels[perm], device=dev)
    print(f"dataset: {feats_t.shape[0]} latents "
          f"({args.n_states} encoded + {feats_t.shape[0]-args.n_states} imagined)")
    nval = feats_t.shape[0] // 10
    vf, vy = feats_t[:nval], labels_t[:nval]
    tf_, ty = feats_t[nval:], labels_t[nval:]

    head = MarginHead().to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4)
    for step in range(args.steps):
        idx = torch.randint(0, tf_.shape[0], (args.batch_size,), device=dev)
        loss = torch.nn.functional.mse_loss(head(tf_[idx]), ty[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 500 == 0 or step == args.steps - 1:
            with torch.no_grad():
                vp = head(vf)
                vmse = torch.nn.functional.mse_loss(vp, vy).item()
                sacc = ((vp >= 0) == (vy >= 0)).float().mean().item()
            print(f"step {step}: train_mse={loss.item():.3f} val_mse={vmse:.3f} "
                  f"val_sign_acc={sacc:.3f}")

    import os
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    torch.save({"margin_gp": head.state_dict()}, args.out_path)
    print(f"saved clearance margin head -> {args.out_path}")


if __name__ == "__main__":
    main()
