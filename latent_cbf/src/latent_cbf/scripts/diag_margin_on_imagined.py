"""Diagnose the HJ-labeled margin head on *imagined* latents.

The safe action map evaluates h on predictor outputs, but the head is trained
on encoder outputs of real frames. This script measures, for random states and
random actions:
  1. h(encode(x))            vs V*(x)         (training distribution)
  2. h(predict(encode(x),a)) vs V*(x_next)    (planner distribution, 1..k steps)

High (1) but low (2) = distribution shift between encoded and imagined latents
is what breaks learned-mode safe planning.

Usage (from latent_cbf/):
  STABLEWM_HOME=/data/seongbin/lewm .venv/bin/python \
      src/latent_cbf/scripts/diag_margin_on_imagined.py \
      --margin_ckpt /data/seongbin/dreamer/lewm/sigreg_only_dubins/margin_hj.pt
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
for _p in ("/home/seongbin/latent", "/home/seongbin/latent/le-wm"):
    if _p not in sys.path:
        sys.path.append(_p)

from configs import get_default_config
from scripts.run_experiment import create_env_from_config
from train_margin_hj_lewm import hj_interp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lewm_ckpt", default="sigreg_only_dubins/weights_epoch_50.pt")
    ap.add_argument("--margin_ckpt", required=True)
    ap.add_argument("--hj_path", default=str(_THIS.parents[3] / "results/hj_truth.npz"))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--rollout_k", type=int, default=8)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dc_mod = __import__("configs", fromlist=["DreamerConfig"])
    dcfg = dc_mod.DreamerConfig()
    dcfg.device = args.device

    from latent_cbf.adapters import LEWMWorldModel
    from latent_cbf.controllers.safe_action_map import GroundTruthH

    wm = LEWMWorldModel(dcfg, args.lewm_ckpt).to(args.device).eval()
    sd = torch.load(args.margin_ckpt, map_location=args.device)
    wm.heads["margin_gp"].load_state_dict(sd["margin_gp"])
    head = wm.heads["margin_gp"]

    config = get_default_config()
    env = create_env_from_config(config)
    env.reset()
    ec = config.environment
    gt = GroundTruthH(
        obstacles=ec.get_obstacles_list(), speed=ec.speed, dt=ec.dt,
        act_mean=0.0, act_std=1.0, act_low=-ec.max_angular_velocity,
        act_high=ec.max_angular_velocity, device=args.device,
        h_source="hj", hj_path=args.hj_path,
    )
    hj = np.load(args.hj_path)
    V, xs, ys, ths = hj["V"], hj["grid_xs"], hj["grid_ys"], hj["grid_thetas"]

    H = wm.history_size
    rng = np.random.default_rng(args.seed)
    pts = rng.uniform([xs[0], ys[0], -np.pi], [xs[-1], ys[-1], np.pi],
                      size=(args.n, 3)).astype(np.float32)
    acts = rng.uniform(-2.0, 2.0, size=(args.n, args.rollout_k)).astype(np.float32)

    h_enc, h_img = [], []          # head on encoded / imagined latents
    v_true0, v_truek = [], []      # V* at x0 / x_k
    bs = 64
    for i in range(0, args.n, bs):
        chunk = pts[i : i + bs]
        b = chunk.shape[0]
        imgs = []
        for x, y, th in chunk:
            env.state = np.array([x, y, th], dtype=np.float32)
            imgs.append(env.render())
        img = np.stack(imgs, 0)[:, None].repeat(H, 1)  # (b, H, Hh, Ww, 3) static history
        with torch.no_grad():
            data = wm.preprocess({"image": torch.from_numpy(img).float()})
            emb = wm.encoder(data)  # (b, H, D)
            a = torch.tensor(acts[i : i + bs], device=args.device)  # (b, k)
            roll = wm.dynamics.imagine_with_action(
                a.unsqueeze(-1),
                {
                    "deter": emb[:, -1],
                    "stoch": torch.zeros(b, 1, device=args.device),
                    "hist_emb": emb,
                    "hist_act": wm.action_encoder(
                        torch.zeros(b, H, 1, device=args.device)
                    ),
                },
            )
            h_enc.append(head(emb[:, -1]).squeeze(-1).cpu().numpy())
            h_img.append(head(roll["deter"][:, -1]).squeeze(-1).cpu().numpy())
        # true rollout
        x_t = torch.tensor(chunk, device=args.device)
        for k in range(args.rollout_k):
            x_t = gt.step(x_t, a[:, k])
        v_true0.append(hj_interp(V, xs, ys, ths, chunk))
        v_truek.append(hj_interp(V, xs, ys, ths, x_t.cpu().numpy()))
    env.close()

    h_enc, h_img = np.concatenate(h_enc), np.concatenate(h_img)
    v0, vk = np.concatenate(v_true0), np.concatenate(v_truek)

    def report(name, pred, true):
        corr = np.corrcoef(pred, true)[0, 1]
        sign = ((pred >= 0) == (true >= 0)).mean()
        # false-safe: predicted safe but truly unsafe (the dangerous error)
        fs = ((pred >= 0) & (true < 0)).mean()
        print(f"{name}: corr={corr:.3f} sign_acc={sign:.3f} false_safe={fs:.3f} "
              f"pred[{pred.min():.2f},{pred.max():.2f}] true[{true.min():.2f},{true.max():.2f}]")

    report("h(enc(x0))      vs V*(x0)", h_enc, v0)
    report(f"h(imag k={args.rollout_k}) vs V*(xk)", h_img, vk)


if __name__ == "__main__":
    main()
