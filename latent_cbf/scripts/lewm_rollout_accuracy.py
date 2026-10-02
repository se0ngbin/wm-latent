"""Model sanity check: how well does a LE-WM model's autoregressive latent
rollout track reality? Encode a real dubins trajectory's history, roll out
`imagine_with_action` with the REAL action blocks, and compare predicted latents
to the encoded true future latents — normalized by the typical latent distance
between distinct frames (so 1.0 ≈ "as wrong as a random other frame").

Runs the same adapter API the planner uses, so it also validates the rollout.
"""
import sys
import numpy as np
import torch
import h5py

for p in ["/home/seongbin/latent", "/home/seongbin/latent/le-wm",
          "/home/seongbin/latent/latent_cbf/src",
          "/home/seongbin/latent/latent_cbf/src/latent_cbf"]:
    if p not in sys.path:
        sys.path.append(p)

from configs import DreamerConfig
from latent_cbf.adapters import LEWMWorldModel

DUB = "/data/seongbin/lewm/datasets/dubins_expert.h5"
AMEAN, ASTD = -0.02112957, 1.1531051
DEV = "cuda:0"


def load(ckpt):
    cfg = DreamerConfig(); cfg.device = DEV; cfg.lewm_cache_dir = None
    return LEWMWorldModel(cfg, ckpt).to(DEV).eval()


@torch.inference_mode()
def enc(wm, frames):  # frames (T,128,128,3)
    img = torch.from_numpy(frames[None]).float()
    return wm.encoder(wm.preprocess({"image": img}))[0]  # (T, D)


@torch.inference_mode()
def rollout_err(wm, fs, n_eps=30, H=3, K=8, seed=0):
    f = h5py.File(DUB, "r")
    epidx, step, off, elen = f["episode_idx"][:], f["step_idx"][:], f["ep_offset"][:], f["ep_len"][:]
    ids = np.unique(epidx)
    rng = np.random.default_rng(seed)
    need = (H + K) * fs
    # scale: mean latent L2 between random distinct frames
    errs = np.zeros(K); cnt = 0; scale_acc = []
    picked = 0
    for e in rng.permutation(len(ids)):
        if picked >= n_eps:
            break
        base, L = int(off[e]), int(elen[e])
        if L < need:
            continue
        picked += 1
        pix = f["pixels"][base:base + need]
        act = np.asarray(f["action"][base:base + need, 0], np.float32)
        kf = pix[np.arange(H + K) * fs]              # (H+K,128,128,3) keyframes spaced fs
        allemb = enc(wm, kf)                          # (H+K, D)
        blocks = ((act - AMEAN) / ASTD).reshape(H + K, fs)  # (H+K, fs) normalized
        # state from history keyframes 0..H-1; hist_act last = placeholder
        histblk = np.concatenate([blocks[:H - 1], np.zeros((1, fs), np.float32)], 0)
        hist_act = wm.action_encoder(torch.tensor(histblk[None], device=DEV))  # (1,H,A_emb)
        state = {"hist_emb": allemb[:H][None], "hist_act": hist_act,
                 "deter": allemb[H - 1][None], "stoch": torch.zeros(1, 1, device=DEV)}
        # real action blocks driving the rollout: from keyframe H-1 onward
        acts = torch.tensor(blocks[H - 1:H - 1 + K][None], device=DEV)  # (1,K,fs)
        roll = wm.dynamics.imagine_with_action(acts, state)
        pred = roll["deter"][0]                       # (K, D)
        real = allemb[H:H + K]                        # (K, D)
        errs += ((pred - real) ** 2).sum(-1).cpu().numpy()
        cnt += 1
        # scale from these keyframes
        d = torch.cdist(allemb, allemb)
        scale_acc.append(d[d > 0].mean().item() ** 2)
    f.close()
    errs /= max(cnt, 1)
    scale = float(np.mean(scale_acc))
    return errs, scale, cnt


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs5", default="sigreg_dubins_fs5/weights_epoch_50.pt")
    ap.add_argument("--fs1", default="sigreg_only_dubins/weights_epoch_50.pt")
    args = ap.parse_args()

    print(f"[scale = mean squared latent L2 between distinct keyframes; err/scale ~1 == as wrong as a random frame]")
    for name, ckpt, fs, K in [("fs5 (8 AR steps = 2.0u)", args.fs5, 5, 8),
                              ("fs1 (8 AR steps = 0.4u)", args.fs1, 1, 8),
                              ("fs1 (40 AR steps = 2.0u)", args.fs1, 1, 40)]:
        wm = load(ckpt)
        errs, scale, cnt = rollout_err(wm, fs, K=K)
        rel = errs / scale
        print(f"\n{name}  [{cnt} eps]")
        pts = [0, 1, 3, min(7, K - 1), K - 1]
        for t in sorted(set(pts)):
            print(f"  step {t+1:2d}: err/scale = {rel[t]:.3f}")
        del wm
        torch.cuda.empty_cache()
