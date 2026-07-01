"""At a fixed (x,y) boundary point, sweep heading θ and plot the learned critic
V_latent(θ) and margin(θ) vs the HJ ground truth V*(θ) and ℓ. Shows whether the
latent encodes orientation-dependent safety. LE-WM only (single backend)."""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_S = Path(__file__).resolve().parent
sys.path.append(str(_S))
import probe_control_authority as P                       # build_lewm, encode_lewm
from compare_critic_to_hj import load_ddpg, eval_V_batch, build_env
from configs import Config

DEV = "cuda:0"
GT = "/home/seongbin/latent/latent_cbf/results/hj_truth.npz"
LEWM_CK = "/data/seongbin/lewm/checkpoints"
LEWM_OUT = "/data/seongbin/dreamer/lewm"

def latest(d):
    import glob, re
    ps = glob.glob(f"{d}/epoch_id_*/policy.pth")
    return max(ps, key=lambda p: int(re.search(r"epoch_id_(\d+)", p).group(1)))

WMS = [  # (run_dir, label, color)
    ("jacobian_w1_dubins", "jacobian (works)", "C0"),
    ("invariance_dubins",  "invariance (good AUC, filter fails)", "C1"),
    ("no_reg_dubins",      "sigreg only (dead)", "C3"),
]

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=float); ap.add_argument("--y", type=float)
    ap.add_argument("--out", default="/home/seongbin/latent/latent_cbf/results/hj_compare/theta_profile.png")
    args = ap.parse_args()

    gt = np.load(GT)
    V = gt["V"]; xs = gt["grid_xs"]; ys = gt["grid_ys"]; ths = gt["grid_thetas"]; obs = gt["obstacles"]
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    ell = np.full(X.shape, np.inf)
    for cx, cy, r in obs:
        ell = np.minimum(ell, np.hypot(X - cx, Y - cy) - r)
    if args.x is not None and args.y is not None:        # snap explicit point to nearest grid cell
        ix = int(np.argmin(np.abs(xs - args.x))); iy = int(np.argmin(np.abs(ys - args.y)))
    else:                                                # auto: balanced boundary point, max θ-swing
        rng = V.max(2) - V.min(2)
        frac_safe = (V > 0).mean(2)
        ok = (ell > 0.0) & (ell < 0.3) & (frac_safe > 0.3) & (frac_safe < 0.7)
        score = np.where(ok, rng, -np.inf)
        ix, iy = np.unravel_index(np.argmax(score), score.shape)
    x0, y0 = float(xs[ix]), float(ys[iy])
    Vstar = V[ix, iy, :]
    ell0 = float(min(np.hypot(x0 - cx, y0 - cy) - r for cx, cy, r in obs))
    print(f"point (x,y)=({x0:.3f},{y0:.3f})  ℓ={ell0:.3f}  V* range [{Vstar.min():.2f},{Vstar.max():.2f}]")

    # render the θ-sweep at this point (same θ grid as GT, 64 angles)
    env = build_env(Config())
    imgs = []
    for th in ths:
        env.state = np.array([x0, y0, th], dtype=np.float32)
        imgs.append(env.render())
    imgs = np.stack(imgs, 0)

    deg = np.degrees(ths)
    order = np.argsort(deg); deg = deg[order]
    Vstar_p = Vstar[order]
    vmax = float(np.abs(Vstar_p).max())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(deg, Vstar_p, "k-", lw=2.6, label="TRUE  V*(θ)  [HJ]", zorder=5)
    ax.axhline(ell0, ls="--", c="gray", lw=1.3, label=f"ℓ (position only) = {ell0:.2f}")
    for run, label, c in WMS:
        ckpt = sorted(Path(f"{LEWM_CK}/{run}").glob("weights_epoch_*.pt"),
                      key=lambda p: int(p.stem.split("_")[-1]))[-1]
        wm, config = P.build_lewm(str(ckpt), f"{LEWM_OUT}/{run}/margin_heads.pt")
        feats = P.encode_lewm(wm, imgs)
        policy = load_ddpg(latest(f"{LEWM_OUT}/{run}/seed0/PyHJ/gp"),
                           int(feats.shape[-1]), 1, DEV, config)
        Vlat = eval_V_batch(policy, feats)[order]
        amax = float(np.abs(Vlat).max())
        if amax > 0.02:                       # scale to V* amplitude, 0 fixed (sign preserved)
            Vlat = Vlat * (vmax / amax); suf = f"  (×{vmax/amax:.2f})"
        else:
            suf = "  (flat, raw)"
        ax.plot(deg, Vlat, "-", color=c, marker=".", ms=3.5, lw=1.4, label=label + suf)
        del wm, policy, feats; torch.cuda.empty_cache()
    ax.axhline(0, c="k", lw=0.7)
    ax.fill_between(deg, *ax.get_ylim(), where=Vstar_p < 0, color="red", alpha=0.10,
                    label="truly-unsafe heading (V*<0)")
    ax.set_xlabel("heading θ (deg)")
    ax.set_ylabel("safety value  (safe = value > 0)")
    ax.set_title(f"Learned vs true safety at fixed point (x,y)=({x0:.2f},{y0:.2f})\n"
                 "learned critics scaled to V* amplitude (0 fixed) — shape & sign comparable, magnitude not")
    ax.set_xticks([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    ax.legend(fontsize=8, loc="lower center", ncol=2)
    ax.margins(x=0)
    fig.tight_layout()
    fig.savefig(args.out, dpi=135, bbox_inches="tight")
    print("saved", args.out)

if __name__ == "__main__":
    main()
