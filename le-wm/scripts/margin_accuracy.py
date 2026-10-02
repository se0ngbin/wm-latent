"""Consistent margin/value-function accuracy vs ground truth for every deployed
learned-CBF head. Same held-out states, same metrics, so the numbers ARE
comparable across models (unlike the per-training-run val logs).

For each model we load its encoder + its deployed margin head, encode a common
set of rendered states, and score the head's prediction h(z) against the ground
truth value:
  * Dubins:  ground truth = HJ avoid value V*(x,y,theta)  (hj_truth.npz)
  * TwoRoom: ground truth = signed wall-clearance         (build_wall_sdf)
Metrics: sign-accuracy (safe/unsafe agreement), ROC-AUC, Pearson r.
NOTE: this is the STATIC (encoded-frame) accuracy — the honest ceiling. The
in-loop filter also queries the head on multi-step IMAGINED latents, which is
where it degrades; that is measured separately by the collision results.
"""
import os, sys
sys.path.insert(0, "/home/seongbin/latent/le-wm")
os.environ.setdefault("STABLEWM_HOME", "/data/seongbin/lewm")
os.environ.setdefault("LOCAL_DATASET_DIR", "/data/seongbin/lewm")
import numpy as np, torch
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
import gymnasium as gym
import stable_worldmodel as swm
from utils import get_img_preprocessor
from tworoom_safe_solver import MarginHead
from tworoom_safety import build_wall_sdf, IMG, BORDER

DEV, N = "cuda:0", 8000
MDIR = "/data/seongbin/dreamer/lewm"
HJ = "/home/seongbin/latent/latent_cbf/results/hj_truth.npz"

DUBINS = [  # label, weights, margin_ckpt
    ("baseline (sigreg)",   "sigreg_only_dubins/weights_epoch_50.pt",   "sigreg_only_dubins/margin_hj_imag.pt"),
    ("jacobian",            "jacobian_w1_dubins/weights_epoch_50.pt",   "jacobian_w1_dubins/margin_hj_imag.pt"),
    ("jacobian (on-dist)",  "jacobian_w1_dubins/weights_epoch_50.pt",   "jacobian_w1_dubins/margin_hj_ondist.pt"),
    ("jacobian+pull",       "lewm_dubins_jacpull50/weights_epoch_50.pt","lewm_dubins_jacpull50/margin_hj_imag.pt"),
    ("jacobian+pull(on-d)", "lewm_dubins_jacpull50/weights_epoch_50.pt","lewm_dubins_jacpull50/margin_hj_ondist.pt"),
]
TWOROOM = [
    ("base (sigreg)", "lewm_tworoom_base/weights_epoch_20.pt", "lewm_tworoom_base/margin_clearance.pt"),
    ("jacobian",      "lewm_tworoom_jac/weights_epoch_20.pt",  "lewm_tworoom_jac/margin_clearance.pt"),
]


def hj_interp(V, xs, ys, ths, pts):
    def idxw(g, v, periodic=False):
        if periodic: v = (v - g[0]) % (2*np.pi) + g[0]
        i = np.clip(np.searchsorted(g, v) - 1, 0, len(g)-2)
        return i, np.clip((v - g[i]) / (g[i+1]-g[i]+1e-9), 0, 1)
    ix, wx = idxw(xs, pts[:,0]); iy, wy = idxw(ys, pts[:,1]); it, wt = idxw(ths, pts[:,2], True)
    out = np.zeros(len(pts))
    for dx in (0,1):
        for dy in (0,1):
            for dt in (0,1):
                w = (wx if dx else 1-wx)*(wy if dy else 1-wy)*(wt if dt else 1-wt)
                out += w * V[np.clip(ix+dx,0,len(xs)-1), np.clip(iy+dy,0,len(ys)-1), (it+dt)%len(ths)]
    return out


def encode(m, frames, bs=64):
    Z = []
    with torch.no_grad():
        for i in range(0, frames.shape[0], bs):
            out = m.encoder(frames[i:i+bs].to(DEV), interpolate_pos_encoding=True)
            Z.append(m.projector(out.last_hidden_state[:, 0]).cpu())
    return torch.cat(Z)


def score(head, Z, y):
    with torch.no_grad():
        p = head(Z.to(DEV)).cpu().numpy()
    acc = float(((p >= 0) == (y >= 0)).mean())
    auc = float(roc_auc_score((y >= 0).astype(int), p)) if 0 < (y>=0).mean() < 1 else float("nan")
    r = float(pearsonr(p, y)[0])
    return acc, auc, r


def run_env(name, models, frames, y, header):
    print(f"\n=== {name}  (N={len(y)}, frac_safe={ (y>=0).mean():.3f}) ===")
    print(f"{'model':22s} {'sign-acc':>9s} {'AUC':>7s} {'pearson r':>10s}")
    for label, wpath, mpath in models:
        m = swm.wm.utils.load_pretrained(wpath).to(DEV).eval()
        Z = encode(m, frames)
        del m; torch.cuda.empty_cache()
        head = MarginHead().to(DEV)
        head.load_state_dict(torch.load(os.path.join(MDIR, mpath), map_location=DEV)["margin_gp"])
        head.eval()
        acc, auc, r = score(head, Z, y)
        print(f"{label:22s} {acc:9.3f} {auc:7.3f} {r:10.3f}")


def main():
    tf = get_img_preprocessor(source="pixels", target="pixels", img_size=IMG)

    # ---- Dubins ----
    from dubins_swm_env import DubinsSwmEnv
    d = np.load(HJ); V, xs, ys, ths = d["V"], d["grid_xs"], d["grid_ys"], d["grid_thetas"]
    rng = np.random.default_rng(0)
    st = np.stack([rng.uniform(xs[0], xs[-1], N), rng.uniform(ys[0], ys[-1], N),
                   rng.uniform(-np.pi, np.pi, N)], 1).astype(np.float32)
    yd = hj_interp(V, xs, ys, ths, st).astype(np.float32)
    env = DubinsSwmEnv()
    imgs = []
    for s in st:
        env.state = s.copy()
        imgs.append(tf({"pixels": env._render_image()[None]})["pixels"][0])
    run_env("Dubins (ground truth: HJ value V*)", DUBINS, torch.stack(imgs).float(), yd, "HJ")

    # ---- TwoRoom ----
    hgrid, _ = build_wall_sdf(agent_r=7.0)
    e = gym.make("swm/TwoRoom-v1").unwrapped; e.reset()
    lo, hi = BORDER + 7.0, IMG - BORDER - 7.0
    pos = rng.uniform(lo, hi, size=(N, 2)).astype(np.float32)
    # 50% near-wall band (where the filter operates), matching training sampling
    nn = N // 2
    pos[:nn, 0] = np.where(rng.integers(0,2,nn)==0, 105-7.0, 119+7.0) + rng.uniform(-3,3,nn)
    pos[:nn, 1] = rng.uniform(lo, hi, nn)
    inside = np.abs(pos[:,0]-112) < (5+7.0-0.5); pos[inside,0] = 105-7.0-1
    yt = hgrid[np.clip(pos[:,1].astype(int),0,IMG-1), np.clip(pos[:,0].astype(int),0,IMG-1)].astype(np.float32)
    imgs = []
    for x, y in pos:
        e.reset()
        e.agent_position = torch.tensor([x, y], dtype=torch.float32)
        e.target_position = torch.tensor(rng.uniform(lo, hi, 2), dtype=torch.float32)
        imgs.append(tf({"pixels": e.render()[None]})["pixels"][0])
    run_env("TwoRoom (ground truth: wall-clearance)", TWOROOM, torch.stack(imgs).float(), yt, "clear")
    print("\ndone")


if __name__ == "__main__":
    main()
