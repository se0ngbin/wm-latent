"""Red -> purple obstacle colour sweep (Dubins), the analogue of the Reacher floor sweep: obstacle RGB
C(a) = round((1 - a) * red + a * purple), red = (255, 0, 0), purple = (128, 0, 128) (PIL's "purple", the purple OOD view).
Per a and per margin function:
  REAL  4000 uniform states (seed 1, as eval_head_color_dubins.py): AUC / balanced sign acc / median h safe, unsafe.
  IMAG  600 held-out expert start frames rendered at C(a), CEM-style 25-step plans, predictor rollout, sim labels
        (as diag_imag_color_dubins.py): AUC / bal acc over all steps + AUC at steps 1, 5, 10, 25.
Also prints the mean pixel change vs red per a (mean |diff| over pixels and channels, 0-255, raw 224 renders).
Usage: [ALPHAS="0 .1 ..."] color_sweep_dubins.py <wm_dir> <head.pt> [...]"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, h5py
from sklearn.metrics import roc_auc_score
from safe_solver import _load_margin_head   # before E: ood_margin_gp_jepa puts the MAIN checkout's le-wm first on sys.path
import ood_margin_gp_jepa as E
import stable_worldmodel as swm

ck, heads = sys.argv[1], sys.argv[2:]
ALPHAS = [float(a) for a in os.environ.get("ALPHAS", "0 0.1 0.2 0.3 0.4 0.5 0.75 1").split()]
RED, PUR = np.array([255, 0, 0]), np.array([128, 0, 128])
col = lambda a: tuple(int(v) for v in np.round((1 - a) * RED + a * PUR))
cond = lambda a: dict(obstacle=col(a), shape="circle", rot=0)
tf = E.get_img_preprocessor(source="pixels", target="pixels", img_size=E.IMG)
tf_fixed = lambda d: tf({"pixels": torch.from_numpy(np.ascontiguousarray(d["pixels"])).permute(0, 3, 1, 2)})
DEV, K, N = E.DEV, 25, 600

def metrics(safe, h):
    return (roc_auc_score(safe, h), ((h[safe] >= 0).mean() + (h[~safe] < 0).mean()) / 2,
            np.median(h[safe]), np.median(h[~safe]))

# real uniform states (identical draw to eval_head_color_dubins.py)
rng = np.random.default_rng(1); x0, x1, y0, y1 = E.WORLD_BOUNDS; n = 4000
s = np.stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n), rng.uniform(-np.pi, np.pi, n)], 1).astype(np.float32)
safe_r = E.inside_obstacle(s[:, :2], "circle") == 0

# imagined: held-out start frames + plans + sim labels (identical to diag_imag_color_dubins.py)
rng = np.random.default_rng(2)
with h5py.File("/data/seongbin/lewm/datasets/dubins_expert.h5", "r") as f:
    off, ln = f["ep_offset"][:], f["ep_len"][:]
    act_all = f["action"][:].astype(np.float32).reshape(len(f["action"]), -1)
    a_mu, a_sd = np.nanmean(act_all, 0), np.nanstd(act_all, 0)
    e = rng.choice(np.arange(int(0.9 * len(ln)), len(ln)), N)
    st0 = f["state"][:][off[e] + rng.integers(0, ln[e])]
A = (rng.standard_normal((N, 1, 1)) + 0.7 * rng.standard_normal((N, K, 1))).astype(np.float32)
env, y = E.DubinsSwmEnv(), np.zeros((N, K), bool)
for i in range(N):
    env.state = st0[i].astype(np.float32).copy()
    for k in range(K):
        env._integrate(float(np.clip(A[i, k, 0] * a_sd[0] + a_mu[0], -env.max_angular_velocity, env.max_angular_velocity)))
        y[i, k] = E.inside_obstacle(env.state[None, :2], "circle")[0]
safe_i = ~y

m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(DEV).eval()
Hs = [(os.path.basename(p), _load_margin_head(p, DEV).eval()) for p in heads]
raw = lambda a: E.render_cond(E.DubinsSwmEnv(), s[:300], cond(a), lambda d: {"pixels": torch.from_numpy(d["pixels"].copy())})
red_px = raw(0.0)
At = torch.from_numpy(A).to(DEV)
for a in ALPHAS:
    dpx = (raw(a) - red_px).abs().mean().item()
    zr = E.encode(m, E.render_cond(E.DubinsSwmEnv(), s, cond(a), tf_fixed)).to(DEV)
    with torch.no_grad():
        emb = E.encode(m, E.render_cond(E.DubinsSwmEnv(), st0, cond(a), tf_fixed)).to(DEV).unsqueeze(1)
        for k in range(K):
            emb = torch.cat([emb, m.predict(emb[:, -3:], m.action_encoder(At[:, :k + 1])[:, -3:])[:, -1:]], 1)
        zi = emb[:, 1:].flatten(0, 1).float()
        for name, hd in Hs:
            hr = hd(zr).cpu().numpy().ravel(); hi = hd(zi).view(N, K).cpu().numpy()
            R, I = metrics(safe_r, hr), metrics(safe_i.ravel(), hi.ravel())
            atk = " ".join(f"{k + 1}:{roc_auc_score(safe_i[:, k], hi[:, k]):.3f}" for k in (0, 4, 9, 24))
            print(f"SWEEP {ck} {name} a={a:.2f} rgb={col(a)} dpx={dpx:.1f} "
                  f"REAL AUC {R[0]:.3f} bal {R[1]:.3f} med {R[2]:.2f}/{R[3]:.2f} "
                  f"IMAG AUC {I[0]:.3f} bal {I[1]:.3f} med {I[2]:.2f}/{I[3]:.2f} AUC@k {atk}", flush=True)
