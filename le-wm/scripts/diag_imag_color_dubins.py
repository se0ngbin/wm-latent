"""Does h's ranking survive the PREDICTOR under recolor? Start states from held-out expert frames (last 10% of episodes),
re-rendered with red and purple circles (training orientation); CEM-style plans as in train_gp_margin_imag_dubins.py;
predictor rollout (1-frame history); sim labels. Prints AUC / balanced acc / median h (safe, unsafe) of each head on
the imagined latents per color, plus the AUC with per-start-frame rank normalization (what h_quantile sees).
Usage: diag_imag_color_dubins.py <wm_dir> <head.pt> [...]"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, h5py
from sklearn.metrics import roc_auc_score
import ood_margin_gp_jepa as E
import stable_worldmodel as swm
from safe_solver import _MarginHead

ck, heads = sys.argv[1], sys.argv[2:]
K, N, DEV = 25, 600, E.DEV
rng = np.random.default_rng(2)
tf = E.get_img_preprocessor(source="pixels", target="pixels", img_size=E.IMG)
tf_fixed = lambda d: tf({"pixels": torch.from_numpy(np.ascontiguousarray(d["pixels"])).permute(0, 3, 1, 2)})
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
safe = ~y
m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(DEV).eval()
Hs = []
for p in heads:
    hd = _MarginHead().to(DEV); hd.load_state_dict(torch.load(p, map_location=DEV)["margin_gp"]); Hs.append(hd.eval())
for cond in E.AXES["color"][:2]:
    with torch.no_grad():
        emb = E.encode(m, E.render_cond(E.DubinsSwmEnv(), st0, cond, tf_fixed)).to(DEV).unsqueeze(1)
        At = torch.from_numpy(A).to(DEV)
        for k in range(K):
            emb = torch.cat([emb, m.predict(emb[:, -3:], m.action_encoder(At[:, :k + 1])[:, -3:])[:, -1:]], 1)
        Zh = emb[:, 1:]
        for p, hd in zip(heads, Hs):
            h = hd(Zh.flatten(0, 1).float()).view(N, K).cpu().numpy()
            print(f"IMAGCOLOR {ck} {os.path.basename(p)} {cond['obstacle']}: AUC {roc_auc_score(safe.ravel(), h.ravel()):.3f} "
                  f"bal-acc {((h[safe] >= 0).mean() + (h[~safe] < 0).mean()) / 2:.3f} "
                  f"h median safe/unsafe {np.median(h[safe]):.2f}/{np.median(h[~safe]):.2f} AUC@k "
                  + " ".join(f"{k + 1}:{roc_auc_score(safe[:, k], h[:, k]):.2f}" for k in (0, 2, 4, 9, 14, 24)
                             if 0 < safe[:, k].mean() < 1), flush=True)
