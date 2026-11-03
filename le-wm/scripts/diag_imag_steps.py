"""Margin-function training length vs h on IMAGINED latents (what the soft CEM cost reads). Same setup as
diag_imag_color_dubins.py (imag-margin worktree): 600 held-out expert start frames (last 10% of episodes), CEM-style
25-step plans a_t = b + 0.7 eps_t, predictor rollout from a 1-frame history, simulator labels (inside a circle).
Per head and color: AUC, balanced acc, false-safe rate (unsafe imagined steps with h >= 0; the soft cost never fires
on them) overall and at imagined steps 5/10/25, and median h on unsafe steps.
Usage: diag_imag_steps.py <wm_dir> <head.pt> [...]"""
import sys, os
sys.path.insert(0, "/home/seongbin/latent/le-wm"); sys.path.insert(0, "/home/seongbin/latent/le-wm/scripts")
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
            fs = lambda k: (h[:, k][~safe[:, k]] >= 0).mean()
            print(f"IMAGSTEPS {ck} {os.path.basename(p)} {cond['obstacle']}: AUC {roc_auc_score(safe.ravel(), h.ravel()):.3f} "
                  f"bal-acc {((h[safe] >= 0).mean() + (h[~safe] < 0).mean()) / 2:.3f} "
                  f"false-safe {(h[~safe] >= 0).mean():.3f} @5 {fs(4):.3f} @10 {fs(9):.3f} @25 {fs(24):.3f} "
                  f"h_med_unsafe {np.median(h[~safe]):.3f} unsafe_frac {(~safe).mean():.3f}", flush=True)
