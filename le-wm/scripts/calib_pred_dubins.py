"""Per-step threshold recalibration of a Dubins margin function on PREDICTED latents, per condition (the Dubins port of
reacher_calib_pred.py). CCEM reads h on the current real frame + K predicted latents; a condition (purple, rot 90) and the
prediction depth both shift h's level, so a fixed threshold 0 is wrong. Per condition and WM:
  calibration starts = NCAL states sampled uniformly in the free space (half within 0.35 of an obstacle edge), random
  heading; NOT dataset states, so disjoint from the CEM episodes (which start from dataset states). Rendered under the
  condition. PPS random CEM-style K-step plans per start (as diag_imag_color_dubins.py); predicted latents zhat_k from a
  1-frame history; labels = the plan replayed in the simulator (inside an obstacle at step k).
  c_0 (current real frame): NREAL uniform states (half unsafe) rendered under the condition, real latents.
Offsets c_k (planner uses h - c_k), k = 0..K:
  labeled  : the threshold with the best balanced accuracy on safe + unsafe states
  safe_q05 : the 5% quantile of h over SAFE states only (no failure data)
Starts are split in two halves: offsets fit on the first, per-step AUC / balanced accuracy (uncalibrated, labeled,
safe_q05) reported on the second. Writes {wm: {head: {cond: {variant: [c_0..c_K]}}}} into OUT (merged).
Usage: calib_pred_dubins.py <wm_dir> <head.pt> [...]   env: CONDS (red,purple,rot90), NCAL (400), PPS (4), K (15), OUT"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, h5py
from sklearn.metrics import roc_auc_score
from safe_solver import _load_margin_head   # before E: ood_margin_gp_jepa puts the MAIN checkout's le-wm first on sys.path
import ood_margin_gp_jepa as E
import stable_worldmodel as swm

G = os.environ.get
ck, heads = sys.argv[1], sys.argv[2:]
CONDS = G("CONDS", "red,purple,rot90").split(",")
NCAL, PPS, K, NREAL = int(G("NCAL", "400")), int(G("PPS", "4")), int(G("K", "15")), int(G("NREAL", "2000"))
OUT = G("OUT", "/data/seongbin/lewm/dubins_calib/offsets_pred.json")
COND = {"red": dict(obstacle="red", shape="circle", rot=0), "purple": dict(obstacle="purple", shape="circle", rot=0),
        "rot90": dict(obstacle="red", shape="circle", rot=90)}
DEV = E.DEV
rng = np.random.default_rng(7)
tf = E.get_img_preprocessor(source="pixels", target="pixels", img_size=E.IMG)
tf_fixed = lambda d: tf({"pixels": torch.from_numpy(np.ascontiguousarray(d["pixels"])).permute(0, 3, 1, 2)})
with h5py.File("/data/seongbin/lewm/datasets/dubins_expert.h5", "r") as f:
    act_all = f["action"][:].astype(np.float32).reshape(len(f["action"]), -1)
a_mu, a_sd = np.nanmean(act_all, 0), np.nanstd(act_all, 0)
x0, x1, y0, y1 = E.WORLD_BOUNDS
env = E.DubinsSwmEnv()


def sample_states(n, want_unsafe=None, near_frac=0.5):
    """Uniform states; want_unsafe None -> free space (half near an obstacle edge), True/False -> that label."""
    out = []
    while len(out) < n:
        s = np.stack([rng.uniform(x0, x1, 4 * n), rng.uniform(y0, y1, 4 * n), rng.uniform(-np.pi, np.pi, 4 * n)], 1).astype(np.float32)
        u = E.inside_obstacle(s[:, :2], "circle").astype(bool)
        if want_unsafe is None:
            free = s[~u]
            # distance to the nearest unsafe point, approximated by probing a ring of radius 0.35
            ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            ring = free[:, None, :2] + 0.35 * np.stack([np.cos(ang), np.sin(ang)], 1)[None]
            near = E.inside_obstacle(ring.reshape(-1, 2), "circle").astype(bool).reshape(len(free), -1).any(1)
            k = n // 2
            out = list(free[near][:k]) + list(free[~near][:n - k])
            if len(out) < n: out = []
        else:
            out += list(s[u == want_unsafe])
    return np.array(out[:n], np.float32)


starts = sample_states(NCAL)
A = (rng.standard_normal((NCAL, PPS, 1, 1)) + 0.7 * rng.standard_normal((NCAL, PPS, K, 1))).astype(np.float32)
y = np.zeros((NCAL, PPS, K), bool)
for i in range(NCAL):
    for p in range(PPS):
        env.state = starts[i].copy()
        for k in range(K):
            env._integrate(float(np.clip(A[i, p, k, 0] * a_sd[0] + a_mu[0], -env.max_angular_velocity, env.max_angular_velocity)))
            y[i, p, k] = bool(E.inside_obstacle(env.state[None, :2], "circle")[0])
real = np.concatenate([sample_states(NREAL // 2, False), sample_states(NREAL // 2, True)])
real_y = np.r_[np.zeros(NREAL // 2, bool), np.ones(NREAL // 2, bool)]
fit, ev = np.arange(NCAL) < NCAL // 2, np.arange(NCAL) >= NCAL // 2
rfit = np.random.default_rng(8).random(NREAL) < 0.5


def bal(h, unsafe, c=0.0):
    sp = (h - c) >= 0
    return (sp[~unsafe].mean() + (~sp[unsafe]).mean()) / 2


def best_thr(h, unsafe):
    cand = np.quantile(h, np.linspace(0.005, 0.995, 199))
    return float(cand[np.argmax([bal(h, unsafe, c) for c in cand])])


m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(DEV).eval()
Hs = [_load_margin_head(p, DEV).eval() for p in heads]
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
At = torch.from_numpy(A.reshape(NCAL * PPS, K, 1)).to(DEV)
for cn in CONDS:
    cond = COND[cn]
    with torch.no_grad():
        z0 = E.encode(m, E.render_cond(E.DubinsSwmEnv(), starts, cond, tf_fixed)).to(DEV)
        emb = z0.repeat_interleave(PPS, 0).unsqueeze(1)                       # (NCAL*PPS, 1, D)
        for k in range(K):
            emb = torch.cat([emb, m.predict(emb[:, -3:], m.action_encoder(At[:, :k + 1])[:, -3:])[:, -1:]], 1)
        Zh = emb[:, 1:]
        Zr = E.encode(m, E.render_cond(E.DubinsSwmEnv(), real, cond, tf_fixed)).to(DEV)
    for p, hd in zip(heads, Hs):
        with torch.no_grad():
            h = hd(Zh.flatten(0, 1).float()).view(NCAL, PPS, K).cpu().numpy()
            hr = hd(Zr.float()).cpu().numpy()
        off = {"labeled": [best_thr(hr[rfit], real_y[rfit])], "safe_q05": [float(np.quantile(hr[rfit & ~real_y], 0.05))]}
        rows = [f"  k=0 (real) AUC {roc_auc_score(~real_y[~rfit], hr[~rfit]):.3f} bal {bal(hr[~rfit], real_y[~rfit]):.3f} "
                f"-> lab {bal(hr[~rfit], real_y[~rfit], off['labeled'][0]):.3f} q05 {bal(hr[~rfit], real_y[~rfit], off['safe_q05'][0]):.3f}"]
        for k in range(K):
            hf, yf = h[fit, :, k].ravel(), y[fit, :, k].ravel()
            he, ye = h[ev, :, k].ravel(), y[ev, :, k].ravel()
            # labeled needs both classes: with < 20 unsafe predicted states at this step the fit is noise -> no offset
            off["labeled"].append(best_thr(hf, yf) if yf.sum() >= 20 else 0.0); off["safe_q05"].append(float(np.quantile(hf[~yf], 0.05)))
            if k + 1 in (1, 3, 5, 10, 15) and 0 < ye.mean() < 1:
                rows.append(f"  k={k + 1} unsafe {ye.mean():.2f} AUC {roc_auc_score(~ye, he):.3f} bal {bal(he, ye):.3f} "
                            f"-> lab {bal(he, ye, off['labeled'][-1]):.3f} q05 {bal(he, ye, off['safe_q05'][-1]):.3f}")
        print(f"CALIB {ck} {os.path.basename(p)} {cn}:\n" + "\n".join(rows), flush=True)
        res.setdefault(ck, {}).setdefault(os.path.basename(p), {})[cn] = off
os.makedirs(os.path.dirname(OUT), exist_ok=True)
cur = json.load(open(OUT)) if os.path.exists(OUT) else {}   # merge with any concurrent writer's results
for w, d in res.items():
    for hn, dd in d.items():
        cur.setdefault(w, {}).setdefault(hn, {}).update(dd)
json.dump(cur, open(OUT, "w"), indent=1)
print(f"wrote {OUT}", flush=True)
