"""Veto-vs-distance curve for Dubins (port of the Reacher session's reacher_veto_diag.py CURVE=1): does h's plan score carry
distance information under a shift, at a MATCHED operating point?
  starts: NS safe states within 0.75 (= 15 steps at speed 1, dt .05) of an obstacle edge, random heading (not dataset
          states); PP random CEM-style 15-step plans each (shared offset + 0.7 noise per step, z-scored like training).
  truth:  each plan simulated step by step; closest distance d = min_t min_obstacles (||p_t - c|| - r); d <= 0 = unsafe plan.
  score:  CCEM's feasibility score, min of h over the current frame + all 15 predicted latents (1-frame start), rendered
          under the condition.
  matched operating point: thr = 0.9-quantile of the score over UNSAFE plans (90% of unsafe plans vetoed); veto = score < thr.
Prints per head: the unsafe share, thr, and the safe-plan veto rate binned by d in units of the obstacle radius r = 0.5
(a sharp fall to ~0 = distance information; a flat curve = none).
Usage: veto_curve_dubins.py <wm_dir> <cond: red|purple|rot90> <head.pt> [...]"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, h5py
from safe_solver import _load_margin_head   # before E (import order, see diag_imag_color_dubins.py)
import ood_margin_gp_jepa as E
import stable_worldmodel as swm
from dubins_swm_env import OBSTACLES

ck, cn, heads = sys.argv[1], sys.argv[2], sys.argv[3:]
NS, PP, K, DEV = int(os.environ.get("NS", "200")), int(os.environ.get("PP", "64")), int(os.environ.get("K", "15")), E.DEV
COND = {"red": dict(obstacle="red", shape="circle", rot=0), "purple": dict(obstacle="purple", shape="circle", rot=0),
        "rot90": dict(obstacle="red", shape="circle", rot=90)}[cn]
rng = np.random.default_rng(5)
tf = E.get_img_preprocessor(source="pixels", target="pixels", img_size=E.IMG)
tf_fixed = lambda d: tf({"pixels": torch.from_numpy(np.ascontiguousarray(d["pixels"])).permute(0, 3, 1, 2)})
with h5py.File("/data/seongbin/lewm/datasets/dubins_expert.h5", "r") as f:
    act_all = f["action"][:].astype(np.float32).reshape(len(f["action"]), -1)
a_mu, a_sd = np.nanmean(act_all, 0), np.nanstd(act_all, 0)


def edge_dist(xy):
    return np.min([np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) - r for cx, cy, r in OBSTACLES], 0)


x0, x1, y0, y1 = E.WORLD_BOUNDS
st = np.zeros((0, 3), np.float32)
while len(st) < NS:
    c = np.stack([rng.uniform(x0, x1, 4 * NS), rng.uniform(y0, y1, 4 * NS), rng.uniform(-np.pi, np.pi, 4 * NS)], 1).astype(np.float32)
    d = edge_dist(c[:, :2])
    st = np.concatenate([st, c[(d > 0.02) & (d < 0.05 * K)]])   # obstacle reachable within K steps
st = st[:NS]
A = (rng.standard_normal((NS, PP, 1, 1)) + 0.7 * rng.standard_normal((NS, PP, K, 1))).astype(np.float32)
env, dmin = E.DubinsSwmEnv(), np.zeros((NS, PP))
for i in range(NS):
    for p in range(PP):
        env.state = st[i].copy(); best = np.inf
        for k in range(K):
            env._integrate(float(np.clip(A[i, p, k, 0] * a_sd[0] + a_mu[0], -env.max_angular_velocity, env.max_angular_velocity)))
            best = min(best, float(edge_dist(env.state[None, :2])[0]))
        dmin[i, p] = best
m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(DEV).eval()
Hs = [_load_margin_head(p, DEV).eval() for p in heads]
At = torch.from_numpy(A.reshape(NS * PP, K, 1)).to(DEV)
with torch.no_grad():
    z0 = E.encode(m, E.render_cond(E.DubinsSwmEnv(), st, COND, tf_fixed)).to(DEV)
    Z = []
    for i in range(0, NS * PP, 1024):
        idx = torch.arange(i, min(i + 1024, NS * PP), device=DEV)
        emb = z0[idx // PP].unsqueeze(1)
        for k in range(K):
            emb = torch.cat([emb, m.predict(emb[:, -3:], m.action_encoder(At[idx, :k + 1])[:, -3:])[:, -1:]], 1)
        Z.append(emb)
    Z = torch.cat(Z)                                                     # (NS*PP, 1 + K, D): current + imagined
unsafe = dmin.ravel() <= 0
r = 0.5
edges = [0, .05, .1, .2, .4, .6, 1.0, np.inf]
for p, hd in zip(heads, Hs):
    with torch.no_grad():
        score = torch.cat([hd(Z[i:i + 4096].flatten(0, 1).float()).view(-1, 1 + K).min(1).values
                           for i in range(0, len(Z), 4096)]).cpu().numpy()
    thr = float(np.quantile(score[unsafe], 0.9))
    veto = score < thr
    ds = dmin.ravel()[~unsafe] / r
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (ds >= lo) & (ds < hi)
        rows.append(f"[{lo},{hi}):{veto[~unsafe][sel].mean():.2f}(n{sel.sum()})" if sel.sum() else f"[{lo},{hi}):-")
    print(f"VETO {ck} {cn} {os.path.basename(p)}: unsafe {unsafe.mean():.2f} thr {thr:+.2f} | thr0 veto: unsafe "
          f"{(score[unsafe] < 0).mean():.2f} safe {(score[~unsafe] < 0).mean():.2f} | safe veto by d/r " + " ".join(rows), flush=True)
