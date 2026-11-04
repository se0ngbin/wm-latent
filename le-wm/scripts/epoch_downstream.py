"""Are downstream metrics still moving between epochs 10..50? (convergence check; checkpoints weights_epoch_{10..50}).
Per checkpoint, on frames rendered once (seed-0 states, training orientation, cached in $CACHE):
  1. margin function (GP, 5000 steps, red uniform states) zero-shot AUC on color / rotate / shape
     (color: 3 margin-function seeds on the same latents; rotate/shape: seed 0), as dub_fixed_margin.py;
  2. the seed-0 red margin function on imagined latents of 600 CEM-style plans (diag_imag_steps.py setup):
     red AUC, red false-safe, purple AUC;
  3. goal-distance fidelity (diag_rollout_safety.py setup): Spearman(latent dist to goal, true xy dist) near the goal
     (true dist < 0.5), real and imagined latents (3-frame history, real actions, 400 windows x 25 steps).
Prints one EPD line per checkpoint. Usage: epoch_downstream.py tag=ckpt [...]   env EPOCHS="10,20,30,40,50"."""
import sys, os
sys.path.insert(0, "/home/seongbin/latent/le-wm"); sys.path.insert(0, "/home/seongbin/latent/le-wm/scripts")
import numpy as np, torch, h5py
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import ood_margin_gp_jepa as E
import stable_worldmodel as swm

CACHE = os.environ.get("CACHE", "/home/seongbin/.claude/jobs/d2d8f7fb/tmp/epd")
os.makedirs(CACHE, exist_ok=True)
EPOCHS = [int(e) for e in os.environ.get("EPOCHS", "10,20,30,40,50").split(",")]
DEV, BS = E.DEV, 32
tf = E.get_img_preprocessor(source="pixels", target="pixels", img_size=E.IMG)


def render_raw(st, cond):  # E.render_cond without the transform (uint8 HWC)
    env = E.DubinsSwmEnv()
    env.colors = {"background": "white", "agent": "blue", "goal": "green", "obstacle": cond["obstacle"]}
    out = []
    for s in st:
        env.state = np.asarray(s, np.float32).copy()
        img = E.render_diamond(env) if cond["shape"] == "diamond" else env._render_image()
        rot = cond.get("rot", 0)
        out.append(np.rot90(img, k=rot // 90) if rot else img)
    return np.ascontiguousarray(np.stack(out)).astype(np.uint8)


def cached(name, fn):
    p = f"{CACHE}/{name}.npz"
    if not os.path.exists(p):
        np.savez(p + ".tmp.npz", **fn()); os.replace(p + ".tmp.npz", p)
    return dict(np.load(p))


# ---- 1. uniform states (seed 0), red + the three shifted views ----
def _uniform():
    rng = np.random.default_rng(0); x0, x1, y0, y1 = E.WORLD_BOUNDS
    s = np.stack([rng.uniform(x0, x1, E.N), rng.uniform(y0, y1, E.N), rng.uniform(-np.pi, np.pi, E.N)], 1).astype(np.float32)
    d = {"s": s, "red": render_raw(s, E.AXES["color"][0]), "y_red": E.inside_obstacle(s[:, :2], "circle")}
    for ax in ("color", "rotate", "shape"):
        nc = E.AXES[ax][1]; d[ax] = render_raw(s, nc); d[f"y_{ax}"] = E.inside_obstacle(s[:, :2], nc["shape"])
    return d
U = cached("uniform_s0", _uniform)

# ---- 2. CEM-style plans from held-out starts (diag_imag_steps.py) ----
K, NP = 25, 600
def _plans():
    rng = np.random.default_rng(2)
    with h5py.File("/data/seongbin/lewm/datasets/dubins_expert.h5", "r") as f:
        off, ln = f["ep_offset"][:], f["ep_len"][:]
        act_all = f["action"][:].astype(np.float32).reshape(len(f["action"]), -1)
        a_mu, a_sd = np.nanmean(act_all, 0), np.nanstd(act_all, 0)
        e = rng.choice(np.arange(int(0.9 * len(ln)), len(ln)), NP)
        st0 = f["state"][:][off[e] + rng.integers(0, ln[e])]
    A = (rng.standard_normal((NP, 1, 1)) + 0.7 * rng.standard_normal((NP, K, 1))).astype(np.float32)
    env, y = E.DubinsSwmEnv(), np.zeros((NP, K), bool)
    for i in range(NP):
        env.state = st0[i].astype(np.float32).copy()
        for k in range(K):
            env._integrate(float(np.clip(A[i, k, 0] * a_sd[0] + a_mu[0], -env.max_angular_velocity, env.max_angular_velocity)))
            y[i, k] = E.inside_obstacle(env.state[None, :2], "circle")[0]
    return {"A": A, "safe": ~y, "red": render_raw(st0, E.AXES["color"][0]), "purple": render_raw(st0, E.AXES["color"][1])}
P = cached("plans", _plans)

# ---- 3. held-out windows for goal fidelity (diag_rollout_safety.py) ----
H, NW = 3, 400
def _windows():
    with h5py.File("/data/seongbin/lewm/datasets/dubins_expert.h5", "r") as f:
        off, ln = f["ep_offset"][:], f["ep_len"][:]
        act_all = f["action"][:].astype(np.float32).reshape(len(f["action"]), -1)
        a_mu, a_sd = np.nanmean(act_all, 0), np.nanstd(act_all, 0)
        rng = np.random.default_rng(0)
        eps = np.where(ln >= H + K)[0]; eps = eps[int(0.9 * len(eps)):]
        starts = [off[e] + rng.integers(0, ln[e] - (H + K) + 1) for e in rng.choice(eps, NW, replace=True)]
        idx = np.concatenate([np.arange(s, s + H + K) for s in starts]); u, inv = np.unique(idx, return_inverse=True)
        return {"px": f["pixels"][u][inv].reshape(NW, H + K, *f["pixels"].shape[1:]),
                "st": f["state"][u][inv].reshape(NW, H + K, -1),
                "act": ((act_all[idx] - a_mu) / (a_sd + 1e-8)).reshape(NW, H + K, -1)}
W = cached("windows", _windows)


@torch.no_grad()
def encode(m, F):  # contiguous CHW, as the training loader / dub_fixed_margin.py
    Z = []
    for i in range(0, len(F), BS):
        x = tf({"pixels": torch.from_numpy(np.ascontiguousarray(F[i:i + BS])).permute(0, 3, 1, 2)})["pixels"].float().contiguous()
        Z.append(m.projector(m.encoder(x.to(DEV), interpolate_pos_encoding=True).last_hidden_state[:, 0]).cpu())
    return torch.cat(Z)


def auc(head, Z, y):  # y: 1 = unsafe; AUC of h for the safe class
    with torch.no_grad():
        h = head(Z.to(DEV)).cpu().numpy().ravel()
    return roc_auc_score((y == 0).astype(int), h)


@torch.no_grad()
def imag(m, z0, A):  # 1-frame start, context grows to 3 (JEPA.rollout / diag_imag_steps.py)
    emb, At = z0.to(DEV).unsqueeze(1), torch.from_numpy(A).to(DEV)
    for k in range(K):
        emb = torch.cat([emb, m.predict(emb[:, -3:], m.action_encoder(At[:, :k + 1])[:, -3:])[:, -1:]], 1)
    return emb[:, 1:]


def run(tag, ck, ep):
    m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_{ep}.pt").to(DEV).eval()
    NTR = E.NTR
    Zr = encode(m, U["red"]); r = {}
    heads = [E.train_margin(Zr[:NTR], U["y_red"][:NTR], "gp", seed=s) for s in (0, 1, 2)]
    for ax in ("color", "rotate", "shape"):
        Zn = encode(m, U[ax])
        hs = heads if ax == "color" else heads[:1]
        r[ax] = [auc(h, Zn[NTR:], U[f"y_{ax}"][NTR:]) for h in hs]
    r["in"] = auc(heads[0], Zr[NTR:], U["y_red"][NTR:])
    h0, safe = heads[0], P["safe"]
    for col in ("red", "purple"):
        Zh = imag(m, encode(m, P[col]), P["A"])
        with torch.no_grad():
            hv = h0(Zh.flatten(0, 1).float()).view(NP, K).cpu().numpy()
        r[f"imag_{col}_auc"] = roc_auc_score(safe.ravel(), hv.ravel())
        r[f"imag_{col}_fs"] = float((hv[~safe] >= 0).mean())
    # goal fidelity
    Z = encode(m, W["px"].reshape(-1, *W["px"].shape[2:])).view(NW, H + K, -1).to(DEV)
    Aw = torch.from_numpy(W["act"]).to(DEV)
    with torch.no_grad():
        emb, a, preds = Z[:, :H].clone(), Aw[:, :H], []
        for k in range(K):
            p = m.predict(emb[:, -H:], m.action_encoder(a[:, -H:]))[:, -1:]
            preds.append(p); emb = torch.cat([emb, p], 1)
            a = torch.cat([a, Aw[:, H + k:H + k + 1]], 1) if H + k < H + K - 1 else a
    Zh, Zt, zg = torch.cat(preds, 1), Z[:, H:], Z[:, -1:]
    dtrue = np.linalg.norm(W["st"][:, H:, :2] - W["st"][:, -1:, :2], axis=-1)[:, :-1].ravel(); near = dtrue < 0.5
    for nm, ZZ in (("real", Zt), ("imag", Zh)):
        dl = (ZZ - zg).pow(2).sum(-1)[:, :-1].cpu().numpy().ravel()
        r[f"goal_near_{nm}"] = float(spearmanr(dl[near], dtrue[near]).correlation)
    del m; torch.cuda.empty_cache()
    c = np.array(r["color"])
    print(f"EPD {tag} ep={ep} in={r['in']:.4f} color={c.mean():.3f}±{c.std():.3f} rotate={r['rotate'][0]:.3f} "
          f"shape={r['shape'][0]:.3f} imag_red_auc={r['imag_red_auc']:.3f} imag_red_fs={r['imag_red_fs']:.3f} "
          f"imag_purple_auc={r['imag_purple_auc']:.3f} goal_near_real={r['goal_near_real']:.3f} "
          f"goal_near_imag={r['goal_near_imag']:.3f}", flush=True)


for spec in sys.argv[1:]:
    t, c = spec.split("=", 1)
    for ep in EPOCHS:
        run(t, c, ep)
