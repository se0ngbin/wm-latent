"""Does h get more optimistic with prediction depth on Reacher (as on Dubins)? Base appearance, held-out episodes (>= 9000).
N starts (half with the box entered within K*5 env steps), one CEM-style plan each (z-scored blocks, a = b + 0.7 eps), K blocks
(default 15 = 75 env steps; the CEM uses 5); zhat_k = predictor rollout from the 1-frame history; label = the plan replayed in the
env (box at block end). Per k: false-safe (unsafe with h >= 0), false-alarm (safe with h < 0), AUC, mean h on unsafe states.
Margin functions: MLP + linear, real-only ($HEADS_REAL) and real+imag ($HEADS_IMAG, trained on k <= 5 only).
Env: WMS (wm:ep,...), N, K."""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch, h5py, gymnasium as gym
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sg_point_eval as E

G = os.environ.get
WMS = [w.split(":") for w in G("WMS", "rc_baseline:10,rc_jacobian:10,lewm_reacher:50").split(",")]
N, K, FS = int(G("N", "1000")), int(G("K", "15")), 5
HD = {"real": G("HEADS_REAL", "/data/seongbin/lewm/reacher_heads"), "imag": G("HEADS_IMAG", "/data/seongbin/lewm/reacher_heads_imag")}
rng = np.random.default_rng(0)
with h5py.File("/data/seongbin/lewm/datasets/reacher.h5", "r") as f:
    y, ep = f["failures"][:].astype(int), f["ep_idx"][:]
    Q, V, A = f["qpos"][:], f["qvel"][:], f["action"][:]
    L = K * FS; cs = np.r_[0, np.cumsum(y)]; t = np.arange(len(y) - L)
    ok = (ep[t] >= 9000) & (ep[t] == ep[t + L]); near = cs[t + L + 1] - cs[t] > 0
    idx = np.sort(np.r_[rng.choice(t[ok & near], N // 2, replace=False), rng.choice(t[ok & ~near], N - N // 2, replace=False)])
    u = np.unique(idx); px = f["pixels"][u][np.searchsorted(u, idx)]
Af = A[~np.isnan(A).any(1)]; A_MU, A_SD = Af.mean(0), Af.std(0)
plan = (rng.standard_normal((N, 1, 2 * FS)) + 0.7 * rng.standard_normal((N, K, 2 * FS))).astype(np.float32)
env = gym.make("swm/ReacherDMControl-v0", task="qpos_match", render_mode="rgb_array").unwrapped; env.reset(seed=0)
def in_box(q): q1 = (q[0] + np.pi) % (2 * np.pi) - np.pi; return 0 <= q1 <= np.pi and -2.88 <= q[1] <= -2.45
yi = np.zeros((N, K), int); raw = np.clip(plan.reshape(N, K * FS, 2) * A_SD + A_MU, -1, 1)
for n, i in enumerate(idx):
    env.set_state(Q[i], V[i])
    for k in range(K * FS):
        env.step(raw[n, k])
        if k % FS == FS - 1: yi[n, k // FS] = in_box(np.asarray(env.env.physics.data.qpos[:2]))
print(f"N={N} K={K} unsafe frac per step {np.round(yi.mean(0), 3).tolist()}", flush=True)

MEAN = torch.tensor([0.485, 0.456, 0.406], device=E.DEV).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=E.DEV).view(1, 3, 1, 1)
def load(path, kind):
    sd = torch.load(path, map_location=E.DEV)
    if kind == "linear": h = torch.nn.Linear(192, 1); h.load_state_dict(sd["linear"])
    else:
        h = E.MarginHead(192)
        h.load_state_dict({k.replace("layers.Margin GP_linear", "l.l").replace("layers.Margin GP_norm", "l.n").replace("mean_layer.", "m."): v for k, v in sd["margin_gp"].items()})
    return h.to(E.DEV).eval()
for wm, epn in WMS:
    m = E.swm.wm.utils.load_pretrained(f"{wm}/weights_epoch_{epn}.pt").to(E.DEV).eval()
    with torch.no_grad():
        zs = []
        for s in range(0, N, 250):
            x = (torch.as_tensor(px[s:s + 250], device=E.DEV).permute(0, 3, 1, 2).float() / 255. - MEAN) / STD
            emb = m.projector(m.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]).unsqueeze(1)
            Ab = torch.from_numpy(plan[s:s + 250]).to(E.DEV)
            for k in range(K):
                act = m.action_encoder(Ab[:, :k + 1]); emb = torch.cat([emb, m.predict(emb[:, -3:], act[:, -3:])[:, -1:]], 1)
            zs.append(emb[:, 1:])
        Z = torch.cat(zs)
    for src in ["real", "imag"]:
        for kind in ["mlp", "linear"]:
            head = load(f"{HD[src]}/{wm}/{kind}_s0.pt", kind)
            with torch.no_grad(): h = head(Z.flatten(0, 1)).view(N, K).cpu().numpy()
            fs = [(h[:, k][yi[:, k] == 1] >= 0).mean() for k in range(K)]
            fa = [(h[:, k][yi[:, k] == 0] < 0).mean() for k in range(K)]
            auc = [roc_auc_score(1 - yi[:, k], h[:, k]) for k in range(K)]
            hu = [h[:, k][yi[:, k] == 1].mean() for k in range(K)]
            f = lambda v: " ".join(f"{x:.3f}" for x in v)
            print(f"DEPTH {wm} {src} {kind}\n  false_safe {f(fs)}\n  false_alarm {f(fa)}\n  auc {f(auc)}\n  mean_h_unsafe {f(hu)}", flush=True)
    del m; torch.cuda.empty_cache()
