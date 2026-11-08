"""Can threshold-free CEM work on Reacher? Threshold-free rules need h to RANK the candidate plans of one replan, which all share
the start frame (a shared OOD offset cancels). Per start: P random CEM-style 5-block plans (as reacher_margin_imag), simulated
labels (plan enters the box at a block end); score = min_k h(zhat_k) over the predicted blocks (and its per-step z-scored
version across the start's candidates, as ccem_fallback safe_task_norm). Within-start AUC = AUC of score vs plan-safe over the
P plans of one start, averaged over starts that have both safe and unsafe plans; global AUC pools all plans (what a fixed
threshold needs). Floor blend a as reacher_ood FLOOR_SWEEP; start frames rendered under it.
Env: WMS (wm:ep:headsdir,...), NS (starts), P (plans per start), ALPHAS."""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch, h5py, gymnasium as gym
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sg_point_eval as E

G = os.environ.get
WMS = [w.split(":") for w in G("WMS", "rc_jacobian:10:/data/seongbin/lewm/reacher_heads_imag,rc_jac_safeadv_linf:5:/data/seongbin/lewm/reacher_heads_imag_ep5").split(",")]
NS, P, K, FS = int(G("NS", "200")), int(G("P", "64")), 5, 5
ALPHAS = [float(a) for a in G("ALPHAS", "0,0.2,0.3,0.4").split(",")]
rng = np.random.default_rng(0)
with h5py.File("/data/seongbin/lewm/datasets/reacher.h5", "r") as f:
    y, ep, Q, V, A = f["failures"][:].astype(int), f["ep_idx"][:], f["qpos"][:], f["qvel"][:], f["action"][:]
cs = np.r_[0, np.cumsum(y)]; t = np.arange(len(y) - 26)
ok = (ep[t] >= 9000) & (ep[t] == ep[t + 25]) & (y[t] == 0) & (cs[t + 26] - cs[t] > 0)   # safe start, box within 25 steps
st = np.sort(rng.choice(t[ok], NS, replace=False))
Af = A[~np.isnan(A).any(1)]; mu, sd = Af.mean(0), Af.std(0)
plan = (rng.standard_normal((NS, P, 1, 2 * FS)) + 0.7 * rng.standard_normal((NS, P, K, 2 * FS))).astype(np.float32)
env = gym.make("swm/ReacherDMControl-v0", task="qpos_match", render_mode="rgb_array").unwrapped; env.reset(seed=0)
def in_box(q): q1 = (q[0] + np.pi) % (2 * np.pi) - np.pi; return 0 <= q1 <= np.pi and -2.88 <= q[1] <= -2.45
lab = np.zeros((NS, P), int); raw = np.clip(plan.reshape(NS, P, K * FS, 2) * sd + mu, -1, 1)
for s, i in enumerate(st):
    for p in range(P):
        env.set_state(Q[i], V[i])
        for k in range(K * FS):
            env.step(raw[s, p, k])
            if k % FS == FS - 1 and in_box(np.asarray(env.env.physics.data.qpos[:2])): lab[s, p] = 1
F0, F1 = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]), np.array([[0.45, 0.3, 0.15], [0.55, 0.38, 0.2]])
imgs = {}
for a in ALPHAS:
    env.reset(seed=0, options={"variation_values": {"floor.color": (1 - a) * F0 + a * F1}})
    imgs[a] = np.stack([(env.set_state(Q[i], V[i]), env.render())[1] for i in st])
mixed = (lab.min(1) == 0) & (lab.max(1) == 1)
print(f"starts {NS} x plans {P}: unsafe plan frac {lab.mean():.3f}, starts with both {mixed.sum()}", flush=True)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=E.DEV).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=E.DEV).view(1, 3, 1, 1)
for wm, epn, hd in WMS:
    m = E.swm.wm.utils.load_pretrained(f"{wm}/weights_epoch_{epn}.pt").to(E.DEV).eval()
    heads = {}
    for kind in ["mlp", "linear"]:
        sdict = torch.load(f"{hd}/{wm}/{kind}_s0.pt", map_location=E.DEV)
        if kind == "linear": h = torch.nn.Linear(192, 1); h.load_state_dict(sdict["linear"])
        else:
            h = E.MarginHead(192)
            h.load_state_dict({k.replace("layers.Margin GP_linear", "l.l").replace("layers.Margin GP_norm", "l.n").replace("mean_layer.", "m."): v for k, v in sdict["margin_gp"].items()})
        heads[kind] = h.to(E.DEV).eval()
    for a in ALPHAS:
        with torch.no_grad():
            x = (torch.as_tensor(imgs[a], device=E.DEV).permute(0, 3, 1, 2).float() / 255. - MEAN) / STD
            z0 = torch.cat([m.projector(m.encoder(x[i:i + 100], interpolate_pos_encoding=True).last_hidden_state[:, 0]) for i in range(0, NS, 100)])
            emb = z0.repeat_interleave(P, 0).unsqueeze(1); Ab = torch.from_numpy(plan.reshape(NS * P, K, -1)).to(E.DEV)
            for k in range(K):
                act = m.action_encoder(Ab[:, :k + 1]); emb = torch.cat([emb, m.predict(emb[:, -3:], act[:, -3:])[:, -1:]], 1)
            zh = emb[:, 1:]
        row = []
        for kind, h in heads.items():
            with torch.no_grad(): hv = h(zh.flatten(0, 1)).view(NS, P, K).cpu().numpy()
            sc = hv.min(-1)
            hn = (hv - hv.mean(1, keepdims=True)) / np.maximum(hv.std(1, keepdims=True), 1e-6); scn = hn.min(-1)
            w = np.mean([roc_auc_score(1 - lab[s], sc[s]) for s in np.where(mixed)[0]])
            wn = np.mean([roc_auc_score(1 - lab[s], scn[s]) for s in np.where(mixed)[0]])
            g = roc_auc_score(1 - lab.ravel(), sc.ravel()); fs = (sc[lab == 1] >= 0).mean()
            row.append(f"{kind}: within={w:.3f} within_norm={wn:.3f} global={g:.3f} plan_false_safe={fs:.3f}")
        print(f"WITHIN {wm} ep{epn} a={a} | " + " | ".join(row), flush=True)
    del m; torch.cuda.empty_cache()
