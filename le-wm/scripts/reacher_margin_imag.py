"""Reacher margin functions (mlp + linear, margin_gp loss) on REAL + IMAGINED latents (Reacher analog of train_gp_margin_imag_dubins.py).
Real: as reacher_margin.py (episodes < 9000, N_PER_CLASS unsafe + N_PER_CLASS safe frames, frozen CLS latents).
Imagined: NPLAN start frames of episodes < 9000 (half drawn from frames whose next 25 steps enter the box, so the box is visited), each with a
  K-block CEM-style plan in z-scored action space, a = b + 0.7 eps (b ~ N(0,1) per plan, eps per step), K = 5 blocks of 5 env steps (= the
  CEM horizon); z_hat = predictor rollout from the 1-frame history exactly as JEPA.rollout / the CEM (history <= 3);
  label = the same plan (de-normalized, clipped to [-1, 1]) replayed in the env from the frame's (qpos, qvel): 1 = joint box at that block's end.
Heads trained on real + imagined latents with balanced safe/unsafe batches (same loss / steps as reacher_margin.py).
Test: (a) reacher_margin.py's real held-out set (5000 frames of episodes >= 9000, natural ratio, same rng), (b) imagined latents of NTE plans
from episodes >= 9000. Prints AUC, balanced sign acc, false-safe for the new heads and the real-only heads ($HEADS/<WM>, if present).
Saves to $HEADS_OUT/<WM>/{mlp,linear}_s<SEED>.pt (same format as reacher_margin.py, so reacher_ood.py / reacher_cem.py take HEADS=$HEADS_OUT).
Env: WM, EP, SEED, N_PER_CLASS, NPLAN, NTE, STEPS, HEADS, HEADS_OUT."""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch, h5py, gymnasium as gym
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sg_point_eval as E

G = os.environ.get
WM, EP, SEED = G("WM", "lewm_reacher"), G("EP", "50"), int(G("SEED", "0"))
NPC, NPLAN, NTE, STEPS = int(G("N_PER_CLASS", "25000")), int(G("NPLAN", "8000")), int(G("NTE", "1000")), int(G("STEPS", "3000"))
HEADS, HEADS_OUT = G("HEADS", "/data/seongbin/lewm/reacher_heads"), G("HEADS_OUT", "/data/seongbin/lewm/reacher_heads_imag")
K, FS, LOOK = 5, 5, 25
rng = np.random.default_rng(SEED)

with h5py.File("/data/seongbin/lewm/datasets/reacher.h5", "r") as f:
    y, ep = f["failures"][:].astype(int), f["ep_idx"][:]
    tr = ep < 9000
    # identical draws to reacher_margin.py (same rng sequence) -> same real train / test frames
    ti = np.sort(np.r_[rng.choice(np.where(tr & (y == 1))[0], NPC, replace=False), rng.choice(np.where(tr & (y == 0))[0], NPC, replace=False)])
    te = np.sort(rng.choice(np.where(~tr)[0], 5000, replace=False))
    Xtr, ytr, Xte, yte = f["pixels"][ti], y[ti], f["pixels"][te], y[te]
    Q, V, A = f["qpos"][:], f["qvel"][:], f["action"][:]
    # plan starts: frames with >= LOOK steps left in the episode; half of them see the box within LOOK steps
    cs = np.r_[0, np.cumsum(y)]; t = np.arange(len(y) - LOOK)
    ok = ep[t] == ep[t + LOOK]; near = cs[t + LOOK + 1] - cs[t] > 0
    def starts(split, n):
        a, b = t[ok & split[t] & near], t[ok & split[t]]
        return np.sort(np.r_[rng.choice(a, n // 2, replace=False), rng.choice(b, n - n // 2, replace=False)])
    s_tr, s_te = starts(tr, NPLAN), starts(~tr, NTE)
    px = lambda s: f["pixels"][np.unique(s)][np.unique(s, return_inverse=True)[1]]   # h5py needs strictly increasing indices; halves can overlap
    P_tr, P_te = px(s_tr), px(s_te)
Af = A[~np.isnan(A).any(1)]; A_MU, A_SD = Af.mean(0), Af.std(0)   # reacher_cem.py's scaler
print(f"real train {len(ti)} | real test {len(te)} (unsafe {yte.mean():.3f}) | plans train {NPLAN} test {NTE}", flush=True)


def plans(n):  # z-scored action blocks (n, K, 2*FS): per-step (a1, a2) flattened, as reacher_cem.py
    return (rng.standard_normal((n, 1, 2 * FS)) + 0.7 * rng.standard_normal((n, K, 2 * FS))).astype(np.float32)


env = gym.make("swm/ReacherDMControl-v0", task="qpos_match", render_mode="rgb_array").unwrapped
env.reset(seed=0)
def in_box(q): q1 = (q[0] + np.pi) % (2 * np.pi) - np.pi; return 0 <= q1 <= np.pi and -2.88 <= q[1] <= -2.45
def simulate(s, a):  # (n, K) labels at each block's end
    raw = np.clip(a.reshape(len(a), K * FS, 2) * A_SD + A_MU, -1, 1); out = np.zeros((len(a), K), np.float32)
    for i, j in enumerate(s):
        env.set_state(Q[j], V[j])
        for k in range(K * FS):
            env.step(raw[i, k])
            if k % FS == FS - 1: out[i, k // FS] = in_box(np.asarray(env.env.physics.data.qpos[:2]))
    return out


m = E.swm.wm.utils.load_pretrained(f"{WM}/weights_epoch_{EP}.pt").to(E.DEV).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=E.DEV).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=E.DEV).view(1, 3, 1, 1)
@torch.no_grad()
def enc(a, bs=256):
    out = []
    for i in range(0, len(a), bs):
        x = (torch.as_tensor(a[i:i + bs], device=E.DEV).permute(0, 3, 1, 2).float() / 255. - MEAN) / STD
        out.append(m.projector(m.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]))
    return torch.cat(out)
@torch.no_grad()
def imagine(px, a, bs=256):  # JEPA.rollout with a 1-frame history: (n, K, D)
    out = []
    for s in range(0, len(px), bs):
        emb, Ab = enc(px[s:s + bs]).unsqueeze(1), torch.from_numpy(a[s:s + bs]).to(E.DEV)
        for k in range(K):
            act = m.action_encoder(Ab[:, :k + 1])
            emb = torch.cat([emb, m.predict(emb[:, -3:], act[:, -3:])[:, -1:]], 1)
        out.append(emb[:, 1:])
    return torch.cat(out)


a_tr, a_te = plans(NPLAN), plans(NTE)
yi_tr, yi_te = simulate(s_tr, a_tr), simulate(s_te, a_te)
Zi_tr, Zi_te = imagine(P_tr, a_tr).flatten(0, 1), imagine(P_te, a_te).flatten(0, 1)
Z, Zte = enc(Xtr), enc(Xte)
print(f"imagined unsafe frac train {yi_tr.mean():.3f} test {yi_te.mean():.3f}", flush=True)
Zall, Yall = torch.cat([Z, Zi_tr]), np.r_[ytr, yi_tr.ravel()].astype(int)
s_idx, u_idx = np.where(Yall == 0)[0], np.where(Yall == 1)[0]


def score(nm, hk, head):
    for tag, Zt, yt in [("real", Zte, yte), ("imag", Zi_te, yi_te.ravel().astype(int))]:
        with torch.no_grad(): h = head(Zt).cpu().numpy().ravel()
        sp = h >= 0
        print(f"REACHER_MARGIN_IMAG {WM} {hk} {nm} test={tag} auc={roc_auc_score(1 - yt, h):.3f} bal_acc={(sp[yt == 0].mean() + (~sp[yt == 1]).mean()) / 2:.3f} "
              f"false_safe={sp[yt == 1].mean():.3f} false_alarm={(~sp[yt == 0]).mean():.3f}", flush=True)


for hk in ["linear", "mlp"]:
    torch.manual_seed(SEED); r = np.random.default_rng(SEED)
    head = (torch.nn.Linear(Z.shape[1], 1) if hk == "linear" else E.MarginHead(Z.shape[1])).to(E.DEV); opt = torch.optim.AdamW(head.parameters(), lr=E.LR)
    for _ in range(STEPS):
        loss = E.gp_loss(head, Zall[r.choice(s_idx, E.BS)], Zall[r.choice(u_idx, E.BS)]); opt.zero_grad(); loss.backward(); opt.step()
    head.eval(); score("real+imag", hk, head)
    sd = head.state_dict()   # same key mapping as reacher_margin.py
    sd = {"margin_gp": {k.replace("l.l", "layers.Margin GP_linear").replace("l.n", "layers.Margin GP_norm").replace("m.", "mean_layer."): v for k, v in sd.items()}} if hk == "mlp" else {"linear": sd}
    os.makedirs(f"{HEADS_OUT}/{WM}", exist_ok=True); torch.save(sd, f"{HEADS_OUT}/{WM}/{hk}_s{SEED}.pt")
    old = f"{HEADS}/{WM}/{hk}_s{SEED}.pt"
    if os.path.exists(old):
        o = torch.load(old, map_location=E.DEV)
        if hk == "linear": oh = torch.nn.Linear(Z.shape[1], 1).to(E.DEV); oh.load_state_dict(o["linear"])
        else:
            oh = E.MarginHead(Z.shape[1]).to(E.DEV)
            oh.load_state_dict({k.replace("layers.Margin GP_linear", "l.l").replace("layers.Margin GP_norm", "l.n").replace("mean_layer.", "m."): v for k, v in o["margin_gp"].items()})
        score("real_only", hk, oh.eval())
