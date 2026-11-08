"""Why does jac survive the floor-colour shift? Per WM and floor blend a (reacher_ood FLOOR_SWEEP colours), on N held-out states
(episodes >= 9000; N/2 unsafe, N/2 safe; each with qpos/qvel of the frame 5 steps later in the same episode):
  static (encoder):
    nsr      = mean ||z_a(s) - z_0(s)|| / mean ||z_0(s') - z_0(s)||, s' = 5 env steps later (nuisance shift / real-motion signal)
    const    = ||mean_s (z_a - z_0)||^2 / mean_s ||z_a - z_0||^2 (1 = the shift is one constant offset for every state)
    dh/gap   = mean_s (h(z_a) - h(z_0)) / (mean_safe h(z_0) - mean_unsafe h(z_0))   (threshold shift in units of the class gap)
    auc, bal = margin function (real+imag MLP, $HEADS) on z_a
  imagined (predictor), one random CEM-style plan per state (5 blocks, z-scored, as reacher_margin_imag), 1-frame history:
    rel_k    = mean ||zhat_k^a - zhat_k^0|| / mean ||z_a - z_0||   (k = 1..5; < 1 = the rollout re-grounds toward the base rollout)
    auc_k, bal_k of h on zhat_k^a, labels = the plan replayed in the env (box at block end).
Env: WMS (wm:ep,...), N, ALPHAS, HEADS."""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch, h5py, gymnasium as gym
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sg_point_eval as E

G = os.environ.get
WMS = [w.split(":") for w in G("WMS", "rc_baseline:10,rc_jacobian:10,lewm_reacher:50").split(",")]
N, ALPHAS = int(G("N", "1000")), [float(a) for a in G("ALPHAS", "0,0.1,0.2,0.3,0.5").split(",")]
HEADS = G("HEADS", "/data/seongbin/lewm/reacher_heads_imag")
K, FS, D5 = 5, 5, 5
rng = np.random.default_rng(0)
with h5py.File("/data/seongbin/lewm/datasets/reacher.h5", "r") as f:
    y, ep = f["failures"][:].astype(int), f["ep_idx"][:]
    Q, V, A = f["qpos"][:], f["qvel"][:], f["action"][:]
t = np.arange(len(y) - D5); ok = (ep[t] >= 9000) & (ep[t] == ep[t + D5])
idx = np.sort(np.r_[rng.choice(t[ok & (y[t] == 1)], N // 2, replace=False), rng.choice(t[ok & (y[t] == 0)], N // 2, replace=False)])
ys = y[idx]
Af = A[~np.isnan(A).any(1)]; A_MU, A_SD = Af.mean(0), Af.std(0)
plan = (rng.standard_normal((N, 1, 2 * FS)) + 0.7 * rng.standard_normal((N, K, 2 * FS))).astype(np.float32)
F0, F1 = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]), np.array([[0.45, 0.3, 0.15], [0.55, 0.38, 0.2]])

env = gym.make("swm/ReacherDMControl-v0", task="qpos_match", render_mode="rgb_array").unwrapped
def in_box(q): q1 = (q[0] + np.pi) % (2 * np.pi) - np.pi; return 0 <= q1 <= np.pi and -2.88 <= q[1] <= -2.45
imgs, imgs5 = {}, None
for a in ALPHAS:
    env.reset(seed=0, options={"variation_values": {"floor.color": (1 - a) * F0 + a * F1}})
    r = []
    for i in idx: env.set_state(Q[i], V[i]); r.append(env.render())
    imgs[a] = np.stack(r)
    if a == 0:
        imgs5 = np.stack([(env.set_state(Q[i + D5], V[i + D5]), env.render())[1] for i in idx])
        yi = np.zeros((N, K))   # plan labels (appearance-independent)
        raw = np.clip(plan.reshape(N, K * FS, 2) * A_SD + A_MU, -1, 1)
        for n, i in enumerate(idx):
            env.set_state(Q[i], V[i])
            for k in range(K * FS):
                env.step(raw[n, k])
                if k % FS == FS - 1: yi[n, k // FS] = in_box(np.asarray(env.env.physics.data.qpos[:2]))
print(f"N={N} unsafe={ys.mean():.2f} | imagined unsafe frac per step {np.round(yi.mean(0), 3).tolist()}", flush=True)

MEAN = torch.tensor([0.485, 0.456, 0.406], device=E.DEV).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=E.DEV).view(1, 3, 1, 1)
def bal(h, lab): sp = h >= 0; return (sp[lab == 0].mean() + (~sp[lab == 1]).mean()) / 2
for wm, epn in WMS:
    m = E.swm.wm.utils.load_pretrained(f"{wm}/weights_epoch_{epn}.pt").to(E.DEV).eval()
    @torch.no_grad()
    def enc(a, bs=250):
        return torch.cat([m.projector(m.encoder(((torch.as_tensor(a[i:i + bs], device=E.DEV).permute(0, 3, 1, 2).float() / 255. - MEAN) / STD),
                                                 interpolate_pos_encoding=True).last_hidden_state[:, 0]) for i in range(0, len(a), bs)])
    @torch.no_grad()
    def imagine(z0):
        emb, Ab = z0.unsqueeze(1), torch.from_numpy(plan).to(E.DEV)
        for k in range(K):
            act = m.action_encoder(Ab[:, :k + 1]); emb = torch.cat([emb, m.predict(emb[:, -3:], act[:, -3:])[:, -1:]], 1)
        return emb[:, 1:]
    sd = torch.load(f"{HEADS}/{wm}/mlp_s0.pt", map_location=E.DEV)
    head = E.MarginHead(192).to(E.DEV).eval()
    head.load_state_dict({k.replace("layers.Margin GP_linear", "l.l").replace("layers.Margin GP_norm", "l.n").replace("mean_layer.", "m."): v for k, v in sd["margin_gp"].items()})
    H = lambda z: head(z).detach().cpu().numpy().ravel()
    z0 = enc(imgs[0]); sig = (enc(imgs5) - z0).norm(dim=-1).mean().item(); h0 = H(z0)
    gap = h0[ys == 0].mean() - h0[ys == 1].mean(); zh0 = imagine(z0)
    for a in ALPHAS:
        za = z0 if a == 0 else enc(imgs[a]); d = za - z0; dn = d.norm(dim=-1).mean().item()
        const = (d.mean(0).norm() ** 2 / d.norm(dim=-1).pow(2).mean().clamp_min(1e-12)).item() if a else float("nan")
        ha = H(za); zha = imagine(za)
        rel = [((zha[:, k] - zh0[:, k]).norm(dim=-1).mean().item() / dn) if a else float("nan") for k in range(K)]
        hk = [H(zha[:, k]) for k in range(K)]
        print(f"FLOORDIAG {wm} a={a} nsr={dn / sig:.3f} const={const:.3f} dh/gap={(ha - h0).mean() / gap:+.3f} "
              f"auc={roc_auc_score(1 - ys, ha):.3f} bal={bal(ha, ys):.3f} | rel_k={' '.join(f'{r:.2f}' for r in rel)} | "
              f"imag auc_k={' '.join(f'{roc_auc_score(1 - yi[:, k], hk[k]):.3f}' for k in range(K))} "
              f"bal_k={' '.join(f'{bal(hk[k], yi[:, k]):.3f}' for k in range(K))}", flush=True)
    del m; torch.cuda.empty_cache()
