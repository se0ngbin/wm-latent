"""Does the predictor pull OOD rollouts toward KNOWN states, and toward the RIGHT ones? (Reacher floor shift; Dubins analog:
support_alias.py / regrounding_gru.py.) Per WM: bank = encoder latents of NBANK training frames (episodes < 9000, base
appearance) with their qpos. Queries: N held-out states (episodes >= 9000, half unsafe) rendered at floor blend a, and one
random CEM-style 5-block plan each. Step 0 = encoder latent z_a; steps 1-5 = predicted zhat_k^a (1-frame history).
Per (a, step):
  knn      = median distance to the 10 nearest bank latents / the same at a = 0 (1 = as close to training latents as in-dist)
  in_sup   = fraction with knn distance <= q99 of the a = 0 distances at that step
  nn_err   = median joint error (max over q1, q2, wrapped) between the true state (simulated for steps >= 1) and the
             nearest bank latent's state; at a = 0 this is the bank's own resolution. Low knn + high nn_err = aliasing.
Env: WMS (wm:ep,...), N, NBANK, ALPHAS."""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch, h5py, gymnasium as gym
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sg_point_eval as E

G = os.environ.get
WMS = [w.split(":") for w in G("WMS", "rc_baseline:10,rc_jacobian:10,lewm_reacher:50").split(",")]
N, NB, K, FS = int(G("N", "1000")), int(G("NBANK", "20000")), 5, 5
ALPHAS = [float(a) for a in G("ALPHAS", "0,0.1,0.2,0.3,0.5").split(",")]
rng = np.random.default_rng(0)
with h5py.File("/data/seongbin/lewm/datasets/reacher.h5", "r") as f:
    y, ep = f["failures"][:].astype(int), f["ep_idx"][:]
    Q, V, A = f["qpos"][:], f["qvel"][:], f["action"][:]
    bi = np.sort(rng.choice(np.where(ep < 9000)[0], NB, replace=False)); Pb, Qb = f["pixels"][bi], Q[bi]
t = np.where(ep >= 9000)[0]
idx = np.sort(np.r_[rng.choice(t[y[t] == 1], N // 2, replace=False), rng.choice(t[y[t] == 0], N // 2, replace=False)])
Af = A[~np.isnan(A).any(1)]; A_MU, A_SD = Af.mean(0), Af.std(0)
plan = (rng.standard_normal((N, 1, 2 * FS)) + 0.7 * rng.standard_normal((N, K, 2 * FS))).astype(np.float32)
F0, F1 = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]), np.array([[0.45, 0.3, 0.15], [0.55, 0.38, 0.2]])
env = gym.make("swm/ReacherDMControl-v0", task="qpos_match", render_mode="rgb_array").unwrapped
imgs = {}
for a in ALPHAS:
    env.reset(seed=0, options={"variation_values": {"floor.color": (1 - a) * F0 + a * F1}})
    imgs[a] = np.stack([(env.set_state(Q[i], V[i]), env.render())[1] for i in idx])
qt = np.zeros((N, K + 1, 2)); qt[:, 0] = Q[idx]   # true qpos at step 0..5
raw = np.clip(plan.reshape(N, K * FS, 2) * A_SD + A_MU, -1, 1)
for n, i in enumerate(idx):
    env.set_state(Q[i], V[i])
    for k in range(K * FS):
        env.step(raw[n, k])
        if k % FS == FS - 1: qt[n, k // FS + 1] = np.asarray(env.env.physics.data.qpos[:2])
wrap = lambda d: np.abs((d + np.pi) % (2 * np.pi) - np.pi)

MEAN = torch.tensor([0.485, 0.456, 0.406], device=E.DEV).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=E.DEV).view(1, 3, 1, 1)
print(f"N={N} bank={NB}", flush=True)
for wm, epn in WMS:
    m = E.swm.wm.utils.load_pretrained(f"{wm}/weights_epoch_{epn}.pt").to(E.DEV).eval()
    @torch.no_grad()
    def enc(a, bs=250):
        return torch.cat([m.projector(m.encoder(((torch.as_tensor(a[i:i + bs], device=E.DEV).permute(0, 3, 1, 2).float() / 255. - MEAN) / STD),
                                                 interpolate_pos_encoding=True).last_hidden_state[:, 0]) for i in range(0, len(a), bs)])
    @torch.no_grad()
    def steps(z0):   # (N, K+1, D): encoder latent + K predicted
        emb, Ab = z0.unsqueeze(1), torch.from_numpy(plan).to(E.DEV)
        for k in range(K):
            act = m.action_encoder(Ab[:, :k + 1]); emb = torch.cat([emb, m.predict(emb[:, -3:], act[:, -3:])[:, -1:]], 1)
        return emb
    Zb = enc(Pb)
    @torch.no_grad()
    def knn(Z):   # (N,) mean distance to 10 NN, index of the nearest
        d = torch.cdist(Z, Zb); v, i = d.topk(10, largest=False)
        return v.mean(1).cpu().numpy(), i[:, 0].cpu().numpy()
    ref = None
    for a in ALPHAS:
        S = steps(enc(imgs[a])); out = []
        res = [knn(S[:, k]) for k in range(K + 1)]
        if a == 0: ref = [(np.median(d), np.quantile(d, 0.99)) for d, _ in res]
        for k, (d, j) in enumerate(res):
            err = np.median(wrap(qt[:, k] - Qb[j]).max(1))
            out.append(f"{np.median(d) / ref[k][0]:.2f}/{(d <= ref[k][1]).mean():.2f}/{err:.2f}")
        print(f"NNDIAG {wm} a={a} step0..5 knn/in_sup/nn_err: " + "  ".join(out), flush=True)
    del m; torch.cuda.empty_cache()
