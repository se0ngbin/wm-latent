"""GP margin function trained on PREDICTED latents of CEM-style plans, labelled in the simulator.
h on imagined latents is optimistic (executed-plan traces: 85-100% of soft-CEM collisions are false-safe) because the GP
margin function only ever saw encoder latents of real frames. Here:
  start = one real frame of a training-split expert episode (eval uses a 1-frame history, as in the CEM rollouts);
  plan  = 25 normalized actions, a = b + 0.7 eps (b ~ N(0,1) per plan, eps ~ N(0,1) per step): CEM's N(0, 1) first
          iteration plus the smooth turning plans it converges to;
  z_hat = predictor rollout of the plan, exactly as JEPA.rollout (history <= 3);
  label = the same plan (de-normalized, clipped) run through the Dubins RK4 sim from the frame's true state;
          1 = inside a red circle at that step.
h is trained (same margin_gp loss / steps as train_gp_margin_dubins.py) on predicted latents + the real uniform-state
latents of the original GP, so it stays valid on real latents. Reports AUC / false-safe on held-out plans (last 10% of
episodes) for the old and new h. Saves {"margin_gp": ...} to <MDIR>/<wm>/margin_gp_imag_red_s<seed>.pt.
Usage: train_gp_margin_imag_dubins.py <wm_dir> [seed]"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, h5py
from sklearn.metrics import roc_auc_score
import ood_margin_gp_jepa as E
import stable_worldmodel as swm
from safe_solver import _MarginHead

ck, seed = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
K, NTR, NTE, DEV, MDIR = 25, 6000, 1000, E.DEV, "/data/seongbin/dreamer/lewm"
tf = E.get_img_preprocessor(source="pixels", target="pixels", img_size=E.IMG)
frames = lambda b: tf({"pixels": torch.from_numpy(np.ascontiguousarray(b)).permute(0, 3, 1, 2)})["pixels"].float()
rng = np.random.default_rng(seed)

with h5py.File("/data/seongbin/lewm/datasets/dubins_expert.h5", "r") as f:
    off, ln = f["ep_offset"][:], f["ep_len"][:]
    act_all = f["action"][:].astype(np.float32).reshape(len(f["action"]), -1)
    a_mu, a_sd = np.nanmean(act_all, 0), np.nanstd(act_all, 0)
    cut = int(0.9 * len(ln))
    def starts(eps, n):
        e = rng.choice(eps, n)
        return np.sort(off[e] + rng.integers(0, ln[e]))
    i_tr, i_te = starts(np.arange(cut), NTR), starts(np.arange(cut, len(ln)), NTE)
    load = lambda i: (f["pixels"][np.unique(i)], f["state"][np.unique(i)], np.unique(i, return_inverse=True)[1])
    (px_tr, st_tr, v_tr), (px_te, st_te, v_te) = load(i_tr), load(i_te)
    px_tr, st_tr, px_te, st_te = px_tr[v_tr], st_tr[v_tr], px_te[v_te], st_te[v_te]


def plans(n):  # normalized actions (n, K, 1)
    return (rng.standard_normal((n, 1, 1)) + 0.7 * rng.standard_normal((n, K, 1))).astype(np.float32)


def simulate(st0, a_norm):  # true states after each action -> failure label (n, K)
    env, y = E.DubinsSwmEnv(), np.zeros(a_norm.shape[:2], np.float32)
    raw = a_norm * a_sd + a_mu                             # env.step clips to +-max_angular_velocity
    for i, s in enumerate(st0):
        env.state = s.astype(np.float32).copy()
        for k in range(K):
            env._integrate(float(np.clip(raw[i, k, 0], -env.max_angular_velocity, env.max_angular_velocity)))
            y[i, k] = E.inside_obstacle(env.state[None, :2], "circle")[0]
    return y


@torch.no_grad()
def imagine(m, px, a_norm, bs=256):  # mirrors JEPA.rollout with a 1-frame history: (n, K, D)
    out = []
    for s in range(0, len(px), bs):
        emb = E.encode(m, frames(px[s:s + bs])).to(DEV).unsqueeze(1)
        A = torch.from_numpy(a_norm[s:s + bs]).to(DEV)
        for k in range(K):
            act = m.action_encoder(A[:, :k + 1])
            emb = torch.cat([emb, m.predict(emb[:, -3:], act[:, -3:])[:, -1:]], 1)
        out.append(emb[:, 1:].cpu())
    return torch.cat(out)


m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(DEV).eval()
a_tr, a_te = plans(NTR), plans(NTE)
Zh_tr, Zh_te = imagine(m, px_tr, a_tr), imagine(m, px_te, a_te)
y_tr, y_te = simulate(st_tr, a_tr), simulate(st_te, a_te)
print(f"{ck}: unsafe frac train {y_tr.mean():.3f} test {y_te.mean():.3f}", flush=True)
# real-frame anchor set = the original GP's training data (uniform states, red circles, training orientation)
og = E.AXES["color"][0]
s = np.stack([rng.uniform(E.WORLD_BOUNDS[0], E.WORLD_BOUNDS[1], E.N), rng.uniform(E.WORLD_BOUNDS[2], E.WORLD_BOUNDS[3], E.N),
              rng.uniform(-np.pi, np.pi, E.N)], 1).astype(np.float32)
tf_fixed = lambda d: tf({"pixels": torch.from_numpy(np.ascontiguousarray(d["pixels"])).permute(0, 3, 1, 2)})
Zr = E.encode(m, E.render_cond(E.DubinsSwmEnv(), s, og, tf_fixed))
yr = E.inside_obstacle(s[:, :2], "circle")
Z = torch.cat([Zh_tr.flatten(0, 1), Zr]); Y = np.concatenate([y_tr.ravel(), yr])
head = E.train_margin(Z, Y, "gp", seed=seed)
out = f"{MDIR}/{ck}/margin_gp_imag_red_s{seed}.pt"
torch.save({"margin_gp": head.state_dict()}, out)


def report(nm, hd):
    with torch.no_grad():
        h = hd(Zh_te.flatten(0, 1).to(DEV).float()).view(NTE, K).cpu().numpy()
    safe = y_te == 0
    fs = lambda a, b: (h[:, a:b][~safe[:, a:b]] >= 0).mean()
    print(f"  {nm}: imag AUC {roc_auc_score(safe.ravel(), h.ravel()):.3f}  false-safe {fs(0, K):.3f} "
          f"(steps 1-5 {fs(0, 5):.3f}, 21-25 {fs(20, K):.3f})  false-alarm {(h[safe] < 0).mean():.3f}", flush=True)


old = _MarginHead().to(DEV); old.load_state_dict(torch.load(f"{MDIR}/{ck}/margin_gp_fail_red_s0.pt", map_location=DEV)["margin_gp"])
report("old h (real latents)", old.eval()); report("new h (+predicted)", head)
print(f"saved {out}", flush=True)
