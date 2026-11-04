"""A/B: h on imagined latents when the start frame is (a) re-rendered by the swm env (goal marker at the default
(1.3, 0); what diag_imag_steps.py / diag_imag_color_dubins.py do) vs (b) the stored dataset frame of the same state
(goal marker where it was during collection; what train_gp_margin_imag_dubins.py / Table 1 do). Same 600 starts,
same plans, same labels, same head. Usage: imag_startframe_ab.py <wm_dir> <head.pt>"""
import sys
sys.path.insert(0, "/home/seongbin/latent/le-wm"); sys.path.insert(0, "/home/seongbin/latent/le-wm/scripts")
import numpy as np, torch, h5py
from sklearn.metrics import roc_auc_score
import ood_margin_gp_jepa as E
import stable_worldmodel as swm
from safe_solver import _MarginHead

ck, hp = sys.argv[1], sys.argv[2]
K, N, DEV = 25, 600, E.DEV
rng = np.random.default_rng(2)
tf = E.get_img_preprocessor(source="pixels", target="pixels", img_size=E.IMG)
tf_fixed = lambda d: tf({"pixels": torch.from_numpy(np.ascontiguousarray(d["pixels"])).permute(0, 3, 1, 2)})
with h5py.File("/data/seongbin/lewm/datasets/dubins_expert.h5", "r") as f:
    off, ln = f["ep_offset"][:], f["ep_len"][:]
    act_all = f["action"][:].astype(np.float32).reshape(len(f["action"]), -1)
    a_mu, a_sd = np.nanmean(act_all, 0), np.nanstd(act_all, 0)
    e = rng.choice(np.arange(int(0.9 * len(ln)), len(ln)), N)
    idx = off[e] + rng.integers(0, ln[e])                      # same draws as diag_imag_steps.py
    st0 = f["state"][:][idx]
    u, inv = np.unique(idx, return_inverse=True)
    px0 = f["pixels"][u][inv]
A = (rng.standard_normal((N, 1, 1)) + 0.7 * rng.standard_normal((N, K, 1))).astype(np.float32)
env, y = E.DubinsSwmEnv(), np.zeros((N, K), bool)
for i in range(N):
    env.state = st0[i].astype(np.float32).copy()
    for k in range(K):
        env._integrate(float(np.clip(A[i, k, 0] * a_sd[0] + a_mu[0], -env.max_angular_velocity, env.max_angular_velocity)))
        y[i, k] = E.inside_obstacle(env.state[None, :2], "circle")[0]
safe = ~y
m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(DEV).eval()
hd = _MarginHead().to(DEV); hd.load_state_dict(torch.load(hp, map_location=DEV)["margin_gp"]); hd.eval()
starts = {"re-rendered": E.encode(m, E.render_cond(E.DubinsSwmEnv(), st0, E.AXES["color"][0], tf_fixed)),
          "dataset frame": E.encode(m, tf_fixed({"pixels": px0})["pixels"].float())}
for nm, z0 in starts.items():
    with torch.no_grad():
        emb, At = z0.to(DEV).unsqueeze(1), torch.from_numpy(A).to(DEV)
        for k in range(K):
            emb = torch.cat([emb, m.predict(emb[:, -3:], m.action_encoder(At[:, :k + 1])[:, -3:])[:, -1:]], 1)
        h = hd(emb[:, 1:].flatten(0, 1).float()).view(N, K).cpu().numpy()
        h0 = hd(z0.to(DEV).float()).cpu().numpy()
    print(f"AB {ck} {nm:13s}: imag AUC {roc_auc_score(safe.ravel(), h.ravel()):.3f}  false-safe {(h[~safe] >= 0).mean():.3f}  "
          f"median h on start frame {np.median(h0):.2f}", flush=True)
