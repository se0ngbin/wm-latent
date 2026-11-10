"""Does camera tilt / shift change the APPARENT joint angles (a calibration change no invariance can undo) or just add noise?
Linear readout of (sin q1, cos q1, sin q2, cos q2) from the latent, fit on clean held-out frames (ridge), applied to shifted frames of
other states. Per joint: systematic error = |mean of the wrapped angle error| (consistent bias, apparent-geometry change) vs its
scatter (std of the error, extra noise), in degrees; clean error as reference. Also the angle error's correlation with the true angle
(a state-dependent, i.e. geometric, distortion). Env: WMS, N."""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch, h5py, gymnasium as gym
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import stable_worldmodel as swm, reacher_shift as RS
G = os.environ.get
WMS = [w.split(":") for w in G("WMS", "jac:rc_jacobian,adv_h varying radius:rc_jac_safeadv_linf_epsmix,whole latent:rc_jac_safeadv_latent").split(",")]
N, DEV = int(G("N", "1200")), "cuda"
rng = np.random.default_rng(0)
with h5py.File("/data/seongbin/lewm/datasets/reacher.h5", "r") as f:
    ep, Q, V = f["ep_idx"][:], f["qpos"][:], f["qvel"][:]
idx = np.sort(rng.choice(np.where(ep >= 9000)[0], N, replace=False)); q = Q[idx]
fit, te = np.arange(N) % 3 != 0, np.arange(N) % 3 == 0
def render(sh):
    env = gym.make("swm/ReacherDMControl-v0", task="qpos_match", render_mode="rgb_array").unwrapped; env.reset(seed=0); RS.apply_physics(env, sh)
    return np.stack([(env.set_state(Q[i], V[i]), env.render())[1] for i in idx])
X = {c: render(c) for c in ["", "cam_tilt", "cam_shift"]}
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
T = np.c_[np.sin(q[:, 0]), np.cos(q[:, 0]), np.sin(q[:, 1]), np.cos(q[:, 1])]
wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
for name, wm in WMS:
    m = swm.wm.utils.load_pretrained(f"{wm}/weights_epoch_10.pt").to(DEV).eval()
    @torch.no_grad()
    def Z(a): return torch.cat([m.projector(m.encoder((torch.as_tensor(a[i:i + 100], device=DEV).permute(0, 3, 1, 2).float() / 255. - MEAN) / STD, interpolate_pos_encoding=True).last_hidden_state[:, 0]) for i in range(0, len(a), 100)]).cpu().numpy()
    Zc = Z(X[""]); A = np.c_[Zc[fit], np.ones(fit.sum())]
    W = np.linalg.solve(A.T @ A + 1e-2 * np.eye(A.shape[1]), A.T @ T[fit])
    row = []
    for c in ["", "cam_tilt", "cam_shift"]:
        Zs = Z(X[c])[te] if c else Zc[te]; P = np.c_[Zs, np.ones(len(Zs))] @ W
        e1 = np.degrees(wrap(np.arctan2(P[:, 0], P[:, 1]) - q[te, 0])); e2 = np.degrees(wrap(np.arctan2(P[:, 2], P[:, 3]) - q[te, 1]))
        cor = np.corrcoef(e1, q[te, 0])[0, 1]
        row.append(f"{c or 'clean'}: q1 bias {e1.mean():+.1f} sd {e1.std():.1f} | q2 bias {e2.mean():+.1f} sd {e2.std():.1f} | corr(err1,q1) {cor:+.2f}")
    print(f"APPARENT {name:22s} || " + " || ".join(row), flush=True)
    del m; torch.cuda.empty_cache()
