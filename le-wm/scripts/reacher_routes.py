"""Route decomposition on Reacher (port of safeadv_results/diagnostics/three_routes.py, the Dubins/Safety Gym analysis).
For a shift dz and the model's safety direction lambda (safe-minus-unsafe latent centroid difference on the eval set):
    along/gap = cos(dz, lambda_n) * (|dz| / spread) / (gap / spread)
cos = orthogonality route, |dz|/spread = invariance route, gap/spread = widening route; spread = RMS radius of the clean latent
cloud. Added: motion/spread = median ||z(s+5 env steps) - z(s)|| / spread (state resolution: does a reg compress real motion?).
Shifts: PGD (eps 8/255 Linf, 2 sign steps maximizing (dz . lambda_n)^2 with best-iterate, as three_routes.py), and real shifts
from paired renders of the same states (floor blends, light, arm colour, reacher_shift.py exposure / white balance / occlusion /
camera shift). N held-out states (episodes >= 9000), half unsafe, each with the state 5 env steps later in the same episode.
Env: WMS (name:ckpt,...), N, CONDS."""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch, h5py, gymnasium as gym
from torch.nn.attention import sdpa_kernel, SDPBackend
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import stable_worldmodel as swm
import reacher_shift as RS

G = os.environ.get
WMS = [w.split(":") for w in G("WMS", "sigreg:rc_baseline/weights_epoch_10.pt,jac ep5:rc_jacobian/weights_epoch_5.pt,jac:rc_jacobian/weights_epoch_10.pt,"
                                      "jac+adv_h ep5:rc_jac_safeadv_linf/weights_epoch_5.pt,official:lewm_reacher/weights_epoch_50.pt").split(",")]
N, EPS, DEV = int(G("N", "1000")), 8 / 255, "cuda"
CONDS = G("CONDS", "floor0.2,floor0.3,light,arm_color,exposure,white_balance,occlusion,cam_shift").split(",")
rng = np.random.default_rng(0)
with h5py.File("/data/seongbin/lewm/datasets/reacher.h5", "r") as f:
    y_all, ep, Q, V = f["failures"][:].astype(int), f["ep_idx"][:], f["qpos"][:], f["qvel"][:]
t = np.arange(len(ep) - 5); ok = (ep[t] >= 9000) & (ep[t] == ep[t + 5])
idx = np.sort(np.r_[rng.choice(t[ok & (y_all[t] == 1)], N // 2, replace=False), rng.choice(t[ok & (y_all[t] == 0)], N // 2, replace=False)])
y = torch.as_tensor(y_all[idx])
F0, F1 = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]), np.array([[0.45, 0.3, 0.15], [0.55, 0.38, 0.2]])
def render(cond, ii):
    env = gym.make("swm/ReacherDMControl-v0", task="qpos_match", render_mode="rgb_array").unwrapped
    var = {}
    if cond.startswith("floor"): var = {"floor.color": (1 - float(cond[5:])) * F0 + float(cond[5:]) * F1}
    elif cond == "light": var = {"light.intensity": np.array([0.3])}
    elif cond == "arm_color": var = {"agent.color": np.array([0.55, 0.2, 0.75])}
    env.reset(seed=0, options={"variation_values": var} if var else None)
    sh = cond if cond in RS.APPEARANCE else ""; RS.apply_physics(env, sh)
    return np.stack([RS.corrupt((env.set_state(Q[i], V[i]), env.render())[1], sh, k) for k, i in enumerate(ii)])
X = {c: render(c, idx) for c in ["clean"] + CONDS}; X5 = render("clean", idx + 5)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
to01 = lambda a: torch.as_tensor(a, device=DEV).permute(0, 3, 1, 2).float() / 255.
print(f"N={N} | model | shift | cos(dz,λn) | |dz|/spread | gap/spread | along/gap | motion/spread", flush=True)
for name, ck in WMS:
    m = swm.wm.utils.load_pretrained(ck).to(DEV).eval().requires_grad_(False)
    def enc(x01): return m.projector(m.encoder((x01 - MEAN) / STD, interpolate_pos_encoding=True).last_hidden_state[:, 0])
    @torch.no_grad()
    def Z(a): return torch.cat([enc(to01(a[i:i + 100])) for i in range(0, len(a), 100)]).cpu()
    zc = Z(X["clean"]); lam = zc[y == 0].mean(0) - zc[y == 1].mean(0); gap = lam.norm(); u = (lam / gap).to(DEV)
    spread = (zc - zc.mean(0)).pow(2).sum(1).mean().sqrt(); motion = (Z(X5) - zc).norm(dim=1).median()
    def pgd():
        out = []
        for i in range(0, N, 32):
            x01 = to01(X["clean"][i:i + 32])
            with torch.no_grad(): zb = enc(x01)
            f_ = lambda P: ((enc((x01 + P).clamp(0, 1)) - zb) @ u).pow(2)
            P = (torch.rand_like(x01) * 2 - 1) * EPS; best, bd = P, None
            for k in range(3):
                if k < 2:
                    P = P.detach().requires_grad_(True)
                    with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH): dk = f_(P); g, = torch.autograd.grad(dk.sum(), P)
                else:
                    with torch.no_grad(): dk = f_(P)
                dk = dk.detach()
                if bd is None: bd = dk
                else: bt = dk > bd; best = torch.where(bt.view(-1, 1, 1, 1), P.detach(), best); bd = torch.maximum(dk, bd)
                if k < 2: P = (P.detach() + 0.5 * EPS * g.sign()).clamp(-EPS, EPS)
            with torch.no_grad(): out.append((enc((x01 + best).clamp(0, 1)) - zb).cpu())
        return torch.cat(out)
    shifts = {"pgd": pgd(), **{c: Z(X[c]) - zc for c in CONDS}}
    for sk, dz in shifts.items():
        along = (dz @ u.cpu()).abs().mean(); tot = dz.norm(dim=1).mean()
        print(f"ROUTE reacher | {name} | {sk} | {float(along / tot):.3f} | {float(tot / spread):.3f} | {float(gap / spread):.3f} | {float(along / gap):.3f} | {float(motion / spread):.3f}", flush=True)
    del m; torch.cuda.empty_cache()
print("done")
