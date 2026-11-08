"""Why does jac break between floor a = 0.2 and 0.3? Direction (anisotropy) or size (nonlinearity)?
Per WM, N held-out frames; d_a = x_a - x_0 (floor blend a, preprocessed pixel space). Directional slopes via finite-difference
JVP, s(v) = ||enc(x + eps v) - enc(x)|| / eps for unit-norm v:
  floor  = s(d_a / ||d_a||)
  rand   = s(u / ||u||), u ~ N(0, I)                     (what the Frobenius / Hutchinson penalty sees on average)
  top    = s(v*) after POWER power iterations on J^T J  (spectral norm estimate, the most sensitive direction)
  nonlin = ||enc(x_a) - enc(x_0)|| / (floor * ||d_a||)   (actual change / first-order prediction; >1 = grows faster than linear)
Medians over frames. floor/rand >> 1 -> the floor direction is a stiff direction the average penalty leaves loose (spectral fix);
nonlin >> 1 at large a -> local smoothness does not extend that far (finite-radius fix).
Env: WMS, N, ALPHAS, POWER."""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch, h5py, gymnasium as gym
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sg_point_eval as E

G = os.environ.get
WMS = [w.split(":") for w in G("WMS", "rc_baseline:10,rc_jacobian:10,lewm_reacher:50").split(",")]
N, POWER, EPS = int(G("N", "200")), int(G("POWER", "10")), 1e-2
ALPHAS = [float(a) for a in G("ALPHAS", "0.1,0.2,0.3,0.5").split(",")]
rng = np.random.default_rng(0)
with h5py.File("/data/seongbin/lewm/datasets/reacher.h5", "r") as f:
    ep, Q, V = f["ep_idx"][:], f["qpos"][:], f["qvel"][:]
idx = np.sort(rng.choice(np.where(ep >= 9000)[0], N, replace=False))
F0, F1 = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]), np.array([[0.45, 0.3, 0.15], [0.55, 0.38, 0.2]])
env = gym.make("swm/ReacherDMControl-v0", task="qpos_match", render_mode="rgb_array").unwrapped
imgs = {}
for a in [0.0] + ALPHAS:
    env.reset(seed=0, options={"variation_values": {"floor.color": (1 - a) * F0 + a * F1}})
    imgs[a] = np.stack([(env.set_state(Q[i], V[i]), env.render())[1] for i in idx])
MEAN = torch.tensor([0.485, 0.456, 0.406], device=E.DEV).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=E.DEV).view(1, 3, 1, 1)
X = {a: (torch.as_tensor(v, device=E.DEV).permute(0, 3, 1, 2).float() / 255. - MEAN) / STD for a, v in imgs.items()}
unit = lambda v: v / v.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
for wm, epn in WMS:
    m = E.swm.wm.utils.load_pretrained(f"{wm}/weights_epoch_{epn}.pt").to(E.DEV).eval()
    enc = lambda x: m.projector(m.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0])
    res = {k: [] for k in ["rand", "top"] + [f"floor{a}" for a in ALPHAS] + [f"nonlin{a}" for a in ALPHAS]}
    for s in range(0, N, 25):
        x0 = X[0.0][s:s + 25]
        with torch.no_grad():
            z0 = enc(x0)
            slope = lambda v: (enc(x0 + EPS * v) - z0).norm(dim=1) / EPS
            res["rand"].append(slope(unit(torch.randn_like(x0))))
            for a in ALPHAS:
                d = X[a][s:s + 25] - x0; fl = slope(unit(d))
                res[f"floor{a}"].append(fl)
                res[f"nonlin{a}"].append((enc(X[a][s:s + 25]) - z0).norm(dim=1) / (fl * d.flatten(1).norm(dim=1)))
        v = unit(torch.randn_like(x0))   # power iteration on J^T J: v <- J^T J v (VJP of the FD-JVP), normalized
        for _ in range(POWER):
            with torch.enable_grad():
                from torch.nn.attention import sdpa_kernel, SDPBackend
                with sdpa_kernel(SDPBackend.MATH):
                    xr = x0.detach().requires_grad_(True); z = enc(xr)
                    jv = (enc(x0 + EPS * v) - z0).detach() / EPS
                    v = unit(torch.autograd.grad(z, xr, grad_outputs=jv)[0].detach())
        with torch.no_grad(): res["top"].append(slope(v))
    r = {k: torch.cat(v).median().item() for k, v in res.items()}
    print(f"ANISO {wm} slope rand={r['rand']:.3f} top={r['top']:.3f} | " +
          " | ".join(f"a={a}: floor={r[f'floor{a}']:.3f} floor/rand={r[f'floor{a}'] / r['rand']:.1f} nonlin={r[f'nonlin{a}']:.2f}" for a in ALPHAS), flush=True)
    del m; torch.cuda.empty_cache()
