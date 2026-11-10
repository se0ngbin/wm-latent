"""When does the adversarial safety reg (adv_h, L-inf 8/255) help? Hypothesis: adv_h buys robustness only against real shifts whose
latent displacement lies in the directions its training adversary exploits. Per benchmark (reacher | dubins), jac vs jac+adv_h:
  1. adversarial subspace: PGD (STEPS sign steps, L-inf EPS on [0, 1] pixels) on jac maximizing ||z(x+d) - z(x)||^2; U = top-K
     right singular vectors of the stacked displacements (uncentered)
  2. alignment of each real shift on jac: mean_s ||U^T dz||^2 / ||dz||^2 (random directions: K / D)
  3. adv_h's benefit per shift: ratio adv_h / jac of median |dh| / gap (MLP real+imag margin functions) and of mean ||dz|| / spread
  4. pixel character of the shift: mean |dx|, median per-image max |dx|, fraction of pixels with |dx| > EPS, high-frequency energy
     fraction of dx (FFT energy beyond 1/8 of the Nyquist radius)
Usage: adv_align.py reacher|dubins. Env: N (states), NA (attacked states), K, EPS (1/255), STEPS."""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import stable_worldmodel as swm
from safe_solver import _load_margin_head

G = os.environ.get
BENCH = sys.argv[1]
N, NA, K, EPS, STEPS, DEV = int(G("N", "400")), int(G("NA", "200")), int(G("K", "10")), float(G("EPS", "8")) / 255, int(G("STEPS", "5")), "cuda"
rng = np.random.default_rng(0)
if BENCH == "reacher":
    import h5py, gymnasium as gym, reacher_shift as RS
    MODELS = {"jac": ("rc_jacobian/weights_epoch_10.pt", "/data/seongbin/lewm/reacher_heads_imag/rc_jacobian/mlp_s0.pt"),
              "jac+adv_h": ("rc_jac_safeadv_linf/weights_epoch_10.pt", "/data/seongbin/lewm/reacher_heads_imag/rc_jac_safeadv_linf/mlp_s0.pt")}
    with h5py.File("/data/seongbin/lewm/datasets/reacher.h5", "r") as f:
        y_all, ep, Q, V = f["failures"][:].astype(int), f["ep_idx"][:], f["qpos"][:], f["qvel"][:]
    ho = np.where(ep >= 9000)[0]
    idx = np.sort(np.r_[rng.choice(ho[y_all[ho] == 1], N // 2, replace=False), rng.choice(ho[y_all[ho] == 0], N // 2, replace=False)])
    y = y_all[idx]
    F0, F1 = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]), np.array([[0.45, 0.3, 0.15], [0.55, 0.38, 0.2]])
    SHIFTS = ["floor0.1", "floor0.2", "floor0.3", "floor0.4", "light", "exposure", "white_balance", "noise", "blur", "cam_zoom", "cam_shift", "cam_tilt", "occlusion"]
    def render(cond):
        env = gym.make("swm/ReacherDMControl-v0", task="qpos_match", render_mode="rgb_array").unwrapped
        var = {}
        if cond.startswith("floor"): var = {"floor.color": (1 - float(cond[5:])) * F0 + float(cond[5:]) * F1}
        elif cond == "light": var = {"light.intensity": np.array([0.3])}
        env.reset(seed=0, options={"variation_values": var} if var else None)
        sh = cond if cond in RS.APPEARANCE else ""; RS.apply_physics(env, sh)
        return np.stack([RS.corrupt((env.set_state(Q[i], V[i]), env.render())[1], sh, k) for k, i in enumerate(idx)])
else:
    sys.path.insert(0, "/home/seongbin/latent/.claude/worktrees/imag-margin/le-wm/scripts")
    import ood_margin_gp_jepa as E
    MODELS = {"jac": ("jacobian_w1_dubins/weights_epoch_50.pt", "/data/seongbin/dreamer/lewm/jacobian_w1_dubins/margin_gp_imag_red_s0.pt"),
              "jac+adv_h": ("lewm_dubins_jac_safeadv_linf50/weights_epoch_50.pt", "/data/seongbin/dreamer/lewm/lewm_dubins_jac_safeadv_linf50/margin_gp_imag_red_s0.pt")}
    x0, x1, y0, y1 = E.WORLD_BOUNDS
    st = np.stack([rng.uniform(x0, x1, 4 * N), rng.uniform(y0, y1, 4 * N), rng.uniform(-np.pi, np.pi, 4 * N)], 1).astype(np.float32)
    lab = E.inside_obstacle(st[:, :2], "circle").astype(int)
    st = np.r_[st[lab == 1][:N // 2], st[lab == 0][:N // 2]]; y = E.inside_obstacle(st[:, :2], "circle").astype(int)
    SHIFTS = ["purple", "diamond", "rot90"]
    COND = {"clean": dict(obstacle="red", shape="circle", rot=0), "purple": dict(obstacle="purple", shape="circle", rot=0),
            "diamond": dict(obstacle="red", shape="diamond", rot=0), "rot90": dict(obstacle="red", shape="circle", rot=90)}
    def render(cond):   # raw uint8 frames, as E.render_cond minus the preprocessor
        env = E.DubinsSwmEnv(); c = COND[cond]
        env.colors = {"background": "white", "agent": "blue", "goal": "green", "obstacle": c["obstacle"]}
        out = []
        for s in st:
            env.state = s.copy(); img = E.render_diamond(env) if c["shape"] == "diamond" else env._render_image()
            out.append(np.ascontiguousarray(np.rot90(img, k=c["rot"] // 90)) if c["rot"] else img)
        return np.stack(out)
X = {c: torch.as_tensor(render(c), device=DEV).permute(0, 3, 1, 2).float() / 255. for c in ["clean"] + SHIFTS}
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
lab_t = torch.as_tensor(y, device=DEV)

def pix_stats(dx):   # dx: (n, 3, H, W) on [0, 1]
    a = dx.abs(); Hh, Ww = dx.shape[-2:]
    fx = torch.fft.fftfreq(Hh, device=DEV).abs().view(-1, 1); fy = torch.fft.fftfreq(Ww, device=DEV).abs().view(1, -1)
    hi = (torch.sqrt(fx ** 2 + fy ** 2) > 0.5 / 8).float()
    P = torch.fft.fft2(dx).abs().pow(2).sum(1)
    return a.mean().item(), a.flatten(1).max(1).values.median().item(), (a.max(1).values > EPS).float().mean().item(), ((P * hi).sum() / P.sum().clamp_min(1e-12)).item()

res, U = {}, None
for name, (ck, hp) in MODELS.items():
    m = swm.wm.utils.load_pretrained(ck).to(DEV).eval().requires_grad_(False)
    head = _load_margin_head(hp, DEV).eval().requires_grad_(False)
    def enc(x01):
        z = (x01 - MEAN) / STD
        if z.shape[-1] != 224: z = F.interpolate(z, size=224, mode="bilinear", antialias=True, align_corners=False)
        return m.projector(m.encoder(z, interpolate_pos_encoding=True).last_hidden_state[:, 0])
    @torch.no_grad()
    def Z(x): return torch.cat([enc(x[i:i + 50]) for i in range(0, len(x), 50)])
    z0 = Z(X["clean"]); h0 = head(z0).view(-1)
    gap = (h0[lab_t == 0].mean() - h0[lab_t == 1].mean()).item(); spread = (z0 - z0.mean(0)).pow(2).sum(1).mean().sqrt().item()
    if name == "jac":   # adversarial subspace of the jac encoder
        dzs, dxs = [], []
        for i in range(0, NA, 25):
            x = X["clean"][i:i + 25]; zc = z0[i:i + 25]; d = (torch.rand_like(x) * 2 - 1) * EPS
            for _ in range(STEPS):
                d.requires_grad_(True)
                with sdpa_kernel(SDPBackend.MATH):
                    g, = torch.autograd.grad((enc((x + d).clamp(0, 1)) - zc).pow(2).sum(), d)
                d = (d.detach() + EPS / 2 * g.sign()).clamp(-EPS, EPS)
            with torch.no_grad(): dzs.append(enc((x + d).clamp(0, 1)) - zc)
            dxs.append(d)
        Dz = torch.cat(dzs); U = torch.linalg.svd(Dz, full_matrices=False).Vh[:K].T   # (D, K)
        res["pgd"] = dict(pix=pix_stats(torch.cat(dxs)), align=((Dz @ U).pow(2).sum(1) / Dz.pow(2).sum(1)).mean().item())
        print(f"{BENCH}: N={N}, adversarial subspace from {NA} PGD displacements, K={K} of D={Dz.shape[1]} (random alignment {K / Dz.shape[1]:.3f}); "
              f"PGD alignment {res['pgd']['align']:.3f}, pixel mean/max/frac>eps/hi-freq {' '.join(f'{v:.3f}' for v in res['pgd']['pix'])}", flush=True)
    for c in SHIFTS:
        zs = Z(X[c]); dz = zs - z0
        with torch.no_grad(): dh = ((head(zs).view(-1) - h0).abs() / gap).median().item()
        r = res.setdefault(c, {}); r[name] = dict(dh=dh, dz=(dz.norm(dim=1).mean() / spread).item())
        if name == "jac":
            r["align"] = ((dz @ U).pow(2).sum(1) / dz.pow(2).sum(1).clamp_min(1e-12)).mean().item(); r["pix"] = pix_stats(X[c] - X["clean"])
    del m; torch.cuda.empty_cache()
print(f"ALIGN {BENCH} | shift | alignment (jac) | |dh|/gap jac -> adv_h (ratio) | ||dz||/spread jac -> adv_h (ratio) | pixel mean / max / frac>eps / hi-freq", flush=True)
for c in SHIFTS:
    r = res[c]; a, b = r["jac"], r["jac+adv_h"]
    print(f"ALIGN {BENCH} | {c} | {r['align']:.3f} | {a['dh']:.3f} -> {b['dh']:.3f} ({b['dh'] / max(a['dh'], 1e-9):.2f}) | {a['dz']:.3f} -> {b['dz']:.3f} ({b['dz'] / max(a['dz'], 1e-9):.2f}) | "
          + " / ".join(f"{v:.3f}" for v in r["pix"]), flush=True)
