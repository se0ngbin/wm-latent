"""Predictor sensitivities of trained Dubins LeWMs, to set the target L of a predictor Jacobian term.
Start states: 600 held-out expert states (last 10% of episodes), red, training orientation, 1-frame context (as at the
start of a CEM episode); actions z-scored like training. Eval mode. Per model prints medians (and 10/90 percentiles) of:
  state gain   ||P(z + d, a) - P(z, a)|| / ||d||,  d ~ N(0, I) * 0.05 * rms(z)        (local, w.r.t. the latent)
  action gain  ||P(z, a + e) - P(z, a)|| / ||e||,  e ~ N(0, I) * 0.05                   (local, w.r.t. the action)
  swap ratio   ||P(z, a) - P(z, a')|| / ||a - a'||, a' = another start's action         (what pull regresses to 1)
Usage: pred_gain_dubins.py <wm_dir> [...]"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, h5py
from safe_solver import _load_margin_head   # noqa: F401  (import order: before E, see diag_imag_color_dubins.py)
import ood_margin_gp_jepa as E
import stable_worldmodel as swm

N, DEV = 600, E.DEV
rng = np.random.default_rng(2)
tf = E.get_img_preprocessor(source="pixels", target="pixels", img_size=E.IMG)
tf_fixed = lambda d: tf({"pixels": torch.from_numpy(np.ascontiguousarray(d["pixels"])).permute(0, 3, 1, 2)})
with h5py.File("/data/seongbin/lewm/datasets/dubins_expert.h5", "r") as f:
    off, ln = f["ep_offset"][:], f["ep_len"][:]
    e = rng.choice(np.arange(int(0.9 * len(ln)), len(ln)), N)
    st0 = f["state"][:][off[e] + rng.integers(0, ln[e])]
A = torch.from_numpy(rng.standard_normal((N, 1, 1)).astype(np.float32)).to(DEV)
cond = E.AXES["color"][0]
q = lambda x: f"{np.median(x):.3f} [{np.percentile(x, 10):.3f}, {np.percentile(x, 90):.3f}]"
for ck in sys.argv[1:]:
    m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(DEV).eval()
    g = torch.Generator(device=DEV).manual_seed(0)
    with torch.no_grad():
        z = E.encode(m, E.render_cond(E.DubinsSwmEnv(), st0, cond, tf_fixed)).to(DEV).unsqueeze(1)   # (N, 1, D)
        P = lambda zz, aa: m.predict(zz, m.action_encoder(aa))[:, -1]
        p0 = P(z, A)
        d = torch.randn(z.shape, device=DEV, generator=g) * 0.05 * z.pow(2).mean().sqrt()
        sg = ((P(z + d, A) - p0).norm(dim=-1) / d.flatten(1).norm(dim=-1)).cpu().numpy()
        ea = torch.randn(A.shape, device=DEV, generator=g) * 0.05
        ag = ((P(z, A + ea) - p0).norm(dim=-1) / ea.flatten(1).norm(dim=-1)).cpu().numpy()
        perm = torch.randperm(N, device=DEV, generator=g)
        da = (A - A[perm]).flatten(1).norm(dim=-1)
        ok = da > 0.1
        sr = ((P(z, A[perm]) - p0).norm(dim=-1) / da)[ok].cpu().numpy()
    print(f"PREDGAIN {ck}: state gain {q(sg)} | action gain {q(ag)} | swap ratio {q(sr)}", flush=True)
