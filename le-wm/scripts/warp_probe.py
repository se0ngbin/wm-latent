"""Geometry probe: is a model's margin function sensitive to small spatial warps (translation, rotation, scaling) of the frame?
Hypothesis: adversarial training against intensity perturbations (L-inf at fixed pixel positions) pushes the encoder to read the arm
through spatial layout, making it MORE sensitive to warps (camera tilt got worse with the varying-radius adversary). Synthetic warps
of rendered held-out frames (affine_grid, border padding), N states (half unsafe): median |dh| / gap and mean ||dz|| / spread per
warp, per model (MLP real+imag margin functions). Env: WMS (name:wm,...), N."""
import os, sys
import numpy as np, torch, torch.nn.functional as F, h5py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import stable_worldmodel as swm
from safe_solver import _load_margin_head

G = os.environ.get
WMS = [w.split(":") for w in G("WMS", "jac:rc_jacobian,adv_h:rc_jac_safeadv_linf,adv_h varying radius:rc_jac_safeadv_linf_epsmix,whole latent:rc_jac_safeadv_latent").split(",")]
N, DEV = int(G("N", "400")), "cuda"
rng = np.random.default_rng(0)
with h5py.File("/data/seongbin/lewm/datasets/reacher.h5", "r") as f:
    y_all, ep = f["failures"][:].astype(int), f["ep_idx"][:]
    ho = np.where(ep >= 9000)[0]
    idx = np.sort(np.r_[rng.choice(ho[y_all[ho] == 1], N // 2, replace=False), rng.choice(ho[y_all[ho] == 0], N // 2, replace=False)])
    X = torch.as_tensor(f["pixels"][idx], device=DEV).permute(0, 3, 1, 2).float() / 255.
y = torch.as_tensor(y_all[idx], device=DEV)
WARPS = [("translate", 2), ("translate", 4), ("translate", 8), ("rotate", 1), ("rotate", 3), ("rotate", 5), ("scale", 1.02), ("scale", 1.05)]
def warp(x, kind, v):
    n = len(x); th = torch.zeros(n, 2, 3, device=DEV); th[:, 0, 0] = th[:, 1, 1] = 1.0
    if kind == "translate": th[:, 0, 2] = 2 * v / x.shape[-1]          # v pixels right (normalized coords span 2)
    elif kind == "rotate":
        a = np.deg2rad(v); th[:, 0, 0] = th[:, 1, 1] = np.cos(a); th[:, 0, 1] = -np.sin(a); th[:, 1, 0] = np.sin(a)
    else: th[:, 0, 0] = th[:, 1, 1] = 1.0 / v                          # zoom in by v
    return F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), padding_mode="border", align_corners=False)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
print(f"N={N} | per warp: median |dh|/gap , mean ||dz||/spread", flush=True)
for name, wm in WMS:
    m = swm.wm.utils.load_pretrained(f"{wm}/weights_epoch_10.pt").to(DEV).eval()
    head = _load_margin_head(f"/data/seongbin/lewm/reacher_heads_imag/{wm}/mlp_s0.pt", DEV).eval()
    @torch.no_grad()
    def Z(x): return torch.cat([m.projector(m.encoder((x[i:i + 100] - MEAN) / STD, interpolate_pos_encoding=True).last_hidden_state[:, 0]) for i in range(0, len(x), 100)])
    z0 = Z(X)
    with torch.no_grad(): h0 = head(z0).view(-1)
    gap = (h0[y == 0].mean() - h0[y == 1].mean()).item(); spread = (z0 - z0.mean(0)).pow(2).sum(1).mean().sqrt().item()
    row = []
    for kind, v in WARPS:
        zw = Z(warp(X, kind, v))
        with torch.no_grad(): dh = ((head(zw).view(-1) - h0).abs() / gap).median().item()
        row.append(f"{kind} {v}: {dh:.3f} , {((zw - z0).norm(dim=1).mean() / spread).item():.3f}")
    print(f"WARP {name:22s} | " + " | ".join(row), flush=True)
    del m; torch.cuda.empty_cache()
