"""sigma1-share of the sg Dreamer encoder Jacobian d feat / d image (for the sg isotropy figure)."""
import os, sys
from pathlib import Path
_THIS = Path("/home/seongbin/latent/latent_cbf/src/latent_cbf/scripts/sg_margin_ood_dreamer.py").resolve()
for p in [str(_THIS.parents[3]), str(_THIS.parents[2]), str(_THIS.parents[1]), str(_THIS.parent)]:
    if p not in sys.path: sys.path.append(p)
import numpy as np, torch
from sg_margin_ood_dreamer import build_wm2d, RSSM, DEV
NF = 4
d = np.load("/data/seongbin/lewm/datasets/sg_ood_eval.npz")
blue, th = d["blue"], d["state"][:, 2]
wm = build_wm2d(RSSM)
def jac_svd(img, t):
    x = torch.tensor(img[None, None], dtype=torch.float32, device=DEV).requires_grad_(True)
    obs_state = np.stack([np.cos([t]), np.sin([t])], -1)[:, None].astype(np.float32)
    batch = {"image": img[None, None].astype(np.float32), "obs_state": obs_state,
             "action": np.zeros((1,1,2), np.float32), "is_first": np.ones((1,1), np.float32),
             "is_terminal": np.zeros((1,1), np.float32)}
    data = wm.preprocess(batch); data["image"] = x / 255.0
    embed = wm.encoder(data)
    st, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
    z = wm.dynamics.get_feat(st)[0, -1]
    J = torch.stack([torch.autograd.grad(z[k], x, retain_graph=True)[0].flatten().detach() for k in range(z.numel())])
    return torch.linalg.svdvals(J.float()).cpu().numpy()
S = np.stack([jac_svd(blue[i], th[i]) for i in range(NF)]).mean(0)
lam = S**2
print(f"SG_DREAMER effrank={float((lam.sum()**2)/(lam**2).sum()):.1f} s1share={float(lam[0]/lam.sum()):.3f} featdim={len(lam)}")
print("DONE")
