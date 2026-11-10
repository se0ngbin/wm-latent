"""Is the real camera-tilt / camera-shift damage from moving the arm, or from newly exposed out-of-distribution content (dark area
beyond the floor edge that never appears in training)? For N held-out states: render clean and shifted frames; exposed region =
pixels that are near-black in the shifted frame (mean RGB < THR) but not in the clean one; 'filled' = shifted frame with that region
replaced by the clean pixels (keeps the arm's displacement, removes the new content). Median |dh| / gap for shifted vs filled, and
the exposed-area fraction. Env: WMS, N, THR."""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch, h5py, gymnasium as gym
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import stable_worldmodel as swm, reacher_shift as RS
from safe_solver import _load_margin_head
G = os.environ.get
WMS = [w.split(":") for w in G("WMS", "jac:rc_jacobian,adv_h:rc_jac_safeadv_linf,adv_h varying radius:rc_jac_safeadv_linf_epsmix,whole latent:rc_jac_safeadv_latent").split(",")]
N, THR, DEV = int(G("N", "400")), float(G("THR", "0.08")), "cuda"
rng = np.random.default_rng(0)
with h5py.File("/data/seongbin/lewm/datasets/reacher.h5", "r") as f:
    y_all, ep, Q, V = f["failures"][:].astype(int), f["ep_idx"][:], f["qpos"][:], f["qvel"][:]
ho = np.where(ep >= 9000)[0]
idx = np.sort(np.r_[rng.choice(ho[y_all[ho] == 1], N // 2, replace=False), rng.choice(ho[y_all[ho] == 0], N // 2, replace=False)])
y = torch.as_tensor(y_all[idx], device=DEV)
def render(sh):
    env = gym.make("swm/ReacherDMControl-v0", task="qpos_match", render_mode="rgb_array").unwrapped; env.reset(seed=0); RS.apply_physics(env, sh)
    return torch.as_tensor(np.stack([(env.set_state(Q[i], V[i]), env.render())[1] for i in idx]), device=DEV).permute(0, 3, 1, 2).float() / 255.
X0 = render("")
S = {}
for sh in ["cam_tilt", "cam_shift"]:
    Xs = render(sh); exposed = ((Xs.mean(1, keepdim=True) < THR) & (X0.mean(1, keepdim=True) >= THR)).float()
    S[sh] = (Xs, Xs * (1 - exposed) + X0 * exposed, exposed.mean().item())
print(f"N={N} exposed-area fraction: " + " ".join(f"{k} {v[2]:.3f}" for k, v in S.items()), flush=True)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
for name, wm in WMS:
    m = swm.wm.utils.load_pretrained(f"{wm}/weights_epoch_10.pt").to(DEV).eval()
    head = _load_margin_head(f"/data/seongbin/lewm/reacher_heads_imag/{wm}/mlp_s0.pt", DEV).eval()
    @torch.no_grad()
    def H(x): return torch.cat([head(m.projector(m.encoder((x[i:i + 100] - MEAN) / STD, interpolate_pos_encoding=True).last_hidden_state[:, 0])).view(-1) for i in range(0, len(x), 100)])
    h0 = H(X0); gap = (h0[y == 0].mean() - h0[y == 1].mean()).item()
    row = [f"{sh}: shifted {((H(a) - h0).abs() / gap).median().item():.3f} -> exposed area filled {((H(b) - h0).abs() / gap).median().item():.3f}" for sh, (a, b, _) in S.items()]
    print(f"EXPOSED {name:22s} | " + " | ".join(row), flush=True)
    del m; torch.cuda.empty_cache()
