"""Do the pull / jac targets (both pinned to 1 DURING TRAINING) hold on the deployed (eval-mode) model?
Measures, on held-out expert windows (last 10% of episodes, red, training orientation; actions z-scored as in training):
  pull ratio   ||P(z, a) - P(z, a')|| / ||a - a'|| on the last predicted step, a' = another window's LAST action
               (exactly ActionSeparationReg's quantity), context = 1 or 3 real frames
  enc Frob     ||J_enc||_F via Hutchinson VJP (exactly JacobianNormReg's quantity) on the first frame
each with BatchNorm in eval mode (running stats, as in planning) and in train mode (batch stats; momentum 0 so the
running stats are not touched). Usage: reg_target_check.py <wm_dir> [...]"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, h5py
from safe_solver import _load_margin_head   # noqa: F401  (import order, see diag_imag_color_dubins.py)
import ood_margin_gp_jepa as E
import stable_worldmodel as swm

N, DEV = 256, E.DEV
rng = np.random.default_rng(3)
tf = E.get_img_preprocessor(source="pixels", target="pixels", img_size=E.IMG)
tf_fixed = lambda d: tf({"pixels": torch.from_numpy(np.ascontiguousarray(d["pixels"])).permute(0, 3, 1, 2)})
with h5py.File("/data/seongbin/lewm/datasets/dubins_expert.h5", "r") as f:
    off, ln = f["ep_offset"][:], f["ep_len"][:]
    act_all = f["action"][:].astype(np.float32).reshape(len(f["action"]), -1)
    a_mu, a_sd = np.nanmean(act_all, 0), np.nanstd(act_all, 0)
    e = rng.choice(np.arange(int(0.9 * len(ln)), len(ln)), N)
    t = off[e] + rng.integers(2, ln[e] - 1)
    idx = np.stack([t - 2, t - 1, t], 1)                     # 3 consecutive frames per window
    S = f["state"][:][idx]                                   # (N, 3, state)
    Aw = np.nan_to_num((act_all[idx] - a_mu) / a_sd)         # (N, 3, A), z-scored
cond = E.AXES["color"][0]
frames = [E.render_cond(E.DubinsSwmEnv(), S[:, k], cond, tf_fixed) for k in range(3)]
A = torch.from_numpy(Aw).to(DEV)
perm = torch.from_numpy(rng.permutation(N)).to(DEV)
q = lambda x: f"{np.median(x):.3f} [{np.percentile(x, 10):.3f},{np.percentile(x, 90):.3f}]"


def set_bn(m, train):
    m.eval()
    if train:
        for mod in m.modules():
            if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                mod.train(); mod.momentum = 0.0


for ck in sys.argv[1:]:
    m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(DEV)
    for bn in ("eval", "train"):
        set_bn(m, bn == "train")
        with torch.no_grad():
            z = torch.stack([E.encode(m, fr).to(DEV) for fr in frames], 1)          # (N, 3, D)
            out = []
            for T in (1, 3):
                zc, ac = z[:, -T:], A[:, -T:]
                ac2 = ac.clone(); ac2[:, -1] = A[perm, -1]
                p = m.predict(torch.cat([zc, zc]), m.action_encoder(torch.cat([ac, ac2])))[:, -1]
                dz = (p[:N] - p[N:]).norm(dim=-1); da = (ac[:, -1] - ac2[:, -1]).norm(dim=-1)
                ok = da > 0.1
                out.append(f"pull ratio ctx{T} {q((dz[ok] / da[ok]).cpu().numpy())}")
        # encoder Frobenius norm, Hutchinson VJP as in JacobianNormReg
        from torch.nn.attention import sdpa_kernel, SDPBackend
        px = E.render_cond(E.DubinsSwmEnv(), S[:64, 2], cond, tf_fixed)
        with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
            pix = (px if torch.is_tensor(px) else torch.as_tensor(px)).to(DEV).float().requires_grad_(True)
            o = m.encoder(pix, interpolate_pos_encoding=True)
            zz = m.projector(o.last_hidden_state[:, 0])
            v = torch.randn_like(zz)
            g = torch.autograd.grad(zz, pix, grad_outputs=v)[0]
            fro = g.flatten(1).norm(dim=1).detach().cpu().numpy()
        out.append(f"enc Frob {q(fro)}")
        print(f"REGCHECK {ck} BN={bn}: " + " | ".join(out), flush=True)
