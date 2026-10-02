"""Test the conditioning->orthogonality hypothesis: does cos_la (appearance shift alignment
with the failure safe/unsafe axis) fall as the encoder Jacobian gets more isotropic
(higher eff_rank / lower σ1-share)? Per encoder: Jacobian spectrum stats + cos_la(color,rotate).
"""
import os, sys
sys.path.insert(0, "/home/seongbin/latent/le-wm")
os.environ.setdefault("STABLEWM_HOME", "/data/seongbin/lewm")
os.environ.setdefault("LOCAL_DATASET_DIR", "/data/seongbin/lewm")
import numpy as np, torch
import stable_worldmodel as swm
from dubins_swm_env import DubinsSwmEnv, WORLD_BOUNDS
from utils import get_img_preprocessor
from ood_margin_gp_jepa import render_cond, encode, inside_obstacle, AXES

DEV, IMG, NF, N = "cuda:0", 224, 6, 3000
MODELS = [
    ("baseline", "orig", "sigreg_only_dubins"), ("jacobian","orig","jacobian_w1_dubins"),
    ("jac+pull","orig","lewm_dubins_jacpull50"),
    ("baseline","color","coloraug_baseline"), ("jacobian","color","coloraug_jacobian"),
    ("jac+pull","color","coloraug_jacpull"),
    ("baseline","color+rot","coloraug_rot_baseline"), ("jacobian","color+rot","coloraug_rot_jacobian"),
    ("jac+pull","color+rot","coloraug_rot_jacpull"),
]

def raw_encode(m, x):
    return m.projector(m.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0])

def jac_svd(m, x1):
    from torch.nn.attention import sdpa_kernel, SDPBackend
    x1 = x1.to(DEV).requires_grad_(True)
    with sdpa_kernel(SDPBackend.MATH):
        z = raw_encode(m, x1).squeeze(0)
        J = torch.stack([torch.autograd.grad(z[k], x1, retain_graph=True)[0].flatten().detach()
                         for k in range(z.numel())])
    return torch.linalg.svdvals(J.float()).cpu().numpy()

def cosla(mu, la):
    return float(abs((mu@la)/(np.linalg.norm(mu)*np.linalg.norm(la)+1e-12)))

def frames(n):
    ds = swm.data.load_dataset("dubins_expert.h5", transform=None, cache_dir=os.environ["LOCAL_DATASET_DIR"],
                               num_steps=1, frameskip=1, keys_to_load=["pixels"], keys_to_cache=[])
    ds.transform = get_img_preprocessor(source="pixels", target="pixels", img_size=IMG)
    idx = np.random.default_rng(0).choice(len(ds), n, replace=False)
    return torch.stack([ds[int(i)]["pixels"][0] for i in idx]).float()

def main():
    x0,x1,y0,y1 = WORLD_BOUNDS
    rng = np.random.default_rng(0)
    st = np.stack([rng.uniform(x0,x1,N), rng.uniform(y0,y1,N), rng.uniform(-np.pi,np.pi,N)],1).astype(np.float32)
    env = DubinsSwmEnv(); tf = get_img_preprocessor(source="pixels", target="pixels", img_size=IMG)
    px = frames(NF)
    for tag, cond_name, ckpt in MODELS:
        m = swm.wm.utils.load_pretrained(f"{ckpt}/weights_epoch_50.pt").to(DEV).eval()
        # Jacobian spectrum (avg over frames)
        S = np.stack([jac_svd(m, px[i:i+1]) for i in range(NF)]).mean(0)
        lam = S**2; effrank = float((lam.sum()**2)/(lam**2).sum()); fro = float(np.sqrt(lam.sum()))
        s1share = float(lam[0]/lam.sum())
        # cos_la for color + rotate (failure-label axis)
        cl = {}
        for axis in ("color","rotate"):
            og, new, *_ = AXES[axis]
            y = inside_obstacle(st[:,:2], og["shape"])   # 1=unsafe(inside)
            Zog = encode(m, render_cond(env, st, og, tf)).cpu().numpy()
            Znew = encode(m, render_cond(env, st, new, tf)).cpu().numpy()
            mu = (Znew-Zog).mean(0)
            la = Zog[y==1].mean(0) - Zog[y==0].mean(0)   # unsafe - safe
            cl[axis] = cosla(mu, la)
        print(f"CJ {tag:9s} {cond_name:9s} effrank={effrank:6.1f} froJ={fro:7.2f} s1share={s1share:.3f} "
              f"cos_la_color={cl['color']:.3f} cos_la_rot={cl['rotate']:.3f}")
        del m; torch.cuda.empty_cache()
    print("CJ_DONE")

if __name__ == "__main__":
    main()
