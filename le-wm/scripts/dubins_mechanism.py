"""Mechanism probe on the dubins LEWM encoders (dubins twin of sg_mechanism.py): encoder-Jacobian
eff_rank / ||J||_F / s1 share + per-sample appearance-shift displacement / state-latent spread
(disp/sigma) for the color / shape / rotate axes of ood_margin_gp_jepa.py (same paired renders).
Override encoders with ENCODERS="name:ckpt,name:ckpt"."""
import os, sys
sys.path.insert(0, "/home/seongbin/latent/le-wm"); sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("STABLEWM_HOME", "/data/seongbin/lewm"); os.environ.setdefault("LOCAL_DATASET_DIR", "/data/seongbin/lewm")
import numpy as np, torch
import stable_worldmodel as swm
from utils import get_img_preprocessor
from dubins_swm_env import DubinsSwmEnv, WORLD_BOUNDS
from ood_margin_gp_jepa import AXES, render_cond, encode
from sg_mechanism import jac_svd

DEV, IMG, N, NF = "cuda:0", 224, 2000, 6
ENC = {"baseline": "sigreg_only_dubins", "jacobian": "jacobian_w1_dubins", "jac+pull": "lewm_dubins_jacpull50"}
if os.environ.get("ENCODERS"):
    ENC = dict(kv.split(":") for kv in os.environ["ENCODERS"].split(","))


def main():
    rng = np.random.default_rng(0); x0, x1, y0, y1 = WORLD_BOUNDS
    st = np.stack([rng.uniform(x0, x1, N), rng.uniform(y0, y1, N),
                   rng.uniform(-np.pi, np.pi, N)], 1).astype(np.float32)
    env = DubinsSwmEnv(); tf = get_img_preprocessor(source="pixels", target="pixels", img_size=IMG)
    og = render_cond(env, st, AXES["color"][0], tf)                    # red circle, rot0
    new = {ax: render_cond(env, st, AXES[ax][1], tf) for ax in AXES}
    for tag, ck in ENC.items():
        m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(DEV).eval()
        S = np.stack([jac_svd(m, og[i:i+1]) for i in range(NF)]).mean(0)
        lam = S**2; effrank = float(lam.sum()**2 / (lam**2).sum()); s1 = float(lam[0] / lam.sum()); pr_sig = float(S.sum()**2 / lam.sum()); fro = float(np.sqrt(lam.sum()))
        Zo = encode(m, og).numpy(); sig = float(np.sqrt(2 * Zo.var(0).sum()))
        disp = {ax: float(np.linalg.norm(encode(m, new[ax]).numpy() - Zo, axis=1).mean()) / sig for ax in AXES}
        print(f"MECH {tag:12s} effrank(sig2)={effrank:6.1f} effrank(sig,report)={pr_sig:6.1f} froJ={fro:6.2f} s1share={s1:.3f} sigma={sig:.3f} | "
              + " | ".join(f"{ax} disp/sig={d:.2f}" for ax, d in disp.items()), flush=True)
        del m; torch.cuda.empty_cache()
    print("MECH_DONE")


if __name__ == "__main__":
    main()
