"""Mechanism probe on the safety-gym LEWM encoders: is jacobian's OOD robustness the SAME
conditioning/nuisance-rejection mechanism as dubins? Encoder-Jacobian eff_rank/cond + the
raw-shift/sigma_red decomposition for color and rotate. Prediction: jac eff_rank >> baseline,
and the OOD div (shift / state-spread) is smaller for the robust encoders."""
import os, sys
sys.path.insert(0, "/home/seongbin/latent/le-wm")
os.environ.setdefault("STABLEWM_HOME", "/data/seongbin/lewm"); os.environ.setdefault("LOCAL_DATASET_DIR", "/data/seongbin/lewm")
import numpy as np, torch
import stable_worldmodel as swm
from utils import get_img_preprocessor
DEV, IMG, NF = "cuda:0", 224, 6
ENC = {"baseline": "sg_baseline", "jacobian": "sg_jacobian", "jac+pull": "sg_jacpull"}
if os.environ.get("ENCODERS"):   # "name:ckpt,name:ckpt" to override (e.g. AC-MTM encoders)
    ENC = dict(kv.split(":") for kv in os.environ["ENCODERS"].split(","))

def raw_encode(m, x): return m.projector(m.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0])

def jac_svd(m, x1):
    from torch.nn.attention import sdpa_kernel, SDPBackend
    x1 = x1.to(DEV).requires_grad_(True)
    with sdpa_kernel(SDPBackend.MATH):
        z = raw_encode(m, x1).squeeze(0)
        J = torch.stack([torch.autograd.grad(z[k], x1, retain_graph=True)[0].flatten().detach() for k in range(z.numel())])
    return torch.linalg.svdvals(J.float()).cpu().numpy()

def enc(m, imgs, tf, bs=128):
    out = []
    for i in range(0, len(imgs), bs):
        b = torch.stack([tf({"pixels": imgs[j]})["pixels"] for j in range(i, min(i+bs, len(imgs)))]).to(DEV)
        with torch.no_grad(): out.append(raw_encode(m, b).cpu().numpy())
    return np.concatenate(out)

def main():
    tf = get_img_preprocessor(source="pixels", target="pixels", img_size=IMG)
    d = np.load("/data/seongbin/lewm/datasets/sg_ood_eval.npz"); blue, purple = d["blue"][:2000], d["ood"][:2000]
    cr = np.load("/data/seongbin/lewm/datasets/sg_camrot_eval.npz"); base, rot = cr["base"][:2000], cr["camrot"][:2000]
    for tag, ck in ENC.items():
        m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(DEV).eval()
        S = np.stack([jac_svd(m, tf({"pixels": blue[i]})["pixels"][None]) for i in range(NF)]).mean(0)
        lam = S**2; effrank = float((lam.sum()**2)/(lam**2).sum()); s1sh = float(lam[0]/lam.sum()); fro = float(np.sqrt(lam.sum()))
        Zb = enc(m, blue, tf); sig = float(np.sqrt(2*Zb.var(0).sum()))
        Zp = enc(m, purple, tf); Zbase = enc(m, base, tf); Zrot = enc(m, rot, tf)
        col_raw = float(np.linalg.norm((Zp-Zb).mean(0)))
        rot_raw = float(np.linalg.norm((Zrot-Zbase).mean(0)))
        # displacement per-sample (mean ||shift||)
        col_disp = float(np.linalg.norm(Zp-Zb, axis=1).mean()); rot_disp = float(np.linalg.norm(Zrot-Zbase, axis=1).mean())
        print(f"MECH {tag:9s} effrank={effrank:6.1f} froJ={fro:6.2f} s1share={s1sh:.3f} sigma_red={sig:.3f} | "
              f"color disp/sig={col_disp/sig:.2f} | rotate disp/sig={rot_disp/sig:.2f}", flush=True)
        del m; torch.cuda.empty_cache()
    print("MECH_DONE")

if __name__ == "__main__":
    main()
