"""Margin OOD eval on the safety-gym CarGoal port (transfer of the dubins finding). Reads
the paired blue/purple eval npz, encodes both with each reg'd WM, trains a probe margin
(WGAN-GP) on BLUE features (failure labels), scores in-dist (blue), zero-shot (purple),
retrain (purple). Also cos_la. Mirrors ood_margin_gp_jepa but on pre-rendered pixels."""
import os, sys
sys.path.insert(0, "/home/seongbin/latent/le-wm")
os.environ.setdefault("STABLEWM_HOME", "/data/seongbin/lewm")
os.environ.setdefault("LOCAL_DATASET_DIR", "/data/seongbin/lewm")
import numpy as np, torch
import stable_worldmodel as swm
from utils import get_img_preprocessor
from sklearn.metrics import roc_auc_score
from ood_margin_gp_jepa import train_margin, score

DEV, IMG, NTR = "cuda:0", 224, 6000
SEED = int(os.environ.get("SG_SEED", "0"))
EVAL = "/data/seongbin/lewm/datasets/sg_ood_eval.npz"
ENCODERS = {"baseline": "sg_baseline", "jacobian": "sg_jacobian", "jacobian+pull": "sg_jacpull"}

def raw_encode(m, x):
    return m.projector(m.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0])

def encode_imgs(m, imgs, tf, bs=128):
    out = []
    for i in range(0, len(imgs), bs):
        batch = torch.stack([tf({"pixels": imgs[j]})["pixels"] for j in range(i, min(i+bs, len(imgs)))]).to(DEV)
        with torch.no_grad():
            out.append(raw_encode(m, batch).cpu())
    return torch.cat(out)   # torch tensor (train_margin/score expect .to(DEV))

def cosla(mu, la): return float(abs((mu@la)/(np.linalg.norm(mu)*np.linalg.norm(la)+1e-12)))

def main():
    d = np.load(EVAL); blue, ood, y = d["blue"], d["ood"], d["failure"].astype(int)
    tf = get_img_preprocessor(source="pixels", target="pixels", img_size=IMG)
    print(f"eval n={len(y)} unsafe_frac={y.mean():.3f} ood_color={d['ood_color']}")
    import os
    AXES = [("color", blue, ood, y)]
    cr = "/data/seongbin/lewm/datasets/sg_camrot_eval.npz"   # REAL 90deg camera rotation (not np.rot90)
    if os.path.exists(cr):
        dr = np.load(cr); AXES.append(("rotate", dr["base"], dr["camrot"], dr["failure"].astype(int)))
    sc, sb = "/data/seongbin/lewm/datasets/sg_shape_circle.npz", "/data/seongbin/lewm/datasets/sg_shape_box.npz"
    if os.path.exists(sc) and os.path.exists(sb):
        dc, db = np.load(sc), np.load(sb)
        AXES.append(("diamond", dc["img"], db["img"], dc["failure"].astype(int)))
    for ename, ck in ENCODERS.items():
        m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(DEV).eval()
        for axis, bi, oi, ay in AXES:
            Zb = encode_imgs(m, bi, tf); Zo = encode_imgs(m, oi, tf)
            Zbn, Zon = Zb.numpy(), Zo.numpy()
            mu = (Zon - Zbn).mean(0); la = Zbn[ay==1].mean(0) - Zbn[ay==0].mean(0)
            for kind in ("gp", "nogp"):
                hb = train_margin(Zb[:NTR], ay[:NTR], kind, seed=SEED); ho = train_margin(Zo[:NTR], ay[:NTR], kind, seed=SEED)
                _, ou = score(hb, Zb[NTR:], ay[NTR:]); _, zu = score(hb, Zo[NTR:], ay[NTR:]); _, nu = score(ho, Zo[NTR:], ay[NTR:])
                print(f"KEY sg_{kind} {ename} axis={axis} ogog_auc={ou:.3f} zs_auc={zu:.3f} or_auc={nu:.3f} cos_la={cosla(mu,la):.3f}", flush=True)
        del m; torch.cuda.empty_cache()
    print("SG_MARGIN_DONE")

if __name__ == "__main__":
    main()
