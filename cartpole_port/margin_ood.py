"""Zero-shot margin_gp OOD on cartpole, in the Dubins/SG report format: head trained on base (tan, native view) frames of
the train episodes, scored on the test episodes in every condition. AUC and sign accuracy are both against the margin's
own label (failed = l < 0). Writes /data/seongbin/lewm/cartpole_eval/<model>_margin.json.
Usage: margin_ood.py MODEL [MODEL...]"""
import json, sys
import numpy as np, torch, h5py
from eval_cartpole import load_data, cond_pixels, encode_pixels, apply, auc, CONDS, H5, OUT, DEV, SEEDS
from ood_margin_gp_jepa import train_margin
from utils import get_img_preprocessor
import stable_worldmodel as swm

COND_NAMES = ["purple", "zoom_out", "pan_x", "pan_y"]
d = load_data(); tr, te = ~d["test"], d["test"]
safe = d["margin"][te] >= 0
tf = get_img_preprocessor(source="pixels", target="pixels", img_size=224)
cp = {c: cond_pixels(d, c) for c in COND_NAMES}
with h5py.File(H5) as f:
    pix = f["pixels"]
    for name in sys.argv[1:]:
        m = swm.wm.utils.load_pretrained(f"{name}/weights_epoch_50.pt").to(DEV).eval()
        Z = encode_pixels(m, pix, tf); Zc = {c: encode_pixels(m, p, tf) for c, p in cp.items()}
        del m; torch.cuda.empty_cache()
        res = {}
        for seed in SEEDS:
            h = train_margin(Z[tr], d["failures"][tr], "gp", seed=seed)
            for c, X in (("base", Z[te]), *Zc.items()):
                s = apply(h, X).flatten().numpy()
                res.setdefault(c, []).append(dict(auc=auc(s, safe), sign_acc=float(((s >= 0) == safe).mean())))
        summ = {c: {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))) for k in rs[0]} for c, rs in res.items()}
        json.dump(dict(model=name, safe_frac=float(safe.mean()), runs=res, summary=summ), open(f"{OUT}/{name}_margin.json", "w"), indent=1)
        print(f"KEY {name} " + " ".join(f"{c}:auc={v['auc'][0]:.3f},sign={v['sign_acc'][0]:.3f}" for c, v in summ.items()), flush=True)
