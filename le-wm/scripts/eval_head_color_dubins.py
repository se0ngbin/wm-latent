"""Real-sample OOD robustness of saved Dubins margin heads (GP MLP or linear, auto-detected): AUC / balanced sign acc on
uniform states rendered in-dist (red circle, rot 0) and on each zero-shot axis (purple / diamond / rot90), same protocol
as train_gp_margin_dubins.py, seed 1 states so they are disjoint from the heads' training draws. Labels follow each
condition's shape. Usage: eval_head_color_dubins.py <wm_dir> <head.pt> [...]"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from safe_solver import _load_margin_head   # before E: ood_margin_gp_jepa puts the MAIN checkout's le-wm first on sys.path
import ood_margin_gp_jepa as E
import stable_worldmodel as swm

ck, heads = sys.argv[1], sys.argv[2:]
tf = E.get_img_preprocessor(source="pixels", target="pixels", img_size=E.IMG)
tf_fixed = lambda d: tf({"pixels": torch.from_numpy(np.ascontiguousarray(d["pixels"])).permute(0, 3, 1, 2)})
rng = np.random.default_rng(1); x0, x1, y0, y1 = E.WORLD_BOUNDS; n = 4000
s = np.stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n), rng.uniform(-np.pi, np.pi, n)], 1).astype(np.float32)
conds = {"red": E.AXES["color"][0], **{a[3]: a[1] for a in E.AXES.values()}}   # red, purple, diamond, rot90
m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(E.DEV).eval()
Z = {k: (E.encode(m, E.render_cond(E.DubinsSwmEnv(), s, c, tf_fixed)), E.inside_obstacle(s[:, :2], c["shape"]) == 0)
     for k, c in conds.items()}
for p in heads:
    hd = _load_margin_head(p, E.DEV).eval()
    r = []
    for c, (z, safe) in Z.items():
        with torch.no_grad():
            h = hd(z.to(E.DEV)).cpu().numpy()
        r.append(f"{c} AUC {roc_auc_score(safe, h):.3f} bal-acc {((h[safe] >= 0).mean() + (h[~safe] < 0).mean()) / 2:.3f} "
                 f"h median safe/unsafe {np.median(h[safe]):.2f}/{np.median(h[~safe]):.2f}")
    print(f"HEADOOD {ck} {os.path.basename(p)}: " + "  ".join(r), flush=True)
