"""Real-sample color robustness of saved Dubins margin heads: AUC / balanced sign acc on uniform states rendered with red
(in-dist) and purple (zero-shot) circles, training orientation (same protocol as train_gp_margin_dubins.py, seed 1 states
so they are disjoint from the heads' training draws). Usage: eval_head_color_dubins.py <wm_dir> <head.pt> [...]"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from sklearn.metrics import roc_auc_score
import ood_margin_gp_jepa as E
import stable_worldmodel as swm
from safe_solver import _MarginHead

ck, heads = sys.argv[1], sys.argv[2:]
tf = E.get_img_preprocessor(source="pixels", target="pixels", img_size=E.IMG)
tf_fixed = lambda d: tf({"pixels": torch.from_numpy(np.ascontiguousarray(d["pixels"])).permute(0, 3, 1, 2)})
rng = np.random.default_rng(1); x0, x1, y0, y1 = E.WORLD_BOUNDS; n = 4000
s = np.stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n), rng.uniform(-np.pi, np.pi, n)], 1).astype(np.float32)
safe = E.inside_obstacle(s[:, :2], "circle") == 0
m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(E.DEV).eval()
Z = {c["obstacle"]: E.encode(m, E.render_cond(E.DubinsSwmEnv(), s, c, tf_fixed)) for c in E.AXES["color"][:2]}
for p in heads:
    hd = _MarginHead().to(E.DEV); hd.load_state_dict(torch.load(p, map_location=E.DEV)["margin_gp"]); hd.eval()
    r = []
    for c, z in Z.items():
        with torch.no_grad():
            h = hd(z.to(E.DEV)).cpu().numpy()
        r.append(f"{c} AUC {roc_auc_score(safe, h):.3f} bal-acc {((h[safe] >= 0).mean() + (h[~safe] < 0).mean()) / 2:.3f} "
                 f"h median safe/unsafe {np.median(h[safe]):.2f}/{np.median(h[~safe]):.2f}")
    print(f"HEADOOD {ck} {os.path.basename(p)}: " + "  ".join(r), flush=True)
