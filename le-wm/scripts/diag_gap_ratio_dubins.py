"""Does a safety reg keep the safe/unsafe SEPARATION under a large appearance shift? Real latents only (no prediction):
4000 uniform states rendered with red and purple circles (training orientation, seed 1). Safety direction u = mean(z safe)
- mean(z unsafe) fit on RED; per color, separation d' = (mean proj safe - mean proj unsafe) / pooled within-class std of
the projection on u (scale-free). Prints d' red, d' purple, their ratio, the raw gap ratio, and AUC of the projection.
Usage: diag_gap_ratio_dubins.py tag=wm_dir [...]"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from sklearn.metrics import roc_auc_score
import ood_margin_gp_jepa as E
import stable_worldmodel as swm
if os.environ.get("CPU"): E.DEV = "cpu"   # GPUs full: the encoder pass is cheap enough on CPU

tf = E.get_img_preprocessor(source="pixels", target="pixels", img_size=E.IMG)
tf_fixed = lambda d: tf({"pixels": torch.from_numpy(np.ascontiguousarray(d["pixels"])).permute(0, 3, 1, 2)})
rng = np.random.default_rng(1); x0, x1, y0, y1 = E.WORLD_BOUNDS; n = 4000
s = np.stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n), rng.uniform(-np.pi, np.pi, n)], 1).astype(np.float32)
safe = E.inside_obstacle(s[:, :2], "circle") == 0
frames = {c["obstacle"]: E.render_cond(E.DubinsSwmEnv(), s, c, tf_fixed) for c in E.AXES["color"][:2]}
print("GAPRATIO model  d'_red  d'_purple  d'_ratio  gap_ratio  auc_red  auc_purple", flush=True)
for spec in sys.argv[1:]:
    tag, ck = spec.split("=", 1)
    m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(E.DEV).eval()
    Z = {c: E.encode(m, f).numpy() for c, f in frames.items()}
    u = Z["red"][safe].mean(0) - Z["red"][~safe].mean(0); u /= np.linalg.norm(u)
    r = {}
    for c, z in Z.items():
        p = z @ u
        gap = p[safe].mean() - p[~safe].mean()
        sd = np.sqrt(((p[safe] - p[safe].mean()) ** 2).sum() + ((p[~safe] - p[~safe].mean()) ** 2).sum()) / np.sqrt(n - 2)
        r[c] = (gap, gap / sd, roc_auc_score(safe, p))
    print(f"GAPRATIO {tag:28s} {r['red'][1]:6.2f}  {r['purple'][1]:8.2f}  {r['purple'][1] / r['red'][1]:7.3f}  "
          f"{r['purple'][0] / r['red'][0]:8.3f}  {r['red'][2]:6.3f}  {r['purple'][2]:9.3f}", flush=True)
    del m; torch.cuda.empty_cache()
