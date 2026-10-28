"""Train + save the GP margin function used in the Dubins OOD tables (margin_gp loss on the failure label, uniform states,
red circles, ood_margin_gp_jepa protocol) on frames in the TRAINING orientation, so CEM can use the same h as the OOD
evals. Saves {"margin_gp": state_dict} to /data/seongbin/dreamer/lewm/<wm>/margin_gp_fail_red_s<seed>.pt (loads in
safe_solver._MarginHead). Usage: train_gp_margin_dubins.py <wm_dir> [seed]"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
import ood_margin_gp_jepa as E
import stable_worldmodel as swm

ck, seed = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
tf = E.get_img_preprocessor(source="pixels", target="pixels", img_size=E.IMG)
# numpy HWC -> tensor CHW before ToImage, as the training loader does (ToImage transposes numpy input)
tf_fixed = lambda d: tf({"pixels": torch.from_numpy(np.ascontiguousarray(d["pixels"])).permute(0, 3, 1, 2)})
og_cond = E.AXES["color"][0]
rng = np.random.default_rng(seed); x0, x1, y0, y1 = E.WORLD_BOUNDS
s = np.stack([rng.uniform(x0, x1, E.N), rng.uniform(y0, y1, E.N), rng.uniform(-np.pi, np.pi, E.N)], 1).astype(np.float32)
y = E.inside_obstacle(s[:, :2], og_cond["shape"])
m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(E.DEV).eval()
Z = E.encode(m, E.render_cond(E.DubinsSwmEnv(), s, og_cond, tf_fixed))
head = E.train_margin(Z[:E.NTR], y[:E.NTR], "gp", seed=seed)
_, auc = E.score(head, Z[E.NTR:], y[E.NTR:])
out = f"/data/seongbin/dreamer/lewm/{ck}/margin_gp_fail_red_s{seed}.pt"
os.makedirs(os.path.dirname(out), exist_ok=True)
torch.save({"margin_gp": head.state_dict()}, out)
print(f"saved {out}  held-out AUC={auc:.4f}", flush=True)
