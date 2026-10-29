"""Control margin functions with no safety knowledge, for the soft-cost CEM control: (a) a randomly initialised MarginHead,
(b) a GP margin function trained exactly like train_gp_margin_dubins.py but on SHUFFLED failure labels.
Saved next to the real one: margin_ctrl_random.pt, margin_ctrl_shuffled.pt. Usage: make_control_heads.py <wm_dir>"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
import ood_margin_gp_jepa as E
import stable_worldmodel as swm
from safe_solver import _MarginHead

ck = sys.argv[1]; out = f"/data/seongbin/dreamer/lewm/{ck}"
torch.manual_seed(0)
torch.save({"margin_gp": _MarginHead().state_dict()}, f"{out}/margin_ctrl_random.pt")
tf = E.get_img_preprocessor(source="pixels", target="pixels", img_size=E.IMG)
tf_fixed = lambda d: tf({"pixels": torch.from_numpy(np.ascontiguousarray(d["pixels"])).permute(0, 3, 1, 2)})
cond = E.AXES["color"][0]
rng = np.random.default_rng(0); x0, x1, y0, y1 = E.WORLD_BOUNDS
s = np.stack([rng.uniform(x0, x1, E.N), rng.uniform(y0, y1, E.N), rng.uniform(-np.pi, np.pi, E.N)], 1).astype(np.float32)
y = E.inside_obstacle(s[:, :2], cond["shape"])
m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(E.DEV).eval()
Z = E.encode(m, E.render_cond(E.DubinsSwmEnv(), s, cond, tf_fixed))
y_shuf = rng.permutation(y[:E.NTR])
head = E.train_margin(Z[:E.NTR], y_shuf, "gp", seed=0)
_, auc = E.score(head, Z[E.NTR:], y[E.NTR:])
torch.save({"margin_gp": head.state_dict()}, f"{out}/margin_ctrl_shuffled.pt")
print(f"saved controls for {ck}; shuffled-label head true-label AUC={auc:.3f} (should be ~0.5)", flush=True)
