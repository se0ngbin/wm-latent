"""LINEAR margin function for Dubins: identical recipe to train_gp_margin_dubins.py (margin_gp loss on the failure label,
8000 uniform states, red circles, training orientation, same steps / lr / batch), only the head is h(z) = w.z + b
instead of the 192-512-512-1 MLP. Saves {"margin_lin": state_dict} to <MDIR>/<wm>/margin_lin_fail_red_s<seed>.pt
(loads in safe_solver via _load_margin_head). Usage: train_lin_margin_dubins.py <wm_dir> [seed]"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from safe_solver import _LinMarginHead   # before E: ood_margin_gp_jepa puts the MAIN checkout's le-wm first on sys.path
import ood_margin_gp_jepa as E
import stable_worldmodel as swm

ck, seed = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
tf = E.get_img_preprocessor(source="pixels", target="pixels", img_size=E.IMG)
tf_fixed = lambda d: tf({"pixels": torch.from_numpy(np.ascontiguousarray(d["pixels"])).permute(0, 3, 1, 2)})
og_cond = E.AXES["color"][0]
rng = np.random.default_rng(seed); x0, x1, y0, y1 = E.WORLD_BOUNDS
s = np.stack([rng.uniform(x0, x1, E.N), rng.uniform(y0, y1, E.N), rng.uniform(-np.pi, np.pi, E.N)], 1).astype(np.float32)
y = E.inside_obstacle(s[:, :2], og_cond["shape"])
m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(E.DEV).eval()
Z = E.encode(m, E.render_cond(E.DubinsSwmEnv(), s, og_cond, tf_fixed))
torch.manual_seed(seed)
head = _LinMarginHead().to(E.DEV)
opt = torch.optim.AdamW(head.parameters(), lr=E.LR)
safe, unsafe = Z[:E.NTR][y[:E.NTR] == 0].to(E.DEV), Z[:E.NTR][y[:E.NTR] == 1].to(E.DEV)
for _ in range(E.STEPS):   # same loop as E.train_margin(kind="gp")
    bs = min(256, safe.shape[0], unsafe.shape[0])
    loss = E.margin_gp_loss(head, safe[torch.randint(0, safe.shape[0], (bs,), device=E.DEV)],
                            unsafe[torch.randint(0, unsafe.shape[0], (bs,), device=E.DEV)])
    opt.zero_grad(); loss.backward(); opt.step()
head.eval()
acc, auc = E.score(head, Z[E.NTR:], y[E.NTR:])
out = f"/data/seongbin/dreamer/lewm/{ck}/margin_lin_fail_red_s{seed}.pt"
os.makedirs(os.path.dirname(out), exist_ok=True)
torch.save({"margin_lin": head.state_dict()}, out)
print(f"saved {out}  held-out AUC={auc:.4f} sign-acc={acc:.4f}", flush=True)
