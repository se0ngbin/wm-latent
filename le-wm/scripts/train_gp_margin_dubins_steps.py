"""train_gp_margin_dubins.py protocol (red, uniform states, per-image fixed-orientation preprocessing, gp loss), with
STEPS overridden; renders once for all models. Saves margin_gp_fail_red_s{seed}_st{STEPS}.pt next to the CEM file
(never overwrites margin_gp_fail_red_s{seed}.pt). Usage: train_gp_st.py <steps> <seed> <wm_dir> [...]"""
import sys
sys.path.insert(0, "/home/seongbin/latent/le-wm"); sys.path.insert(0, "/home/seongbin/latent/le-wm/scripts")
import numpy as np, torch
import ood_margin_gp_jepa as E
import stable_worldmodel as swm

E.STEPS, seed, cks = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
tf = E.get_img_preprocessor(source="pixels", target="pixels", img_size=E.IMG)
tf_fixed = lambda d: tf({"pixels": torch.from_numpy(np.ascontiguousarray(d["pixels"])).permute(0, 3, 1, 2)})
og_cond = E.AXES["color"][0]
rng = np.random.default_rng(seed); x0, x1, y0, y1 = E.WORLD_BOUNDS
s = np.stack([rng.uniform(x0, x1, E.N), rng.uniform(y0, y1, E.N), rng.uniform(-np.pi, np.pi, E.N)], 1).astype(np.float32)
y = E.inside_obstacle(s[:, :2], og_cond["shape"])
F = E.render_cond(E.DubinsSwmEnv(), s, og_cond, tf_fixed)
for ck in cks:
    m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(E.DEV).eval()
    Z = E.encode(m, F); del m; torch.cuda.empty_cache()
    head = E.train_margin(Z[:E.NTR], y[:E.NTR], "gp", seed=seed)
    _, auc = E.score(head, Z[E.NTR:], y[E.NTR:])
    out = f"/data/seongbin/dreamer/lewm/{ck}/margin_gp_fail_red_s{seed}_st{E.STEPS}.pt"
    torch.save({"margin_gp": head.state_dict(), "steps": E.STEPS}, out)
    print(f"saved {out}  held-out AUC={auc:.4f}", flush=True)
