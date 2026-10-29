"""Dubins margin OOD with FIXED frame orientation (tensor CHW into ToImage, as the training loader does). ood_margin_gp_jepa
protocol (gp margin function, failure labels, uniform states). One axis + seed per process; renders once, then all 6 models.
Prints KEY lines: AUC and balanced sign accuracy (TPR_safe + TPR_unsafe)/2 + balanced predicted-safe fraction."""
import sys, os
sys.path.insert(0, "/home/seongbin/latent/le-wm"); sys.path.insert(0, "/home/seongbin/latent/le-wm/scripts")
import numpy as np, torch
import ood_margin_gp_jepa as E
import stable_worldmodel as swm

axis, seed = sys.argv[1], int(sys.argv[2])
ENC = {"sigreg": "sigreg_only_dubins", "jac": "jacobian_w1_dubins", "jac+pull": "lewm_dubins_jacpull50",
       "adv_h": "lewm_dubins_safeadv_linf50", "jac+pull+adv_h": "lewm_dubins_jacpull_safeadv_linf50",
       "adv_λ": "lewm_dubins_lam_bdet50"}
if os.environ.get("DUB_ENC"):   # DUB_ENC="name=ckpt[,name=ckpt]" evaluates only those models
    ENC = dict(kv.split("=", 1) for kv in os.environ["DUB_ENC"].split(","))
tf = E.get_img_preprocessor(source="pixels", target="pixels", img_size=E.IMG)
tf_fixed = lambda d: tf({"pixels": torch.from_numpy(np.ascontiguousarray(d["pixels"])).permute(0, 3, 1, 2)})

og_cond, new_cond, _, _ = E.AXES[axis]
rng = np.random.default_rng(seed); x0, x1, y0, y1 = E.WORLD_BOUNDS
s = np.stack([rng.uniform(x0, x1, E.N), rng.uniform(y0, y1, E.N), rng.uniform(-np.pi, np.pi, E.N)], 1).astype(np.float32)
y_og, y_new = E.inside_obstacle(s[:, :2], og_cond["shape"]), E.inside_obstacle(s[:, :2], new_cond["shape"])
env = E.DubinsSwmEnv()
F_og, F_new = E.render_cond(env, s, og_cond, tf_fixed), E.render_cond(env, s, new_cond, tf_fixed)


def metrics(head, Z, y):  # y: 1 = unsafe
    with torch.no_grad():
        h = head(Z.to(E.DEV)).cpu().numpy().ravel()
    safe = y == 0; pred_safe = h >= 0
    auc = float(E.roc_auc_score(safe.astype(int), h))
    tpr_s, tpr_u = pred_safe[safe].mean(), (~pred_safe[~safe]).mean()
    return auc, float((tpr_s + tpr_u) / 2), float((pred_safe[safe].mean() + pred_safe[~safe].mean()) / 2)


for name, ck in ENC.items():
    m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(E.DEV).eval()
    Z_og, Z_new = E.encode(m, F_og), E.encode(m, F_new)
    del m; torch.cuda.empty_cache()
    h = E.train_margin(Z_og[:E.NTR], y_og[:E.NTR], "gp", seed=seed)
    a_in, b_in, p_in = metrics(h, Z_og[E.NTR:], y_og[E.NTR:])
    a_zs, b_zs, p_zs = metrics(h, Z_new[E.NTR:], y_new[E.NTR:])
    print(f"KEY {name} axis={axis} seed={seed} in_auc={a_in:.4f} in_bacc={b_in:.4f} in_psafe={p_in:.4f} "
          f"zs_auc={a_zs:.4f} zs_bacc={b_zs:.4f} zs_psafe={p_zs:.4f}", flush=True)
