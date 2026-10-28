"""Where does an OOD shift send the latent? Across the margin's boundary inside the training support (aliasing),
or off the support (the margin extrapolates)? Frozen encoders, paired renders: z_og(s) and z_sh(s) are the
same state s under the native and shifted rendering.

Per model and shift, on the test states:
  support   kNN distance (k=10, mean) from z to the in-dist training latents (ref). "off" = above the 99th
            percentile of the native test latents' kNN distance.
  flip      sign(h(z_sh)) != sign(h(z_og)), h = margin_gp trained on native training latents (env's own protocol).
  2x2       fraction of test states in {in, off} x {kept, flipped}.
  alias     among in-support z_sh: majority label of the k nearest ref latents != true label
            (the shifted frame sits among training frames of the other class). Compared with the same rate for z_og.
  nn_err    median true-state distance between s and the state of z's nearest ref latent (Dubins/cartpole only;
            Point layouts differ per episode, so pose alone means nothing there).
Usage: support_alias.py dubins|cartpole|point   -> /data/seongbin/lewm/support_alias/<env>.json + KEY lines
"""
import os, sys, json
import numpy as np, torch
from sklearn.metrics import roc_auc_score

K, Q, OUT = 10, 0.99, "/data/seongbin/lewm/support_alias"
DEV = "cuda:0"
WT = "/home/seongbin/latent/.claude/worktrees"


def knn(Z, ref, k=K, bs=2048):
    """mean distance to and indices of the k nearest ref rows."""
    R = ref.to(DEV); D, I = [], []
    for i in range(0, len(Z), bs):
        d = torch.cdist(Z[i:i + bs].to(DEV), R)
        v, j = d.topk(k, largest=False)
        D.append(v.mean(1).cpu()); I.append(j.cpu())
    return torch.cat(D).numpy(), torch.cat(I).numpy()


def analyze(ref, y_ref, s_ref, Zog, Zsh, y, s, h_og, h_sh):
    """ref/y_ref/s_ref: in-dist training latents, labels (1 = unsafe), states (None to skip nn_err).
    Zog/Zsh: paired test latents; y, s: their labels and states; h_og/h_sh: margin values on Zog/Zsh."""
    d_og, i_og = knn(Zog, ref); d_sh, i_sh = knn(Zsh, ref)
    thr = np.quantile(d_og, Q)
    off, flip = d_sh > thr, (h_sh >= 0) != (h_og >= 0)
    lab = lambda I: (y_ref[I].mean(1) > 0.5).astype(int)
    a_og, a_sh = lab(i_og) != y, lab(i_sh) != y
    u = y == 1
    # kNN score = fraction of safe neighbours; its AUC asks whether the shifted latent sits among same-class training
    # latents at all (a head-free read of the neighbourhood), unlike the raw mismatch rate it ignores the base rate
    ks = lambda I: 1 - y_ref[I].mean(1)
    au = lambda sc: float(roc_auc_score(1 - y, sc))
    r = dict(off=off.mean(), dist_ratio=np.median(d_sh) / np.median(d_og), flip=flip.mean(),
             in_kept=(~off & ~flip).mean(), in_flip=(~off & flip).mean(),
             off_kept=(off & ~flip).mean(), off_flip=(off & flip).mean(),
             alias_og=a_og.mean(), alias_sh=a_sh[~off].mean() if (~off).any() else float("nan"),
             alias_unsafe_og=a_og[u].mean(), alias_unsafe_sh=a_sh[u].mean(),
             knn_auc_og=au(ks(i_og)), knn_auc_sh=au(ks(i_sh)), head_auc_og=au(h_og), head_auc_sh=au(h_sh))
    if s_ref is not None:
        e = lambda I: np.median(np.linalg.norm(s_ref[I[:, 0]] - s, axis=1))
        r.update(nn_err_og=e(i_og), nn_err_sh=e(i_sh))
    return {k: float(v) for k, v in r.items()}


# ---------------- Dubins: uniform states, exact geometric labels (ood_gp_one.py protocol) ----------------
def dubins():
    sys.path.insert(0, "/data/seongbin/lewm/safeadv_results/ablation_eval")
    os.environ.setdefault("ENC", "x=x")
    import ood_gp_one as G
    from dubins_swm_env import DubinsSwmEnv, WORLD_BOUNDS
    from utils import get_img_preprocessor
    models = {"sigreg": "sigreg_only_dubins", "jacobian": "jacobian_w1_dubins", "jac+pull": "lewm_dubins_jacpull50",
              "L∞ safety reg": "lewm_dubins_safeadv_linf50", "jac+pull + L∞": "lewm_dubins_jacpull_safeadv_linf50",
              "λ batch-detached": "lewm_dubins_lam_bdet50"}
    rng = np.random.default_rng(0); x0, x1, y0, y1 = WORLD_BOUNDS
    st = np.stack([rng.uniform(x0, x1, G.N), rng.uniform(y0, y1, G.N), rng.uniform(-np.pi, np.pi, G.N)], 1).astype(np.float32)
    # state metric: position + heading on the unit circle (heading moves the rendered car)
    sm = np.concatenate([st[:, :2], np.cos(st[:, 2:]), np.sin(st[:, 2:])], 1)
    env = DubinsSwmEnv(); tf = get_img_preprocessor(source="pixels", target="pixels", img_size=G.IMG)
    tr, te = slice(0, G.NTR), slice(G.NTR, None)
    res = {}
    for name, ck in models.items():
        m = G.swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(DEV).eval()
        og_cond = G.AXES["color"][0]
        Z = {"og": G.encode(m, G.render_cond(env, st, og_cond, tf))}
        for ax in ("color", "rotate", "shape"):
            Z[ax] = G.encode(m, G.render_cond(env, st, G.AXES[ax][1], tf))
        del m; torch.cuda.empty_cache()
        y_og = G.inside_obstacle(st[:, :2], "circle").astype(int)
        h = G.train_margin(Z["og"][tr], y_og[tr].astype(np.float32), "gp", seed=0)
        hv = lambda X: h(X.to(DEV)).detach().cpu().numpy().ravel()
        with torch.no_grad():
            for ax in ("color", "rotate", "shape"):
                y_sh = G.inside_obstacle(st[:, :2], G.AXES[ax][1]["shape"]).astype(int)
                res[f"{name}|{ax}"] = analyze(Z["og"][tr], y_og[tr], sm[tr], Z["og"][te], Z[ax][te], y_sh[te], sm[te],
                                              hv(Z["og"][te]), hv(Z[ax][te]))
                print(f"KEY dubins {name}|{ax} " + " ".join(f"{k}={v:.3f}" for k, v in res[f'{name}|{ax}'].items()), flush=True)
    return res


# ---------------- Cartpole: dataset test episodes, cached OOD re-renders (eval_cartpole.py protocol) -------------
def cartpole():
    sys.path[:0] = [f"{WT}/cartpole-port/cartpole_port", "/home/seongbin/latent/le-wm/scripts"]
    import h5py, eval_cartpole as C
    from utils import get_img_preprocessor
    models = {"sigreg": "cp_baseline", "jacobian": "cp_jacobian", "jac+pull": "cp_jacpull",
              "L∞ safety reg": "cp_safeadv_linf_g0", "jacobian + L∞": "cp_jac_safeadv_linf",
              "jac+pull + L∞": "cp_jacpull_safeadv_linf50", "λ batch-detached": "cp_lam_bdet50",
              "jacobian + λ": "cp_jac_lam_bdet", "jac+pull + λ": "cp_jacpull_lam_bdet"}
    conds = ["purple", "zoom_out", "pan_x", "pan_y"]
    d = C.load_data(); tr, te = np.where(~d["test"])[0], np.where(d["test"])[0]
    ref_i = np.sort(np.random.default_rng(0).choice(tr, 20000, replace=False))
    sm = d["state"] / np.array([2.4, 9.0, 0.2094, 4.0], np.float32)   # eval_cartpole's state-ceiling scaling
    fail = d["failures"].astype(int)
    cp = {c: C.cond_pixels(d, c) for c in conds}
    tf = get_img_preprocessor(source="pixels", target="pixels", img_size=224)
    res = {}
    with h5py.File(C.H5) as f:
        pix = f["pixels"]
        for name, ck in models.items():
            m = C.swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(DEV).eval()
            Z = C.encode_pixels(m, pix, tf); Zc = {c: C.encode_pixels(m, p, tf) for c, p in cp.items()}
            del m; torch.cuda.empty_cache()
            h = C.train_margin(Z[tr], d["failures"][tr], "gp", seed=0)
            hv = lambda X: C.apply(h, X).flatten().numpy()
            for c in conds:
                r = analyze(Z[ref_i], fail[ref_i], sm[ref_i], Z[te], Zc[c], fail[te], sm[te], hv(Z[te]), hv(Zc[c]))
                res[f"{name}|{c}"] = r
                print(f"KEY cartpole {name}|{c} " + " ".join(f"{k}={v:.3f}" for k, v in r.items()), flush=True)
    return res


# ---------------- Point overhead@224: layout-disjoint (sg_point_bigprobe.py protocol) ----------------
def point():
    sys.path.insert(0, f"{WT}/point-dynamics-probe/le-wm/scripts")
    os.environ["SG_DATA"] = "/data/seongbin/lewm/datasets/sg_point_ovh224_multi_eval.npz"
    import h5py, sg_point_eval as E
    names = {"sigreg": "pt_baseline", "jacobian": "pt_jacobian", "jac+pull": "pt_jacpull",
             "L∞ safety reg": "pt_safeadv_linf_g0", "jacobian + L∞": "pt_jac_safeadv_linf",
             "jac+pull + L∞": "pt_jacpull_safeadv_linf", "λ batch-detached": "pt_lam_bdet",
             "jacobian + λ": "pt_jac_lam_bdet", "jac+pull + λ": "pt_jacpull_lam_bdet"}
    N_TR, N_HO, STEPS = 30000, 5000, 20000
    with h5py.File("/data/seongbin/lewm/datasets/sg_point_ovh224.h5", "r") as f:
        idx = np.sort(np.random.default_rng(0).choice(f["pixels"].shape[0], N_TR + N_HO, replace=False))
        X, yX, ep = f["pixels"][idx], f["failures"][idx].astype(int), f["episode_idx"][idx]
    # hold out whole episodes: within-episode neighbours are near-duplicates and would shrink the in-dist kNN distance
    ho = np.isin(ep, np.random.default_rng(1).choice(np.unique(ep), int(len(np.unique(ep)) * N_HO / len(idx)), replace=False))
    itr, iho = np.where(~ho)[0], np.where(ho)[0]
    d = np.load(E.DATA); y = d["failure"].astype(int)
    tf = E.get_img_preprocessor(source="pixels", target="pixels", img_size=224)
    res = {}
    for name, ck in names.items():
        ck = ck + "_ovh"
        m = E.swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(DEV).eval()
        def enc(imgs, bs=256):
            out = []
            for i in range(0, len(imgs), bs):
                b = torch.stack([tf({"pixels": imgs[j]})["pixels"] for j in range(i, min(i + bs, len(imgs)))]).to(DEV)
                with torch.no_grad(): out.append(m.projector(m.encoder(b, interpolate_pos_encoding=True).last_hidden_state[:, 0]).cpu())
            return torch.cat(out)
        ZX = enc(X); Zv = {ax: enc(d[ax]) for ax in ["base"] + E.AXES}
        del m; torch.cuda.empty_cache()
        Ztr, ytr = ZX[itr].to(DEV), yX[itr]
        torch.manual_seed(0); rng = np.random.default_rng(0)
        s_idx, u_idx = np.where(ytr == 0)[0], np.where(ytr == 1)[0]
        head = E.MarginHead(Ztr.shape[1]).to(DEV); opt = torch.optim.AdamW(head.parameters(), lr=E.LR)
        for _ in range(STEPS):
            loss = E.gp_loss(head, Ztr[rng.choice(s_idx, E.BS)], Ztr[rng.choice(u_idx, E.BS)])
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        hv = lambda Z: head(Z.to(DEV)).detach().cpu().numpy().ravel()
        # base: native view, unseen layouts, vs held-out training frames (seen-layout distribution) as "og"
        Zho, yho = ZX[iho], yX[iho]
        d_ho, _ = knn(Zho, ZX[itr]); d_b, i_b = knn(Zv["base"], ZX[itr]); thr = np.quantile(d_ho, Q)
        lab = (yX[itr][i_b].mean(1) > 0.5).astype(int); off_b = d_b > thr
        res[f"{name}|layout"] = dict(off=float(off_b.mean()), dist_ratio=float(np.median(d_b) / np.median(d_ho)),
                                     wrong=float(((hv(Zv["base"]) >= 0) == (y == 1)).mean()),
                                     alias_sh=float((lab != y)[~off_b].mean()) if (~off_b).any() else float("nan"))
        print(f"KEY point {name}|layout " + " ".join(f"{k}={v:.3f}" for k, v in res[f'{name}|layout'].items()), flush=True)
        for ax in E.AXES:   # paired with base (same pose, same unseen layout)
            r = analyze(ZX[itr], ytr, None, Zv["base"], Zv[ax], y, None, hv(Zv["base"]), hv(Zv[ax]))
            res[f"{name}|{ax}"] = r
            print(f"KEY point {name}|{ax} " + " ".join(f"{k}={v:.3f}" for k, v in r.items()), flush=True)
    return res


if __name__ == "__main__":
    env = sys.argv[1]
    sys.path.insert(0, "/home/seongbin/latent/le-wm")
    os.environ.setdefault("STABLEWM_HOME", "/data/seongbin/lewm"); os.environ.setdefault("LOCAL_DATASET_DIR", "/data/seongbin/lewm")
    res = dict(dubins=dubins, cartpole=cartpole, point=point)[env]()
    os.makedirs(OUT, exist_ok=True); json.dump(res, open(f"{OUT}/{env}.json", "w"), indent=1)
    print("done", flush=True)
