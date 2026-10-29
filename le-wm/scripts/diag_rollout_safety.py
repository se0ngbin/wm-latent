"""Why do the safety regs plan worse? Per Dubins WM, on held-out expert windows (frames in the TRAINING orientation):
  1. rollout fidelity: open-loop K-step prediction error ||z_hat - z||^2, normalized by the WM's latent variance;
  2. h on imagined latents: AUC of the GP margin function / HJ value function on z_hat_{t+k} vs on real z_{t+k}
     (failure label at t+k), i.e. whether the soft CEM cost reads imagined futures correctly;
  3. goal-cost fidelity: Spearman(latent ||z - z_goal||^2, true ||xy - xy_goal||) on real and imagined latents.
Rollout mirrors JEPA.rollout (history 3, action at frame i drives i -> i+1). Writes one JSON per model.
Usage: diag_rollout_safety.py tag=ckpt [tag=ckpt ...]"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch, h5py
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import stable_worldmodel as swm
from utils import get_img_preprocessor
from safe_solver import _MarginHead

DEV, H, K, NW = "cuda", 3, 25, 400
OUT = "/data/seongbin/lewm/safeadv_results/rollout_diag"; os.makedirs(OUT, exist_ok=True)
MDIR = "/data/seongbin/dreamer/lewm"
tf = get_img_preprocessor(source="pixels", target="pixels", img_size=224)

# ---- held-out windows (same for every model) ----
with h5py.File("/data/seongbin/lewm/datasets/dubins_expert.h5", "r") as f:
    off, ln = f["ep_offset"][:], f["ep_len"][:]
    act_all = f["action"][:].astype(np.float32).reshape(len(f["action"]), -1)
    a_mu, a_sd = np.nanmean(act_all, 0), np.nanstd(act_all, 0)
    rng = np.random.default_rng(0)
    eps = np.where(ln >= H + K)[0]
    eps = eps[int(0.9 * len(eps)):]                      # last 10% of episodes
    starts = []
    for e in rng.choice(eps, NW, replace=True):
        starts.append(off[e] + rng.integers(0, ln[e] - (H + K) + 1))
    idx = np.concatenate([np.arange(s, s + H + K) for s in starts])
    u, inv = np.unique(idx, return_inverse=True)          # h5py needs increasing, unique indices
    px = f["pixels"][u][inv].reshape(NW, H + K, *f["pixels"].shape[1:])
    st = f["state"][u][inv].reshape(NW, H + K, -1)
    fail = f["failures"][u][inv].reshape(NW, H + K)
    act = ((act_all[idx] - a_mu) / (a_sd + 1e-8)).reshape(NW, H + K, -1)
safe = (fail[:, H:] == 0).astype(int)                     # targets for k = 1..K (frames H..H+K-1)


def frames(b):  # uint8 (n,h,w,3) -> normalized (n,3,224,224), tensor path as in training
    return tf({"pixels": torch.from_numpy(np.ascontiguousarray(b)).permute(0, 3, 1, 2)})["pixels"].float()


@torch.no_grad()
def run(tag, ck):
    m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(DEV).eval()
    enc = lambda x: m.projector(m.encoder(x.to(DEV), interpolate_pos_encoding=True).last_hidden_state[:, 0])
    flat = px.reshape(-1, *px.shape[2:])
    Z = torch.cat([enc(frames(flat[i:i + 256])) for i in range(0, len(flat), 256)]).view(NW, H + K, -1)
    A = torch.from_numpy(act).to(DEV)
    emb, a = Z[:, :H].clone(), A[:, :H]
    preds = []
    for k in range(K):                                   # predict frame H+k from the last H latents + actions
        p = m.predict(emb[:, -H:], m.action_encoder(a[:, -H:]))[:, -1:]
        preds.append(p); emb = torch.cat([emb, p], 1)
        a = torch.cat([a, A[:, H + k:H + k + 1]], 1) if H + k < H + K - 1 else a
    Zh, Zr = torch.cat(preds, 1), Z[:, H:]               # (NW, K, D) imagined / real
    var = (Z.flatten(0, 1) - Z.flatten(0, 1).mean(0)).pow(2).sum(-1).mean()
    nerr = ((Zh - Zr).pow(2).sum(-1).mean(0) / var).cpu().numpy()
    r = {"tag": tag, "ckpt": ck, "nerr": nerr.tolist()}
    # goal-cost fidelity: goal = real last frame of the window
    zg, xy, xyg = Z[:, -1:], st[:, H:, :2], st[:, -1:, :2]
    dtrue = np.linalg.norm(xy - xyg, axis=-1)[:, :-1].ravel()
    for nm, ZZ in (("real", Zr), ("imag", Zh)):
        dl = (ZZ - zg).pow(2).sum(-1)[:, :-1].cpu().numpy().ravel()
        r[f"goal_spearman_{nm}"] = float(spearmanr(dl, dtrue).correlation)
        near = dtrue < 0.5
        r[f"goal_spearman_near_{nm}"] = float(spearmanr(dl[near], dtrue[near]).correlation)
    # h on imagined vs real latents
    for hn, fn in (("gp", "margin_gp_fail_red_s0.pt"), ("hj", "margin_hj_color_red_rk0_s0.pt")):
        p = f"{MDIR}/{ck}/{fn}"
        if not os.path.exists(p): continue
        head = _MarginHead().to(DEV); head.load_state_dict(torch.load(p, map_location=DEV)["margin_gp"]); head.eval()
        for nm, ZZ in (("real", Zr), ("imag", Zh)):
            hv = head(ZZ.flatten(0, 1).float()).view(NW, K).cpu().numpy()
            r[f"{hn}_auc_{nm}"] = [float(roc_auc_score(safe[:, k], hv[:, k])) if 0 < safe[:, k].mean() < 1 else None
                                   for k in range(K)]
            r[f"{hn}_auc_{nm}_all"] = float(roc_auc_score(safe.ravel(), hv.ravel()))
    json.dump(r, open(f"{OUT}/{tag}.json", "w"))
    print(f"DIAG {tag} nerr@1={nerr[0]:.4f} @5={nerr[4]:.4f} @25={nerr[-1]:.4f} "
          f"goal_rho real={r['goal_spearman_real']:.3f} imag={r['goal_spearman_imag']:.3f} "
          + " ".join(f"{hn}_auc real={r[f'{hn}_auc_real_all']:.3f} imag={r[f'{hn}_auc_imag_all']:.3f}"
                     for hn in ("gp", "hj") if f"{hn}_auc_real_all" in r), flush=True)
    del m; torch.cuda.empty_cache()


for spec in sys.argv[1:]:
    t, c = spec.split("=", 1)
    run(t, c)
