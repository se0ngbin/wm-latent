"""Does the imagined rollout wash out an appearance shift ("re-grounding")? Held-out expert windows as in
diag_rollout_safety.py, rendered red (original) and purple (same frames, obstacle recolored with the ObstacleRecolor rule
out = x + a*(C - RED), a = relu(min(R-G, R-B))/255). Per horizon k: GP margin function AUC and balanced sign accuracy on
real purple frames vs on latents imagined from a purple context, and the purple-red latent gap imagined vs real.
Usage: diag_regrounding.py tag=ckpt [...]"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch, h5py
from sklearn.metrics import roc_auc_score
import stable_worldmodel as swm
from utils import get_img_preprocessor
from safe_solver import _MarginHead

DEV, H, K, NW = "cuda", 3, 25, 400
OUT = "/data/seongbin/lewm/safeadv_results/regrounding"; os.makedirs(OUT, exist_ok=True)
tf = get_img_preprocessor(source="pixels", target="pixels", img_size=224)
with h5py.File("/data/seongbin/lewm/datasets/dubins_expert.h5", "r") as f:   # same windows as diag_rollout_safety.py
    off, ln = f["ep_offset"][:], f["ep_len"][:]
    act_all = f["action"][:].astype(np.float32).reshape(len(f["action"]), -1)
    a_mu, a_sd = np.nanmean(act_all, 0), np.nanstd(act_all, 0)
    rng = np.random.default_rng(0)
    eps = np.where(ln >= H + K)[0]; eps = eps[int(0.9 * len(eps)):]
    starts = [off[e] + rng.integers(0, ln[e] - (H + K) + 1) for e in rng.choice(eps, NW, replace=True)]
    idx = np.concatenate([np.arange(s, s + H + K) for s in starts])
    u, inv = np.unique(idx, return_inverse=True)
    px = f["pixels"][u][inv].reshape(NW, H + K, *f["pixels"].shape[1:])
    fail = f["failures"][u][inv].reshape(NW, H + K)
    act = ((act_all[idx] - a_mu) / (a_sd + 1e-8)).reshape(NW, H + K, -1)
safe = (fail[:, H:] == 0).astype(int)
x = px.astype(np.float32)
a = np.clip(np.minimum(x[..., 0] - x[..., 1], x[..., 0] - x[..., 2]), 0, None)[..., None] / 255.0
px_purple = np.clip(x + a * (np.array([128.0, 0.0, 128.0]) - np.array([255.0, 0.0, 0.0])), 0, 255).astype(np.uint8)


def frames(b):
    return tf({"pixels": torch.from_numpy(np.ascontiguousarray(b)).permute(0, 3, 1, 2)})["pixels"].float()


def bacc(y, h):  # balanced sign accuracy, safe = y==1, h>=0 called safe
    p = h >= 0; return float((p[y == 1].mean() + (~p[y == 0]).mean()) / 2)


@torch.no_grad()
def run(tag, ck):
    m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(DEV).eval()
    enc = lambda xx: m.projector(m.encoder(xx.to(DEV), interpolate_pos_encoding=True).last_hidden_state[:, 0])
    def encode(P):
        flat = P.reshape(-1, *P.shape[2:])
        return torch.cat([enc(frames(flat[i:i + 256])) for i in range(0, len(flat), 256)]).view(NW, H + K, -1)
    def rollout(Z):
        A = torch.from_numpy(act).to(DEV); emb, aa = Z[:, :H].clone(), A[:, :H]; out = []
        for k in range(K):
            p = m.predict(emb[:, -H:], m.action_encoder(aa[:, -H:]))[:, -1:]
            out.append(p); emb = torch.cat([emb, p], 1)
            if H + k < H + K - 1: aa = torch.cat([aa, A[:, H + k:H + k + 1]], 1)
        return torch.cat(out, 1)
    Zr, Zp = encode(px), encode(px_purple)
    Ir, Ip = rollout(Zr), rollout(Zp)
    head = _MarginHead().to(DEV)
    head.load_state_dict(torch.load(f"/data/seongbin/dreamer/lewm/{ck}/margin_gp_fail_red_s0.pt", map_location=DEV)["margin_gp"])
    head.eval()
    hv = lambda Z: head(Z.flatten(0, 1).float()).view(NW, K).cpu().numpy()
    r = {"tag": tag}
    for nm, Z in (("real_red", Zr[:, H:]), ("real_purple", Zp[:, H:]), ("imag_red", Ir), ("imag_purple", Ip)):
        h = hv(Z)
        r[f"auc_{nm}"] = [float(roc_auc_score(safe[:, k], h[:, k])) for k in range(K)]
        r[f"bacc_{nm}"] = [bacc(safe[:, k], h[:, k]) for k in range(K)]
        r[f"psafe_{nm}"] = [float((h[:, k] >= 0).mean()) for k in range(K)]
    r["gap_real"] = (Zp[:, H:] - Zr[:, H:]).norm(dim=-1).mean(0).cpu().tolist()       # purple-red distance, real frames
    r["gap_imag"] = (Ip - Ir).norm(dim=-1).mean(0).cpu().tolist()                      # purple-red distance, imagined
    json.dump(r, open(f"{OUT}/{tag}.json", "w"))
    f = lambda key, k: r[key][k - 1]
    for k in (1, 5, 25):
        print(f"REGROUND {tag} k={k:2d} gap imag/real={f('gap_imag', k) / f('gap_real', k):.3f} | AUC red real {f('auc_real_red', k):.3f} "
              f"purple real {f('auc_real_purple', k):.3f} purple imag {f('auc_imag_purple', k):.3f} | bacc purple real "
              f"{f('bacc_real_purple', k):.3f} purple imag {f('bacc_imag_purple', k):.3f} red imag {f('bacc_imag_red', k):.3f}", flush=True)
    del m; torch.cuda.empty_cache()


for spec in sys.argv[1:]:
    t, c = spec.split("=", 1); run(t, c)
