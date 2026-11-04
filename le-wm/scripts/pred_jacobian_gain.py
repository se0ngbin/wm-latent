"""How much does the LeWM predictor amplify latent perturbations? (motivation check for a non-expansion penalty)
Held-out-agnostic random Dubins windows (H=3 real context frames + K real actions), rollout mirrors
diag_rollout_safety.py / JEPA.rollout (history 3, autoregressive). Per model, per-window power iteration
(Jv by central finite differences, J^T u by autograd) for:
  one-step  : sigma_max of d zhat_{t+1} / d(window: 3 x D)   and   d zhat_{t+1} / d z_t (last slot only)
  K-step    : sigma_max of d zhat_{t+K} / d(window), K in KS, plus the gain along a random direction.
Gains are dimensionless (||J v|| / ||v||); > 1 means amplification. Usage: pred_jacobian_gain.py tag=ckpt [...]"""
import sys, os
sys.path.insert(0, "/home/seongbin/latent/le-wm")
import numpy as np, torch, h5py
import stable_worldmodel as swm
from utils import get_img_preprocessor

DEV, H, NW, CH, ITERS, EPS, KS = "cuda", 3, 128, 16, 15, 1e-3, (1, 5, 10, 25)
K = max(KS)
tf = get_img_preprocessor(source="pixels", target="pixels", img_size=224)
with h5py.File("/data/seongbin/lewm/datasets/dubins_expert.h5", "r") as f:
    off, ln = f["ep_offset"][:], f["ep_len"][:]
    act_all = f["action"][:].astype(np.float32).reshape(len(f["action"]), -1)
    a_mu, a_sd = np.nanmean(act_all, 0), np.nanstd(act_all, 0)
    rng = np.random.default_rng(0)
    eps_ok = np.where(ln >= H + K)[0]
    starts = [off[e] + rng.integers(0, ln[e] - (H + K) + 1) for e in rng.choice(eps_ok, NW)]
    idx = np.concatenate([np.arange(s, s + H + K) for s in starts]); u, inv = np.unique(idx, return_inverse=True)
    px = f["pixels"][u][inv].reshape(NW, H + K, *f["pixels"].shape[1:])
    act = ((act_all[idx] - a_mu) / (a_sd + 1e-8)).reshape(NW, H + K, -1)


def bnorm(x):  # per-window norm over all non-batch dims
    return x.flatten(1).norm(dim=1)


@torch.no_grad()
def run(tag, ck):
    m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(DEV).eval()
    res = {}
    for c0 in range(0, NW, CH):   # ponytail: chunked so it fits next to other jobs on a nearly full GPU
        fr = torch.from_numpy(np.ascontiguousarray(px[c0:c0 + CH, :H].reshape(-1, *px.shape[2:]))).permute(0, 3, 1, 2)
        x = tf({"pixels": fr})["pixels"].float().contiguous().to(DEV)
        W0 = m.projector(m.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]).view(len(fr) // H, H, -1)
        for key, val in chunk(m, W0, torch.from_numpy(act[c0:c0 + CH]).to(DEV)).items():
            res.setdefault(key, []).append(val)
    res = {k: np.concatenate(v) for k, v in res.items()}
    out = [f"PJG {tag}  1-step sigma_max: window {np.median(res['s1']):.3f}  last-slot {np.median(res['s1_last']):.3f}"]
    for k in KS[1:]:
        out.append(f"K={k}: sigma_max med {np.median(res[f's{k}']):.2f} p90 {np.percentile(res[f's{k}'], 90):.2f}  "
                   f"random-dir med {np.median(res[f'r{k}']):.3f}")
    print("  ".join(out), flush=True)
    del m; torch.cuda.empty_cache()


@torch.no_grad()
def chunk(m, W0, A):
    def roll(win, k, last_only=False):  # win: (n, H, D) -> zhat_{t+k}
        if last_only:
            win = torch.cat([W0[:, :-1], win], 1)
        emb, a = win, A[:, :H]
        for s in range(k):
            p = m.predict(emb[:, -H:], m.action_encoder(a[:, -H:]))[:, -1:]
            emb = torch.cat([emb, p], 1)
            a = torch.cat([a, A[:, H + s:H + s + 1]], 1)
        return emb[:, -1]

    def gain(k, last_only=False):
        base = W0[:, -1:] if last_only else W0
        F = lambda w: roll(w, k, last_only)
        Jv = lambda v: (F(base + EPS * v) - F(base - EPS * v)) / (2 * EPS)
        def JTu(uu):
            with torch.enable_grad():
                w = base.clone().requires_grad_(True)
                return torch.autograd.grad(F(w), w, grad_outputs=uu)[0]
        g = torch.Generator(device=DEV).manual_seed(0)
        v = torch.randn(base.shape, device=DEV, generator=g); v = v / bnorm(v).view(-1, 1, 1)
        rand_gain = bnorm(Jv(v))                                  # typical direction
        for _ in range(ITERS):
            uu = Jv(v); v = JTu(uu); v = v / (bnorm(v).view(-1, 1, 1) + 1e-12)
        return bnorm(Jv(v)).cpu().numpy(), rand_gain.cpu().numpy()

    r = {"s1_last": gain(1, last_only=True)[0], "s1": gain(1)[0]}
    for k in KS[1:]:
        r[f"s{k}"], r[f"r{k}"] = gain(k)
    return r


for spec in sys.argv[1:]:
    t, c = spec.split("=", 1)
    run(t, c)
