"""Cartpole safety eval of a frozen LeWM encoder against the exact value oracle.

Per model (WM frozen; heads trained on train episodes, scored on held-out test episodes, 3 head seeds):
  margin  l(z_t): margin_gp head (dubins loss/arch, 50/50 safe/failed batches) on `failures` = "currently safe".
  critic  Q(h_t, a): offline discrete safety-Bellman fitted Q-iteration on dataset transitions,
          target (1-g) l(z_t) + g min(l(z_t), max_a' Q'(h_{t+1}, a')), l = the frozen learned margin.
          h_t = [z_{t-2}, z_{t-1}, z_t] (the predictor's context): one frame has no velocity, and doom needs it.
          No value/doomed labels used in training.
  probe   supervised regression h_t -> oracle V (is the information in the latent at all?).
Scored: failed AUC (margin), unsafe AUC (V<0) and doomed-vs-safe AUC (among l>=0) for critic/probe/margin,
in-dist (tan pole) and zero-shot on the SAME test states re-rendered with a held-out purple pole.
`--state_ceiling` runs the same critic on true 4D state history with the true margin (method ceiling).
Usage: eval_cartpole.py MODEL [MODEL...]   |   eval_cartpole.py --state_ceiling
"""
import os, sys, json, argparse
sys.path[:0] = ["/home/seongbin/latent/le-wm", "/home/seongbin/latent/le-wm/scripts", os.path.dirname(os.path.abspath(__file__))]
os.environ.setdefault("STABLEWM_HOME", "/data/seongbin/lewm"); os.environ.setdefault("LOCAL_DATASET_DIR", "/data/seongbin/lewm")
import numpy as np, torch, torch.nn as nn, h5py
from sklearn.metrics import roc_auc_score
import stable_worldmodel as swm
from utils import get_img_preprocessor
from ood_margin_gp_jepa import train_margin, DEV
from render import render, POLE_PALETTE

H5 = "/data/seongbin/lewm/datasets/cartpole_expert.h5"
OUT = "/data/seongbin/lewm/cartpole_eval"
PURPLE = f"{OUT}/test_purple_pixels.npy"
GAMMA, Q_STEPS, SEEDS = 0.98, 20000, (0, 1, 2)


def load_data():
    with h5py.File(H5) as f:
        d = {k: f[k][:] for k in ("state", "action", "failures", "margin", "value", "episode_idx", "step_idx")}
    test = (d["episode_idx"] % 10) >= 7                                    # 30% of episodes held out
    t = np.arange(len(test)); s = d["step_idx"]
    d["hist"] = np.stack([t - np.minimum(k, s) for k in (2, 1, 0)], 1)     # [t-2, t-1, t], clamped to ep start
    nxt = np.append(d["episode_idx"][1:] == d["episode_idx"][:-1], False)   # t+1 in the same episode
    d["test"], d["has_next"] = test, nxt
    return d


def purple_pixels(d):
    if not os.path.exists(PURPLE):
        os.makedirs(OUT, exist_ok=True)
        idx = np.where(d["test"])[0]
        col = dict(pole=POLE_PALETTE["purple"])
        np.save(PURPLE, np.stack([render(s, col) for s in d["state"][idx]]))
    return np.load(PURPLE, mmap_mode="r")


def encode_pixels(m, pix, tf, bs=256):
    Z = []
    with torch.no_grad():
        for i in range(0, len(pix), bs):
            x = tf({"pixels": np.asarray(pix[i:i + bs])})["pixels"].float().to(DEV)
            Z.append(m.projector(m.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]).cpu())
    return torch.cat(Z)


class QNet(nn.Module):
    def __init__(self, inp, out=2, units=512):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(inp, units), nn.LayerNorm(units), nn.SiLU(),
                               nn.Linear(units, units), nn.LayerNorm(units), nn.SiLU(), nn.Linear(units, out))

    def forward(self, x): return self.f(x)


def fqi(H, Hn, l, a, seed):
    """Discounted safety-Bellman fitted Q-iteration. H, Hn: (N, D) history features at t, t+1; l: (N,) margin at t;
    a: (N,) action index taken at t."""
    torch.manual_seed(seed)
    q, qt = QNet(H.shape[1]).to(DEV), QNet(H.shape[1]).to(DEV); qt.load_state_dict(q.state_dict())
    opt = torch.optim.AdamW(q.parameters(), lr=3e-4)
    H, Hn, l, a = H.to(DEV), Hn.to(DEV), l.to(DEV), a.to(DEV)
    for _ in range(Q_STEPS):
        i = torch.randint(0, len(H), (512,), device=DEV)
        with torch.no_grad():
            y = (1 - GAMMA) * l[i] + GAMMA * torch.minimum(l[i], qt(Hn[i]).max(1).values)
        loss = ((q(H[i]).gather(1, a[i, None])[:, 0] - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for p, pt in zip(q.parameters(), qt.parameters()): pt.lerp_(p, 0.005)
    return q.eval()


def probe(H, v, seed, steps=5000):
    torch.manual_seed(seed)
    p = QNet(H.shape[1], out=1).to(DEV); opt = torch.optim.AdamW(p.parameters(), lr=3e-4)
    H, v = H.to(DEV), torch.as_tensor(np.clip(v, -1, 1), device=DEV)
    for _ in range(steps):
        i = torch.randint(0, len(H), (512,), device=DEV)
        loss = ((p(H[i])[:, 0] - v[i]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return p.eval()


def auc(score, safe):
    return float(roc_auc_score(safe, score)) if 0 < safe.mean() < 1 else float("nan")


def metrics(score, mar, val):
    """score >= 0 means 'safe'. failed AUC: safe=l>=0. unsafe AUC: safe=V>=0. doomed AUC: among l>=0, safe=V>=0."""
    alive = mar >= 0
    return dict(failed_auc=auc(score, mar >= 0), unsafe_auc=auc(score, val >= 0),
                doomed_auc=auc(score[alive], val[alive] >= 0), sign_acc=float(((score >= 0) == (val >= 0)).mean()))


def apply(net, X, bs=8192):
    with torch.no_grad():
        return torch.cat([net(X[i:i + bs].to(DEV)) for i in range(0, len(X), bs)]).cpu()


def run(name, d, Z, Zp):
    """Z: latents for all frames (in-dist); Zp: latents for test frames re-rendered purple (test order)."""
    tr, te = ~d["test"], d["test"]
    hist = torch.as_tensor(d["hist"])
    Hall = Z[hist].flatten(1)                                                     # (N, 3*192)
    te_idx = np.where(te)[0]; pos = np.full(len(te), -1); pos[te_idx] = np.arange(len(te_idx))
    Hp = Zp[torch.as_tensor(pos[d["hist"][te_idx]])].flatten(1)                  # purple history (test order)
    mar, val = d["margin"][te], d["value"][te]
    trn = tr & d["has_next"]; ti = np.where(trn)[0]
    a = torch.as_tensor((d["action"][:, 0] > 0).astype(np.int64))
    res = {}
    for seed in SEEDS:
        m = train_margin(Z[tr], d["failures"][tr], "gp", seed=seed)
        l_all = apply(m, Z).flatten()
        q = fqi(Hall[ti], Hall[ti + 1], l_all[ti], a[ti], seed)
        pr = probe(Hall[tr], d["value"][tr], seed)
        for cond, Zc, Hc in (("tan", Z[te], Hall[te]), ("purple", Zp, Hp)):
            res.setdefault(cond, {}).setdefault("margin", []).append(metrics(apply(m, Zc).flatten().numpy(), mar, val))
            res[cond].setdefault("critic", []).append(metrics(apply(q, Hc).max(1).values.numpy(), mar, val))
            res[cond].setdefault("probe", []).append(metrics(apply(pr, Hc)[:, 0].numpy(), mar, val))
    summ = {c: {h: {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))) for k in rs[0]}
                for h, rs in hs.items()} for c, hs in res.items()}
    json.dump(dict(model=name, seeds=list(SEEDS), runs=res, summary=summ), open(f"{OUT}/{name}.json", "w"), indent=1)
    for c, hs in summ.items():
        for h, ks in hs.items():
            print(f"KEY {name} {c:6s} {h:6s} " + " ".join(f"{k}={m:.3f}±{s:.3f}" for k, (m, s) in ks.items()), flush=True)


def state_ceiling(d):
    tr, te = ~d["test"], d["test"]
    S = torch.as_tensor(d["state"] / np.array([2.4, 9.0, 0.2094, 4.0], np.float32))
    H = S[torch.as_tensor(d["hist"])].flatten(1)
    ti = np.where(tr & d["has_next"])[0]
    a = torch.as_tensor((d["action"][:, 0] > 0).astype(np.int64)); l = torch.as_tensor(d["margin"])
    rs = [metrics(apply(fqi(H[ti], H[ti + 1], l[ti], a[ti], s), H[te]).max(1).values.numpy(), d["margin"][te], d["value"][te]) for s in SEEDS]
    summ = {k: (float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))) for k in rs[0]}
    json.dump(dict(model="state_ceiling", runs=rs, summary=summ), open(f"{OUT}/state_ceiling.json", "w"), indent=1)
    print("KEY state_ceiling critic " + " ".join(f"{k}={m:.3f}±{s:.3f}" for k, (m, s) in summ.items()), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("models", nargs="*"); ap.add_argument("--state_ceiling", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    d = load_data()
    if a.state_ceiling: state_ceiling(d)
    if not a.models: return
    pp = purple_pixels(d)
    tf = get_img_preprocessor(source="pixels", target="pixels", img_size=224)
    with h5py.File(H5) as f:
        pix = f["pixels"]
        for name in a.models:
            m = swm.wm.utils.load_pretrained(f"{name}/weights_epoch_50.pt").to(DEV).eval()
            Z, Zp = encode_pixels(m, pix, tf), encode_pixels(m, pp, tf)
            del m; torch.cuda.empty_cache()
            run(name, d, Z, Zp)


if __name__ == "__main__":
    main()
