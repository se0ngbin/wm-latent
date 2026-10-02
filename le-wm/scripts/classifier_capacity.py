"""Classifier-capacity sweep: is the baseline LE-WM latent's weaker CBF a
CLASSIFIER problem or a LATENT problem?

Fix the encoder (baseline vs jacobian), render dubins states, encode, and train
heads of increasing capacity to regress the HJ value V* — report sign-accuracy
and AUC vs the true safe set. If a bigger head on the baseline latent climbs to
jacobian's level, it was the classifier; if it plateaus below, it's the latent.
"""
import os, sys
sys.path.insert(0, "/home/seongbin/latent/le-wm")
os.environ.setdefault("STABLEWM_HOME", "/data/seongbin/lewm")
os.environ.setdefault("LOCAL_DATASET_DIR", "/data/seongbin/lewm")
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
import stable_worldmodel as swm
from dubins_swm_env import DubinsSwmEnv
from utils import get_img_preprocessor

DEV, IMG, N = "cuda:0", 224, 9000
HJ = "/home/seongbin/latent/latent_cbf/results/hj_truth.npz"
MODELS = {"baseline": "sigreg_only_dubins", "jacobian": "jacobian_w1_dubins",
          "jacobian+pull": "lewm_dubins_jacpull50"}
HEADS = {"linear": [], "mlp-256": [256], "mlp-512x2": [512, 512], "mlp-1024x3": [1024, 1024, 1024]}

def hj_interp(V, xs, ys, ths, pts):
    # trilinear on (x,y) + periodic θ; pts (M,3)
    def idxw(g, v, periodic=False):
        if periodic:
            v = (v - g[0]) % (2*np.pi) + g[0]
        i = np.clip(np.searchsorted(g, v) - 1, 0, len(g)-2)
        w = (v - g[i]) / (g[i+1] - g[i] + 1e-9)
        return i, np.clip(w, 0, 1)
    ix, wx = idxw(xs, pts[:,0]); iy, wy = idxw(ys, pts[:,1]); it, wt = idxw(ths, pts[:,2], True)
    out = np.zeros(len(pts))
    for dx in (0,1):
        for dy in (0,1):
            for dt in (0,1):
                w = (wx if dx else 1-wx)*(wy if dy else 1-wy)*(wt if dt else 1-wt)
                out += w * V[np.clip(ix+dx,0,len(xs)-1), np.clip(iy+dy,0,len(ys)-1), (it+dt)%len(ths)]
    return out

def encode(m, x):
    return m.projector(m.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0])

def main():
    d = np.load(HJ); V, xs, ys, ths = d["V"], d["grid_xs"], d["grid_ys"], d["grid_thetas"]
    rng = np.random.default_rng(0)
    st = np.stack([rng.uniform(xs[0], xs[-1], N), rng.uniform(ys[0], ys[-1], N),
                   rng.uniform(-np.pi, np.pi, N)], 1).astype(np.float32)
    vstar = hj_interp(V, xs, ys, ths, st).astype(np.float32)
    print(f"N={N}  frac_safe(V*>=0)={(vstar>=0).mean():.3f}")
    env = DubinsSwmEnv(); tf = get_img_preprocessor(source="pixels", target="pixels", img_size=IMG)
    imgs = []
    for s in st:
        env.state = s.copy()
        imgs.append(tf({"pixels": env._render_image()[None]})["pixels"][0])
    frames = torch.stack(imgs).float()
    ntr = 7000; y = torch.tensor(vstar)
    print(f"\n{'encoder':10s} {'head':12s} {'test sign-acc':>13s} {'test AUC':>9s}")
    for mname, ck in MODELS.items():
        m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(DEV).eval()
        with torch.no_grad():
            Z = torch.cat([encode(m, frames[i:i+64].to(DEV)).cpu() for i in range(0, N, 64)])
        del m; torch.cuda.empty_cache()
        mu, sd = Z[:ntr].mean(0), Z[:ntr].std(0)+1e-6; Zn = (Z-mu)/sd
        Ztr, Zte, ytr, yte = Zn[:ntr].to(DEV), Zn[ntr:].to(DEV), y[:ntr].to(DEV), y[ntr:].to(DEV)
        for hname, hid in HEADS.items():
            torch.manual_seed(0)
            layers, dprev = [], Z.shape[1]
            for h in hid: layers += [nn.Linear(dprev, h), nn.SiLU()]; dprev = h
            layers += [nn.Linear(dprev, 1)]
            head = nn.Sequential(*layers).to(DEV)
            opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
            for step in range(3000):
                idx = torch.randint(0, ntr, (512,), device=DEV)
                loss = nn.functional.mse_loss(head(Ztr[idx]).squeeze(-1), ytr[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                p = head(Zte).squeeze(-1).cpu().numpy()
            yt = yte.cpu().numpy()
            acc = ((p >= 0) == (yt >= 0)).mean()
            auc = roc_auc_score((yt >= 0).astype(int), p)
            print(f"{mname:10s} {hname:12s} {acc:13.3f} {auc:9.3f}")
    print("done")

if __name__ == "__main__":
    main()
