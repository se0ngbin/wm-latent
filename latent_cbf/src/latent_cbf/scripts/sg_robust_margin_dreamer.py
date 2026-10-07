"""Stage-1 robust-margin (Ilyas) on the sg DREAMER encoder (frozen). Same 4 arms + layout-controlled
multicolor set as scripts/sg_robust_margin.py (LEWM), but features = sg RSSM observe->get_feat (544-d)
via build_wm2d/feats. Arms: plain / aug / robust(min-max) / proj(D_R nuisance-subspace projection)."""
import os, sys
from pathlib import Path
_THIS = Path(__file__).resolve()
for p in [str(_THIS.parents[3]), str(_THIS.parents[2]), str(_THIS.parents[1]), str(_THIS.parent)]:
    if p not in sys.path: sys.path.append(p)
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sg_margin_ood_dreamer import build_wm2d, feats, RSSM, DEV

TRAIN = "/data/seongbin/lewm/datasets/sg_multicolor_train.npz"
STEPS, LR, BS = 5000, 3e-4, 256
GRAD_THR, ZS_W, RELU_W, GP_W = 0.1, 0.1, 1.0, 10.0
SEED = int(os.environ.get("SG_SEED", "0")); TEST_FRAC = 0.2

class MarginHead(nn.Module):
    def __init__(self, d, u=512):
        super().__init__(); self.l = nn.Sequential(); k = d
        for i in range(2):
            self.l.add_module(f"l{i}", nn.Linear(k, u, bias=False)); self.l.add_module(f"n{i}", nn.LayerNorm(u, eps=1e-3))
            self.l.add_module(f"a{i}", nn.SiLU()); k = u
        self.m = nn.Linear(u, 1)
    def forward(self, x): return self.m(self.l(x))

def gp_loss(head, safe, unsafe):
    pos, neg = head(safe), head(unsafe); N = max(pos.shape[0], neg.shape[0])
    def rs(x, n): return x if x.shape[0] == n else x[torch.randint(0, x.shape[0], (n,), device=x.device)]
    sd, ud = rs(safe, N), rs(unsafe, N)
    a = torch.rand(N, 1, device=DEV); interp = (a*sd + (1-a)*ud).requires_grad_(True)
    g = torch.autograd.grad(head(interp), interp, torch.ones(N, 1, device=DEV), create_graph=True)[0]
    gp = ((torch.sqrt((g**2).sum(1)+1e-12) - GRAD_THR)**2).mean()
    return ZS_W*(neg.mean()-pos.mean()) + RELU_W*(torch.relu(neg).mean()+torch.relu(-pos).mean()) + GP_W*gp

def score(head, Z, y):
    with torch.no_grad(): h = head(Z.to(DEV)).cpu().numpy()
    return float(roc_auc_score((y == 0).astype(int), h)) if 0 < (y == 0).mean() < 1 else float("nan")

def main():
    d = np.load(TRAIN); imgs, y, colors = d["imgs"], d["failure"].astype(int), list(d["colors"])
    purple, camrot, th = d["purple"], d["camrot"], d["state"][:, 2]
    N, K = imgs.shape[0], imgs.shape[1]; blue_i = colors.index("blue")
    n_test = int(N*TEST_FRAC); tr = np.arange(N-n_test); te = np.arange(N-n_test, N)
    ytr, yte = y[tr], y[te]
    print(f"dreamer n={N} K={K} train={len(tr)} test={len(te)} unsafe={y.mean():.3f}", flush=True)
    wm = build_wm2d(RSSM)
    Ztr = torch.stack([feats(wm, imgs[tr, c], th[tr]) for c in range(K)], 1)   # (Ntr,K,D)
    Zte_blue = feats(wm, imgs[te, blue_i], th[te]); Zte_purp = feats(wm, purple[te], th[te]); Zte_rot = feats(wm, camrot[te], th[te])
    D = Ztr.shape[2]; s_idx = np.where(ytr == 0)[0]; u_idx = np.where(ytr == 1)[0]; Ztr_d = Ztr.to(DEV)
    class S:
        def __init__(s, mode): s.mode = mode
        def __call__(s, head):
            si = s_idx[np.random.randint(0, len(s_idx), BS)]; ui = u_idx[np.random.randint(0, len(u_idx), BS)]
            def pick(idx, want_safe):
                Z = Ztr_d[idx]
                if s.mode == "blue": return Z[:, blue_i]
                if s.mode == "rand": return Z[torch.arange(len(idx)), torch.randint(0, K, (len(idx),), device=DEV)]
                with torch.no_grad(): h = head(Z.reshape(-1, D)).reshape(len(idx), K)
                pc = h.argmax(1) if want_safe else h.argmin(1)
                return Z[torch.arange(len(idx)), pc]
            return pick(si, True), pick(ui, False)
    def train(sampler):
        torch.manual_seed(SEED); head = MarginHead(D).to(DEV); opt = torch.optim.AdamW(head.parameters(), lr=LR)
        for _ in range(STEPS):
            s, u = sampler(head); loss = gp_loss(head, s, u); opt.zero_grad(); loss.backward(); opt.step()
        return head.eval()
    for tag, mode in [("plain", "blue"), ("aug", "rand"), ("robust", "worst")]:
        head = train(S(mode))
        print(f"KEY robmarg dreamer arm={tag} indist_blue={score(head,Zte_blue,yte):.3f} "
              f"zs_purple={score(head,Zte_purp,yte):.3f} zs_rotate={score(head,Zte_rot,yte):.3f}", flush=True)
    deltas = (Ztr - Ztr.mean(1, keepdim=True)).reshape(-1, D).numpy()
    _, Sv, Vt = np.linalg.svd(deltas - deltas.mean(0), full_matrices=False)
    k = int(np.searchsorted(np.cumsum(Sv**2)/np.sum(Sv**2), 0.95) + 1)
    Vk = torch.tensor(Vt[:k], dtype=torch.float32)
    def proj(Z): return Z - (Z @ Vk.T) @ Vk
    ZtrP = proj(Ztr).to(DEV)
    torch.manual_seed(SEED); head = MarginHead(D).to(DEV); opt = torch.optim.AdamW(head.parameters(), lr=LR)
    for _ in range(STEPS):
        si = s_idx[np.random.randint(0, len(s_idx), BS)]; ui = u_idx[np.random.randint(0, len(u_idx), BS)]
        loss = gp_loss(head, ZtrP[si, blue_i], ZtrP[ui, blue_i]); opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    print(f"KEY robmarg dreamer arm=proj(k={k}) indist_blue={score(head,proj(Zte_blue),yte):.3f} "
          f"zs_purple={score(head,proj(Zte_purp),yte):.3f} zs_rotate={score(head,proj(Zte_rot),yte):.3f}", flush=True)
    print("SG_ROBMARG_DREAMER_DONE")

if __name__ == "__main__":
    main()
