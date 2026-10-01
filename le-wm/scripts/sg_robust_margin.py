"""Stage-1 robust-margin study (Ilyas-style), LEWM encoders FROZEN. LAYOUT-CONTROLLED: one
multi-color collection where every appearance variant (6 train colors + held-out purple + real
camera-rotation) is a render of the SAME pose. Split frames 80/20; train the margin on train-frames,
test on test-frames under blue (in-dist), purple (same-type OOD), camrot (cross-type OOD). The
appearance gap = indist_blue - zs_* on identical test frames, so it isolates appearance from the
(layout-dependent) safety signal. Four arms:
  A plain  : train on BLUE only.            B aug   : random train color per sample/step.
  C robust : worst-case color (min-max).    D proj  : project out the color-nuisance subspace (D_R analog), plain margin.
"""
import os, sys
sys.path.insert(0, "/home/seongbin/latent/le-wm")
os.environ.setdefault("STABLEWM_HOME", "/data/seongbin/lewm"); os.environ.setdefault("LOCAL_DATASET_DIR", "/data/seongbin/lewm")
import numpy as np, torch, torch.nn as nn
import stable_worldmodel as swm
from utils import get_img_preprocessor
from sklearn.metrics import roc_auc_score

DEV, IMG = "cuda:0", 224
TRAIN = "/data/seongbin/lewm/datasets/sg_multicolor_train.npz"
ENCODERS = {"baseline": "sg_baseline", "jacobian": "sg_jacobian", "jacobian+pull": "sg_jacpull"}
if os.environ.get("SG_ENCODERS"):   # "name:ckpt,name:ckpt" to override (e.g. Stage-2 retrained encoders)
    ENCODERS = dict(kv.split(":") for kv in os.environ["SG_ENCODERS"].split(","))
STEPS, LR, BS = 5000, 3e-4, 256
GRAD_THR, ZS_W, RELU_W, GP_W = 0.1, 0.1, 1.0, 10.0
SEED = int(os.environ.get("SG_SEED", "0"))
TEST_FRAC = 0.2

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

def raw_encode(m, x): return m.projector(m.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0])

def encode(m, imgs, tf, bs=128):
    out = []
    for i in range(0, len(imgs), bs):
        b = torch.stack([tf({"pixels": imgs[j]})["pixels"] for j in range(i, min(i+bs, len(imgs)))]).to(DEV)
        with torch.no_grad(): out.append(raw_encode(m, b).cpu())
    return torch.cat(out)

def score(head, Z, y):
    with torch.no_grad(): h = head(Z.to(DEV)).cpu().numpy()
    return float(roc_auc_score((y == 0).astype(int), h)) if 0 < (y == 0).mean() < 1 else float("nan")

def main():
    d = np.load(TRAIN); imgs, y, colors = d["imgs"], d["failure"].astype(int), list(d["colors"])
    purple, camrot = d["purple"], d["camrot"]
    N, K = imgs.shape[0], imgs.shape[1]; blue_i = colors.index("blue")
    n_test = int(N * TEST_FRAC); tr = np.arange(N - n_test); te = np.arange(N - n_test, N)
    ytr, yte = y[tr], y[te]
    print(f"n={N} K={K} colors={[str(c) for c in colors]} train={len(tr)} test={len(te)} unsafe={y.mean():.3f}", flush=True)
    tf = get_img_preprocessor(source="pixels", target="pixels", img_size=IMG)
    for ename, ck in ENCODERS.items():
        m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(DEV).eval()
        Ztr = torch.stack([encode(m, imgs[tr, c], tf) for c in range(K)], 1)      # (Ntr,K,D)
        Zte_blue = encode(m, imgs[te, blue_i], tf)                                # test blue (in-dist)
        Zte_purp = encode(m, purple[te], tf)                                      # test purple (same-type OOD)
        Zte_rot  = encode(m, camrot[te], tf)                                      # test camrot (cross-type OOD)
        del m; torch.cuda.empty_cache()
        D = Ztr.shape[2]; s_idx = np.where(ytr == 0)[0]; u_idx = np.where(ytr == 1)[0]
        Ztr_d = Ztr.to(DEV)
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
            print(f"KEY robmarg {ename} arm={tag} indist_blue={score(head,Zte_blue,yte):.3f} "
                  f"zs_purple={score(head,Zte_purp,yte):.3f} zs_rotate={score(head,Zte_rot,yte):.3f}", flush=True)
        # arm D: nuisance-subspace projection (Ilyas D_R analog)
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
        print(f"KEY robmarg {ename} arm=proj(k={k}) indist_blue={score(head,proj(Zte_blue),yte):.3f} "
              f"zs_purple={score(head,proj(Zte_purp),yte):.3f} zs_rotate={score(head,proj(Zte_rot),yte):.3f}", flush=True)
    print("SG_ROBMARG_DONE")

if __name__ == "__main__":
    main()
