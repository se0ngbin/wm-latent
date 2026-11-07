"""Is 'Point inside a hazard' learnable from clean 128px pixels across UNSEEN layouts, and is it in the WM encoder?
Train on N_TR frames of sg_point_clean.h5 (~850 layouts); test on sg_point_multi_eval.npz base view (separate collection,
all layouts unseen), plus the other OOD views.
  cnn    supervised small CNN on raw pixels (upper bound for the benchmark itself)
  patch  supervised attention-pool head on the FROZEN encoder's patch tokens (spatial info before CLS pooling)
  cls    same head budget on the pooled CLS-projector latent (what all margin evals use)"""
import os, sys, h5py, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sg_point_eval as E

DEV = "cuda:0"; N_TR = int(os.environ.get("N_TR", "50000")); EPOCHS = int(os.environ.get("EPOCHS", "8"))
WM = os.environ.get("WM", "pt_jacpull"); MODES = os.environ.get("MODES", "cnn,patch,cls").split(",")
torch.manual_seed(0); rng = np.random.default_rng(0)
with h5py.File("/data/seongbin/lewm/datasets/sg_point_clean.h5", "r") as f:
    idx = np.sort(rng.choice(f["pixels"].shape[0], N_TR, replace=False)); X, y = f["pixels"][idx], f["failures"][idx].astype(np.float32)
d = np.load(E.DATA); yt = d["failure"].astype(int); VIEWS = ["base"] + E.AXES
print(f"train={len(y)} unsafe={y.mean():.3f} | test poses={len(yt)} (unseen layouts)", flush=True)

def fit(model, feats, labels, bs=256):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4); n = len(labels); pw = torch.tensor((1 - labels.mean()) / labels.mean(), device=DEV)
    for ep in range(EPOCHS):
        perm = rng.permutation(n); tot = 0
        for i in range(0, n, bs):
            j = perm[i:i + bs]; xb = feats(j); yb = torch.as_tensor(labels[j], device=DEV)
            loss = F.binary_cross_entropy_with_logits(model(xb).squeeze(-1), yb, pos_weight=pw); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(j)
        print(f"  ep{ep} loss={tot / n:.4f}", flush=True)
    return model.eval()

def report(tag, predict):
    out = []
    for v in VIEWS:
        with torch.no_grad(): s = np.concatenate([predict(d[v][i:i + 500]) for i in range(0, len(yt), 500)])
        out.append(f"{v}={roc_auc_score(yt, s):.3f}")
    print(f"LEARN {tag} unseen-layout AUC (unsafe score): " + " ".join(out), flush=True)

if "cnn" in MODES:
    def px(a): return torch.as_tensor(a, device=DEV).permute(0, 3, 1, 2).float() / 255.
    cnn = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(128, 1)).to(DEV)
    fit(cnn, lambda j: px(X[j]), y)
    report("cnn(pixels)", lambda a: cnn(px(a)).squeeze(-1).cpu().numpy())

if "patch" in MODES or "cls" in MODES:
    m = E.swm.wm.utils.load_pretrained(f"{WM}/weights_epoch_50.pt").to(DEV).eval()
    tf = E.get_img_preprocessor(source="pixels", target="pixels", img_size=224)
    def enc(a, bs=256):
        P, C = [], []
        for i in range(0, len(a), bs):
            b = torch.stack([tf({"pixels": a[k]})["pixels"] for k in range(i, min(i + bs, len(a)))]).to(DEV)
            with torch.no_grad():
                h = m.encoder(b, interpolate_pos_encoding=True).last_hidden_state
                P.append(h[:, 1:].half().cpu()); C.append(m.projector(h[:, 0]).cpu())
        return torch.cat(P), torch.cat(C)
    Ptr, Ctr = enc(X)
    class AttnPool(nn.Module):   # learned-query attention over patch tokens (keeps "where" information)
        def __init__(s, dim, q=4):
            super().__init__(); s.q = nn.Parameter(torch.randn(q, dim) * .02); s.att = nn.MultiheadAttention(dim, 4, batch_first=True)
            s.mlp = nn.Sequential(nn.LayerNorm(q * dim), nn.Linear(q * dim, 256), nn.SiLU(), nn.Linear(256, 1))
        def forward(s, x): o, _ = s.att(s.q.expand(len(x), -1, -1), x, x); return s.mlp(o.flatten(1))
    if "patch" in MODES:
        hp = fit(AttnPool(Ptr.shape[-1]).to(DEV), lambda j: Ptr[j].float().to(DEV), y)
        report(f"patch({WM})", lambda a: hp(enc(a)[0].float().to(DEV)).squeeze(-1).cpu().numpy())
    if "cls" in MODES:
        hc = fit(nn.Sequential(nn.Linear(Ctr.shape[-1], 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 1)).to(DEV), lambda j: Ctr[j].to(DEV), y)
        report(f"cls({WM})", lambda a: hc(enc(a)[1].to(DEV)).squeeze(-1).cpu().numpy())
