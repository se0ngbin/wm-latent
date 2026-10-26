"""Benchmark learnability: supervised CNN on raw pixels, trained on train shards, scored on a test shard collected with the
SAME collector but different seeds (unseen layouts). Per epoch: train AUC (fit) and test AUC (layout generalization).
usage: python sg_point_cnn_check.py '<train glob>' <test.h5> [epochs]"""
import sys, glob, h5py, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
DEV = "cuda:0"; torch.manual_seed(0); rng = np.random.default_rng(0)
def load(paths):
    X, Y = [], []
    for p in paths:
        with h5py.File(p, "r") as f: X.append(f["pixels"][:]); Y.append(f["failures"][:])
    return np.concatenate(X), np.concatenate(Y).astype(np.float32)
Xtr, ytr = load(sorted(glob.glob(sys.argv[1]))); Xte, yte = load([sys.argv[2]]); EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 15
print(f"train {Xtr.shape} unsafe={ytr.mean():.3f} | test {Xte.shape} unsafe={yte.mean():.3f}", flush=True)
px = lambda a: torch.as_tensor(a, device=DEV).permute(0, 3, 1, 2).float() / 255.
def block(i, o): return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2)]
cnn = nn.Sequential(*block(3, 32), *block(32, 64), *block(64, 128), *block(128, 128), nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(128, 1)).to(DEV)
opt = torch.optim.AdamW(cnn.parameters(), lr=1e-3, weight_decay=1e-4); pw = torch.tensor((1 - ytr.mean()) / ytr.mean(), device=DEV)
def scores(X, bs=500):
    cnn.eval()
    with torch.no_grad(): s = np.concatenate([cnn(px(X[i:i + bs])).squeeze(-1).cpu().numpy() for i in range(0, len(X), bs)])
    cnn.train(); return s
for ep in range(EPOCHS):
    perm = rng.permutation(len(ytr))
    for i in range(0, len(perm), 128):
        j = np.sort(perm[i:i + 128])
        loss = F.binary_cross_entropy_with_logits(cnn(px(Xtr[j])).squeeze(-1), torch.as_tensor(ytr[j], device=DEV), pos_weight=pw)
        opt.zero_grad(); loss.backward(); opt.step()
    sub = rng.choice(len(ytr), 5000, replace=False)
    print(f"ep{ep:2d} train_auc={roc_auc_score(ytr[sub], scores(Xtr[sub])):.3f}  TEST(unseen layouts)_auc={roc_auc_score(yte, scores(Xte)):.3f}", flush=True)
