"""Margin functions (mlp + linear, sg_point_eval's margin_gp loss) on a Reacher WM's frozen CLS latents, labels = SLS-squared joint box
(`failures` in datasets/reacher.h5). Train: episodes < 9000, 25k unsafe + 25k safe frames; test: 5000 frames of episodes >= 9000 at the
natural ratio (episode-disjoint). Prints AUC + balanced sign acc; saves heads for CEM to $HEADS/<WM>/{mlp,linear}_s<SEED>.pt.
Env: WM (checkpoint dir, epoch file via EP), SEED, N_PER_CLASS, STEPS, HEADS."""
import os, sys, numpy as np, torch, h5py
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sg_point_eval as E

WM, EP = os.environ.get("WM", "lewm_reacher"), os.environ.get("EP", "50")
SEED, NPC, STEPS = int(os.environ.get("SEED", "0")), int(os.environ.get("N_PER_CLASS", "25000")), int(os.environ.get("STEPS", "20000"))
HEADS = os.environ.get("HEADS", "/data/seongbin/lewm/reacher_heads")
rng = np.random.default_rng(SEED)
with h5py.File("/data/seongbin/lewm/datasets/reacher.h5", "r") as f:
    y, ep = f["failures"][:].astype(int), f["ep_idx"][:]
    tr = ep < 9000
    ti = np.sort(np.r_[rng.choice(np.where(tr & (y == 1))[0], NPC, replace=False), rng.choice(np.where(tr & (y == 0))[0], NPC, replace=False)])
    te = np.sort(rng.choice(np.where(~tr)[0], 5000, replace=False))
    Xtr, ytr, Xte, yte = f["pixels"][ti], y[ti], f["pixels"][te], y[te]
print(f"train {len(ti)} (unsafe {ytr.mean():.2f}) | test {len(te)} held-out-episode frames (unsafe {yte.mean():.3f})", flush=True)
m = E.swm.wm.utils.load_pretrained(f"{WM}/weights_epoch_{EP}.pt").to(E.DEV).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=E.DEV).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=E.DEV).view(1, 3, 1, 1)
def enc(a, bs=256):   # training orientation: HDF5 loader permute(0,3,1,2) + ImageNet norm (frames already 224)
    out = []
    for i in range(0, len(a), bs):
        x = (torch.as_tensor(a[i:i + bs], device=E.DEV).permute(0, 3, 1, 2).float() / 255. - MEAN) / STD
        with torch.no_grad(): out.append(m.projector(m.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]))
    return torch.cat(out)
Z, Zte = enc(Xtr), enc(Xte)
s_idx, u_idx = np.where(ytr == 0)[0], np.where(ytr == 1)[0]
for hk in ["linear", "mlp"]:
    torch.manual_seed(SEED); r = np.random.default_rng(SEED)
    head = (torch.nn.Linear(Z.shape[1], 1) if hk == "linear" else E.MarginHead(Z.shape[1])).to(E.DEV); opt = torch.optim.AdamW(head.parameters(), lr=E.LR)
    for _ in range(STEPS):
        si, ui = r.choice(s_idx, E.BS), r.choice(u_idx, E.BS)
        loss = E.gp_loss(head, Z[si], Z[ui]); opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    with torch.no_grad(): h = head(Zte).cpu().numpy().ravel()
    sp = h >= 0; bal = (sp[yte == 0].mean() + (~sp[yte == 1]).mean()) / 2
    print(f"REACHER_MARGIN {WM} {hk} seed={SEED} test_auc={roc_auc_score(1 - yte, h):.3f} bal_acc={bal:.3f} predsafe={sp.mean():.3f}", flush=True)
    sd = head.state_dict()
    sd = {"margin_gp": {k.replace("l.l", "layers.Margin GP_linear").replace("l.n", "layers.Margin GP_norm").replace("m.", "mean_layer."): v for k, v in sd.items()}} if hk == "mlp" else {"linear": sd}
    os.makedirs(f"{HEADS}/{WM}", exist_ok=True); torch.save(sd, f"{HEADS}/{WM}/{hk}_s{SEED}.pt")
