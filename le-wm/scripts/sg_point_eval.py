"""Zero-shot margin eval for the CLEAN SafetyPointGoal1 WMs (layout-controlled protocol of
/data/seongbin/lewm/safeadv_results/sg_eval/sg_plain_eval.py, same head / split / steps / scoring).
Data: sg_point_multi_eval.npz (collect_point_multi.py) - every variant is a clean render of the SAME pose.
Frames split 80/20; margin_gp head trained on BASE train frames (5000 steps); scored on the test frames
under base (in-dist) and each OOD axis: color (purple hazards), rotate (real 90deg camera), shape (box
hazards), agent (Car robot). Reports sign accuracy (h>=0 = safe), AUC, predicted-safe fraction.
Usage: SG_SEED=s [SG_ENCODERS="name:ckptdir,.."] python sg_point_eval.py"""
import os, sys
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "/home/seongbin/latent/le-wm")
os.environ.setdefault("STABLEWM_HOME", "/data/seongbin/lewm"); os.environ.setdefault("LOCAL_DATASET_DIR", "/data/seongbin/lewm")
import stable_worldmodel as swm
from utils import get_img_preprocessor

DEV = "cuda:0"
DATA = os.environ.get("SG_DATA", "/data/seongbin/lewm/datasets/sg_point_multi_eval.npz")
NAMES = ["pt_baseline", "pt_jacobian", "pt_jacpull", "pt_safeadv_linf_g0", "pt_jac_safeadv_linf", "pt_jacpull_safeadv_linf",
         "pt_lam_bdet", "pt_jac_lam_bdet", "pt_jacpull_lam_bdet"]
ENCODERS = dict(kv.split(":", 1) for kv in os.environ["SG_ENCODERS"].split(",")) if os.environ.get("SG_ENCODERS") else {n: n for n in NAMES}
AXES = ["color", "rotate", "shape", "agent"]
SEED = int(os.environ.get("SG_SEED", "0"))
STEPS, LR, BS = 5000, 3e-4, 256
GRAD_THR, ZS_W, RELU_W, GP_W = 0.1, 0.1, 1.0, 10.0
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
    a = torch.rand(N, 1, device=DEV); interp = (a * sd + (1 - a) * ud).requires_grad_(True)
    g = torch.autograd.grad(head(interp), interp, torch.ones(N, 1, device=DEV), create_graph=True)[0]
    gp = ((torch.sqrt((g ** 2).sum(1) + 1e-12) - GRAD_THR) ** 2).mean()
    return ZS_W * (neg.mean() - pos.mean()) + RELU_W * (torch.relu(neg).mean() + torch.relu(-pos).mean()) + GP_W * gp


def score(head, Z, y):
    with torch.no_grad(): h = head(Z.to(DEV)).cpu().numpy().ravel()
    safe = (y == 0)
    return float(((h >= 0) == safe).mean()), float(roc_auc_score(safe.astype(int), h)), float((h >= 0).mean())


def main():
    d = np.load(DATA); y = d["failure"].astype(int); N = len(y)
    n_test = int(N * TEST_FRAC); tr = np.arange(N - n_test); te = np.arange(N - n_test, N); ytr, yte = y[tr], y[te]
    print(f"seed={SEED} train={len(tr)} test={len(te)} test_unsafe={yte.mean():.3f}", flush=True)
    tf = get_img_preprocessor(source="pixels", target="pixels", img_size=224)
    for name, ck in ENCODERS.items():
        path = f"/data/seongbin/lewm/checkpoints/{ck}/weights_epoch_50.pt"
        if not os.path.exists(path): print(f"SKIP {name}: no {path}", flush=True); continue
        m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(DEV).eval()
        def enc(imgs, bs=128):
            out = []
            for i in range(0, len(imgs), bs):
                b = torch.stack([tf({"pixels": imgs[j]})["pixels"] for j in range(i, min(i + bs, len(imgs)))]).to(DEV)
                with torch.no_grad(): out.append(m.projector(m.encoder(b, interpolate_pos_encoding=True).last_hidden_state[:, 0]).cpu())
            return torch.cat(out)
        Ztr = enc(d["base"][tr]).to(DEV)
        Zte = {k: enc(d[k][te]) for k in ["base"] + AXES}
        s_idx, u_idx = np.where(ytr == 0)[0], np.where(ytr == 1)[0]
        torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
        head = MarginHead(Ztr.shape[1]).to(DEV); opt = torch.optim.AdamW(head.parameters(), lr=LR)
        for _ in range(STEPS):
            si, ui = rng.choice(s_idx, BS), rng.choice(u_idx, BS)
            loss = gp_loss(head, Ztr[si], Ztr[ui]); opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        ia, iu, ip = score(head, Zte["base"], yte)
        for ax in AXES:
            za, zu, zp = score(head, Zte[ax], yte)
            print(f"SGPT {name} axis={ax} seed={SEED} in_acc={ia:.3f} in_auc={iu:.3f} in_predsafe={ip:.3f} "
                  f"zs_acc={za:.3f} zs_auc={zu:.3f} zs_predsafe={zp:.3f}", flush=True)
        del m; torch.cuda.empty_cache()
    print("done")


if __name__ == "__main__":
    main()
