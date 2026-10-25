"""Layout-disjoint margin eval: does the safety probe generalize to UNSEEN layouts when trained on many?
Head (sg_point_eval's margin_gp, more steps) trained on N_TR frames of the WM training set sg_point_clean.h5 (~850 random
layouts); scored on ALL 5000 poses of sg_point_multi_eval.npz, a separate collection (seed 321) -> every layout unseen.
Axes: base (same view, new layouts) / color / rotate / shape / agent. AUC + sign acc + predicted-safe frac."""
import os, sys, h5py, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sg_point_eval as E

N_TR = int(os.environ.get("N_TR", "50000")); STEPS = int(os.environ.get("STEPS", "20000")); SEED = int(os.environ.get("SG_SEED", "0"))
names = os.environ.get("DIAG_ENCODERS", "pt_baseline,pt_jacobian,pt_jacpull,pt_jacpull_safeadv_linf").split(",")
with h5py.File("/data/seongbin/lewm/datasets/sg_point_clean.h5", "r") as f:
    idx = np.sort(np.random.default_rng(SEED).choice(f["pixels"].shape[0], N_TR, replace=False))
    Xtr, ytr = f["pixels"][idx], f["failures"][idx].astype(int)
d = np.load(E.DATA); y = d["failure"].astype(int)
print(f"train frames={N_TR} unsafe={ytr.mean():.3f} | eval poses={len(y)} unsafe={y.mean():.3f} safe frac={1 - y.mean():.3f}", flush=True)
tf = E.get_img_preprocessor(source="pixels", target="pixels", img_size=224)
for name in names:
    m = E.swm.wm.utils.load_pretrained(f"{name}/weights_epoch_50.pt").to(E.DEV).eval()
    def enc(imgs, bs=256):
        out = []
        for i in range(0, len(imgs), bs):
            b = torch.stack([tf({"pixels": imgs[j]})["pixels"] for j in range(i, min(i + bs, len(imgs)))]).to(E.DEV)
            with torch.no_grad(): out.append(m.projector(m.encoder(b, interpolate_pos_encoding=True).last_hidden_state[:, 0]).cpu())
        return torch.cat(out)
    Z = enc(Xtr).to(E.DEV)
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    s_idx, u_idx = np.where(ytr == 0)[0], np.where(ytr == 1)[0]
    head = E.MarginHead(Z.shape[1]).to(E.DEV); opt = torch.optim.AdamW(head.parameters(), lr=E.LR)
    for _ in range(STEPS):
        si, ui = rng.choice(s_idx, E.BS), rng.choice(u_idx, E.BS)
        loss = E.gp_loss(head, Z[si], Z[ui]); opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    res = {ax: E.score(head, enc(d[ax]), y) for ax in ["base"] + E.AXES}
    print(f"BIG {name} " + " ".join(f"{ax}: auc={r[1]:.3f} acc={r[0]:.3f} predsafe={r[2]:.3f} |" for ax, r in res.items()), flush=True)
    del m, Z; torch.cuda.empty_cache()
