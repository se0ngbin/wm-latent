"""Layout-disjoint margin eval: does the safety probe generalize to UNSEEN layouts when trained on many?
Head (sg_point_eval's margin_gp, more steps) trained on N_TR frames of the WM training set sg_point_clean.h5 (~850 random
layouts); scored on ALL 5000 poses of sg_point_multi_eval.npz, a separate collection (seed 321) -> every layout unseen.
Axes: base (same view, new layouts) / color / rotate / shape / agent. AUC + sign acc + predicted-safe frac."""
import os, sys, h5py, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sg_point_eval as E

N_TR = int(os.environ.get("N_TR", "50000")); STEPS = int(os.environ.get("STEPS", "20000")); SEED = int(os.environ.get("SG_SEED", "0"))
names = os.environ.get("DIAG_ENCODERS", "pt_baseline,pt_jacobian,pt_jacpull,pt_jacpull_safeadv_linf").split(",")
TRAIN_H5 = os.environ.get("TRAIN_H5", "/data/seongbin/lewm/datasets/sg_point_clean.h5")   # eval set = SG_DATA (sg_point_eval.DATA)
with h5py.File(TRAIN_H5, "r") as f:
    idx = np.sort(np.random.default_rng(SEED).choice(f["pixels"].shape[0], N_TR, replace=False))
    Xtr, ytr = f["pixels"][idx], f["failures"][idx].astype(int)
d = np.load(E.DATA); y = d["failure"].astype(int)
d = {k: d[k] for k in ["base"] + E.AXES}   # decompress each view once (npz access re-decompresses on every read)
print(f"train frames={N_TR} unsafe={ytr.mean():.3f} | eval poses={len(y)} unsafe={y.mean():.3f} safe frac={1 - y.mean():.3f}", flush=True)
tf = E.get_img_preprocessor(source="pixels", target="pixels", img_size=224)
for name in names:
    m = E.swm.wm.utils.load_pretrained(f"{name}/weights_epoch_50.pt").to(E.DEV).eval()
    MEAN = torch.tensor([0.485, 0.456, 0.406], device=E.DEV).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=E.DEV).view(1, 3, 1, 1)
    def prep(a):   # == get_img_preprocessor (ImageNet ToImage + Resize 224) for frames already at 224: batched on GPU
        if a.shape[1] != 224: return torch.stack([tf({"pixels": x})["pixels"] for x in a]).to(E.DEV)   # per-frame CPU path
        # (0,3,2,1), not (0,3,1,2): the training ToImage yields W/H transposed relative to the HWC frame (checked: 7e-7 vs 4.1)
        return (torch.as_tensor(a, device=E.DEV).permute(0, 3, 2, 1).float() / 255. - MEAN) / STD
    def enc(imgs, bs=256):
        out = []
        for i in range(0, len(imgs), bs):
            with torch.no_grad(): out.append(m.projector(m.encoder(prep(imgs[i:i + bs]), interpolate_pos_encoding=True).last_hidden_state[:, 0]).cpu())
        return torch.cat(out)
    if not getattr(E, "_prep_checked", False):   # exactness check vs the training preprocessor
        chk = d["base"][:8]; ref = torch.stack([tf({"pixels": x})["pixels"] for x in chk]).to(E.DEV)
        err = (prep(chk) - ref).abs().max().item(); print(f"prep check: max|gpu - training preprocessor| = {err:.2e}", flush=True)
        assert err < 1e-4, "fast preprocessing does not match the training preprocessor"; E._prep_checked = True
    Z = enc(Xtr).to(E.DEV)
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    s_idx, u_idx = np.where(ytr == 0)[0], np.where(ytr == 1)[0]
    head = E.MarginHead(Z.shape[1]).to(E.DEV); opt = torch.optim.AdamW(head.parameters(), lr=E.LR)
    for _ in range(STEPS):
        si, ui = rng.choice(s_idx, E.BS), rng.choice(u_idx, E.BS)
        loss = E.gp_loss(head, Z[si], Z[ui]); opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    sub = rng.choice(len(ytr), 5000, replace=False); fit_auc = E.score(head, Z[sub].cpu(), ytr[sub])[1]   # does it fit the training layouts?
    res = {ax: E.score(head, enc(d[ax]), y) for ax in ["base"] + E.AXES}
    print(f"BIG {name} train_fit_auc={fit_auc:.3f} | " + " ".join(f"{ax}: auc={r[1]:.3f} acc={r[0]:.3f} predsafe={r[2]:.3f} |" for ax, r in res.items()), flush=True)
    del m, Z; torch.cuda.empty_cache()
