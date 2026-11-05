"""Layout-disjoint margin eval: does the safety probe generalize to UNSEEN layouts when trained on many?
Head (sg_point_eval's margin_gp, more steps) trained on N_TR frames of the WM training set sg_point_clean.h5 (~850 random
layouts); scored on ALL 5000 poses of sg_point_multi_eval.npz, a separate collection (seed 321) -> every layout unseen.
Axes: base (same view, new layouts) / color / rotate / shape / agent. AUC + sign acc + predicted-safe frac."""
import os, sys, h5py, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sg_point_eval as E

N_TR = int(os.environ.get("N_TR", "50000")); STEPS = int(os.environ.get("STEPS", "20000")); SEED = int(os.environ.get("SG_SEED", "0"))
HEAD = os.environ.get("HEAD", "mlp")   # mlp = sg_point_eval.MarginHead; linear = one nn.Linear (same gp loss: grad norm = |w| -> GRAD_THR)
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
    def prep(a):   # == training preprocessing: HDF5 loader makes a tensor + permute(0,3,1,2), then ImageNet ToImage (+ Resize 224)
        if a.shape[1] != 224: return torch.stack([tf({"pixels": torch.from_numpy(x[None]).permute(0, 3, 1, 2)})["pixels"][0] for x in a]).to(E.DEV)
        return (torch.as_tensor(a, device=E.DEV).permute(0, 3, 1, 2).float() / 255. - MEAN) / STD
    def enc(imgs, bs=256):
        out = []
        for i in range(0, len(imgs), bs):
            with torch.no_grad(): out.append(m.projector(m.encoder(prep(imgs[i:i + bs]), interpolate_pos_encoding=True).last_hidden_state[:, 0]).cpu())
        return torch.cat(out)
    if not getattr(E, "_prep_checked", False):   # exactness check vs the training preprocessor
        chk = d["base"][:8]; ref = torch.stack([tf({"pixels": torch.from_numpy(x[None]).permute(0, 3, 1, 2)})["pixels"][0] for x in chk]).to(E.DEV)   # tensor input: no ToImage transpose
        err = (prep(chk) - ref).abs().max().item(); print(f"prep check: max|gpu - training preprocessor| = {err:.2e}", flush=True)
        assert err < 1e-4, "fast preprocessing does not match the training preprocessor"; E._prep_checked = True
    Z = enc(Xtr).to(E.DEV)
    Zev = {ax: enc(d[ax]).to(E.DEV) for ax in ["base"] + E.AXES}
    s_idx, u_idx = np.where(ytr == 0)[0], np.where(ytr == 1)[0]
    for hk in (["linear", "mlp"] if HEAD == "both" else [HEAD]):
        torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
        head = (torch.nn.Linear(Z.shape[1], 1) if hk == "linear" else E.MarginHead(Z.shape[1])).to(E.DEV); opt = torch.optim.AdamW(head.parameters(), lr=E.LR)
        for _ in range(STEPS):
            si, ui = rng.choice(s_idx, E.BS), rng.choice(u_idx, E.BS)
            loss = E.gp_loss(head, Z[si], Z[ui]); opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        if os.environ.get("SAVE_HEADS"):   # for CEM: mlp in safe_solver._MarginHead naming ({"margin_gp": ...}), linear as {"linear": ...}
            sd = head.state_dict()
            if hk == "mlp": sd = {"margin_gp": {k.replace("l.l", "layers.Margin GP_linear").replace("l.n", "layers.Margin GP_norm").replace("m.", "mean_layer."): v for k, v in sd.items()}}
            else: sd = {"linear": sd}
            os.makedirs(f"{os.environ['SAVE_HEADS']}/{name}", exist_ok=True); torch.save(sd, f"{os.environ['SAVE_HEADS']}/{name}/{hk}_s{SEED}.pt")
        sub = rng.choice(len(ytr), 5000, replace=False); fit_auc = E.score(head, Z[sub].cpu(), ytr[sub])[1]   # does it fit the training layouts?
        res = {}
        for ax, z in Zev.items():
            acc, auc, ps = E.score(head, z, y)
            with torch.no_grad(): sp = (head(z).cpu().numpy().ravel() >= 0)
            res[ax] = (acc, auc, ps, ((sp & (y == 0)).sum() / (y == 0).sum() + (~sp & (y == 1)).sum() / (y == 1).sum()) / 2)   # balanced sign acc
        print(f"BIG{'_LIN' if hk == 'linear' else ''} {name} train_fit_auc={fit_auc:.3f} | " + " ".join(f"{ax}: auc={r[1]:.3f} acc={r[0]:.3f} predsafe={r[2]:.3f} bal={r[3]:.3f} |" for ax, r in res.items()), flush=True)
    del m, Z, Zev; torch.cuda.empty_cache()
