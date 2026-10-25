"""Is the clean-Point 'rotate' failure a viewpoint problem or a NEW-LAYOUT problem?
A 90deg camera turn about world-z renders like the base camera on a rotated layout. Head = sg_point_eval's margin_gp,
trained on base-view train frames of sg_point_multi_eval.npz (test frames share layouts with them: shuffled split).
Scored on: base test frames (shared layouts) | rotate test frames | base-view frames from FRESH layouts (sg_point_clean.h5,
different episodes) | a head retrained on rotate train frames (is the info there at all?)."""
import os, sys, h5py, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sg_point_eval as E

N_FRESH = 5000
names = os.environ.get("DIAG_ENCODERS", "pt_baseline,pt_jacobian,pt_jacpull,pt_jacpull_safeadv_linf,pt_jac_lam_bdet").split(",")
d = np.load(E.DATA); y = d["failure"].astype(int); N = len(y); n_test = int(N * E.TEST_FRAC)
tr, te = np.arange(N - n_test), np.arange(N - n_test, N)
with h5py.File("/data/seongbin/lewm/datasets/sg_point_clean.h5", "r") as f:
    idx = np.sort(np.random.default_rng(0).choice(f["pixels"].shape[0], N_FRESH, replace=False))
    fresh, yf = f["pixels"][idx], f["failures"][idx].astype(int)
print(f"fresh-layout frames n={len(yf)} unsafe={yf.mean():.3f}", flush=True)
tf = E.get_img_preprocessor(source="pixels", target="pixels", img_size=224)

def train_head(Z, yy, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); Z = Z.to(E.DEV)
    s_idx, u_idx = np.where(yy == 0)[0], np.where(yy == 1)[0]
    head = E.MarginHead(Z.shape[1]).to(E.DEV); opt = torch.optim.AdamW(head.parameters(), lr=E.LR)
    for _ in range(E.STEPS):
        si, ui = rng.choice(s_idx, E.BS), rng.choice(u_idx, E.BS)
        loss = E.gp_loss(head, Z[si], Z[ui]); opt.zero_grad(); loss.backward(); opt.step()
    return head.eval()

for name in names:
    m = E.swm.wm.utils.load_pretrained(f"{name}/weights_epoch_50.pt").to(E.DEV).eval()
    def enc(imgs, bs=128):
        out = []
        for i in range(0, len(imgs), bs):
            b = torch.stack([tf({"pixels": imgs[j]})["pixels"] for j in range(i, min(i + bs, len(imgs)))]).to(E.DEV)
            with torch.no_grad(): out.append(m.projector(m.encoder(b, interpolate_pos_encoding=True).last_hidden_state[:, 0]).cpu())
        return torch.cat(out)
    Zb, Zr, Zf = enc(d["base"]), enc(d["rotate"]), enc(fresh)
    hb = train_head(Zb[tr], y[tr]); hr = train_head(Zr[tr], y[tr])
    _, a_in, _ = E.score(hb, Zb[te], y[te]); _, a_rot, _ = E.score(hb, Zr[te], y[te])
    _, a_fresh, p_fresh = E.score(hb, Zf, yf); _, a_rr, _ = E.score(hr, Zr[te], y[te])
    # fresh layouts, head trained ON fresh-layout-free data but evaluated by held-out EPISODES of the fresh set itself
    hf = train_head(Zf[:4000], yf[:4000]); _, a_ff, _ = E.score(hf, Zf[4000:], yf[4000:])
    print(f"DIAG {name}: base->base(shared layouts)={a_in:.3f}  base->rotate={a_rot:.3f}  base->FRESH layouts={a_fresh:.3f} "
          f"(predsafe {p_fresh:.3f})  rotate->rotate={a_rr:.3f}  fresh->fresh(random split)={a_ff:.3f}", flush=True)
    del m; torch.cuda.empty_cache()
