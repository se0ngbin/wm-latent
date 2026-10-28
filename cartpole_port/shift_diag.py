"""Is an OOD condition's safety info lost or just relocated in the latent? On the test episodes only (cached
re-renders), split episodes A/B and compare, per head:
  in-dist   head trained on native A, scored on native B
  zero-shot head trained on native A, scored on shifted B
  retrained head trained on shifted A, scored on shifted B   (high = info still in the latent)
Heads: margin (failed AUC, z_t) and the supervised V probe (doomed AUC, 3-latent history).
Also the latent displacement |z_shift(s) - z(s)| / mean |z(s1) - z(s2)| (paired states vs random pairs).
Usage: shift_diag.py COND MODEL [MODEL...]"""
import sys, numpy as np, torch, h5py
from eval_cartpole import load_data, cond_pixels, encode_pixels, probe, apply, metrics, H5, DEV, SEEDS
from ood_margin_gp_jepa import train_margin
from utils import get_img_preprocessor
import stable_worldmodel as swm

cond, models = sys.argv[1], sys.argv[2:]
d = load_data(); te = np.where(d["test"])[0]
A = (d["episode_idx"][te] % 2 == 0); B = ~A
pos = np.full(len(d["test"]), -1); pos[te] = np.arange(len(te))
hte = torch.as_tensor(pos[d["hist"][te]])
mar, val, fail = d["margin"][te], d["value"][te], d["failures"][te]
tf = get_img_preprocessor(source="pixels", target="pixels", img_size=224)
sp = cond_pixels(d, cond)
with h5py.File(H5) as f:
    nat_pix = f["pixels"][np.sort(te)]                          # te is sorted already
for name in models:
    m = swm.wm.utils.load_pretrained(f"{name}/weights_epoch_50.pt").to(DEV).eval()
    Zn, Zs = encode_pixels(m, nat_pix, tf), encode_pixels(m, sp, tf)
    del m; torch.cuda.empty_cache()
    Hn, Hs = Zn[hte].flatten(1), Zs[hte].flatten(1)
    rng = np.random.default_rng(0); p = rng.permutation(len(Zn))
    disp = float((Zs - Zn).norm(dim=1).mean() / (Zn - Zn[p]).norm(dim=1).mean())
    out = {}
    for seed in SEEDS:
        mA_n = train_margin(Zn[A], fail[A], "gp", seed=seed); mA_s = train_margin(Zs[A], fail[A], "gp", seed=seed)
        pA_n = probe(Hn[A], val[A], seed); pA_s = probe(Hs[A], val[A], seed)
        for k, (h, X) in dict(m_in=(mA_n, Zn), m_zs=(mA_n, Zs), m_re=(mA_s, Zs)).items():
            out.setdefault(k, []).append(metrics(apply(h, X[B]).flatten().numpy(), mar[B], val[B])["failed_auc"])
        for k, (h, X) in dict(p_in=(pA_n, Hn), p_zs=(pA_n, Hs), p_re=(pA_s, Hs)).items():
            out.setdefault(k, []).append(metrics(apply(h, X[B])[:, 0].numpy(), mar[B], val[B])["doomed_auc"])
    s = " ".join(f"{k}={np.mean(v):.3f}±{np.std(v):.3f}" for k, v in out.items())
    print(f"KEY {cond} {name} disp={disp:.3f} {s}", flush=True)
