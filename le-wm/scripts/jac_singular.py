"""Concrete singular-value stats of the encoder Jacobian, both envs/models.

JacobianNormReg penalizes ‖J‖_F² = Σ σ_i².  It does NOT directly constrain
individual σ_i or the condition number — yet conditioning improves. This
quantifies exactly how the singular spectrum is reshaped.
"""
import os, sys
sys.path.insert(0, "/home/seongbin/latent/le-wm")
os.environ.setdefault("STABLEWM_HOME", "/data/seongbin/lewm")
os.environ.setdefault("LOCAL_DATASET_DIR", "/data/seongbin/lewm")
import numpy as np, torch
import stable_worldmodel as swm
from utils import get_img_preprocessor

DEV, IMG, NF = "cuda:0", 224, 16
CFG = {
    "dubins": dict(ds="dubins_expert.h5", ep=50,
                   models={"baseline": "sigreg_only_dubins", "jacobian": "jacobian_w1_dubins"}),
    "pusht":  dict(ds="pusht_expert_train.h5", ep=10,
                   models={"baseline": "sigreg_pusht_sched10", "jacobian": "lewm_pusht_jac10"}),
    "tworoom": dict(ds="tworoom.h5", ep=20,
                   models={"baseline": "lewm_tworoom_base", "jacobian": "lewm_tworoom_jac"}),
    "dubins_seeds": dict(ds="dubins_expert.h5", ep=50,
                   models={"base_s3072": "sigreg_only_dubins",
                           "jac_s3072": "jacobian_w1_dubins",
                           "jac_s100": "lewm_dubins_jacw1_s100",
                           "jac_s200": "lewm_dubins_jacw1_s200"}),
    "acmtm_dubins": dict(ds="dubins_expert.h5", ep=50,
                   models={"baseline": "sigreg_only_dubins", "jacobian": "jacobian_w1_dubins",
                           "jac+pull": "lewm_dubins_jacpull50", "acmtm": "dubins_acmtm",
                           "acmtm+jac": "dubins_acmtm_jac"}),
}
if len(sys.argv) > 1:   # restrict to the named CFG entries, e.g. `jac_singular.py acmtm_dubins`
    CFG = {k: CFG[k] for k in sys.argv[1:]}

def encode(m, x):
    return m.projector(m.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0])

def frames(name, n):
    ds = swm.data.load_dataset(name, transform=None, cache_dir=os.environ["LOCAL_DATASET_DIR"],
                               num_steps=1, frameskip=1, keys_to_load=["pixels"], keys_to_cache=[])
    ds.transform = get_img_preprocessor(source="pixels", target="pixels", img_size=IMG)
    idx = np.random.default_rng(0).choice(len(ds), n, replace=False)
    return torch.stack([ds[int(i)]["pixels"][0] for i in idx]).float()

def jac_svd(m, x1):
    from torch.nn.attention import sdpa_kernel, SDPBackend
    x1 = x1.to(DEV).requires_grad_(True)
    with sdpa_kernel(SDPBackend.MATH):
        z = encode(m, x1).squeeze(0)
        J = torch.stack([torch.autograd.grad(z[k], x1, retain_graph=True)[0].flatten().detach()
                         for k in range(z.numel())])
    return torch.linalg.svdvals(J.float()).cpu().numpy()

print(f"{'env/model':22s} {'σ1':>7s} {'σ2':>7s} {'σ5':>7s} {'σ10':>7s} {'‖J‖_F':>7s} "
      f"{'σ1/‖J‖_F':>9s} {'top1%E':>7s} {'top5%E':>7s} {'#σ>0.1σ1':>9s}")
for env, c in CFG.items():
    for label, ckpt in c["models"].items():
        m = swm.wm.utils.load_pretrained(f"{ckpt}/weights_epoch_{c['ep']}.pt").to(DEV).eval()
        px = frames(c["ds"], NF)
        S = np.stack([jac_svd(m, px[i:i+1]) for i in range(NF)])  # (NF, D)
        s = S.mean(0)
        fro = np.sqrt((s**2).sum())
        E = (s**2) / (s**2).sum()
        top1 = E[0]; top5 = E[:5].sum()
        n_signif = int((s > 0.1 * s[0]).sum())
        print(f"{env+'/'+label:22s} {s[0]:7.3f} {s[1]:7.3f} {s[4]:7.3f} {s[9]:7.3f} {fro:7.3f} "
              f"{s[0]/fro:9.3f} {top1:7.3f} {top5:7.3f} {n_signif:9d}  effrank={s.sum()**2/(s**2).sum():6.1f}")
        del m; torch.cuda.empty_cache()
