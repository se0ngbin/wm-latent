"""Mechanism diagnostics: why does JacobianNormReg help dubins planning?

Compares jacobian_w1_dubins vs sigreg_only_dubins (baseline) on held-out frames:
  1. Encoder Jacobian Frobenius norm  ‖∂z/∂x‖  (did the reg actually bind?)
  2. Jacobian singular-value spectrum (the "matrix": conditioning / anisotropy)
  3. Latent-metric fidelity: corr(‖z_i−z_j‖, ‖s_i−s_j‖)  (the core smoothness claim)
  4. Local sensitivity: latent response to a fixed pixel perturbation
Outputs a PNG + prints a summary table.
"""
import os, sys
sys.path.insert(0, "/home/seongbin/latent/le-wm")
os.environ.setdefault("STABLEWM_HOME", "/data/seongbin/lewm")
os.environ.setdefault("LOCAL_DATASET_DIR", "/data/seongbin/lewm")
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import stable_worldmodel as swm
import stable_pretraining as spt
from utils import get_img_preprocessor

DEV = "cuda:0"
N = 256          # frames for distance/norm stats
N_SVD = 12       # frames for full-Jacobian SVD (expensive)
IMG = 224

ENV = sys.argv[1] if len(sys.argv) > 1 else "dubins"
CFG = {
    "dubins": dict(dataset="dubins_expert.h5", epoch=50,
                   models={"sigreg (baseline)": "sigreg_only_dubins", "jacobian_w1": "jacobian_w1_dubins"}),
    "pusht":  dict(dataset="pusht_expert_train.h5", epoch=10,
                   models={"sigreg (baseline)": "sigreg_pusht_sched10", "jacobian_w1": "lewm_pusht_jac10"}),
    "tworoom": dict(dataset="tworoom.h5", epoch=20, state_key="proprio",
                   models={"sigreg (baseline)": "lewm_tworoom_base", "jacobian_w1": "lewm_tworoom_jac"}),
}[ENV]
STATE_KEY = CFG.get("state_key", "state")

def load_model(name):
    m = swm.wm.utils.load_pretrained(f"{name}/weights_epoch_{CFG['epoch']}.pt")
    return m.to(DEV).eval()

def get_frames(n):
    ds = swm.data.load_dataset(CFG["dataset"], transform=None,
                               cache_dir=os.environ["LOCAL_DATASET_DIR"],
                               num_steps=1, frameskip=1,
                               keys_to_load=["pixels", STATE_KEY], keys_to_cache=[STATE_KEY])
    ds.transform = get_img_preprocessor(source="pixels", target="pixels", img_size=IMG)
    g = np.random.default_rng(0)
    idx = np.sort(g.choice(len(ds), size=n, replace=False))
    px = torch.stack([ds[int(i)]["pixels"][0] for i in idx]).float()   # (n,3,H,W) normalized
    st = torch.stack([torch.as_tensor(ds[int(i)][STATE_KEY][0]) for i in idx]).float()  # (n, S)
    return px, st

def encode(model, x):
    out = model.encoder(x, interpolate_pos_encoding=True)
    return model.projector(out.last_hidden_state[:, 0])

@torch.no_grad()
def embed_all(model, px, bs=64):
    zs = []
    for i in range(0, len(px), bs):
        zs.append(encode(model, px[i:i+bs].to(DEV)).cpu())
    return torch.cat(zs)

@torch.no_grad()
def jac_fro_fd(model, x, sigma=0.05, probes=8):
    # E‖J u‖² = ‖J‖_F²  for u~N(0,I); finite-diff JVP
    x = x.to(DEV)
    z0 = encode(model, x)
    acc = torch.zeros(x.size(0), device=DEV)
    for _ in range(probes):
        u = torch.randn_like(x)
        z1 = encode(model, x + sigma * u)
        acc += (z1 - z0).flatten(1).pow(2).sum(1) / sigma**2
    return (acc / probes).sqrt().cpu()

def full_jacobian(model, x1):  # x1: (1,3,H,W) -> J: (D, 3*H*W)
    x1 = x1.to(DEV).requires_grad_(True)
    from torch.nn.attention import sdpa_kernel, SDPBackend
    with sdpa_kernel(SDPBackend.MATH):
        z = encode(model, x1).squeeze(0)   # (D,)
        rows = []
        for k in range(z.numel()):
            g, = torch.autograd.grad(z[k], x1, retain_graph=True)
            rows.append(g.flatten().detach())
    return torch.stack(rows).cpu()          # (D, 3HW)

@torch.no_grad()
def snr(model, px):
    z = embed_all(model, px)
    sig = (z[1:] - z[:-1]).norm(dim=1).median().item()   # consecutive-frame motion
    x = px[:64].to(DEV)
    noise = (encode(model, x * 1.02) - encode(model, x)).norm(dim=1).median().item()  # 2% brightness
    return sig / (noise + 1e-9)

def main():
    px, st = get_frames(N)
    models = CFG["models"]
    sd = torch.cdist(st, st).numpy(); iu = np.triu_indices(N, 1)
    summary, spectra, fro = {}, {}, {}

    for label, ckpt in models.items():
        m = load_model(ckpt)
        z = embed_all(m, px)
        ld = torch.cdist(z, z).numpy()
        rho = spearmanr(ld[iu], sd[iu]).correlation           # latent-vs-state fidelity
        f = jac_fro_fd(m, px[:64]).numpy()                    # ‖J‖ distribution
        svs = np.stack([np.linalg.svd(full_jacobian(m, px[i:i+1]).numpy(),
                                      compute_uv=False) for i in range(N_SVD)])
        cond = (svs[:, 0] / (svs[:, min(50, svs.shape[1]-1)] + 1e-9)).mean()
        eff_rank = np.mean([(s.sum()**2) / (s**2).sum() for s in svs])  # participation ratio
        snr_v = snr(m, px[:128])
        summary[label] = dict(rho=rho, fro_med=np.median(f), cond=cond, eff_rank=eff_rank, snr=snr_v)
        spectra[label] = svs.mean(0); fro[label] = f
        del m; torch.cuda.empty_cache()

    print(f"\n=== {ENV} ===")
    print(f"{'model':20s} {'‖J‖_F med':>12s} {'cond(J)':>10s} {'eff_rank':>10s} {'SNR':>8s}")
    for k, v in summary.items():
        print(f"{k:20s} {v['fro_med']:12.3f} {v['cond']:10.1f} {v['eff_rank']:10.1f} {v['snr']:8.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for label in models:
        ax[0].hist(fro[label], bins=25, alpha=0.6, label=label)
        ax[1].plot(np.arange(1, len(spectra[label])+1), spectra[label], marker=".", label=label)
    ax[0].set(title=f"[{ENV}] Encoder Jacobian norm ‖J‖_F", xlabel="‖J‖_F", ylabel="frames"); ax[0].legend()
    ax[1].set(title=f"[{ENV}] Jacobian singular spectrum (mean)", xlabel="index", ylabel="σ", yscale="log"); ax[1].legend()
    plt.tight_layout()
    out = f"/home/seongbin/latent/le-wm/outputs/jac_diagnostics_{ENV}.png"
    plt.savefig(out, dpi=120); print("saved", out)

if __name__ == "__main__":
    main()
