"""Planning-relevant diagnostic: does latent cost track true progress-to-goal?

CEM ranks action candidates by ‖z_pred − z_goal‖. It only works if that latent
cost decreases (smoothly, monotonically) as the agent's true position approaches
the goal. We test this directly on expert trajectories: goal = final frame, then
plot/measure ‖z_t − z_goal‖ vs true position-distance ‖pos_t − pos_goal‖.

Metrics per model (over K episodes):
  - Spearman(cost, true_dist)   : does cost track progress? (higher = better)
  - roughness = mean|Δ²cost|/range : jaggedness of the cost curve (lower = better)
  - SNR = ‖Δz‖(real 1-step move) / ‖Δz‖(rendering nuisance) : signal vs amplified noise
"""
import os, sys
sys.path.insert(0, "/home/seongbin/latent/le-wm")
os.environ.setdefault("STABLEWM_HOME", "/data/seongbin/lewm")
os.environ.setdefault("LOCAL_DATASET_DIR", "/data/seongbin/lewm")
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import stable_worldmodel as swm
from utils import get_img_preprocessor

DEV, IMG, K, MINLEN = "cuda:0", 224, 40, 30

def load_model(name):
    return swm.wm.utils.load_pretrained(f"{name}/weights_epoch_50.pt").to(DEV).eval()

def encode(m, x):
    return m.projector(m.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0])

def get_episodes():
    ds = swm.data.load_dataset("dubins_expert.h5", transform=None, cache_dir=os.environ["LOCAL_DATASET_DIR"],
                               num_steps=1, frameskip=1, keys_to_load=["pixels", "state"], keys_to_cache=["state"])
    ds.transform = get_img_preprocessor(source="pixels", target="pixels", img_size=IMG)
    import h5py
    with h5py.File(os.path.join(os.environ["LOCAL_DATASET_DIR"], "datasets/dubins_expert.h5"), "r") as f:
        ep = np.asarray(f["episode_idx"]).reshape(-1)
    uniq, first, cnt = np.unique(ep, return_index=True, return_counts=True)
    order = np.argsort(first); uniq, first, cnt = uniq[order], first[order], cnt[order]
    g = np.random.default_rng(0); pick = g.choice(np.where(cnt >= MINLEN)[0], K, replace=False)
    eps = []
    for p in pick:
        rows = list(range(first[p], first[p] + cnt[p]))
        px = torch.stack([ds[r]["pixels"][0] for r in rows]).float()
        st = torch.stack([torch.as_tensor(ds[r]["state"][0]) for r in rows]).float()
        eps.append((px, st))
    return eps

@torch.no_grad()
def embed(m, px, bs=64):
    return torch.cat([encode(m, px[i:i+bs].to(DEV)).cpu() for i in range(0, len(px), bs)])

@torch.no_grad()
def snr(m, px):
    # signal: consecutive-frame latent change (real 1-step motion)
    z = embed(m, px)
    sig = (z[1:] - z[:-1]).norm(dim=1).median().item()
    # nuisance: same frames, tiny brightness perturbation (rendering-scale, no state change)
    x = px[:64].to(DEV)
    zn = encode(m, x); zp = encode(m, (x * 1.02))
    noise = (zp - zn).norm(dim=1).median().item()
    return sig / (noise + 1e-9)

def main():
    eps = get_episodes()
    models = {"sigreg (baseline)": "sigreg_only_dubins", "jacobian_w1": "jacobian_w1_dubins"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for j, (label, ckpt) in enumerate(models.items()):
        m = load_model(ckpt)
        rhos, roughs, curves = [], [], []
        for px, st in eps:
            z = embed(m, px).numpy(); pos = st[:, :2].numpy()
            cost = np.linalg.norm(z - z[-1], axis=1)
            td = np.linalg.norm(pos - pos[-1], axis=1)
            rhos.append(spearmanr(cost, td).correlation)
            c = cost / (cost.max() + 1e-9)
            roughs.append(np.abs(np.diff(c, 2)).mean())
            order = np.argsort(td); curves.append((td[order], c[order]))
        snr_v = snr(m, eps[0][0])
        print(f"{label:20s}  Spearman(cost,truedist)={np.nanmean(rhos):.3f}  "
              f"roughness={np.mean(roughs):.4f}  SNR(signal/nuisance)={snr_v:.2f}")
        # left: representative cost curves; right: aggregate
        for td, c in curves[:8]:
            ax[0].plot(td, c, color=f"C{j}", alpha=0.35, lw=1)
        allx = np.concatenate([c[0] for c in curves]); ally = np.concatenate([c[1] for c in curves])
        b = np.linspace(0, allx.max(), 12); bi = np.digitize(allx, b)
        mean = [ally[bi == k].mean() for k in range(1, len(b))]
        ax[1].plot(0.5*(b[1:]+b[:-1]), mean, color=f"C{j}", marker="o", label=label)
        del m; torch.cuda.empty_cache()
    ax[0].set(title="per-episode: normalized latent cost vs true dist-to-goal",
              xlabel="‖pos − goal‖ (true)", ylabel="normalized ‖z − z_goal‖")
    ax[1].set(title="binned mean (monotone ⇒ CEM-usable)", xlabel="‖pos − goal‖", ylabel="norm. cost"); ax[1].legend()
    plt.tight_layout()
    out = "/home/seongbin/latent/le-wm/outputs/jac_costsurface.png"
    plt.savefig(out, dpi=120); print("saved", out)

if __name__ == "__main__":
    main()
