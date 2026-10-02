"""Spectral placement of the safety function ell in LE-WM latent space.

Question: is ell(x) = distance-to-obstacle a LOW-FREQUENCY function of the
latent manifold geometry?

  1. Collect held-out Dubins rollouts once (frames + privileged states).
  2. Per checkpoint: encode -> latent point cloud {z_i} with labels ell(x_i).
  3. Graph Laplacian on the latents: kNN graph with self-tuned Gaussian
     weights (Zelnik-Manor), UNION with temporal transition edges (z_t,z_t+1),
     symmetric normalized L = I - D^-1/2 W D^-1/2.
  4. Bottom m eigenfunctions (skip the trivial constant); regress ell onto
     them; report R^2 for m in {5, 10, 20, 50}.
  5. Controls: R^2 of ell on top-10 PCA of z, and on random 10-dim
     projections of z (linear decodability =/= spectral smoothness).

Read-out:
  R^2(10) high (>~0.8)  -> ell is low-frequency: anisotropic regularization of
                           head-gradient / predictor gain in high modes has a
                           concrete basis.
  R^2(10) low           -> the ENCODER scrambled ell into high frequencies;
                           certifiability dies at the encoder.

Usage (from latent_cbf/):
  STABLEWM_HOME=/data/seongbin/lewm .venv/bin/python \
      src/latent_cbf/scripts/spectral_ell_analysis.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

_THIS = Path(__file__).resolve()
for p in [str(_THIS.parents[3]), str(_THIS.parents[2]), str(_THIS.parents[1])]:
    if p not in sys.path:
        sys.path.append(p)
for _p in ("/home/seongbin/latent", "/home/seongbin/latent/le-wm"):
    if _p not in sys.path:
        sys.path.append(_p)

from configs import get_default_config
from scripts.run_experiment import create_env_from_config
from measure_unmodeled_error import collect_episodes, ell


def build_laplacian(Z: np.ndarray, edges_t: np.ndarray, knn: int = 10):
    """Self-tuned Gaussian kNN graph + transition edges; symmetric normalized
    Laplacian. Z: (n, d); edges_t: (m, 2) temporal pairs."""
    from sklearn.neighbors import NearestNeighbors

    n = Z.shape[0]
    nn = NearestNeighbors(n_neighbors=knn + 1).fit(Z)
    dist, idx = nn.kneighbors(Z)          # includes self at col 0
    sigma = dist[:, min(7, knn)]          # local scale: dist to 7th neighbor
    sigma = np.maximum(sigma, 1e-12)

    rows = np.repeat(np.arange(n), knn)
    cols = idx[:, 1:].ravel()
    d2 = (dist[:, 1:] ** 2).ravel()
    w = np.exp(-d2 / (sigma[rows] * sigma[cols]))

    # temporal transition edges with the same kernel
    tr, tc = edges_t[:, 0], edges_t[:, 1]
    d2t = ((Z[tr] - Z[tc]) ** 2).sum(1)
    wt = np.exp(-d2t / (sigma[tr] * sigma[tc]))

    W = sp.coo_matrix(
        (np.concatenate([w, wt]),
         (np.concatenate([rows, tr]), np.concatenate([cols, tc]))),
        shape=(n, n),
    ).tocsr()
    W = W.maximum(W.T)                    # symmetrize (union)
    # restrict to the giant connected component: otherwise the bottom
    # eigenvectors are component indicators, not smooth modes.
    ncomp, labels = sp.csgraph.connected_components(W, directed=False)
    keep = np.arange(n)
    if ncomp > 1:
        giant = np.bincount(labels).argmax()
        keep = np.where(labels == giant)[0]
        W = W[keep][:, keep]
        print(f"  graph had {ncomp} components; keeping giant "
              f"({len(keep)}/{n} = {len(keep)/n:.1%} of nodes)")
    m = W.shape[0]
    d = np.asarray(W.sum(1)).ravel()
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    Dis = sp.diags(d_inv_sqrt)
    return sp.identity(m) - Dis @ W @ Dis, keep


def r2(y: np.ndarray, X: np.ndarray) -> float:
    """R^2 of least-squares regression of y on X (with intercept)."""
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    res = y - X1 @ beta
    return float(1.0 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", default=[
        "sigreg_only_dubins/weights_epoch_50.pt",
        "pixel_lipschitz_dubins/weights_epoch_50.pt",
        "jacobian_w1_dubins/weights_epoch_50.pt",
    ])
    ap.add_argument("--n_eps", type=int, default=100)
    ap.add_argument("--ep_len", type=int, default=40)
    ap.add_argument("--knn", type=int, default=10)
    ap.add_argument("--n_modes", type=int, default=51)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_prefix",
                    default=str(_THIS.parents[3] / "results" / "spectral_ell"))
    args = ap.parse_args()

    # ---- collect ONE shared set of held-out rollouts ----
    env = create_env_from_config(get_default_config())
    rng = np.random.default_rng(args.seed)
    print("collecting shared held-out rollouts...")
    episodes = collect_episodes(env, args.n_eps, args.ep_len, rng)
    env.close()
    all_states, ep_slices = [], []
    off = 0
    for frames, states, _ in episodes:
        all_states.append(states)
        ep_slices.append((off, off + len(states)))
        off += len(states)
    states_all = np.concatenate(all_states)
    ell_all = ell(states_all)
    edges_t = np.concatenate(
        [np.stack([np.arange(a, b - 1), np.arange(a + 1, b)], 1)
         for a, b in ep_slices])
    print(f"{len(states_all)} latent nodes, {len(edges_t)} transition edges")

    dc_mod = __import__("configs", fromlist=["DreamerConfig"])
    results = {}
    ms = [5, 10, 20, 50]
    for ckpt in args.ckpts:
        dcfg = dc_mod.DreamerConfig()
        dcfg.device = args.device
        from latent_cbf.adapters import LEWMWorldModel

        wm = LEWMWorldModel(dcfg, ckpt).to(args.device).eval()
        Zs = []
        with torch.no_grad():
            for frames, _, _ in episodes:
                data = wm.preprocess({"image": torch.from_numpy(frames[None]).float()})
                Zs.append(wm.encoder(data)[0].cpu().numpy())
        del wm
        torch.cuda.empty_cache()
        Z = np.concatenate(Zs).astype(np.float64)

        print(f"\n=== {ckpt} ===  building Laplacian...")
        L, keep = build_laplacian(Z, edges_t, knn=args.knn)
        y = ell_all[keep]
        vals, vecs = spla.eigsh(L.tocsc(), k=args.n_modes, sigma=0, which="LM")
        order = np.argsort(vals)
        vals, vecs = vals[order], vecs[:, order]
        print("eigenvalues[:12]:", np.round(vals[:12], 5))

        row = {"eigenvalues": vals.tolist(),
               "giant_component_frac": float(len(keep) / len(ell_all))}
        for m in ms:
            row[f"R2_bottom{m}"] = r2(y, vecs[:, 1 : m + 1])
        # controls (on the same node subset)
        Zk = Z[keep]
        Zc = Zk - Zk.mean(0)
        _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
        row["R2_pca10"] = r2(y, Zc @ Vt[:10].T)
        rp = [r2(y, Zc @ np.random.default_rng(s).standard_normal((Z.shape[1], 10)))
              for s in range(5)]
        row["R2_randproj10_mean"] = float(np.mean(rp))
        results[ckpt] = row
        print("  ".join(f"R2@{m}={row[f'R2_bottom{m}']:.3f}" for m in ms)
              + f"  |  PCA10={row['R2_pca10']:.3f}  randproj10={row['R2_randproj10_mean']:.3f}")

    out_json = f"{args.out_prefix}.json"
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    grid = np.arange(1, args.n_modes)
    for ckpt, row in results.items():
        # cumulative R^2 vs #modes (recompute per point, cheap)
        tag = Path(ckpt).parts[0]
        r2s = [row[f"R2_bottom{m}"] for m in ms]
        ax.plot(ms, r2s, marker="o", label=tag)
    ax.set_xlabel("# bottom Laplacian eigenfunctions")
    ax.set_ylabel("R² of ell (distance-to-obstacle)")
    ax.set_ylim(0, 1)
    ax.axhline(0.8, color="gray", lw=0.8, ls="--")
    ax.legend()
    ax.set_title("spectral placement of ell in latent space")
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}.png", dpi=130)
    print(f"\nsaved {out_json}\nsaved {args.out_prefix}.png")


if __name__ == "__main__":
    main()
