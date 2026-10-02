"""Nuisance-subspace projection at the readout (lever #3) + per-latent-space
safe/unsafe separability.

Idea: the appearance shift D = z(new) - z(og) spans a small 'nuisance subspace'.
Estimate it (top-k right-singular vectors of the centered D cloud), build the
projector P = I - U U^T, and re-score the FROZEN margin head on the orthogonal
complement:  h_P(z) = head(P z).  If projecting out the recolor direction
recovers OOD AUC WITHOUT hurting in-dist AUC, then the nuisance really was a
separable off-axis direction (color entangled but removable). If in-dist AUC
also collapses, color is genuinely fused with the safety signal.

Two bases compared:
  mean  : k=1 along the global recolor offset mu = mean(D)   (bias-fixable part)
  svd-k : top-k of the full D cloud (offset + state-dependent deformation)

Separability block (axis-independent, uses og/in-dist features Z0):
  probe_auc/acc : 5-fold linear (logistic) probe safe-vs-unsafe on Z0
  fisher        : ||mu_u - mu_s||^2 / mean within-class variance (trace)
  classgap_sig  : ||mu_u - mu_s|| / sigma_pairwise(Z0)   (scale-free gap)
  head_auc      : the DEPLOYED margin head's own in-dist AUC on Z0
Usage mirrors latent_probe_matrix.sh's arg set.
"""
import sys, argparse
from pathlib import Path
import numpy as np, torch
_THIS = Path(__file__).resolve()
for p in [str(_THIS.parents[3]), str(_THIS.parents[2]), str(_THIS.parents[1]), str(_THIS.parent)]:
    if p not in sys.path: sys.path.append(p)
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from latent_shift_probe import run_lewm, run_dreamer, sigma_pairwise, DEV


def nuisance_basis(D, k, mode):
    Dc = D - D.mean(0)
    if mode == "mean":
        mu = D.mean(0); n = np.linalg.norm(mu) + 1e-12
        return (mu / n)[None, :]                 # 1 x d
    _, _, Vt = np.linalg.svd(Dc, full_matrices=False)
    return Vt[:k]                                # k x d, orthonormal rows


def project_out(Z, U):
    return Z - (Z @ U.T) @ U                     # remove component in span(U)


def head_score(head, Z):
    with torch.no_grad():
        return head(torch.tensor(Z, dtype=torch.float32, device=DEV)).view(-1).cpu().numpy()


def auc_safe(h, y):
    safe = (y == 0).astype(int)                  # margin >= 0 => safe
    return roc_auc_score(safe, h) if 0 < safe.mean() < 1 else float("nan")


def separability(Z0, y, h0=None, h1=None):
    """Existence-of-boundary (probe_auc) + how EASY/wide the separation is.
    head_dprime : (mean h_safe - mean h_unsafe) / pooled_std   -- margin in the
                  head's own output, in units of its spread (bigger = easier).
    nearbdry    : frac of points with |h0| below the 25th pctile of |h0|, i.e.
                  mass sitting on the fence (smaller = more clearance).
    robust_ratio: perturbation move in head output / the safe-unsafe head gap.
                  >1 means the nuisance shoves the margin further than the whole
                  class separation -> it can flip points (fragile)."""
    safe = (y == 0).astype(int)
    out = dict(probe_auc=float("nan"), probe_acc=float("nan"), fisher=float("nan"),
               classgap_sig=float("nan"), head_dprime=float("nan"),
               nearbdry=float("nan"), robust_ratio=float("nan"))
    if not (0 < safe.mean() < 1):
        return out
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    p = cross_val_predict(clf, Z0, safe, cv=5, method="predict_proba")[:, 1]
    out["probe_auc"] = float(roc_auc_score(safe, p))
    out["probe_acc"] = float(((p >= 0.5).astype(int) == safe).mean())
    mu_s, mu_u = Z0[y == 0].mean(0), Z0[y == 1].mean(0)
    within = Z0[y == 0].var(0).sum() + Z0[y == 1].var(0).sum()
    out["fisher"] = float((np.linalg.norm(mu_u - mu_s) ** 2) / (within + 1e-12))
    out["classgap_sig"] = float(np.linalg.norm(mu_u - mu_s) / sigma_pairwise(Z0))
    if h0 is not None:
        hs, hu = h0[y == 0], h0[y == 1]
        gap = float(hs.mean() - hu.mean())                      # safe scores higher
        pooled = float(np.sqrt(0.5 * (hs.var() + hu.var())) + 1e-12)
        out["head_dprime"] = gap / pooled
        eps = float(np.quantile(np.abs(h0), 0.25))
        out["nearbdry"] = float((np.abs(h0) < max(eps, 1e-6)).mean())
        if h1 is not None:
            out["robust_ratio"] = float(np.abs(h1 - h0).mean() / (abs(gap) + 1e-12))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["lewm", "dreamer"])
    ap.add_argument("--enc"); ap.add_argument("--margin"); ap.add_argument("--rssm")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--axis", default="color", choices=["color", "shape", "rotate"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ks", default="1,2,4,8")
    a = ap.parse_args()

    if a.backend == "lewm":
        _, Z0, Z1, y, head = run_lewm(a.enc, a.margin, a.axis, a.seed)
    else:
        _, Z0, Z1, y, head = run_dreamer(a.rssm, a.axis, a.seed)

    # separability of safe/unsafe in this latent space (in-dist features)
    h0 = head_score(head, Z0); h1 = head_score(head, Z1)
    sep = separability(Z0, y, h0=h0, h1=h1)
    sep["head_auc"] = auc_safe(h0, y)
    print(f"SEP {a.tag} axis={a.axis} dim={Z0.shape[1]} " +
          " ".join(f"{k}={v:.3f}" for k, v in sep.items()))

    # baseline (no projection) OOD/in-dist AUC
    base_in, base_ood = auc_safe(h0, y), auc_safe(h1, y)
    print(f"PROJ {a.tag} axis={a.axis} mode=none k=0 in_auc={base_in:.3f} ood_auc={base_ood:.3f}")

    D = Z1 - Z0
    for mode, k in [("mean", 1)] + [("svd", int(k)) for k in a.ks.split(",")]:
        U = nuisance_basis(D, k, mode)
        in_auc = auc_safe(head_score(head, project_out(Z0, U)), y)
        ood_auc = auc_safe(head_score(head, project_out(Z1, U)), y)
        print(f"PROJ {a.tag} axis={a.axis} mode={mode} k={U.shape[0]} "
              f"in_auc={in_auc:.3f} ood_auc={ood_auc:.3f} "
              f"d_ood={ood_auc - base_ood:+.3f} d_in={in_auc - base_in:+.3f}")


if __name__ == "__main__":
    main()
