"""Nuisance-projection recovery, margin vs value function, on the color axis.
For each encoder: color-OOD AUC and cos_la BEFORE (hollow) and AFTER (filled)
projecting out the color subspace, connected by a DIAGONAL arrow. Both endpoints
recomputed in the same space so the arrow shows how projection moves BOTH the
alignment (x) and the classifier (y). Two recovery mechanisms are visible:
 - Dreamer margin: mean-mode kills the global color offset -> cos_la collapses
   (0.82->0.05) yet AUC jumps (removing the offset that dominated the shift).
 - baseline margin: svd-mode strips the color DEFORMATION -> residual shift is
   MORE aligned (0.71->0.90) and AUC recovers.
The critic barely moves either way (projection does not recover the value fn).
x = cos_la (margin: failure axis; value: V* axis)."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (encoder, marker, cos_la_orig, cos_la_proj, AUC_orig, AUC_proj)  -- cos_la_proj
# from compute_proj_cosla.py (best projection per cell); no-proj cells repeat cos_la.
MARGIN = [
    ("baseline", "o", 0.710, 0.899, 0.427, 0.671),   # svd k2
    ("jacobian", "s", 0.670, 0.670, 0.606, 0.606),   # none helps
    ("jac+pull", "^", 0.636, 0.631, 0.689, 0.698),   # svd k2 (tiny)
    ("dreamer",  "D", 0.821, 0.052, 0.565, 0.796),   # mean k1
]
VALUE = [  # dreamer critic-projection needs a dreamer script (lewm-only) -> omitted
    ("baseline", "o", 0.621, 0.621, 0.643, 0.643),   # none helps
    ("jacobian", "s", 0.574, 0.605, 0.619, 0.655),   # svd k4
    ("jac+pull", "^", 0.610, 0.671, 0.204, 0.234),   # svd k4
]
MK = {"baseline": "o", "jacobian": "s", "jac+pull": "^", "dreamer": "D"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.0), sharey=True)
for ax, data, title in [(ax1, MARGIN, "Margin"), (ax2, VALUE, "Value function (critic)")]:
    for enc, mk, xo, xp, o, p in data:
        up = p - o > 0.02
        c = "#2ca02c" if up else "#999999"
        if abs(p - o) > 0.01 or abs(xp - xo) > 0.01:
            ax.annotate("", xy=(xp, p), xytext=(xo, o),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8, alpha=.9))
        ax.scatter(xo, o, s=130, marker=mk, facecolors="white", edgecolors="k", linewidths=1.3, zorder=3)
        ax.scatter(xp, p, s=130, marker=mk, facecolors=c, edgecolors="k", linewidths=0.8, zorder=4)
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.text(ax.get_xlim()[1], 0.5, " chance / inverts below", fontsize=8, c="gray", va="bottom", ha="right")
    ax.set_xlabel("cos_la  (shift alignment with safe→unsafe axis)", fontsize=9.5)
    ax.set_title(title, fontsize=11); ax.grid(alpha=0.25); ax.set_xlim(-0.02, 0.98)
ax1.set_ylabel("color-OOD AUC", fontsize=11); ax1.set_ylim(0.15, 0.9)
handles = [plt.Line2D([0], [0], marker=MK[e], color="w", markerfacecolor="#bbb",
           markeredgecolor="k", markersize=10, label=e) for e in MK]
handles += [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="k",
            markersize=10, label="hollow = original"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", markeredgecolor="k",
            markersize=10, label="filled = after projection")]
fig.suptitle("Nuisance projection moves alignment AND AUC: Dreamer kills the offset (left), baseline strips deformation (right)",
             y=0.995, fontsize=11.5)
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=6, fontsize=8.5, frameon=False)
fig.tight_layout(rect=[0, 0, 1, 0.86])
out = "/home/seongbin/latent/figs/proj_recovery.png"
fig.savefig(out, dpi=130); print("saved", out)
