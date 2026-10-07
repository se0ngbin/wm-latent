"""Summary figure for the latent-shift probe: zero-shot survival (frozen-margin AUC)
vs (left) displacement and (right) alignment with the safe/unsafe axis. Makes the
'direction, not distance' point visually — left is a flat cloud (distance doesn't
predict), right trends down (alignment does). Data are the 12 probe cells
(4 encoders x 3 axes); jacpull-color uses the retrained head (zs_auc .69)."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (encoder, axis, displ, cos_la, zs_auc)
D = [
    ("baseline", "color", 1.365, 0.720, 0.427), ("jacobian", "color", 0.820, 0.666, 0.606),
    ("jac+pull", "color", 0.782, 0.656, 0.689), ("dreamer", "color", 0.768, 0.820, 0.566),
    ("baseline", "shape", 0.742, 0.892, 0.613), ("jacobian", "shape", 0.113, 0.279, 0.996),
    ("jac+pull", "shape", 0.071, 0.443, 0.995), ("dreamer", "shape", 0.041, 0.266, 0.994),
    ("baseline", "rotate", 1.005, 0.281, 0.929), ("jacobian", "rotate", 0.970, 0.045, 0.987),
    ("jac+pull", "rotate", 0.957, 0.129, 0.975), ("dreamer", "rotate", 0.854, 0.387, 0.433),
]
# axis -> COLOR, encoder -> MARKER SHAPE (so no text labels needed)
AXCOL = {"color": "#d62728", "shape": "#2ca02c", "rotate": "#1f77b4"}
ENCMK = {"baseline": "o", "jacobian": "s", "jac+pull": "^", "dreamer": "D"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.0), sharey=True)
for ax, use_cosla, xlab, title in [
    (ax1, False, "displacement  ‖D‖ / σ   —  how FAR the latent moves", "Distance does NOT predict AUC"),
    (ax2, True, "cos_la   —  shift alignment with the safe→unsafe axis", "Direction (alignment) DOES"),
]:
    for enc, axis, displ, cos_la, zs in D:
        x = cos_la if use_cosla else displ
        ax.scatter(x, zs, s=115, marker=ENCMK[enc], c=AXCOL[axis],
                   edgecolors="k", linewidths=0.6, alpha=0.9, zorder=3)
    ax.set_xlabel(xlab, fontsize=9.5); ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25)
ax1.set_ylabel("zero-shot AUC", fontsize=11)
ax1.set_ylim(0.35, 1.03)
col_h = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=AXCOL[a],
         markeredgecolor="k", markersize=10, label=a) for a in AXCOL]
mk_h = [plt.Line2D([0], [0], marker=ENCMK[e], color="w", markerfacecolor="#999",
        markeredgecolor="k", markersize=9, label=e) for e in ENCMK]
leg1 = fig.legend(handles=col_h, title="axis (color)", loc="upper left",
                  bbox_to_anchor=(0.10, 0.955), ncol=3, fontsize=8.5, frameon=False, title_fontsize=8.5)
fig.add_artist(leg1)
fig.legend(handles=mk_h, title="encoder (shape)", loc="upper right",
           bbox_to_anchor=(0.93, 0.955), ncol=4, fontsize=8.5, frameon=False, title_fontsize=8.5)
fig.suptitle("Zero-shot margin AUC under a shift: set by direction, not distance",
             y=0.995, fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.83])
out = "/home/seongbin/latent/figs/latent_probe_summary.png"
fig.savefig(out, dpi=130); print("saved", out)
