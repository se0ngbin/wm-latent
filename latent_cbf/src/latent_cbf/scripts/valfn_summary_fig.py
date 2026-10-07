"""Value-function analogue of the probe summary figure: the reachability CRITIC's
zero-shot AUC (vs HJ V*) plotted against displacement (left) and against the shift's
alignment with the V*-defined safe/unsafe axis (right). Same encoding as the margin
figure: axis=color, encoder=shape. displ/cos_la from valfn_geometry.py; AUC from the
value-function critic table (n=2, zero-shot column)."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (encoder, axis, displ, cos_la[V*], critic zero-shot AUC vs V*, margin zero-shot AUC)
D = [
    ("baseline", "color", 1.374, 0.621, 0.587, 0.427), ("jacobian", "color", 0.823, 0.574, 0.572, 0.606),
    ("jac+pull", "color", 0.784, 0.610, 0.205, 0.689), ("dreamer", "color", 0.771, 0.854, 0.627, 0.566),
    ("baseline", "shape", 0.743, 0.808, 0.715, 0.613), ("jacobian", "shape", 0.113, 0.141, 0.961, 0.996),
    ("jac+pull", "shape", 0.071, 0.252, 0.976, 0.995), ("dreamer", "shape", 0.040, 0.443, 0.997, 0.994),
    ("baseline", "rotate", 1.007, 0.271, 0.815, 0.929), ("jacobian", "rotate", 0.970, 0.037, 0.911, 0.987),
    ("jac+pull", "rotate", 0.956, 0.235, 0.894, 0.978), ("dreamer", "rotate", 0.858, 0.039, 0.488, 0.433),
]
AXCOL = {"color": "#d62728", "shape": "#2ca02c", "rotate": "#1f77b4"}
ENCMK = {"baseline": "o", "jacobian": "s", "jac+pull": "^", "dreamer": "D"}

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 5.0), sharey=True)
for ax, key, xlab, title in [
    (ax1, "displ", "displacement  ‖D‖ / σ   —  how FAR the latent moves", "Distance does NOT predict AUC"),
    (ax2, "cos_la", "cos_la   —  shift alignment with the safe→unsafe axis", "Alignment predicts LESS cleanly than for the margin"),
    (ax3, "margin", "margin zero-shot AUC   —  does the critic inherit the margin?", "Critic coarsely tracks the margin, except jac+pull color"),
]:
    if key == "margin":
        ax.plot([0.15, 1.0], [0.15, 1.0], ls="--", c="gray", lw=1, zorder=1)  # y = x
    for enc, axis, displ, cos_la, zs, mauc in D:
        x = {"displ": displ, "cos_la": cos_la, "margin": mauc}[key]
        ax.scatter(x, zs, s=115, marker=ENCMK[enc], c=AXCOL[axis],
                   edgecolors="k", linewidths=0.6, alpha=0.9, zorder=3)
    ax.set_xlabel(xlab, fontsize=9.5); ax.set_title(title, fontsize=10.5)
    ax.grid(alpha=0.25)
ax1.set_ylabel("critic zero-shot AUC", fontsize=11)
ax1.set_ylim(0.15, 1.03)
col_h = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=AXCOL[a],
         markeredgecolor="k", markersize=10, label=a) for a in AXCOL]
mk_h = [plt.Line2D([0], [0], marker=ENCMK[e], color="w", markerfacecolor="#999",
        markeredgecolor="k", markersize=9, label=e) for e in ENCMK]
leg1 = fig.legend(handles=col_h, title="axis (color)", loc="upper left",
                  bbox_to_anchor=(0.10, 0.955), ncol=3, fontsize=8.5, frameon=False, title_fontsize=8.5)
fig.add_artist(leg1)
fig.legend(handles=mk_h, title="encoder (shape)", loc="upper right",
           bbox_to_anchor=(0.93, 0.955), ncol=4, fontsize=8.5, frameon=False, title_fontsize=8.5)
fig.suptitle("Value function (critic vs V*): zero-shot AUC under a shift", y=0.995, fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.83])
out = "/home/seongbin/latent/figs/valfn_summary.png"
fig.savefig(out, dpi=130); print("saved", out)
