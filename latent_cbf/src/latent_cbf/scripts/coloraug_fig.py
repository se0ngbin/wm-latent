"""Obstacle color-aug (purple held out): purple OOD AUC before (original) vs after
(color-aug) for the MARGIN (probe) and the VALUE function (critic), across the 4
encoders. Color-aug fixes purple for margin (all but jacobian) AND for the value fn of
the PREDICTIVE JEPA encoders (baseline/jacobian/jacpull) — but NOT Dreamer's value
(reconstruction keeps color encoded). Rotation is traded away (shown as text)."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENC = ["baseline", "jacobian", "jac+pull", "dreamer"]
# purple AUC (original, coloraug)
MARGIN = {"baseline": (.136, .880), "jacobian": (.757, .631), "jac+pull": (.707, .891), "dreamer": (.651, .732)}
VALUE  = {"baseline": (.643, .915), "jacobian": (.619, .970), "jac+pull": (.204, .881), "dreamer": (.649, .642)}
COL = {"baseline": "#1f77b4", "jacobian": "#2ca02c", "jac+pull": "#9467bd", "dreamer": "#d62728"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
for ax, data, title in [(ax1, MARGIN, "Margin (probe)"), (ax2, VALUE, "Value function (critic)")]:
    for i, e in enumerate(ENC):
        o, a = data[e]; c = COL[e]; up = a - o > 0.02
        ax.annotate("", xy=(i, a), xytext=(i, o),
                    arrowprops=dict(arrowstyle="-|>", color=c if up else "#b0b0b0", lw=2.2, alpha=.9))
        ax.scatter(i, o, s=150, facecolors="white", edgecolors=c, linewidths=2, zorder=3)
        ax.scatter(i, a, s=150, facecolors=c, edgecolors="k", linewidths=0.8, zorder=4)
        ax.text(i, a + (0.03 if up else -0.06), f"{a:.2f}", ha="center", fontsize=8.5,
                color=c, fontweight="bold")
        ax.text(i, o - 0.055, f"{o:.2f}", ha="center", fontsize=8, color="#777")
    ax.axhline(0.5, ls="--", c="gray", lw=1); ax.set_xticks(range(len(ENC)))
    ax.set_xticklabels(ENC, fontsize=9.5); ax.set_title(title, fontsize=11.5)
    ax.grid(alpha=0.2, axis="y"); ax.set_xlim(-0.5, len(ENC) - 0.5)
ax1.set_ylabel("purple (color-OOD) AUC", fontsize=11); ax1.set_ylim(0.05, 1.02)
ax2.text(3, .642, "  ← Dreamer value\n     unchanged", fontsize=8, color="#d62728", va="center")
h = [plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="white", markeredgecolor="k", markersize=10, label="original"),
     plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#555", markeredgecolor="k", markersize=10, label="+ color-aug (purple held out)")]
fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, fontsize=9, frameon=False)
fig.suptitle("Obstacle color-aug fixes purple for the margin and the PREDICTIVE encoders' value fn — not Dreamer's (reconstruction)",
             y=0.995, fontsize=10.8)
fig.text(0.5, 0.015, "Trade-off (not shown): rotation regresses under color-aug — value rot90 baseline .80→.58, jacobian .91→.61, jac+pull .89→.61 (Dreamer ~flat).",
         ha="center", fontsize=8, color="#555")
fig.tight_layout(rect=[0, 0.035, 1, 0.91])
out = "/home/seongbin/latent/figs/coloraug_purple.png"
fig.savefig(out, dpi=130); print("saved", out)
