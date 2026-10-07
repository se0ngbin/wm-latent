"""Can color+rotation aug give BOTH color and rotation robustness? Value-function
(critic) purple AUC (x) vs rot90 AUC (y), one point per (encoder, training condition),
trajectory original -> color-aug -> color+rot-aug. Only jac+pull reaches the top-right
(both robust) under color+rot; baseline/jacobian trade one for the other (anti-diagonal)
and their critics degenerate under the combined aug. Dashed = chance."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# value AUC (purple, rot90) per encoder x condition
DATA = {
 "baseline": {"orig": (.643, .804), "color": (.915, .579), "color+rot": (.359, .261)},
 "jacobian": {"orig": (.619, .913), "color": (.970, .613), "color+rot": (.204, .150)},
 "jac+pull": {"orig": (.204, .886), "color": (.881, .610), "color+rot": (.930, .982)},
}
COND_ORDER = ["orig", "color", "color+rot"]
COL = {"baseline": "#1f77b4", "jacobian": "#2ca02c", "jac+pull": "#9467bd"}
MK = {"orig": "o", "color": "s", "color+rot": "*"}
SZ = {"orig": 130, "color": 150, "color+rot": 360}

fig, ax = plt.subplots(figsize=(7.2, 6.6))
for enc, d in DATA.items():
    pts = [d[c] for c in COND_ORDER]; xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    ax.plot(xs, ys, "-", color=COL[enc], lw=1.2, alpha=.5, zorder=1)
    for c in COND_ORDER:
        x, y = d[c]
        ax.scatter(x, y, s=SZ[c], marker=MK[c], facecolors=COL[enc], edgecolors="k",
                   linewidths=0.9, zorder=3)
    lx, ly = d["color+rot"]
    ax.annotate(enc, (lx, ly), textcoords="offset points", xytext=(10, 8),
                fontsize=9.5, color=COL[enc], fontweight="bold")
ax.axhline(0.5, ls="--", c="gray", lw=1); ax.axvline(0.5, ls="--", c="gray", lw=1)
ax.text(.97, .97, "BOTH robust", ha="right", va="top", fontsize=10, color="#444",
        transform=ax.transAxes, style="italic")
ax.set_xlabel("value-fn purple (color-OOD) AUC", fontsize=11)
ax.set_ylabel("value-fn rot90 (rotation-OOD) AUC", fontsize=11)
ax.set_xlim(0.08, 1.02); ax.set_ylim(0.08, 1.02); ax.grid(alpha=0.25)
mk_h = [plt.Line2D([0],[0], marker=MK[c], color="w", markerfacecolor="#888", markeredgecolor="k",
        markersize=(9 if c!="color+rot" else 15), label=c) for c in COND_ORDER]
en_h = [plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=COL[e], markeredgecolor="k",
        markersize=10, label=e) for e in DATA]
ax.legend(handles=en_h + mk_h, loc="lower left", fontsize=8.5, ncol=2, frameon=True)
ax.set_title("Color+rotation aug: only jac+pull's value fn gets BOTH\n(others trade color↔rotation; their critics degenerate)",
             fontsize=11.5)
fig.tight_layout()
out = "/home/seongbin/latent/figs/coloraug_rot_both.png"
fig.savefig(out, dpi=130); print("saved", out)
