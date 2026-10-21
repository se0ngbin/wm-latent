"""Scatter of the zero-shot margin head vs the true distance to the nearest obstacle (Dubins, 2000 test states,
training seed 3072): one figure with y = margin score h, one with y = signed latent distance to the decision boundary
h / ||grad_z h|| (class-gap units). Rows: in-dist, color shift. Single hue, no legend."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SURF, INK, INK2, BLUE, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#e6e5e1"
d = np.load("/data/seongbin/lewm/safeadv_results/routes/safety_scores.npz"); x = d["dist"]
MODELS = ["sigreg only", "jac+pull", "L∞ w/ h", "L∞ w/o h", "jac+pull + L∞"]; ROWS = [("in-dist", "in-dist"), ("color", "color shift")]
for suffix, ylabel, out in [("", "margin score", "figs/safety_margin_scatter.png"), ("|bdist", "dist to decision boundary", "figs/safety_bdist_scatter.png")]:
    fig, axes = plt.subplots(2, len(MODELS), figsize=(2.6 * len(MODELS), 5.2), sharex=True, facecolor=SURF)
    for r, (cond, cname) in enumerate(ROWS):
        for c, m in enumerate(MODELS):
            ax = axes[r, c]; yv = d[f"{m}|{cond}{suffix}"]; ax.set_facecolor(SURF)
            ax.axhline(0, color=INK2, lw=0.6); ax.axvline(0, color=INK2, lw=0.6)
            ax.scatter(x, yv, s=2, color=BLUE, alpha=0.35, linewidths=0)
            ax.set_title(f"{m}, {cname}", fontsize=8, color=INK)
            ax.tick_params(colors=INK2, labelsize=7, length=0); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
            for sp in ax.spines.values(): sp.set_visible(False)
            if r == 1: ax.set_xlabel("dist to obstacle", fontsize=8, color=INK2)
            if c == 0: ax.set_ylabel(ylabel, fontsize=8, color=INK)
    fig.tight_layout(); fig.savefig(out, dpi=160, facecolor=SURF); print("wrote", out)
