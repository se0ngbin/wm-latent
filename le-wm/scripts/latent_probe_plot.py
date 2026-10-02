"""4x3 grid of 2D-PCA scatters (encoders x axes) for the latent-shift probe.
Each panel: red-safe / red-unsafe / purple-safe / purple-unsafe. Shows whether the
appearance shift (red->purple) moves ALONG the safe/unsafe axis (harmful) or across
it (harmless). Reads run_logs/probe_coords/<enc>_<axis>.npz."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

CD = Path("/home/seongbin/latent/run_logs/probe_coords")
ENC = ["baseline", "jacobian", "jacpull", "dreamer"]
AX = ["color", "shape", "rotate"]
# appear 0=og(red/circle/rot0), 1=new ; label 0=safe,1=unsafe
STY = {(0, 0): ("#1f77b4", ".", "og-safe"), (0, 1): ("#d62728", ".", "og-unsafe"),
       (1, 0): ("#17becf", "x", "new-safe"), (1, 1): ("#ff7f0e", "x", "new-unsafe")}

fig, axes = plt.subplots(len(ENC), len(AX), figsize=(11, 13))
for i, enc in enumerate(ENC):
    for j, ax in enumerate(AX):
        p = axes[i][j]; f = CD / f"{enc}_{ax}.npz"
        if not f.exists():
            p.set_axis_off(); continue
        d = np.load(f); xy, ap, lb = d["xy"], d["appear"], d["label"]
        idx = np.random.default_rng(0).choice(len(xy), min(1500, len(xy)), replace=False)
        for (a_, l_), (c, mk, _) in STY.items():
            m = (ap[idx] == a_) & (lb[idx] == l_)
            p.scatter(xy[idx][m, 0], xy[idx][m, 1], s=5, c=c, marker=mk, alpha=.5, linewidths=.5)
        p.set_xticks([]); p.set_yticks([])
        if i == 0: p.set_title(ax, fontsize=12)
        if j == 0: p.set_ylabel(enc, fontsize=12)
handles = [plt.Line2D([0], [0], marker=mk, color="w", markerfacecolor=c, markeredgecolor=c,
           markersize=8, label=lab) for (c, mk, lab) in STY.values()]
fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=11, frameon=False)
fig.suptitle("Latent 2D-PCA: appearance shift (og→new) vs safe/unsafe axis", y=0.995, fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.965])
out = "/home/seongbin/latent/latent_probe_pca.png"
fig.savefig(out, dpi=110)
print("saved", out)
