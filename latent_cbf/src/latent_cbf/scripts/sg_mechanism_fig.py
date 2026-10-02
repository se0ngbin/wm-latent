"""sg mechanism (corrected). Left: zs AUC vs disp/sigma (shift magnitude / state spread) —
explains COLOR (baseline amplifies) but NOT rotate. Right: zs AUC vs cos vs the TRUE safety
direction (LDA/logistic DISCRIMINANT) — all ~0 (every shift orthogonal to 1-D safety for every
encoder), so orthogonality is the DEFAULT and does not distinguish the encoders. (The mean-diff
cos_la proxy misleadingly showed .2-.43.) color=red, rotate=blue; o=baseline s=jacobian ^=jac+pull."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (encoder, axis, disp/sigma, cos_discriminant, zs_auc)  -- sg LEWM
D = [
 ("baseline","color", 0.70, 0.023, 0.568), ("jacobian","color", 0.28, 0.007, 0.988), ("jac+pull","color", 0.32, 0.123, 0.990),
 ("baseline","rotate",0.90, 0.016, 0.531), ("jacobian","rotate",1.12, 0.031, 0.737), ("jac+pull","rotate",1.13, 0.012, 0.754),
]
AXC = {"color": "#d62728", "rotate": "#1f77b4"}; MK = {"baseline":"o","jacobian":"s","jac+pull":"^"}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.0), sharey=True)
for ax, key, xlab, note in [
    (ax1, 2, "disp / σ_red   (shift magnitude ÷ state spread)", "MAGNITUDE explains color, not rotate"),
    (ax2, 3, "cos vs TRUE safety direction (LDA discriminant)", "all ≈ 0: orthogonal for everyone — not a distinguisher")]:
    for enc, axis, disp, cd, auc in D:
        ax.scatter(disp if key==2 else cd, auc, s=150, marker=MK[enc], c=AXC[axis], edgecolors="k", linewidths=0.8, zorder=3)
    if key == 2:
        for axis in ("color","rotate"):
            pts=sorted([(d[2], d[4]) for d in D if d[1]==axis]); xs,ys=zip(*pts); ax.plot(xs,ys,"-",color=AXC[axis],lw=1.2,alpha=.4)
    else:
        ax.set_xlim(-0.02, 0.5); ax.axvspan(-0.02, 0.15, color="#eee", zorder=0)
        ax.text(0.25, 0.62, "(mean-diff proxy\nmisleadingly showed .2–.43)", fontsize=8, c="gray", ha="left")
    ax.axhline(0.5, ls="--", c="gray", lw=1); ax.set_xlabel(xlab, fontsize=9.5); ax.set_title(note, fontsize=10.5); ax.grid(alpha=0.22)
ax1.set_ylabel("zero-shot OOD margin AUC", fontsize=11); ax1.set_ylim(0.45, 1.03)
ax_h = [plt.Line2D([0],[0], marker="s", color="w", markerfacecolor=AXC[a], markeredgecolor="k", markersize=11, label=a) for a in AXC]
mk_h = [plt.Line2D([0],[0], marker=MK[e], color="w", markerfacecolor="#999", markeredgecolor="k", markersize=10, label=e) for e in MK]
fig.legend(handles=ax_h+mk_h, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=5, fontsize=9, frameon=False)
fig.suptitle("sg mechanism: shifts are orthogonal to the 1-D safety direction for ALL encoders (default);\n"
             "color robustness = magnitude, rotation = margin's off-axis sensitivity (neither is alignment)", y=0.9, fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.88])
out = "/home/seongbin/latent/figs/sg_mechanism.png"
fig.savefig(out, dpi=130); print("saved", out)
