"""Safety-gym CarGoal transfer: zero-shot margin OOD AUC (2-seed mean) for 4 encoders across
3 appearance shifts. Color transfers the dubins finding (baseline fails, structured hold);
diamond is easy (all hold); real 90-deg camera rotation is hard for all but the Jacobian
family beats baseline AND Dreamer."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# zs AUC per (model, axis): (seed0, seed1)
D = {
 "baseline": {"color": (.568,.571), "rotate": (.531,.529), "diamond": (.826,.828)},
 "jacobian": {"color": (.988,.988), "rotate": (.737,.754), "diamond": (.999,.999)},
 "jac+pull": {"color": (.990,.992), "rotate": (.754,.767), "diamond": (.999,.999)},
 "dreamer":  {"color": (.977,.977), "rotate": (.537,.521), "diamond": (.998,.997)},
}
AXES = ["color", "rotate", "diamond"]
MODELS = ["baseline", "jacobian", "jac+pull", "dreamer"]
COL = {"baseline": "#7f7f7f", "jacobian": "#2ca02c", "jac+pull": "#9467bd", "dreamer": "#d62728"}

fig, ax = plt.subplots(figsize=(9, 5.2))
w = 0.2
x = np.arange(len(AXES))
for i, mdl in enumerate(MODELS):
    means = [np.mean(D[mdl][a]) for a in AXES]
    errs = [(max(D[mdl][a]) - min(D[mdl][a])) / 2 for a in AXES]
    ax.bar(x + (i - 1.5) * w, means, w, yerr=errs, capsize=2, color=COL[mdl],
           edgecolor="k", linewidth=0.5, label=mdl)
    for xi, m in zip(x + (i - 1.5) * w, means):
        ax.text(xi, m + 0.015, f"{m:.2f}", ha="center", fontsize=7.5)
ax.axhline(0.5, ls="--", c="gray", lw=1); ax.text(2.45, 0.5, "chance", fontsize=8, c="gray", va="bottom", ha="right")
ax.set_xticks(x); ax.set_xticklabels(["color\n(hazard recolor)", "rotate\n(90° camera)", "diamond\n(box hazards)"], fontsize=10)
ax.set_ylabel("zero-shot OOD margin AUC (2-seed mean)", fontsize=11); ax.set_ylim(0.4, 1.03)
ax.legend(fontsize=9, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.28), frameon=False)
ax.set_title("Safety-gym CarGoal — appearance-OOD transfer: color reproduces dubins (baseline fails,\n"
             "structured hold); diamond trivial; real camera-rotation hard for all but Jacobian beats baseline & Dreamer",
             fontsize=10)
ax.grid(alpha=0.2, axis="y")
fig.tight_layout()
out = "/home/seongbin/latent/figs/sg_transfer.png"
fig.savefig(out, dpi=130, bbox_inches="tight"); print("saved", out)
