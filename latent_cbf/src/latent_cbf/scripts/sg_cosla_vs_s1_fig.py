"""Safety-gym analog of the dubins cosla_vs_jac figure. Does cos_la fall as the encoder Jacobian gets
more isotropic? x = sigma1-share of ∂z/∂x (dimension-invariant, LOWER = more isotropic), y = cos_la.
Two panels (color / rotate shift). 4 models (3 LEWM regs + Dreamer). No aug variants exist for sg, so
this is the isotropy sweep only (fewer points than dubins). Data: run_logs/sg/mechanism.log +
sg margin cos_la + sg_dreamer_s1. Punchline mirrors dubins: no clean isotropy→cos_la trend, and the
most isotropic model (Dreamer, s1share .12) is NOT the least aligned on color."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# encoder: (s1share, cos_la_color, cos_la_rot)   sg, gp margin, real camrot
D = {
 "baseline": (0.543, 0.241, 0.426),
 "jacobian": (0.167, 0.288, 0.236),
 "jac+pull": (0.241, 0.430, 0.272),
 "dreamer":  (0.123, 0.185, 0.084),
}
MK = {"baseline":"o","jacobian":"s","jac+pull":"^","dreamer":"*"}
PANEL = [(1, "color shift", "#d62728"), (2, "rotation shift", "#1f77b4")]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
for ax, (ci, title, col) in zip(axes, PANEL):
    xs = [v[0] for v in D.values()]; ys = [v[ci] for v in D.values()]
    r = np.corrcoef(xs, ys)[0,1]
    for enc, v in D.items():
        ax.scatter(v[0], v[ci], s=360 if enc=="dreamer" else 175, marker=MK[enc], c=col,
                   edgecolors="k", linewidths=0.9, zorder=3)
    ax.set_title(f"cos_la vs Jacobian isotropy — {title}\nPearson r = {r:+.2f}  (n=4)", fontsize=10.5)
    ax.set_xlabel("σ₁-share of encoder Jacobian J   (← more isotropic)", fontsize=9.5)
    ax.grid(alpha=0.25); ax.set_xlim(0.05, 0.62); ax.set_ylim(0.0, 0.55)
axes[0].set_ylabel("cos_la = |cos(shift, safe→unsafe mean-diff λ)|", fontsize=10)
enc_h = [plt.Line2D([0],[0],marker=MK[k],color="w",markerfacecolor="#bbb",markeredgecolor="k",
                    markersize=14 if k=="dreamer" else 11,label=("dreamer (RSSM)" if k=="dreamer" else k)) for k in MK]
fig.legend(handles=enc_h, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=9, frameon=False)
fig.suptitle("Safety-gym (n=4, no aug variants): color cos_la unrelated to isotropy (r=−.03); rotate cos_la DOES fall with isotropy (r=+.92, baseline-driven)\n"
             "— but even so cos_la doesn't predict robustness: Dreamer is the MOST isotropic w/ the LOWEST rotate cos_la (.08) yet FAILS rotation (AUC .53)",
             y=0.92, fontsize=8.4)
fig.tight_layout(rect=[0, 0, 1, 0.84])
out = "/home/seongbin/latent/figs/sg_cosla_vs_s1.png"
fig.savefig(out, dpi=130); print("saved", out)
