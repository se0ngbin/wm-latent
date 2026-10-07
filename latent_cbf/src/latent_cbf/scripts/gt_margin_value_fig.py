"""Predictor-isolation for the value function (Post Sept 16). Train the critic with the
GROUND-TRUTH margin as reward (oracle-clean) instead of the learned margin head, so any
residual OOD failure is the PREDICTOR's latent geometry alone. Value-fn zero-shot AUC (vs
HJ V*) plotted against three x's: displacement, cos_la, and the instantaneous margin_gp AUC.
hollow = learned-margin critic -> filled = GT-margin critic (vertical arrow = the reward
swap; every x is fixed because it doesn't touch the encoder). Panels 1-2: cos_la explains
value OOD (r tightens once the reward confound is removed); displacement doesn't. Panel 3
(y=x): the critic does NOT simply inherit the margin. color = axis, shape = encoder."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# encoder, axis, displ, cos_la, margin_gp AUC, learned-critic AUC, GT-critic AUC
ROWS = [
 ("baseline","color", 1.374,0.621,0.427,0.643,0.808), ("baseline","shape",0.743,0.808,0.613,0.712,0.150),
 ("baseline","rotate",1.007,0.271,0.929,0.804,0.883), ("jacobian","color",0.823,0.574,0.606,0.619,0.530),
 ("jacobian","shape", 0.113,0.141,0.996,0.964,0.986), ("jacobian","rotate",0.970,0.037,0.987,0.913,0.928),
 ("jacpull", "color", 0.784,0.610,0.689,0.204,0.356), ("jacpull","shape",0.071,0.252,0.995,0.977,0.987),
 ("jacpull", "rotate",0.956,0.235,0.978,0.886,0.913), ("dreamer","color",0.771,0.854,0.566,0.649,0.603),
 ("dreamer","shape", 0.040,0.443,0.994,0.997,0.997), ("dreamer","rotate",0.858,0.039,0.433,0.483,0.579),
]
AXC = {"color": "#d62728", "shape": "#ff7f0e", "rotate": "#1f77b4"}
ENM = {"baseline": "o", "jacobian": "s", "jacpull": "^", "dreamer": "D"}
I = {"displ": 2, "cos_la": 3, "margin": 4}

def rr(a, b): return np.corrcoef(a, b)[0, 1]
gt = np.array([r[6] for r in ROWS])

fig, axes = plt.subplots(1, 3, figsize=(16, 5.3), sharey=True)
panels = [("displ", "displacement  ‖Δz‖/σ"),
          ("cos_la", "cos_la  (shift alignment with safe→unsafe axis)"),
          ("margin", "margin_gp zero-shot AUC")]
for ax, (key, xlab) in zip(axes, panels):
    if key == "margin":
        ax.plot([0.05, 1.0], [0.05, 1.0], ls="--", c="gray", lw=1, zorder=1)
        ax.text(0.98, 1.0, "value = margin", color="gray", fontsize=8, ha="right", va="top", rotation=38)
    for enc, ax_, d, c, m, lc, gc in ROWS:
        x = {"displ": d, "cos_la": c, "margin": m}[key]
        col = AXC[ax_]; mk = ENM[enc]
        ax.annotate("", xy=(x, gc), xytext=(x, lc),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4, alpha=.6))
        ax.scatter(x, lc, s=95, marker=mk, facecolors="white", edgecolors=col, linewidths=1.5, zorder=3)
        ax.scatter(x, gc, s=110, marker=mk, facecolors=col, edgecolors="k", linewidths=0.7, zorder=4)
    ax.axhline(0.5, ls=":", c="#bbb", lw=1); ax.grid(alpha=0.22)
    ax.set_xlabel(xlab, fontsize=9.5)
    xv = np.array([r[I[key]] for r in ROWS])
    extra = "  (value inherits margin?)" if key == "margin" else ""
    ax.set_title(f"vs {key}   (GT r={rr(xv,gt):+.2f}){extra}", fontsize=10.5)
axes[0].set_ylabel("value-function OOD AUC (vs HJ V*)", fontsize=11); axes[0].set_ylim(0.05, 1.03)
h_enc = [plt.Line2D([0],[0], marker=ENM[e], color="w", markerfacecolor="#bbb",
         markeredgecolor="k", markersize=10, label=e) for e in ENM]
h_ax = [plt.Line2D([0],[0], marker="s", color="w", markerfacecolor=AXC[a],
        markeredgecolor="k", markersize=10, label=a) for a in AXC]
h_hf = [plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="white", markeredgecolor="k",
        markersize=10, label="hollow = learned margin"),
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#555", markeredgecolor="k",
        markersize=10, label="filled = GT margin")]
fig.suptitle("Isolate the predictor (GT-margin reward): value OOD is explained by cos_la, not distance, and the critic does not just inherit the margin",
             y=0.995, fontsize=10.8)
fig.legend(handles=h_enc + h_ax + h_hf, loc="upper center", bbox_to_anchor=(0.5, 0.95),
           ncol=8, fontsize=8, frameon=False)
fig.tight_layout(rect=[0, 0, 1, 0.88])
out = "/home/seongbin/latent/figs/gt_margin_value.png"
fig.savefig(out, dpi=130); print("saved", out)
