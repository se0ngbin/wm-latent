"""cos_la vs zero-shot GP-margin AUC, dubins (left) and safety-gymnasium (right), SAME symbol/colour
scheme so the two domains are directly comparable. Marker = encoder (baseline o / jacobian s /
jac+pull ^ / dreamer *); colour = shift axis (color red / rotate blue / shape-diamond green).
Point: cos_la (mean-diff alignment) does NOT order OOD AUC in either domain. Dubins color is the
ENTANGLED case (cos_la high .66-.82, AUC modest for all) because appearance & safety are colocated in
pixels (cos_pix .82); sg color is orthogonal at input (cos_pix .02) so conditioned models hit .99.
dubins: model's trained margin_gp head (latent_shift_probe KEY probe). sg: freshly-trained gp probe
margin (sg_margin_ood, real camrot for rotate). Both = a GP safety margin's zero-shot AUC."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (encoder, axis, cos_la, zs_auc)
DUB = [
 ("baseline","color",0.720,0.427),("jacobian","color",0.666,0.606),("jac+pull","color",0.656,0.420),("dreamer","color",0.820,0.566),
 ("baseline","rotate",0.281,0.929),("jacobian","rotate",0.045,0.987),("jac+pull","rotate",0.129,0.975),("dreamer","rotate",0.387,0.433),
 ("baseline","shape",0.892,0.613),("jacobian","shape",0.279,0.996),("jac+pull","shape",0.443,0.995),("dreamer","shape",0.266,0.994),
]
SG = [
 ("baseline","color",0.241,0.570),("jacobian","color",0.288,0.988),("jac+pull","color",0.430,0.991),("dreamer","color",0.185,0.977),
 ("baseline","rotate",0.426,0.530),("jacobian","rotate",0.236,0.746),("jac+pull","rotate",0.272,0.761),("dreamer","rotate",0.084,0.529),
 ("baseline","shape",0.137,0.827),("jacobian","shape",0.452,0.999),("jac+pull","shape",0.290,0.999),("dreamer","shape",0.180,0.998),
]
MK = {"baseline":"o","jacobian":"s","jac+pull":"^","dreamer":"*"}
AC = {"color":"#d62728","rotate":"#1f77b4","shape":"#2ca02c"}

def rval(D):
    x=np.array([d[2] for d in D]); y=np.array([d[3] for d in D]); return np.corrcoef(x,y)[0,1]

fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
for ax, D, name in [(axes[0], DUB, "dubins  (2D top-down)"), (axes[1], SG, "safety-gymnasium  (3D physics, Car)")]:
    for enc, axis, cl, auc in D:
        ax.scatter(cl, auc, s=340 if enc=="dreamer" else 165, marker=MK[enc], c=AC[axis],
                   edgecolors="k", linewidths=0.9, zorder=3)
    ax.axhline(0.5, ls="--", c="gray", lw=1); ax.text(0.80, 0.515, "chance", fontsize=8, color="gray")
    ax.set_title(f"{name}\ncos_la vs OOD AUC:  Pearson r = {rval(D):+.2f}", fontsize=11)
    ax.set_xlabel("cos_la = |cos(appearance shift, safe→unsafe mean-diff λ)|", fontsize=10)
    ax.set_xlim(-0.02, 0.95); ax.set_ylim(0.40, 1.03); ax.grid(alpha=0.25)
axes[0].set_ylabel("zero-shot OOD GP-margin AUC", fontsize=11)
enc_h = [plt.Line2D([0],[0],marker=MK[k],color="w",markerfacecolor="#bbb",markeredgecolor="k",
                    markersize=13 if k=="dreamer" else 11,label=("dreamer (RSSM)" if k=="dreamer" else k)) for k in MK]
ax_h = [plt.Line2D([0],[0],marker="o",color="w",markerfacecolor=AC[k],markeredgecolor="k",markersize=11,
                   label=("shape/diamond" if k=="shape" else k)) for k in AC]
fig.legend(handles=enc_h+ax_h, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=7, fontsize=9, frameon=False)
fig.suptitle("Dubins: cos_la tracks difficulty ACROSS axes (r=-0.75) — the entangled color axis is high-cos_la & hard.  "
             "Safety-gym: no relation (r=+0.22), all shifts input-orthogonal.\n"
             "In BOTH, cos_la fails to order the ENCODERS within an axis (e.g. dubins color: jac+pull .66→.42 vs dreamer .82→.57).",
             y=0.955, fontsize=8.6)
fig.tight_layout(rect=[0, 0, 1, 0.86])
out = "/home/seongbin/latent/figs/cosla_vs_auc_both.png"
fig.savefig(out, dpi=130); print("saved", out, "r_dub=%.3f r_sg=%.3f" % (rval(DUB), rval(SG)))
