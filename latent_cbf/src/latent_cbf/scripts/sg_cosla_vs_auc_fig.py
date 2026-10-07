"""Safety-gymnasium: does cos_la predict zero-shot OOD margin AUC? Scatter x=cos_la (mean-diff axis)
vs y=zero-shot margin AUC, gp margin, real camera-rotation. 4 encoders x 3 shift axes. If cos_la
governed OOD robustness the points would trend down-right; instead there is no relationship (r~0):
baseline color cos_la .24 -> AUC .57 while Dreamer color cos_la .18 (LOWER) -> AUC .98, and on rotate
baseline (.43, fails) and Dreamer (.08, fails) sit at opposite cos_la extremes with the same AUC."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (encoder, axis, cos_la, zs_auc)  -- sg_gp, 2-seed mean, real camrot for rotate
D = [
 ("baseline","color",0.241,0.570), ("jacobian","color",0.288,0.988), ("jac+pull","color",0.430,0.991), ("dreamer","color",0.185,0.977),
 ("baseline","rotate",0.426,0.530),("jacobian","rotate",0.236,0.746),("jac+pull","rotate",0.272,0.761),("dreamer","rotate",0.084,0.529),
 ("baseline","diamond",0.137,0.827),("jacobian","diamond",0.452,0.999),("jac+pull","diamond",0.290,0.999),("dreamer","diamond",0.180,0.998),
]
MK = {"baseline":"o","jacobian":"s","jac+pull":"^","dreamer":"*"}
AC = {"color":"#d62728","rotate":"#1f77b4","diamond":"#2ca02c"}

x = np.array([d[2] for d in D]); y = np.array([d[3] for d in D])
r_all = np.corrcoef(x, y)[0,1]
cr = [d for d in D if d[1] != "diamond"]  # color+rotate only (diamond is trivially easy)
xc = np.array([d[2] for d in cr]); yc = np.array([d[3] for d in cr]); r_cr = np.corrcoef(xc, yc)[0,1]

fig, ax = plt.subplots(figsize=(8.8, 5.4))
for enc, axis, cl, auc in D:
    ax.scatter(cl, auc, s=340 if enc=="dreamer" else 165, marker=MK[enc], c=AC[axis],
               edgecolors="k", linewidths=0.9, zorder=3)
ax.axhline(0.5, ls="--", c="gray", lw=1); ax.text(0.46, 0.515, "chance", fontsize=8, color="gray")
ax.set_xlabel("cos_la  = |cos(appearance shift, safe→unsafe mean-diff λ)|", fontsize=10.5)
ax.set_ylabel("zero-shot OOD margin AUC", fontsize=11)
ax.set_title(f"Safety-gymnasium: cos_la does NOT predict OOD AUC\n"
             f"Pearson r = {r_all:+.2f} (all 12) / {r_cr:+.2f} (color+rotate only)", fontsize=11.5)
ax.set_xlim(0.02, 0.49); ax.set_ylim(0.45, 1.03); ax.grid(alpha=0.25)
enc_h = [plt.Line2D([0],[0],marker=MK[k],color="w",markerfacecolor="#bbb",markeredgecolor="k",
                    markersize=13 if k=="dreamer" else 11,label=("dreamer (RSSM)" if k=="dreamer" else k)) for k in MK]
ax_h = [plt.Line2D([0],[0],marker="o",color="w",markerfacecolor=AC[k],markeredgecolor="k",markersize=11,label=k) for k in AC]
ax.legend(handles=enc_h+ax_h, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, frameon=True, framealpha=0.95)
fig.tight_layout()
out = "/home/seongbin/latent/figs/sg_cosla_vs_auc.png"
fig.savefig(out, dpi=130); print("saved", out, f"r_all={r_all:+.3f} r_cr={r_cr:+.3f}")
