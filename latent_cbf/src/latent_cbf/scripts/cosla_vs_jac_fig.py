"""Does cos_la fall as the encoder Jacobian gets more isotropic? Parse run_logs/cosla_vs_jac.log
(9 dubins LEWM models = 3 regs x {orig, color-aug, color+rot-aug}, + dubins Dreamer). x = sigma1-share
(lambda1 / sum lambda of the encoder Jacobian ∂z/∂x) — a DIMENSION-INVARIANT isotropy measure in [0,1]
(LOWER = more isotropic), so the 544-d Dreamer RSSM is comparable to the ViT encoders. y = cos_la.
Two panels (color / rotate shift). Marker = model family, colour = aug condition. Punchline: isotropy
does NOT control cos_la — Dreamer is as isotropic as jacobian (sigma1-share ~0.10) yet has the HIGHEST
color cos_la (.81), and color aug collapses cos_la at any isotropy. Conditioning != orthogonalization."""
import re, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = []   # (tag, cond, s1share, cos_la_color, cos_la_rot)
for ln in open("/home/seongbin/latent/run_logs/cosla_vs_jac.log"):
    m = re.match(r"CJ\s+(\S+)\s+(\S+)\s+effrank=\s*([\d.]+)\s+froJ=\s*([\d.]+)\s+s1share=([\d.]+)\s+cos_la_color=([\d.]+)\s+cos_la_rot=([\d.]+)", ln)
    if m:
        rows.append((m[1], m[2], float(m[5]), float(m[6]), float(m[7])))
assert rows, "no CJ rows parsed"

MK = {"baseline":"o", "jacobian":"s", "jac+pull":"^", "dreamer":"*"}
CC = {"orig":"#888888", "color":"#d62728", "color+rot":"#1f77b4"}
CLABEL = {"orig":"no aug", "color":"color aug", "color+rot":"color+rot aug"}
DREAMER_C = "#111111"   # dreamer drawn distinctly (its own architecture, no reg)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True)
for ax, ci, title in [(axes[0], 3, "color shift"), (axes[1], 4, "rotation shift")]:
    for tag, cond, s1, cc, crot in rows:
        y = [None, None, None, cc, crot][ci]
        is_dr = tag == "dreamer"
        ax.scatter(s1, y, s=340 if is_dr else 170, marker=MK[tag],
                   c=DREAMER_C if is_dr else CC[cond], edgecolors="k", linewidths=0.9, zorder=4 if is_dr else 3)
    ax.set_title(f"cos_la vs Jacobian isotropy — {title}", fontsize=11)
    ax.set_xlabel("σ₁-share of encoder Jacobian J   (← more isotropic)", fontsize=9.5)
    ax.grid(alpha=0.25); ax.set_ylim(-0.03, 0.88); ax.set_xlim(-0.02, 0.83)
axes[0].set_ylabel("cos_la = |cos(shift, safe→unsafe mean-diff λ)|", fontsize=10)

reg_h = [plt.Line2D([0],[0], marker=MK[k], color="w", markerfacecolor="#bbb", markeredgecolor="k",
                    markersize=13 if k=="dreamer" else 11, label=("dreamer (RSSM)" if k=="dreamer" else k)) for k in MK]
aug_h = [plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=CC[k], markeredgecolor="k", markersize=11, label=CLABEL[k]) for k in CC]
fig.legend(handles=reg_h+aug_h, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=7, fontsize=8.5, frameon=False)
fig.suptitle("Isotropy does NOT control cos_la: Dreamer is as isotropic as jacobian yet most-aligned;\n"
             "the DATA lever (color aug, red/blue) collapses cos_la→0 at any isotropy — conditioning ≠ orthogonalization",
             y=0.90, fontsize=9.5)
fig.tight_layout(rect=[0, 0, 1, 0.83])
out = "/home/seongbin/latent/figs/cosla_vs_jac.png"
fig.savefig(out, dpi=130); print("saved", out, "rows=", len(rows))
