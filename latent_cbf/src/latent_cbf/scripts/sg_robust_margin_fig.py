"""Stage-1 robust-margin (Ilyas) results, sg, frozen encoders. Two OOD panels (purple=same-type,
rotate=cross-type); x=encoder, grouped bars by arm (plain/aug/robust/proj). Shows: (1) color-OOD is
an ENCODER property — no readout arm rescues baseline (~.55), jacobian/jacpull already ~.99 at plain;
(2) worst-case 'robust' readout training is net-harmful; (3) the D_R nuisance projection helps ROTATION
only on well-conditioned encoders (jacobian .64->.77). in-dist blue ~.99 for all except robust."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENC = ["baseline", "jacobian", "jac+pull", "dreamer"]
ARMS = ["plain", "aug", "robust", "proj"]
# [plain, aug, robust, proj] per encoder
PURP = {"baseline":[.604,.584,.536,.521], "jacobian":[.992,.994,.969,.997], "jac+pull":[.996,.997,.932,.994], "dreamer":[.961,.976,.950,.573]}
ROT  = {"baseline":[.574,.559,.456,.563], "jacobian":[.643,.686,.519,.771], "jac+pull":[.684,.699,.438,.743], "dreamer":[.480,.514,.456,.500]}
COL = {"plain":"#7f7f7f","aug":"#1f77b4","robust":"#d62728","proj":"#2ca02c"}

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, DATA, title in [(axes[0], PURP, "purple  (same-type color OOD)"), (axes[1], ROT, "camera-rotation  (cross-type OOD)")]:
    x = np.arange(len(ENC)); w = 0.2
    for j, arm in enumerate(ARMS):
        vals = [DATA[e][j] for e in ENC]
        ax.bar(x + (j-1.5)*w, vals, w, label=arm, color=COL[arm], edgecolor="k", linewidth=0.6)
    ax.axhline(0.5, ls="--", c="gray", lw=1); ax.text(3.3, .51, "chance", fontsize=8, color="gray")
    ax.set_xticks(x); ax.set_xticklabels(ENC); ax.set_title(title, fontsize=11)
    ax.set_ylim(0.4, 1.03); ax.grid(axis="y", alpha=0.25)
axes[0].set_ylabel("zero-shot OOD margin AUC (frozen encoder)", fontsize=11)
axes[0].legend(title="margin arm", fontsize=9, ncol=2, loc="lower right")
fig.suptitle("Stage 1 (robust readout on FROZEN encoders): the encoder determines everything. No readout arm rescues baseline color (~.55);\n"
             "jacobian/jac+pull/dreamer already transfer color at 'plain'. D_R projection gains ROTATION only on DISENTANGLED encoders (jacobian .64→.77),\n"
             "does nothing on baseline, and DAMAGES dreamer (reconstruction entangles color↔safety: purple .96→.57).",
             y=1.0, fontsize=8.6)
fig.tight_layout(rect=[0, 0, 1, 0.92])
out = "/home/seongbin/latent/figs/sg_robust_margin.png"
fig.savefig(out, dpi=130); print("saved", out)
