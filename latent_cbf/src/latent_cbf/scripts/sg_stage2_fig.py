"""Stage-2 encoder-level robust training (sg), plain-readout zero-shot. Bars: in-dist blue + purple
(same-type color OOD) for baseline / +color-aug / +adversarial-invariance, with jacobian as the
conditioning ceiling. Story: color-OOD IS fixable at the ENCODER (baseline .60->.79 via color-aug,
keeping in-dist high), but the METHOD matters — the adversarial min-max reg over-invariantizes and
collapses the in-dist safety signal (.99->.67). Data lever robust; min-max lever fragile (cf Stage 1)."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENC = ["baseline", "+color-aug\n(data)", "+adversarial\n(min-max)", "jacobian\n(conditioning ref)"]
INDIST = [0.991, 0.978, 0.673, 0.996]
PURPLE = [0.599, 0.791, 0.724, 0.992]
x = np.arange(len(ENC)); w = 0.38
fig, ax = plt.subplots(figsize=(9, 5.2))
b1 = ax.bar(x - w/2, INDIST, w, label="in-dist blue", color="#9ecae1", edgecolor="k", linewidth=0.7)
b2 = ax.bar(x + w/2, PURPLE, w, label="zero-shot purple (color OOD)", color="#d62728", edgecolor="k", linewidth=0.7)
ax.axhline(0.5, ls="--", c="gray", lw=1); ax.text(3.3, .51, "chance", fontsize=8, color="gray")
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+.008, f"{b.get_height():.2f}", ha="center", fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(ENC, fontsize=9.5)
ax.set_ylabel("plain-readout AUC (frozen encoder)", fontsize=11); ax.set_ylim(0.4, 1.06)
ax.set_title("Stage 2 — color-OOD is fixable at the ENCODER, but the method matters\n"
             "color-aug lifts baseline purple .60→.79 (in-dist intact); adversarial min-max over-invariantizes (in-dist .99→.67)",
             fontsize=10.5)
ax.legend(fontsize=9.5, loc="lower left"); ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
out = "/home/seongbin/latent/figs/sg_stage2_encoder.png"
fig.savefig(out, dpi=130); print("saved", out)
