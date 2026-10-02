"""Re-grounding curves for the color+rot WMs: red<->{purple,rot90} latent divergence
(in units of red spread) over a zero-action predictor rollout. Log y (magnitudes span
2.5..80). The dominant effect is at t=0 (the ENCODER): jac+pull keeps OOD ~2.5 sigma from
red; baseline ~16-20; jacobian ~60-66. Over the rollout jacobian's rot90 even DIVERGES."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURVES = {
 ("baseline","purple"): [20.504,20.467,20.533,20.680,20.912,21.155,21.286,21.083,20.659,20.305,20.224,20.389,20.727],
 ("baseline","rot90"):  [15.757,15.848,16.014,16.156,16.261,16.294,16.151,15.666,15.022,14.546,14.411,14.557,14.824],
 ("jacobian","purple"): [60.781,58.892,58.119,57.663,57.383,57.243,57.225,57.320,57.522,57.826,58.232,58.738,59.343],
 ("jacobian","rot90"):  [66.092,65.445,65.763,66.486,67.446,68.609,69.964,71.499,73.202,75.058,77.044,79.133,81.293],
 ("jac+pull","purple"): [2.527,2.521,2.508,2.491,2.470,2.447,2.424,2.401,2.378,2.357,2.338,2.324,2.316],
 ("jac+pull","rot90"):  [2.685,2.662,2.658,2.653,2.651,2.651,2.653,2.657,2.664,2.673,2.685,2.699,2.715],
}
COL = {"baseline": "#1f77b4", "jacobian": "#2ca02c", "jac+pull": "#9467bd"}
LS = {"purple": "-", "rot90": "--"}

fig, ax = plt.subplots(figsize=(7.6, 5.2))
for (enc, sh), c in CURVES.items():
    ax.plot(range(len(c)), c, LS[sh], color=COL[enc], lw=2, marker="o", ms=3,
            label=f"{enc} · {sh}")
ax.set_yscale("log"); ax.set_xlabel("predictor rollout step (zero action)", fontsize=11)
ax.set_ylabel("red↔OOD latent divergence / σ_red  (log)", fontsize=11)
ax.grid(alpha=0.25, which="both")
ax.set_title("div = ‖shift‖/σ_red; RAW shift ~equal for all 3 (~16–19) — the gap is σ_red\n"
             "(state-latent spread: jacpull 6.9, baseline .85, jacobian .27 collapsed); jacobian rot90 also DIVERGES",
             fontsize=9.6)
ax.legend(fontsize=8.5, ncol=3, loc="center right")
fig.tight_layout()
out = "/home/seongbin/latent/figs/regrounding_coloraug_rot.png"
fig.savefig(out, dpi=130); print("saved", out)
