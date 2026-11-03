"""Do CEM's elites converge? Reads SOFTCEM_SPREAD files (safe_solver._EliteSpread). Action std is in CEM's
sampling units (initial std = var_scale = 1). Usage: diag_elite_spread.py <file.pt> [...]"""
import sys
from collections import defaultdict
import numpy as np
import torch

for p in sys.argv[1:]:
    d = torch.load(p, weights_only=False)
    by_step = defaultdict(list)
    for step, sd in d["curve"]:
        by_step[step].append(sd)
    curve = {s: torch.cat(v).median().item() for s, v in sorted(by_step.items())}
    cat = lambda k: torch.cat([r[k] for r in d["rows"]]).numpy()
    sd_h, b2m, rng, pc1 = cat("sd_h"), cat("best_to_mean"), cat("cost_range"), cat("pc1")
    md = lambda a: np.median(a, 0)
    print(f"== {p.rsplit('/', 1)[-1]}  (n = {len(b2m)} solves x envs)")
    print("median elite std by CEM iteration:", " ".join(f"{s}:{v:.3f}" for s, v in curve.items() if s % 5 == 0 or s == max(curve)))
    print("final elite std by horizon step (median):", " ".join(f"t{t + 1}:{md(sd_h)[t]:.3f}" for t in (0, 4, 9, 14, 19, 24)))
    print(f"final elite std, all steps: median {np.median(sd_h.mean(1)):.3f}, p90 {np.percentile(sd_h.mean(1), 90):.3f}")
    print(f"RMS best elite - returned mean: median {np.median(b2m):.3f}, p90 {np.percentile(b2m, 90):.3f}")
    print(f"elite cost range: median {np.median(rng):.3f}, p90 {np.percentile(rng, 90):.3f}")
    print(f"share of elite variance on top direction: median {np.median(pc1):.2f}, p90 {np.percentile(pc1, 90):.2f}")
    if d["rows"][0]["all_safe"] is not None:
        print(f"all 30 elites predicted safe: {cat('all_safe').mean():.3f}")
