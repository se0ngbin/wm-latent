"""Colour-sweep CEM tables from run_logs/softcem_*: raw CEM (w0) and CCEM (minh, real+imag GP margin function), crossing
goals, 100 episodes, K = 10 / 15, alpha 0 (red), 0.3 (a03), 0.4 (a04), 1 (purple). Mean ± std over WM seeds.
Usage: color_sweep_cem_tables.py [log_dir]"""
import os, re, sys
import numpy as np

L = sys.argv[1] if len(sys.argv) > 1 else "/home/seongbin/latent/run_logs"
MODELS = {"sigreg": ["sigreg", "sigreg_s1", "sigreg_s2"], "jac": ["jacobian", "jacobian_s1", "jacobian_s100"],
          "jac+adv_h": ["jac_linf", "jac_linf_s1", "jac_linf_s2"]}
SCENS = [("red", "0"), ("a03", "0.3"), ("a04", "0.4"), ("purple", "1")]
fmt = lambda v: "" if not v else (f"{np.mean(v):.3f} ± {np.std(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}")


def rates(path):
    if not os.path.exists(path):
        return None
    t = open(path).read()
    s, c = re.search(r"'success_rate': ([\d.]+)", t), re.search(r"'collision_rate': ([\d.]+)", t)
    return (float(s.group(1)) / 100, float(c.group(1)) / 100) if s and c else None


for k in (10, 15):
    print(f"\nK = {k}: success / collision (n = WM seeds done)\n")
    print("| WM | planner | " + " | ".join(f"alpha {a}" for _, a in SCENS) + " |")
    print("|---|---|" + "---|" * len(SCENS))
    for m, tags in MODELS.items():
        for pl, pat in (("raw", "softcem_{t}_gr_w0_k{k}_cross_n100_{s}.log"),
                        ("CCEM", "softcem_{t}_gpimag_gr_w1000_k{k}_cross_ccemminh_n100_{s}.log")):
            cells = []
            for s, _ in SCENS:
                r = [x for x in (rates(os.path.join(L, pat.format(t=t, k=k, s=s))) for t in tags) if x]
                cells.append(f"{fmt([x[0] for x in r])} / {fmt([x[1] for x in r])} (n={len(r)})" if r else "")
            print(f"| {m} | {pl} | " + " | ".join(cells) + " |")
