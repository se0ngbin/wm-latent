"""Markdown tables from run_logs/color_sweep/*.log (color_sweep_dubins.py): mean ± std over WM seeds per model, per alpha.
Usage: color_sweep_tables.py [log_dir]"""
import re, glob, os, sys, collections
import numpy as np

L = sys.argv[1] if len(sys.argv) > 1 else "/home/seongbin/latent/run_logs/color_sweep"
M = {"sigreg": "sigreg_only_dubins", "jac": "jacobian_w1_dubins|lewm_dubins_jacw1_s100", "jac+adv_h": "lewm_dubins_jac_safeadv_linf50"}
PAT = (r"REAL AUC ([\d.]+) bal ([\d.]+) .*IMAG AUC ([\d.]+) bal ([\d.]+) .*"
       r"AUC@k 1:([\d.]+) 5:([\d.]+) 10:([\d.]+) 25:([\d.]+)")
NAMES = ["real AUC", "real bal acc", "imag AUC (all steps)", "imag bal acc", "imag AUC k=1", "imag AUC k=5",
         "imag AUC k=10", "imag AUC k=25"]
D, DPX = collections.defaultdict(list), {}
for f in glob.glob(os.path.join(L, "*.log")):
    for l in open(f):
        if not l.startswith("SWEEP"):
            continue
        ck = l.split()[1]
        mod = [k for k, v in M.items() if re.fullmatch(f"({v})(_s\\d)?", ck)]
        if not mod:
            continue
        hd = "gp" if "margin_gp" in l else "lin"
        a = float(re.search(r"a=([\d.]+)", l).group(1))
        DPX[a] = re.search(r"dpx=([\d.]+)", l).group(1)
        D[(mod[0], hd, a)].append([float(x) for x in re.search(PAT, l).groups()])
fmt = lambda v: "" if not v else (f"{np.mean(v):.3f} ± {np.std(v):.3f}" if len(v) > 1 else f"{v[0]:.3f}")
A = sorted(DPX)
for hd in ("gp", "lin"):
    for j, name in enumerate(NAMES):
        print(f"\n{hd}: {name}\n")
        print("| WM | " + " | ".join(f"{a:g} ({DPX[a]})" for a in A) + " |")
        print("|---|" + "---|" * len(A))
        for mod in M:
            n = len(D[(mod, hd, A[0])])
            print(f"| {mod} (n={n}) | " + " | ".join(fmt([x[j] for x in D[(mod, hd, a)]]) for a in A) + " |")
