"""Table + cross-model correlations: rollout_diag JSONs vs raw CEM (sg25clean, red) success / collision."""
import json, glob, re, os
import numpy as np
from scipy.stats import spearmanr
L = "/home/seongbin/latent/run_logs"
R = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob("/data/seongbin/lewm/safeadv_results/rollout_diag/*.json")}


def cem(tag):
    for f in (f"{L}/rawcem_sg25_{tag}_red.log", f"{L}/softcem_{tag}_w0_red.log"):
        if os.path.exists(f):
            t = open(f, errors="ignore").read()
            s, c = re.findall(r"'success_rate': ([0-9.]+)", t), re.findall(r"'collision_rate': ([0-9.]+)", t)
            if s and c: return float(s[-1]), float(c[-1])
    return None


rows = []
print("| model | nerr@5 | nerr@25 | goal ρ real | goal ρ imag | GP AUC real | GP AUC imag | HJ AUC real | HJ AUC imag | raw CEM |")
print("|---|---|---|---|---|---|---|---|---|---|")
for t, r in sorted(R.items(), key=lambda kv: -(cem(kv[0]) or (0, 0))[0]):
    c = cem(t); g = lambda k: f"{r[k]:.3f}" if k in r else ""
    print(f"| {t} | {r['nerr'][4]:.4f} | {r['nerr'][-1]:.3f} | {g('goal_spearman_real')} | {g('goal_spearman_imag')} | "
          f"{g('gp_auc_real_all')} | {g('gp_auc_imag_all')} | {g('hj_auc_real_all')} | {g('hj_auc_imag_all')} | "
          f"{'' if c is None else f'{c[0]:.1f} / {c[1]:.1f}'} |")
    if c: rows.append((t, r, c))
print(f"\nSpearman across {len(rows)} models with raw CEM (success, collision):")
for name, f in (("nerr@5", lambda r: r["nerr"][4]), ("nerr@25", lambda r: r["nerr"][-1]),
                ("goal rho real", lambda r: r["goal_spearman_real"]), ("goal rho imag", lambda r: r["goal_spearman_imag"]),
                ("goal rho near real", lambda r: r["goal_spearman_near_real"]),
                ("goal rho near imag", lambda r: r["goal_spearman_near_imag"])):
    x = [f(r) for _, r, _ in rows]
    print(f"  {name:20s} success {spearmanr(x, [c[0] for *_, c in rows]).correlation:+.2f}   "
          f"collision {spearmanr(x, [c[1] for *_, c in rows]).correlation:+.2f}")
