"""Markdown tables for the cartpole report section, straight from /data/seongbin/lewm/cartpole_eval/*.json and the
shift_diag logs."""
import json, re

D = "/data/seongbin/lewm/cartpole_eval"
LOGS = "/home/seongbin/latent/run_logs/cartpole_eval"
ROWS = [("sigreg", "cp_baseline"), ("jacobian", "cp_jacobian"), ("jac+pull", "cp_jacpull"),
        ("L∞ safety reg", "cp_safeadv_linf_g0"), ("jacobian + L∞", "cp_jac_safeadv_linf"),
        ("jac+pull + L∞", "cp_jacpull_safeadv_linf50"), ("λ batch-detached", "cp_lam_bdet50"),
        ("jacobian + λ", "cp_jac_lam_bdet"), ("jac+pull + λ", "cp_jacpull_lam_bdet")]
COLS = [("base", "base"), ("color", "purple"), ("zoom_out", "zoom_out"), ("pan_x", "pan_x"), ("pan_y", "pan_y")]


def f(ms): return f"{ms[0]:.3f} ± {ms[1]:.3f}"


def table(title, get):
    print(f"{title}\n\n| Model | " + " | ".join(c for c, _ in COLS) + " |\n|---|" + "---|" * len(COLS))
    for lbl, m in ROWS:
        print(f"| {lbl} | " + " | ".join(get(m, k) for _, k in COLS) + " |")
    print()


mar = {m: json.load(open(f"{D}/{m}_margin.json")) for _, m in ROWS}
sf = mar[ROWS[0][1]]["safe_frac"]
table("AUC", lambda m, k: f(mar[m]["summary"][k]["auc"]))
table(f"Sign accuracy (h >= 0 = safe; the test set is {sf:.3f} not failed, so {sf:.3f} = everything called safe, "
      f"{1 - sf:.3f} = everything unsafe)", lambda m, k: f(mar[m]["summary"][k]["sign_acc"]))

# value-function section: tan/purple from <model>.json, views from <model>_zoom_out+pan_x+pan_y.json
val = {m: {**json.load(open(f"{D}/{m}.json"))["summary"], **json.load(open(f"{D}/{m}_zoom_out+pan_x+pan_y.json"))["summary"]}
       for _, m in ROWS}
for head in ("critic", "probe"):
    table(f"{head} doomed AUC", lambda m, k: f(val[m]["tan" if k == "base" else k][head]["doomed_auc"]))
sc = json.load(open(f"{D}/state_ceiling.json"))["summary"]
print("state ceiling: " + " ".join(f"{k}={f(v)}" for k, v in sc.items()) + "\n")

# lost vs relocated (shift_diag.py): retrained heads on the shifted latents
lbl = dict((m, l) for l, m in ROWS)
print("| Model | shift | latent move | margin zero-shot | margin retrained | probe zero-shot | probe retrained |\n|---|---|---|---|---|---|---|")
for cond, log in (("zoom_out", "diag_zoom.log"), ("color", "diag_purple.log")):
    for line in open(f"{LOGS}/{log}"):
        if not line.startswith("KEY"): continue
        t = line.split(); kv = dict(re.findall(r"(\w+)=([\d.±]+)", line))
        g = lambda k: kv[k].replace("±", " ± ")
        print(f"| {lbl[t[2]]} | {cond} | {float(kv['disp']):.3f} | {g('m_zs')} | {g('m_re')} | {g('p_zs')} | {g('p_re')} |")
