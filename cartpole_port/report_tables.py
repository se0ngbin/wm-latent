"""Markdown tables for the cartpole report section, straight from /data/seongbin/lewm/cartpole_eval/*.json."""
import json

D = "/data/seongbin/lewm/cartpole_eval"
ROWS = [("sigreg", "cp_baseline"), ("jacobian", "cp_jacobian"), ("jac+pull", "cp_jacpull"),
        ("sigreg + L∞ safety reg", "cp_safeadv_linf_g0"), ("jacobian + L∞ safety reg", "cp_jac_safeadv_linf"),
        ("jac+pull + L∞ safety reg", "cp_jacpull_safeadv_linf50"), ("sigreg + λ batch-detached", "cp_lam_bdet50"),
        ("jacobian + λ batch-detached", "cp_jac_lam_bdet"), ("jac+pull + λ batch-detached", "cp_jacpull_lam_bdet")]
VIEWS = ("zoom_out", "pan_x", "pan_y")


def f(ms): return f"{ms[0]:.3f} ± {ms[1]:.3f}"


def table(head, cols, get):
    print("| model | " + " | ".join(head) + " |\n|---|" + "---|" * len(head))
    for lbl, m in ROWS:
        print(f"| {lbl} | " + " | ".join(get(m, c) for c in cols) + " |")
    print()


main = {m: json.load(open(f"{D}/{m}.json"))["summary"] for _, m in ROWS}
vp = {m: json.load(open(f"{D}/{m}_zoom_out+pan_x+pan_y.json"))["summary"] for _, m in ROWS}
cols_c = [("margin", "failed_auc"), ("margin", "doomed_auc"), ("critic", "doomed_auc"), ("critic", "unsafe_auc"),
          ("probe", "doomed_auc"), ("critic", "sign_acc")]
hdr = ["margin failed AUC", "margin doomed AUC", "critic doomed AUC", "critic unsafe AUC", "probe doomed AUC", "critic sign acc"]
for cond in ("tan", "purple"):
    print(f"#### {cond}\n")
    table(hdr, cols_c, lambda m, c: f(main[m][cond][c[0]][c[1]]))
sc = json.load(open(f"{D}/state_ceiling.json"))["summary"]
print("state ceiling: " + " ".join(f"{k}={f(v)}" for k, v in sc.items()) + "\n")
for head, key in (("critic", "doomed_auc"), ("margin", "failed_auc"), ("probe", "doomed_auc"), ("critic", "sign_acc")):
    print(f"#### viewpoint {head} {key}\n")
    table([*VIEWS, "mean of views"], [*VIEWS, "mean"],
          lambda m, c: f"{sum(vp[m][v][head][key][0] for v in VIEWS) / 3:.3f}" if c == "mean" else f(vp[m][c][head][key]))
