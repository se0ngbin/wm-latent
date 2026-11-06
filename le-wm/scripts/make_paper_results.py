"""Build paper-results.tex (Dubins paper grid) from the run logs: 21 models x {red, purple, rotated 90°} x 3 WM seeds.
Tables: real-frame AUC and balanced sign accuracy (linear and MLP margin functions), raw CEM and safety-first CEM
(open loop, final-step goal cost), and replanning every 5 steps (to run). Means (± std) over the seeds done; a blank
cell is not done; a dagger marks rows with fewer than 3 seeds done. CEM cells use only 50-episode runs.
Usage: python make_paper_results.py [out.tex]   (default /home/seongbin/latent/paper-results.tex)"""
import os, re, sys, numpy as np
L = "/home/seongbin/latent/run_logs/"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/seongbin/latent/paper-results.tex"
D = "lewm_dubins_"
# row, LaTeX name, [(ckpt dir, seed label, CEM tag)] for the 3 paper seeds
def seeds(base, tag, extra=None):
    s = [(base, "3072", tag), (base + "_s1", "1", tag + "_s1"), (base + "_s2", "2", tag + "_s2")]
    return s if extra is None else s[:2] + [extra]
ROWS = [
 ("1a", "sigreg", seeds("sigreg_only_dubins", "sigreg")),
 ("1b", "adv$_h$", seeds(D + "safeadv_linf50", "linf")),
 ("1c", "adv$_h$, lin.\\ head", seeds(D + "safeadv_linf_linhead50", "linf_linhead")),
 ("1d", "adv$_\\lambda$", seeds(D + "lam_bdet50", "lam_bdet")),
 ("1e", "adv$_\\lambda$, enc.", seeds(D + "lam_bdet_enc50", "lam_bdet_enc")),
 ("1f", "adv$_h$, enc.", seeds(D + "sa_enc50", "sa_enc")),
 ("1g", "adv$_h$, lin.\\ head, enc.", seeds(D + "sa_enc_linhead50", "sa_enc_linhead")),
 ("2a", "jac", seeds("jacobian_w1_dubins", "jacobian", (D + "jacw1_s100", "100", "jacobian_s100"))),
 ("2b", "jac+adv$_h$", seeds(D + "jac_safeadv_linf50", "jac_linf")),
 ("2c", "jac+adv$_h$, lin.\\ head", seeds(D + "jac_safeadv_linf_linhead50", "jac_linf_linhead")),
 ("2d", "jac+adv$_\\lambda$", seeds(D + "jac_lam_bdet50", "jac_lam_bdet")),
 ("2e", "jac+adv$_\\lambda$, enc.", seeds(D + "jac_lam_bdet_enc50", "jac_lam_bdet_enc")),
 ("2f", "jac+adv$_h$, enc.", seeds(D + "jac_safeadv_linf_enc50", "jac_linf_enc")),
 ("2g", "jac+adv$_h$, lin.\\ head, enc.", seeds(D + "jac_safeadv_linf_enc_linhead50", "jac_linf_enc_linhead")),
 ("3a", "jac+pull", seeds(D + "jacpull50", "jacpull")),
 ("3b", "jac+pull+adv$_h$", seeds(D + "jacpull_safeadv_linf50", "jacpull_linf")),
 ("3c", "jac+pull+adv$_h$, lin.\\ head", seeds(D + "jacpull_safeadv_linf_linhead50", "jacpull_linf_linhead")),
 ("3d", "jac+pull+adv$_\\lambda$", seeds(D + "jacpull_lam_bdet50", "jacpull_lam_bdet")),
 ("3e", "jac+pull+adv$_\\lambda$, enc.", seeds(D + "jacpull_lam_bdet_enc50", "jacpull_lam_bdet_enc")),
 ("3f", "jac+pull+adv$_h$, enc.", seeds(D + "jacpull_safeadv_linf_enc50", "jacpull_linf_enc")),
 ("3g", "jac+pull+adv$_h$, lin.\\ head, enc.", seeds(D + "jacpull_safeadv_linf_enc_linhead50", "jacpull_linf_enc_linhead")),
]
CONDS = [("red", "red", "red"), ("purple", "purple", "purple"), ("rot90", "fullrot90", "rot.\\ 90$^\\circ$")]

# ---- real-frame margin-function metrics
R = {}
for f in sorted(os.listdir(L)):
    if f in ("lin_h_metrics.log",) or f.startswith("seed_head_eval_"):
        for l in open(L + f, errors="ignore"):
            if l.startswith("HEADOOD"):
                ck, hd = l.split()[1], ("lin" if "margin_lin" in l else "gp")
                for c, a, b in re.findall(r"(\w+) AUC ([\d.]+) bal-acc ([\d.]+)", l):
                    R[(ck, hd, c)] = (float(a), float(b))


def num(v, p=3):
    if not v:
        return ""
    return f"{v[0]:.{p}f}" if len(v) == 1 else f"${np.mean(v):.{p}f} \\pm {np.std(v, ddof=1):.{p}f}$"


def seeds_cell(done):
    s = ", ".join(done)
    return s + ("" if len(done) == 3 else "$^\\dagger$")


def real_rows(k):
    out = []
    for r, name, ss in ROWS:
        done = [(ck, lab) for ck, lab, _ in ss if (ck, "gp", "red") in R]
        cells = [num([R[(ck, h, c)][k] for ck, _ in done]) for c, _, _ in CONDS for h in ("lin", "gp")]
        out.append(f"{r} & {seeds_cell([lab for _, lab in done])} & " + " & ".join(cells) + " \\\\")
    return out


# ---- CEM (50-episode runs only)
def cem_val(tag, cond, planner, rec):
    suf = {"raw": f"{tag}_gr_w0", "sf": f"{tag}_gp_gr_w1000_c10000"}[planner]
    suf += "" if rec == 25 else f"_rec{rec}_gmean"
    p = f"{L}softcem_{suf}_n50_{cond}.log"
    if not os.path.exists(p):
        return None
    t = open(p, errors="ignore").read()
    s, c = re.findall(r"'success_rate': ([\d.]+)", t), re.findall(r"'collision_rate': ([\d.]+)", t)
    return (float(s[-1]), float(c[-1])) if s and c else None


def cem_rows(planner, rec):
    out = []
    for r, name, ss in ROWS:
        per = {c: [(lab, cem_val(tag, cl, planner, rec)) for _, lab, tag in ss] for c, cl, _ in CONDS}
        done = [lab for _, lab, _ in ss if all(cem_val(t, cl, planner, rec) for (_, l2, t) in ss if l2 == lab for _, cl, _ in CONDS)]
        cells = []
        for c, _, _ in CONDS:
            v = [x for _, x in per[c] if x]
            cells.append("" if not v else f"{np.mean([a for a, _ in v]):.1f} / {np.mean([b for _, b in v]):.1f}" + ("" if len(v) == 3 else f" ({len(v)})"))
        out.append(f"{r} & {seeds_cell(done)} & " + " & ".join(cells) + " \\\\")
    return out


def table(caption, label, header, rows, colspec):
    return "\n".join(["\\begin{table*}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
                      f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header, "\\midrule"]
                     + [r + ("\n\\midrule" if r.startswith(("1g", "2g")) else "") for r in rows]
                     + ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])


REAL_H = ("Row & Seeds & \\multicolumn{2}{c}{Red} & \\multicolumn{2}{c}{Purple} & \\multicolumn{2}{c}{Rot.\\ 90$^\\circ$} \\\\\n"
          "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n & & linear & MLP & linear & MLP & linear & MLP \\\\")
CEM_H = "Row & Seeds & Red & Purple & Rot.\\ 90$^\\circ$ \\\\"
NOTE = ("Dubins, 21 world models (base sigreg / jac / jac+pull $\\times$ safety reg), three world-model seeds each "
        "(3072, 1, 2; jac uses 3072, 1, 100), as in LeWM (results averaged over 3 training seeds). Every world model is "
        "trained for 50 epochs on the same dataset. Mean ($\\pm$ std) over the seeds done; a blank cell is not done yet; "
        "$^\\dagger$ = fewer than 3 seeds done (the Seeds column lists them; a number in parentheses in a CEM cell is the "
        "number of seeds averaged). Rows: 1 = sigreg, 2 = jac, 3 = jac+pull base; a = no safety reg, b = adv$_h$, c = adv$_h$ with a linear internal head, d = adv$_\\lambda$, e = adv$_\\lambda$ encoder-only, f = adv$_h$ encoder-only, g = adv$_h$ with a linear internal head, encoder-only.")
doc = ["% Generated by le-wm/scripts/make_paper_results.py from run_logs; do not edit by hand, rerun the script.",
       "\\documentclass{article}", "\\usepackage[margin=0.7in,landscape]{geometry}", "\\usepackage{booktabs}",
       "\\begin{document}", "", "\\section*{Dubins paper grid}", "", NOTE, "",
       table("Real-frame AUC of the margin function (trained on red frames only), linear ($w^\\top z + b$) and MLP (192-512-512-1), both trained with the same gradient-penalty loss. 4000 uniform states "
             "per condition, disjoint from training.", "tab:dubins_auc", REAL_H, real_rows(0), "lp{2.2cm}cccccc"),
       table("Real-frame balanced sign accuracy (0.5 = every state on one side of 0).", "tab:dubins_signacc", REAL_H,
             real_rows(1), "lp{2.2cm}cccccc"),
       table("Raw CEM (no safety cost), open loop, success / collision (\\%), 50 episodes per seed.", "tab:dubins_cem_raw",
             CEM_H, cem_rows("raw", 25), "lp{2.2cm}ccc"),
       table("Safety-first CEM (MLP margin function, $w = 1000$, $C = 10^4$), open loop, success / collision (\\%), "
             "50 episodes per seed.", "tab:dubins_cem_sf", CEM_H, cem_rows("sf", 25), "lp{2.2cm}ccc"),
       table("Safety-first CEM, replanning every 5 steps (goal cost at every imagined step), success / collision (\\%), "
             "50 episodes per seed. To be run.", "tab:dubins_cem_rec5", CEM_H,
             [f"{r} & & & & \\\\" for r, _, _ in ROWS], "lp{2.2cm}ccc"),
       "\\end{document}", ""]
open(OUT, "w").write("\n".join(doc))
print("wrote", OUT)
