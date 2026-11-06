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
GOALK = int(os.environ.get("GOALK", 25))   # goal distance of the paper grid (soft_cem.sh GOALK); crossing goals


def cem_val(tag, cond, planner, rec):
    k = "" if GOALK == 25 else f"_k{GOALK}"
    suf = {"raw": f"{tag}_gr_w0{k}_cross", "ccem": f"{tag}_gp_gr_w1000{k}_cross_ccem"}[planner]
    suf += "" if rec == GOALK else f"_rec{rec}"
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
METHODS = r"""\section{CEM planners}\label{sec:cem}

Paths are relative to the \texttt{le-wm} directory unless noted; \texttt{stable\_worldmodel} is version 0.1.1
(\texttt{le-wm/.venv/lib/python3.10/site-packages/stable\_worldmodel}).

\paragraph{Episodes.} A start state $s_0$ and goal $s_g$ are taken from one recorded episode of
\texttt{dubins\_expert.h5}: $s_g$ is the state $K$ steps after $s_0$ ($K = """ + str(GOALK) + r"""$ here). Pairs are kept
only if the source episode never collides and ends in the goal region, the goal is outside every obstacle, and the
straight segment from the start position to the goal position intersects an obstacle disk, so driving straight at the
goal collides (\texttt{eval.py}:150--246, options \texttt{goal\_clean\_source}, \texttt{goal\_crosses\_obstacle}). The
recorded episode is therefore a collision-free solution in $K$ steps. The episode budget is $2K$ environment steps.
Success: the agent's position comes within $0.2$ of the goal position (the episode then ends); collision: the agent's
center is inside an obstacle disk at any step (the episode continues) (\texttt{dubins\_swm\_env.py}:117--130, 179--183).
Both are reported in \% of 50 episodes per world-model seed. Overrides: \texttt{scripts/soft\_cem.sh}.

\paragraph{World model and rollout.} The current observation $o_t$ and the goal frame $o_g$ are encoded,
$z_0 = f(o_t)$, $z_g = f(o_g)$ (one real frame of context). A candidate is a sequence of $H = K$ steering commands
$a = (a_0, \dots, a_{H-1})$ in the dataset's normalized action units. The predictor rolls out autoregressively,
$\hat z_0 = z_0$, $\hat z_{k+1} = P\big(\hat z_{\max(0,k-2):k},\, a_{\max(0,k-2):k}\big)$, $k = 0, \dots, H-1$, with at
most 3 frames of history (\texttt{jepa.py}:61--110).

\paragraph{Goal cost (LeWM).} $J(a) = \lVert \hat z_H - z_g \rVert_2^2$, summed over the 192 latent dimensions; only the
last imagined state counts (\texttt{jepa.py}:112--131).

\paragraph{CEM (LeWM).} \texttt{stable\_worldmodel/solver/cem.py}:95--283, settings
\texttt{config/eval/solver/cem.yaml} (identical in \texttt{soft\_cem.yaml}). Initialize $\mu = 0$, $\sigma = 1$
($\mu, \sigma \in \mathbb{R}^{H \times 1}$). For $i = 1, \dots, 30$: draw $N = 300$ candidates
$a^{(n)} = \mu + \sigma \odot \varepsilon^{(n)}$, $\varepsilon^{(n)} \sim \mathcal N(0, I)$, with $a^{(1)} = \mu$;
score each, $s_n$; let $E$ be the indices of the $M = 30$ smallest scores; set $\mu \leftarrow \frac{1}{M}\sum_{n \in E}
a^{(n)}$ and $\sigma \leftarrow \mathrm{std}_{n \in E}\, a^{(n)}$ (elementwise, unbiased), with no momentum and no clipping.
The plan is the final $\mu$. The policy executes its first $H$ actions open loop and then replans
(\texttt{receding\_horizon} $= H$; \texttt{stable\_worldmodel/policy.py}:379--431); nothing of the previous plan is left,
so every plan starts from $\mu = 0$. Actions are de-normalized and the steering rate is clipped to $[-2, 2]$ by the
environment.

\paragraph{Raw CEM.} $s_n = J(a^{(n)})$: LeWM's planner, unchanged (\texttt{solver=cem} semantics; run as
\texttt{soft\_cem} with weight 0 and no constraint).

\paragraph{CCEM.} The constrained elite rule of SafeDreamer's online safety-reward planner (OSRP,
\texttt{SafeDreamer/SafeDreamer/behaviors.py}:316--391), inside LeWM's CEM above. This is a planner: CEM searches over
action sequences, and each candidate's cost is that of its own imagined trajectory. The same elite rule appears in the
constrained cross-entropy method of Wen \& Topcu (2018), which is a policy-search method (CEM over policy parameters, with
a constraint on each policy's expected cumulative cost estimated from rollouts); we use only the rule, in SafeDreamer's
trajectory-search form, and the constraint here is on the predicted cost of a single plan, not an expected cost.
The cost model is the margin function $h : \mathbb{R}^{192} \to \mathbb{R}$, $h \ge 0$ read as safe (MLP 192-512-512-1,
trained on red frames, frozen; \texttt{scripts/train\_gp\_margin\_dubins.py}, loss \texttt{scripts/ood\_margin\_gp\_jepa.py}:96).
Per state $c_k = \mathbb 1[h(\hat z_k) < 0]$, $k = 0, \dots, H$ (the current, real state included, as in SafeDreamer's
imagined trajectory), and per candidate
\[ C(a) = \frac{1000}{H} \sum_{k=0}^{H} c_k, \qquad \mathcal F = \{\, n : C(a^{(n)}) < d \,\},\quad d = 2 . \]
If $|\mathcal F| \ge M$, the elites are the $M$ members of $\mathcal F$ with the smallest $J$; otherwise they are the $M$
candidates with the smallest $C$ (ties: lower index first, as \texttt{jax.lax.top\_k}). The mean/std update and execution
are unchanged. Implemented as the score CEM minimizes (\texttt{safe\_solver.py}:323--334; check:
\texttt{test\_ccem.py}):
\[ s_n = \begin{cases} J_n & |\mathcal F| \ge M,\ n \in \mathcal F \\ J_n + 10^9 & |\mathcal F| \ge M,\ n \notin \mathcal F \\
C_n + 10^{-6}\, n & |\mathcal F| < M . \end{cases} \]
Because $c_k \in \{0, 1\}$ and $1000/H \ge 40 > d$ for $H \le 25$, a candidate is feasible exactly when $h \ge 0$ on every
state of its plan. Kept from SafeDreamer: the binary per-state cost, the $1000/H$ scaling, $d = 2$ (its \texttt{osrp}
default), and the elite rule. Different, because our world models have no reward, critic or cost head: the objective is
$-J$ (SafeDreamer: the critic's $\lambda$-return) and $c_k$ comes from the margin function (SafeDreamer: a learned cost
model). LeWM's CEM settings are kept, so the safe and raw planners differ only in the elite rule (SafeDreamer uses 500
samples, 50 elites, 6 iterations, mean momentum 0.1, actions clipped to $[-1, 1]$, warm start, and executes only the first
action). SafeDreamer pads its feasible-index list with index 0, which can repeat candidate 0 among the elites; this is not
reproduced.
"""
doc = ["% Generated by le-wm/scripts/make_paper_results.py from run_logs; do not edit by hand, rerun the script.",
       "\\documentclass{article}", "\\usepackage[margin=0.7in,landscape]{geometry}", "\\usepackage{booktabs}", "\\usepackage{amsmath,amssymb}",
       "\\begin{document}", "", "\\section*{Dubins paper grid}", "", NOTE, "",
       table("Real-frame AUC of the margin function (trained on red frames only), linear ($w^\\top z + b$) and MLP (192-512-512-1), both trained with the same gradient-penalty loss. 4000 uniform states "
             "per condition, disjoint from training.", "tab:dubins_auc", REAL_H, real_rows(0), "lp{2.2cm}cccccc"),
       table("Real-frame balanced sign accuracy (0.5 = every state on one side of 0).", "tab:dubins_signacc", REAL_H,
             real_rows(1), "lp{2.2cm}cccccc"),
       table(f"Raw CEM (LeWM), open loop, goals {GOALK} steps ahead whose straight-line path crosses an obstacle, "
             "success / collision (\\%), 50 episodes per seed (Section~\\ref{sec:cem}).", "tab:dubins_cem_raw",
             CEM_H, cem_rows("raw", GOALK), "lp{2.2cm}ccc"),
       table(f"CCEM (LeWM's CEM with SafeDreamer's constrained elite rule; MLP margin function as the cost model), open "
             f"loop, same episodes, success / collision (\\%), 50 episodes per seed (Section~\\ref{{sec:cem}}).",
             "tab:dubins_cem_ccem", CEM_H, cem_rows("ccem", GOALK), "lp{2.2cm}ccc"),
       table("CCEM, replanning every 5 steps, same episodes, success / collision (\\%), 50 episodes per seed. To be run.",
             "tab:dubins_cem_rec5", CEM_H, [f"{r} & & & & \\\\" for r, _, _ in ROWS], "lp{2.2cm}ccc"),
       "\\clearpage", METHODS,
       "\\end{document}", ""]
open(OUT, "w").write("\n".join(doc))
print("wrote", OUT)
