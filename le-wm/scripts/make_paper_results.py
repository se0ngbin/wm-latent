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


def seeds_cell(done, partial=()):
    """done: seeds with every cell of the row; partial: seeds with some but not all cells (still running)."""
    s = ", ".join(done) + ("" if len(done) == 3 else "$^\\dagger$")
    return s + (f"; {', '.join(partial)} partial" if partial else "")


def real_rows(k):
    out = []
    for r, name, ss in ROWS:
        done = [(ck, lab) for ck, lab, _ in ss if (ck, "gp", "red") in R]
        cells = [num([R[(ck, h, c)][k] for ck, _ in done]) for c, _, _ in CONDS for h in ("lin", "gp")]
        out.append(f"{r} & {seeds_cell([lab for _, lab in done])} & " + " & ".join(cells) + " \\\\")
    return out


# ---- CEM (50-episode runs only)
GOALK = int(os.environ.get("GOALK", 25))   # goal distance of the paper grid (soft_cem.sh GOALK); crossing goals


def cem_val(tag, cond, planner, rec, K=None):
    K = K or GOALK
    k = "" if K == 25 else f"_k{K}"
    rule = os.environ.get("CCEMRULE", "minh")   # minh: all feasible, then highest worst-case h; "" = SafeDreamer's rule
    suf = {"raw": f"{tag}_gr_w0{k}_cross", "ccem": f"{tag}_gp_gr_w1000{k}_cross_ccem{rule}",
           "ccemlin": f"{tag}_lin_gr_w1000{k}_cross_ccem{rule}",
           "ccemlimag": f"{tag}_limag_gr_w1000{k}_cross_ccem{rule}"}[planner]
    suf += "" if rec == K else f"_rec{rec}"
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
        partial = [lab for _, lab, t in ss if lab not in done and any(cem_val(t, cl, planner, rec) for _, cl, _ in CONDS)]
        out.append(f"{r} & {seeds_cell(done, partial)} & " + " & ".join(cells) + " \\\\")
    return out


def sweep_rows(K):
    """Success / collision for raw, CCEM-MLP, CCEM-linear on red / purple / rot 90 at goal distance K, mean over the
    world-model seeds done (Seeds column: seeds with all 9 cells done; (n) in a cell = number of seeds it averages)."""
    out = []
    for r, name, ss in ROWS:
        cells, done = [], []
        for _, lab, tag in ss:
            if all(cem_val(tag, cl, p, K, K) for p in ("raw", "ccem", "ccemlin") for _, cl, _ in CONDS):
                done.append(lab)
        for planner in ("raw", "ccem", "ccemlin"):
            for _, cl, _ in CONDS:
                v = [x for x in (cem_val(tag, cl, planner, K, K) for _, _, tag in ss) if x]
                cells.append("" if not v else f"{np.mean([a for a, _ in v]):.0f} / {np.mean([b for _, b in v]):.0f}"
                             + ("" if len(v) == 3 else f" ({len(v)})"))
        partial = [lab for _, lab, tag in ss if lab not in done
                   and any(cem_val(tag, cl, p, K, K) for p in ("raw", "ccem", "ccemlin") for _, cl, _ in CONDS)]
        out.append(f"{r} & {seeds_cell(done, partial)} & " + " & ".join(cells) + " \\\\")
    return out


SWEEP_H = ("Row & Seeds & \\multicolumn{3}{c}{Raw CEM} & \\multicolumn{3}{c}{CCEM, MLP $h$} & \\multicolumn{3}{c}{CCEM, linear $h$} \\\\\n"
           "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}\n"
           " & & red & purple & rot. & red & purple & rot. & red & purple & rot. \\\\")


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
        "$^\\dagger$ = fewer than 3 seeds done (the Seeds column lists the seeds with every cell of the row done, and "
        "\"partial\" seeds with only some cells done so far; a number in parentheses in a CEM cell is the number of seeds "
        "that cell averages, which can exceed the fully done seeds while runs are in progress). Rows: 1 = sigreg, 2 = jac, 3 = jac+pull base; a = no safety reg, b = adv$_h$, c = adv$_h$ with a linear internal head, d = adv$_\\lambda$, e = adv$_\\lambda$ encoder-only, f = adv$_h$ encoder-only, g = adv$_h$ with a linear internal head, encoder-only.")
METHODS = r"""\section{CEM planners}\label{sec:cem}

Both planners choose the car's next $H$ steering commands by imagining their outcome with the world model. They share
everything (episodes, world model, goal cost, CEM loop) and differ in one step: which candidates are kept as elites.
Paths are relative to the \texttt{le-wm} directory unless noted; \texttt{stable\_worldmodel} is version 0.1.1
(\texttt{le-wm/.venv/lib/python3.10/site-packages/stable\_worldmodel}).

\subsection{Task}

A start state and a goal state are taken from one recorded episode of \texttt{dubins\_expert.h5}: the goal is where that
episode was $K$ steps after the start ($K = """ + str(GOALK) + r"""$ here). A pair is used only if (i) the recorded
episode never collides and finishes in the goal region, (ii) the goal is outside every obstacle, and (iii) the straight
line from start to goal passes through an obstacle, so steering straight at the goal crashes and the planner must detour.
The recorded episode is thus a collision-free solution in $K$ steps (\texttt{eval.py}:150--246; options
\texttt{goal\_clean\_source}, \texttt{goal\_crosses\_obstacle}). The car gets $2K$ steps. \emph{Success}: its position comes
within $0.2$ of the goal position (the episode then ends). \emph{Collision}: its center is inside an obstacle at any step
(the episode continues) (\texttt{dubins\_swm\_env.py}:117--130, 179--183). Both are \% of 50 episodes per world-model
seed. Settings: \texttt{scripts/soft\_cem.sh}.

\subsection{Imagining a plan}

A candidate plan is a sequence of $H = K$ steering commands $a = (a_0, \dots, a_{H-1})$ (in the dataset's normalized
action units). The planner sees only images. It encodes the current image and the goal image,
\[ z_0 = f(o_t), \qquad z_g = f(o_g), \]
and rolls the predictor forward one step at a time from $z_0$, feeding back its own predictions:
\[ \hat z_0 = z_0, \qquad \hat z_{k+1} = P\big(\hat z_{\max(0,k-2):k},\ a_{\max(0,k-2):k}\big), \quad k = 0, \dots, H-1, \]
using at most the last 3 latents and actions (\texttt{jepa.py}:61--110). $\hat z_k$ is the predicted latent after $k$
commands.

\subsection{Goal cost $J$}

$J$ scores how far from the goal a plan ends, measured in latent space: the squared Euclidean distance between the
predicted latent after all $H$ commands and the latent of the goal image,
\[ J(a) = \big\lVert \hat z_H - z_g \big\rVert_2^2 = \sum_{d=1}^{192} \big(\hat z_{H,d} - z_{g,d}\big)^2 . \]
Lower is better. Only the end of the plan counts; the path to it is not scored, and $J$ knows nothing about obstacles.
This is LeWM's planning cost, unchanged (\texttt{jepa.py}:112--131).

\subsection{CEM loop (shared)}

LeWM's cross-entropy method (\texttt{stable\_worldmodel/solver/cem.py}:95--283; settings
\texttt{config/eval/solver/cem.yaml}, identical in \texttt{soft\_cem.yaml}). It keeps a Gaussian over plans, one mean
$\mu_k$ and standard deviation $\sigma_k$ per step $k$, and repeats 30 times:
\begin{enumerate}
\item Sample $N = 300$ plans $a^{(n)} = \mu + \sigma \odot \varepsilon^{(n)}$, $\varepsilon^{(n)} \sim \mathcal N(0, I)$;
      plan 1 is the current mean, $a^{(1)} = \mu$.
\item Imagine each plan and score it.
\item Keep $M = 30$ elites (\textbf{this is the only step where the two planners differ}).
\item Refit: $\mu \leftarrow$ mean of the elites, $\sigma \leftarrow$ standard deviation of the elites (per step, unbiased;
      no momentum, no clipping).
\end{enumerate}
It starts from $\mu = 0$, $\sigma = 1$. The plan is the final $\mu$. The car executes all $H$ commands open loop, then
the planner runs again from the new image (\texttt{receding\_horizon} $= H$, \texttt{stable\_worldmodel/policy.py}:379--431;
nothing of the old plan remains, so every plan starts from $\mu = 0$). The environment de-normalizes commands and clips
the steering rate to $[-2, 2]$.

\subsection{Raw CEM}

Elites are the 30 plans with the smallest $J$: LeWM's planner as published. Safety plays no role; any obstacle avoidance
comes from what the world model learned from the data.

\subsection{CCEM: CEM with a safety constraint}

\paragraph{Safety check.} The margin function $h : \mathbb{R}^{192} \to \mathbb{R}$ reads a latent and says
\emph{safe} if $h \ge 0$, \emph{unsafe} if $h < 0$ (MLP 192-512-512-1, trained on red frames with failure labels, then
frozen; \texttt{scripts/train\_gp\_margin\_dubins.py}, loss \texttt{scripts/ood\_margin\_gp\_jepa.py}:96). A plan's
cost counts how many of its states $h$ calls unsafe, the current state included:
\[ C(a) = \frac{1000}{H} \sum_{k=0}^{H} \mathbb 1\big[h(\hat z_k) < 0\big], \qquad
\text{plan is \emph{feasible} if } C(a) < 2 . \]
For $H \le 25$ one unsafe state already gives $C \ge 40$, so \emph{feasible} simply means $h$ calls every state of the
plan safe.

\paragraph{Elite rule.} Let $\mathcal F$ be the feasible plans among the 300.
\begin{itemize}
\item If at least 30 plans are feasible: the elites are the 30 feasible plans with the smallest $J$ (closest to the goal
      among plans predicted safe; unsafe plans are ignored).
\item Otherwise: the elites are the 30 plans with the smallest $C$, i.e.\ the fewest predicted-unsafe states, regardless
      of $J$ (ties: lower index first, as \texttt{jax.lax.top\_k}). The search first moves toward safety, then toward the
      goal.
\end{itemize}
Everything else is the shared loop. In code the rule is a score that the unchanged CEM minimizes
(\texttt{safe\_solver.py}:323--334; check: \texttt{test\_ccem.py}):
\[ s_n = \begin{cases} J_n & |\mathcal F| \ge 30,\ n \in \mathcal F \\ J_n + 10^9 & |\mathcal F| \ge 30,\ n \notin \mathcal F \\
C_n + 10^{-6}\, n & |\mathcal F| < 30 . \end{cases} \]

\paragraph{Source.} The cost, its $1000/H$ scaling, the limit $2$ (its \texttt{osrp} default) and the elite rule are
SafeDreamer's online safety-reward planner (OSRP; \texttt{SafeDreamer/SafeDreamer/behaviors.py}:316--391), which
searches over action sequences in a world model as we do. The same elite rule appears in the constrained cross-entropy
method of Wen \& Topcu (2018), but there it is policy search (CEM over policy parameters, constraining each policy's
expected cumulative cost estimated from rollouts); we use only the rule, in SafeDreamer's trajectory-search form, so the
constraint is on the predicted cost of one plan. Differences from SafeDreamer, because our world models have no reward,
critic or cost head: the objective is $J$ instead of the critic's return, and the per-state cost comes from $h$ instead
of a learned cost model. LeWM's CEM settings are kept so raw and constrained planning differ only in the elite rule
(SafeDreamer: 500 samples, 50 elites, 6 iterations, mean momentum 0.1, actions clipped to $[-1, 1]$, warm start, executes
only the first action). SafeDreamer pads its feasible-index list with index 0, which can repeat plan 0 among the elites;
not reproduced. This is a \emph{constrained} CEM, not a robust one: the world model and $h$ are trusted as they are, with
no hedging against their errors.
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
       table(f"CCEM with the MLP margin function (constrained elite rule: safe plans first; if fewer than 30 are safe, "
             f"all safe plans plus the unsafe ones with the highest worst-case $h$), open loop, same episodes, success / "
             f"collision (\\%), 50 episodes per seed.", "tab:dubins_cem_ccem", CEM_H, cem_rows("ccem", GOALK), "lp{2.2cm}ccc"),
       table("CCEM with the linear margin function, otherwise as the previous table.", "tab:dubins_cem_ccemlin", CEM_H,
             cem_rows("ccemlin", GOALK), "lp{2.2cm}ccc"),
       table("CCEM with the linear margin function trained on real \\emph{and predicted} latents (6000 CEM-style "
             "25-step plans rolled out by the world model, each imagined step labelled by running the plan in the "
             "simulator; plus the 8000 uniform real states), otherwise as the previous table. Run so far: adv$_h$ with "
             "a linear internal head, encoder-only (rows g; 1g and 2g are not trained) and jac (2a) as reference; "
             "blank = not run.", "tab:dubins_cem_ccemlimag", CEM_H, cem_rows("ccemlimag", GOALK), "lp{2.2cm}ccc"),
       "\\clearpage", "\\section*{Goal-distance sweep}", "",
       "How hard the task is as a function of goal distance $K$ (goals $K$ steps ahead, straight-line path through an "
       "obstacle, plan horizon $K$ open loop, budget $2K$; protocol of Section~\\ref{sec:cem}). 50 episodes per cell and "
       "world-model seed, the same 50 episodes for every model and seed at a given $K$. Cells: success / collision (\\%), "
       "mean over the seeds done; the Seeds column lists the seeds with every cell of the row done ($^\\dagger$ = fewer "
       "than 3) and \"partial\" seeds with only some cells done so far, and a number in parentheses in a cell is the number "
       "of seeds that cell averages. Blank: not run (world model "
       "not trained).", "",
       *[table(f"Goal distance $K = {K}$.", f"tab:dubins_sweep_k{K}", SWEEP_H, sweep_rows(K), "lp{2.2cm}ccccccccc")
         for K in (10, 15, 20, 25)],
       "\\clearpage", METHODS,
       "\\end{document}", ""]
open(OUT, "w").write("\n".join(doc))
print("wrote", OUT)
