# Paper summary: what can go in the paper (Dubins, as of 2026-11-08)

Distilled from `post-oct-21.md`, `paper-results.tex` and the 2026-11-07/08 runs (branch `worktree-imag-margin`). Only results that hold across seeds or in paired tests are listed as claims; single-seed and pending items are marked. OOD conditions (purple obstacles, 90° rotation) are test-only: no model, margin function or threshold is trained on them, except where a section says "test-time" and states exactly what it uses.

## Setup in one paragraph

LeWM world model (ViT-tiny encoder $f$, transformer predictor $P$, frameskip 1, 50 epochs, 3 world-model seeds) on a Dubins car with two circular obstacles. A margin function $h$ (MLP 192-512-512-1 or linear $w^\top z + b$, gradient-penalty margin loss) is trained afterwards on frozen latents of red, training-orientation frames with failure labels. Planning: LeWM's CEM (300 samples, 30 elites, 30 iterations, open loop), goal cost $J = \|\hat z_H - z_g\|^2$; CCEM keeps plans with $h > 0$ on every state (current + imagined) and, if fewer than 30 are feasible, adds the infeasible ones with the highest $\min_k h(\hat z_k)$. Dangerous episodes: goals $K$ steps out whose straight line crosses an obstacle. Paired tests: exact McNemar on the same episodes.

## Claim 1: jac makes a red-trained margin function transfer to shifted real frames

Real-frame AUC of $h$ (4000 uniform states per condition, 3 world-model seeds, `paper-results.tex` Table 1):

| Model | Red, linear | Red, MLP | Purple, linear | Purple, MLP | Rot. 90°, linear | Rot. 90°, MLP |
| --- | --- | --- | --- | --- | --- | --- |
| sigreg (LeWM default) | 0.981 ± 0.005 | 0.987 ± 0.006 | 0.543 ± 0.100 | 0.170 ± 0.116 | 0.637 ± 0.503 | 0.591 ± 0.428 |
| jac | 0.998 ± 0.001 | 0.998 ± 0.000 | 0.970 ± 0.017 | 0.648 ± 0.088 | 0.984 ± 0.010 | 0.986 ± 0.008 |
| jac+pull | 0.997 ± 0.000 | 0.998 ± 0.001 | 0.574 ± 0.362 | 0.423 ± 0.396 | 0.983 ± 0.004 | 0.983 ± 0.004 |
| jac+pull+adv_h | 0.998 ± 0.000 | 0.999 ± 0.000 | 0.986 ± 0.015 | 0.925 ± 0.030 | 0.994 ± 0.002 | 0.994 ± 0.001 |

- jac (encoder Jacobian Frobenius norm pinned to 1; a sigreg encoder sits at about 10) keeps the ranking under both shifts on every seed with a linear $h$; sigreg loses it.
- Thresholds do not transfer for any model: purple balanced sign accuracy is 0.500 everywhere (all states on one side of 0). Rotation keeps it partly (jac 0.891 ± 0.017 linear).
- A linear readout is the robust one: it matches the MLP on red and ranks purple far better on every jac-base model (jac 0.970 vs 0.648).

## Claim 2: that transfer is what makes safe planning work out of distribution

Goal-distance sweep, 3 seeds, 50 episodes per seed (`paper-results.tex` Table "K = 15"), success / collision (%):

| Model | Raw CEM, purple | CCEM MLP, purple | CCEM linear, purple | Raw CEM, rot. | CCEM MLP, rot. | CCEM linear, rot. |
| --- | --- | --- | --- | --- | --- | --- |
| sigreg | 39 / 69 | 40 / 71 | 40 / 67 | 45 / 56 | 45 / 56 | 43 / 57 |
| jac | 22 / 74 | 43 / 32 | 49 / 31 | 46 / 69 | 50 / 33 | 53 / 37 |

- Without a safety term both collide about equally on purple; the red-trained $h$ cuts jac's purple collisions by more than half and does nothing for sigreg. Paired per seed (50 episodes, CCEM, purple): jac beats sigreg on all 3 seed pairs with both $h$ (p ≤ 0.024); rotation favors jac on all 6 pairs (3 significant).
- In distribution (red) CCEM keeps jac at 7-13% collision.

## Claim 3 (negative): safety regularizers on top of jac do not improve planning

- Over the 21-model grid, no model beats jac in paired CCEM tests at any goal distance (10/15/20). jac+pull, the encoder-only safety regs and jac+pull+adv_λ tie it; full safety regs on the jac base are mostly worse.
- 200 paired episodes, goals 15 steps out, linear $h$ on uniform + predicted latents: jac+pull+adv_h encoder-only (3f) and its linear-head variant (3g) collide 2-4× more than jac on red (10.5-18.5% vs 4-8.5%); jac+pull collides more than jac under rotation (30.0% vs 20.0%, p = 0.006).
- So the robustness on real frames from adv_h (Claim 1, last row) does not reach planning.
- Pull: its target 1 was never tuned; the unregularized predictor's action ratio is 2.2-2.3, and pull only constrains the 3-frame context (1-frame context stays at 2.04), so it halves control authority at full context. Measured (`reg_target_check.py`).

## Claim 4 (mechanism): out of distribution, $h$ loses its distance information within about 10 environment steps of imagination

Per-step AUC of $h$ on imagined latents (600 held-out starts, random CEM-style plans, simulator labels), jac:

| Start, h | step 1 | 5 | 10 | 15 | 25 |
| --- | --- | --- | --- | --- | --- |
| red, MLP | 1.00 | 1.00 | 0.99 | 0.97 | 0.90 |
| purple, linear | 0.93 | 0.74 | 0.41 | 0.46 | 0.53 |
| purple, MLP | 0.53 | 0.51 | 0.45 | 0.46 | 0.54 |
| rot. 90°, linear | 0.97 | 0.76 | 0.49 | 0.46 | 0.42 |

- The same holds for all 14 models screened (best OOD AUC at step 10: 0.60; at step 15: 0.61).
- Cause (purple): the predictor tracks the purple world about as well as the red one (error 0.57 vs 0.53 of the latent variance at step 10), but in the purple region $h$'s safe/unsafe gap is about 30× smaller (0.06 vs 1.73), so prediction noise erases the ranking.
- Plan level (veto-vs-distance curve, `veto_curve_dubins.py`; threshold set per model to veto 90% of truly unsafe plans; safe-plan veto rate binned by the plan's closest true distance to an obstacle): on red it falls from 0.76 near the boundary to 0.12 by 0.2 radii; on purple it is flat at 0.87-0.92, so the planner cannot tell a near miss from a crash. This is why jac's purple CCEM falls back to "safest-looking plans" (h flags nearly every candidate unsafe) and stalls.
- The decay follows environment time, not the number of predictor steps: a frameskip-5 jac+pull model (each predictor step = 5 env steps) loses the purple ranking at the same env step (0.53-0.56 at env step 10).

## Claim 5 (negative, planner side): no thresholding or planner rule recovers it

Each tested in paired runs; none beats CCEM with threshold 0 out of distribution:

- per-step threshold calibration on predicted latents (labeled or safe-only quantile offsets; port of the Reacher procedure): no model passes the gate (AUC ≥ 0.8 at step 15) under purple or rotation; on red it never lowers collisions and raises them in one cell (jac, $K = 15$: 4% → 8-9%).
- threshold-free rules (safest-50-then-best-goal, best-goal-50-then-safest, per-step z-scored $h$, a dynamic threshold, a cost relative to the current frame): all collide as much or more.
- $h$ trained on predicted latents: helps on red, not under purple.
- frameskip 5 (jac+pull, pull target 2.2): matches jac on red at $K$ = 15-20 but collides about twice as often on purple (35-46% vs 17-28%) and rotation (34-50% vs 21-28%).

## Claim 6 (test-time, label-free): BatchNorm statistics adaptation fixes the per-step ranking but not planning beyond short horizons

Procedure (`bn_adapt_save.py`): recompute the running statistics of the two BatchNorm layers (encoder projector; predictor output projector, from 25-step random-action imagined rollouts) on unlabeled frames of the shifted condition; weights and $h$ unchanged. Realistic variant: only safe frames (500 suffice).

- Imagined purple ranking (jac, MLP $h$): 0.45 → 0.87 at step 10, 0.46 → 0.79 at step 15. Rotation is not fixed (0.48 at step 10).
- But the plan-level veto curve on purple stays flat for 15-step plans; for 10-step plans it gains a slope (0.77 near → 0.31 beyond one radius).
- CCEM on purple (100 paired episodes): at $K = 10$ success rises 73% → 85-86% (p = 0.024 / 0.036) with collisions unchanged (13-14% → 15%); at $K$ = 15-25 collisions roughly double (36-62% vs 17-28%), because adaptation removes the conservative fallback while plans still carry no distance information. A threshold from the safe frames (10% quantile) does not help.
- sigreg: adaptation lowers purple collisions in 17 of 18 cells (up to −27 points, 7 significant), as on Reacher, but adapted sigreg (38-70%) stays far worse than unadapted jac (17-28%).

## Suggested paper framing

- Main result: an encoder Jacobian regularizer (jac) makes a margin function trained on in-distribution frames transfer to appearance and geometric shifts on real frames, which is what lets constrained CEM stay safer out of distribution than the LeWM default (Claims 1-2).
- Negative results that sharpen it: label-based adversarial safety regularizers improve real-frame robustness but not planning (Claim 3).
- Limitation, with a precise diagnosis: imagined OOD states lose distance information within about 10 environment steps (Claim 4); planner-side fixes cannot recover information that is not there (Claim 5); test-time BN adaptation recovers the average ranking and helps only for short horizons (Claim 6).

## Unconfirmed or pending (not paper-ready yet)

- jac + predictor-gap term (imagined latents must stay on the correct side of an online safety head over a 5-step rollout): at 15 epochs it beat its 15-epoch jac control by 12-15 collision points in every condition (p ≤ 0.005), not purple-specific, single seed. 50-epoch run training (`train_jac_predgap50.sh`, ETA about 11:30 pm PST 2026-11-08).
- Seed variance is large for planning: jac's three seeds span 15.5-29.5% purple collision under one protocol, so single-seed model comparisons within about 10 points are not claims.
- Rows 2c, 2g, 3c and several second/third seeds of the 21-model grid are not trained; CEM tables in `paper-results.tex` are mostly seed 3072.
- Not run: diamond in the planning tables; replanning (every 5 steps) beyond the archived ablation.

## Where things are

- Tables: `paper-results.tex` (generated by `le-wm/scripts/make_paper_results.py`); full report `post-oct-21.md`; superseded results `archive.md`.
- Recent scripts (le-wm/scripts): `head_decision_table.py` (paired tests), `calib_pred_dubins.py`, `diag_imag_fs_dubins.py` (per-step screen, any frameskip, `BNADAPT=1`), `veto_curve_dubins.py`, `bn_adapt_save.py`, `bn_safe_cem.sh`, `bn_sigreg_purple.sh`, `fs5_cem.sh`, `train_fs5_jacpull.sh`, `train_jac_predgap50.sh`, `reg_target_check.py`.
