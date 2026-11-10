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
- Label-free and label-based gap terms (readout consistency, encoder gap, predictor gap, predictor Jacobian; 15-epoch screen) do not keep the purple ranking either; the one that beat its 15-epoch control (predictor gap) is worse than jac at 50 epochs in every cell (see Pending, resolved).
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

## Claim 7 (test-time, label-free): BN adaptation restores the CBF safety filter under purple

Dreamer-framework CBF on the LeWM (red-trained margin function → reachability critic trained by DDPG in the world model's imagination → CBF-filtered diffusion policy, 1000 trajectories, `cbf-results.md`). The filter only reads the current frame and one predicted step, the depth where BN adaptation recovers the ranking. jac, purple obstacles, red-trained margin head and critic unchanged; world model unadapted vs BN-adapted on 500 safe purple frames (`le-wm/scripts/cbf_bn_purple.sh`, outcomes `cbf_outcomes.py`), rates over 1000 trajectories:

| Run | Collision | Success | Safe success | Safe fail | Unsafe success |
| --- | --- | --- | --- | --- | --- |
| no filter | 0.537 | 0.869 | 0.456 | 0.007 | 0.413 |
| jac, red (in distribution), gpfail | 0.065 | 0.761 | 0.759 | 0.176 | 0.002 |
| jac, red (in distribution), gpimag | 0.006 | 0.785 | 0.785 | 0.209 | 0.000 |
| jac, purple, gpfail, unadapted | 0.529 | 0.827 | 0.430 | 0.041 | 0.397 |
| jac, purple, gpfail, BN-adapted | 0.280 | 0.614 | 0.480 | 0.240 | 0.134 |
| jac, purple, gpimag, unadapted | 0.507 | 0.851 | 0.459 | 0.034 | 0.392 |
| jac, purple, gpimag, BN-adapted | 0.211 | 0.576 | 0.461 | 0.328 | 0.115 |

- Unadapted, the filter does nothing under purple (collision equals no filter): the red-trained critic reads purple states as safe.
- BN adaptation halves collision (gpfail 0.529 → 0.280, gpimag 0.507 → 0.211; paired McNemar on the same 1000 starts, p < 1e-30 for both). The drop in raw success is entirely unsafe successes (reaching the goal after driving through an obstacle); safe success is held or higher (0.430 → 0.480, 0.459 → 0.461). The cost is more safe failures (the filter stops the car short of the goal).
- Gap to in distribution: red collision 0.6-6.5%, so adaptation recovers about half to two thirds of the lost safety. Single world-model seed; rotation not tested (BN adaptation does not fix rotation on the imagined screen).
- Contrast with CEM (Claim 6): there BN adaptation hurt beyond 10-step plans, because CCEM scores 10-25 imagined steps, where distance information stays lost.

## Suggested paper framing

- Main result: an encoder Jacobian regularizer (jac) makes a margin function trained on in-distribution frames transfer to appearance and geometric shifts on real frames, which is what lets constrained CEM stay safer out of distribution than the LeWM default (Claims 1-2).
- Negative results that sharpen it: label-based adversarial safety regularizers improve real-frame robustness but not planning (Claim 3).
- Limitation, with a precise diagnosis: imagined OOD states lose distance information within about 10 environment steps (Claim 4); planner-side fixes cannot recover information that is not there (Claim 5); test-time BN adaptation recovers the average ranking and helps only for short horizons (Claim 6).
- Positive OOD result for short-horizon safety: a CBF filter, which needs only the current frame and one predicted step, recovers under purple with label-free BN adaptation on 500 safe frames (collision halved, safe success held; Claim 7). This fits the diagnosis: the shift is fixable where imagination is shallow.

## Unconfirmed or pending (not paper-ready yet)

- jac + predictor-gap term (imagined latents must stay on the correct side of an online safety head over a 5-step rollout): resolved, negative. At 15 epochs it beat its 15-epoch jac control (12-15 fewer collision points), but at 50 epochs it is worse than jac 50 in all 18 cells (goals 10/15/20 steps out, red / purple / rotation, linear and MLP $h$, 100 paired episodes; collisions +11 to +27 points, p ≤ 0.03). On red its veto curve falls off further from the boundary (safe plans 0.1-0.2 radii out vetoed 44% vs 5% for jac at 10 steps) and its $h$ is optimistic on imagined states; purple stays flat (`predgap50_eval.sh`, logs `run_logs/predgap50_*`, `softcem_jacpredgap50_*`). The 15-epoch screen's gain reflected an undertrained control.
- Seed variance is large for planning: jac's three seeds span 15.5-29.5% purple collision under one protocol, so single-seed model comparisons within about 10 points are not claims.
- Rows 2c, 2g, 3c and several second/third seeds of the 21-model grid are not trained; CEM tables in `paper-results.tex` are mostly seed 3072.
- Not run: diamond in the planning tables; replanning (every 5 steps) beyond the archived ablation.

## Where things are

- Tables: `paper-results.tex` (generated by `le-wm/scripts/make_paper_results.py`); full report `post-oct-21.md`; superseded results `archive.md`.
- Recent scripts (le-wm/scripts): `head_decision_table.py` (paired tests), `calib_pred_dubins.py`, `diag_imag_fs_dubins.py` (per-step screen, any frameskip, `BNADAPT=1`), `veto_curve_dubins.py`, `bn_adapt_save.py`, `bn_safe_cem.sh`, `bn_sigreg_purple.sh`, `fs5_cem.sh`, `train_fs5_jacpull.sh`, `train_jac_predgap50.sh`, `reg_target_check.py`.

# Paper Summary pt 2 (Reacher, as of 11-09)

Distilled from `reacher-results.md` (branch `worktree-reacher-imag-margin`, merged into main). Every Reacher world model is a single training seed (3072), so every model comparison below is one seed; paired tests are over the same start/goal pairs. OOD conditions are test-only: no model, margin function or threshold is trained on them, except the test-time items, which state what they use.

## Setup

DMC Reacher with SLS-squared's invisible joint-space box: unsafe iff $\mathrm{wrap}(q_1) \in [0, \pi]$ and $q_2 \in [-2.88, -2.45]$ (3.25% of frames; nothing in the image marks it, so safety is pure arm geometry). LeWM's `reacher_random` data (10k episodes x 200 steps); LeWM world models with LeWM's Reacher settings (ViT-tiny, frameskip 5, history 3, 10 epochs). Models: sigreg (LeWM default), jac, jac+adv_h (MLP adversary head, L∞ 8/255, 2 PGD steps); evaluations in progress for jac+pull(target 2.4)+adv_h and jac+adv_λ; training: jac + whole-latent adversary. Margin functions (MLP 192-512-512-1 and linear) are trained on real plus imagined latents (predictor rollouts of CEM-style plans, labelled by replaying the plan in the simulator). Planning: LeWM's Reacher CEM (300 samples, 30 elites, 10 iterations, 5-block horizon = 25 env steps, replan every 5 blocks, 50-step budget) and CCEM with the Dubins rule (feasible iff $h > 0$ on the current and every predicted state; fallback = highest $\min_k h$). Episodes: held-out start/goal pairs 25 steps apart whose recorded path enters the box (so the recorded route is unsafe); 50 pairs, or 150 for the paired tests. Shifts: floor colour blended toward brown (α = 0.1-1), light intensity, arm colour, and sim-to-real shifts (camera shift 3 cm / tilt 5° / zoom 10 cm, light direction, sensor noise, blur, JPEG, exposure, white balance, occlusion patch; dynamics: weaker motors, damping, 1-step latency, actuation noise).

## R1 (agrees with Claim 1, stronger): jac makes $h$ transfer to shifted frames, thresholds included

Balanced sign accuracy (AUC) on 5000 held-out states, real+imag margin functions:

| Shift | sigreg, MLP | jac, MLP | jac, linear |
| --- | --- | --- | --- |
| floor α = 0.2 | 0.500 (0.493) | 0.997 (0.999) | 0.990 (0.999) |
| light direction | 0.500 (0.906) | 0.997 (1.000) | 0.992 (0.999) |
| JPEG | 0.500 (0.802) | 0.996 (1.000) | 0.991 (0.999) |
| exposure | 0.500 (0.583) | 0.735 (0.998) | 0.980 (0.998) |
| white balance | 0.500 (0.498) | 0.997 (1.000) | 0.991 (0.999) |
| floor α = 0.3 | 0.500 (0.499) | 0.500 (0.873) | 0.578 (0.952) |
| camera shift | 0.500 (0.838) | 0.506 (0.935) | 0.860 (0.973) |
| occlusion | 0.500 (0.691) | 0.592 (0.956) | 0.707 (0.955) |

- Unlike Dubins purple (sign accuracy 0.500 for every model), on Reacher jac keeps the threshold under every photometric shift; it loses it only under large floor shifts (α ≥ 0.3), camera geometry and occlusion, where the ranking still survives. The linear $h$ is the more robust readout, as on Dubins.
- Mechanism (route decomposition, port of the Dubins analysis): jac works by plain invariance, which transfers here (the latent moves 10-20x less than sigreg's under light, floor 0.2, white balance), and does not compress the safe/unsafe gap (unlike Dubins, ×0.44).

## R2 (agrees with Claim 2): that transfer is what makes OOD planning safe

CCEM-MLP on 50 pairs, collision (success): sigreg vs jac. In distribution: sigreg 0.04 (0.60), jac 0.06 (0.60); raw CEM 0.54-0.58.

| Shift | sigreg | jac |
| --- | --- | --- |
| floor α = 0.1 | 0.48 (0.38) | 0.06 (0.66) |
| light 0.3 | 0.42 (0.16) | 0.06 (0.58) |
| light direction | 0.56 (0.16) | 0.06 (0.62) |
| exposure | 0.28 (0.14) | 0.08 (0.68) |
| white balance | 0.60 (0.10) | 0.06 (0.56) |

- Wherever sigreg's $h$ calls everything safe, its CCEM equals raw CEM; jac's CCEM stays at 0.04-0.12 collision under every photometric and sensor shift.

## R3 (differs from Claim 3): jac+adv_h beats jac under camera shift

150 paired episodes, CCEM, collision / safe success; exact McNemar vs jac (episodes better for jac / for the other model, p). Split = jac+adv_h planning with its BN-adapted model while $h$ reads the unadapted model's rollout (see R6).

| Condition | Margin fn | jac | jac+adv_h | jac+adv_h + BN split |
| --- | --- | --- | --- | --- |
| camera shift | MLP | 0.427 / 0.207 | 0.207 / 0.207 (collision 47 / 14, p < 0.001) | 0.193 / 0.193 (49 / 14, p < 0.001) |
| camera shift | linear | 0.253 / 0.133 | 0.147 / 0.173 (29 / 13, p = 0.020) | 0.133 / 0.233 (p = 0.005; safe success p = 0.024) |
| camera tilt | MLP | 0.373 / 0.087 | 0.313 / 0.120 (p = 0.31) | 0.333 / 0.120 (p = 0.52) |
| floor α = 0.3 | MLP | 0.233 / 0.447 | 0.140 / 0.327 (collision p = 0.016; safe success p = 0.018 worse) | 0.267 / 0.500 (n.s.) |
| floor α = 0.4 | MLP | 0.340 / 0.100 | 0.227 / 0.060 (collision p = 0.008; progress 0.27 vs 0.57) | 0.473 / 0.160 (collision p = 0.005 worse) |

- Under camera shift adv_h halves collisions at equal safe success, with both margin functions and both planners. Under floor shifts it is a more conservative operating point, not a better one (fewer collisions because it reaches the goal less); the BN split restores progress and gives the collisions back.
- Why adv_h helps: it sharpens the margin function's boundary under shift (veto-vs-distance curve, 12,800 plans per condition: at floor 0.3 the 50% veto point moves from 0.25 rad for jac to 0.05 rad, in-distribution level; also sharper under camera shift) and halves $h$'s sensitivity to camera shift at every prediction depth.
- adv_h barely costs Reacher's predictor (prediction loss 0.0220 vs jac 0.0218 at epoch 10), unlike Dubins (1.7-6x). On Reacher $h$ is more movable by the 8/255 adversary than on Dubins (24% vs 9% of unsafe states flipped), so the low cost is not because there was nothing to remove.
- Consistent with Dubins: a linear adversary head is worse than jac; the pull target must be calibrated (Reacher's free action ratio is 2.4, not 1). jac + predictor-gap was stopped at epoch 8: on Reacher its term is almost inactive (0-2% of predicted states on the wrong side of the margin, loss about 1e-3) while prediction loss is 8-13% above jac.

## R4 (differs from Claim 4): out of distribution the latent blurs near the boundary; imagined states keep a usable ranking

- Plan-level ranking survives where the threshold fails: at floor 0.3, within-start AUC of the plan score is 0.90 (jac) / 0.97 (jac+adv_h), with 28-35% of unsafe plans passing the 0 threshold.
- The failure is a blur in state resolution near the box, not a flat curve: in distribution the safe-plan veto (at 90% unsafe caught) falls off within about 0.05-0.1 rad of the box; jac at floor 0.3 pushes the 50% point to about 0.25 rad, matching its nearest-neighbour state error (0.29-0.41 rad). Only sigreg, and jac at α ≥ 0.4, go flat like Dubins purple.
- $h$ errs optimistic under shift (calls everything safe), so CCEM degrades to raw CEM; on Dubins it errs pessimistic and CCEM falls back to stalling.
- jac+adv_h in CCEM (epoch 10; plan veto at threshold 0, which is what CCEM applies, as unsafe caught / near-safe vetoed / far-safe vetoed; MLP): at floor 0.3 it is exactly as optimistic as jac (0.61 / 0.06 / 0.02 vs 0.62 / 0.07 / 0.02; in distribution 0.91 / 0.31 / 0.06), so its sharper ranking (veto-vs-distance curve, which sets the threshold per model) does not reach CCEM there, and its fewer floor collisions come with less goal reaching (R3). Under camera shift its $h$ is genuinely less optimistic: 0.84 / 0.49 / 0.16 vs jac 0.62 / 0.34 / 0.10, the mechanism behind its camera-shift win. Attribution over the 150-pair runs: under floor 0.3 / 0.4 every collision of both models is "fooled" (the returned plan looked safe; 1.00) and $h$ never reads the box once inside (0-0.10); under camera shift / tilt 0.84-1.00 fooled; the returned plan was infeasible in only 0-6% of replans, so no collision comes from CCEM's fallback.
- The predictor damps perturbations: under the adversary and under the real shifts, the change in $h$ is largest on the encoded start frame and shrinks 3-20x by predicted step 5. The rollout re-grounds toward the training manifold, and toward the right state only when the encoder kept it (sigreg re-grounds onto wrong states).
- Depth alone (in distribution) makes only the MLP $h$ mildly optimistic (false-safe 0.06-0.10 at steps 1-5, 0.12-0.18 at 6-15); the linear $h$ does not drift.
- The information is missing from the latent, not the readout: an oracle margin function regressing the true signed distance to the box sharpens the boundary in distribution but is identical to the classification $h$ under shift.

## R5 (agrees with Claim 5): no threshold or planner rule recovers it

- Threshold recalibration (real-frame, per-step on predicted latents, plan-level; labeled or safe-only): static sign accuracy is recovered (safe-only 5% quantile of $h$ on 100 safe frames: jac+adv_h at floor 0.3 0.50-0.57 → 0.97-0.98, no failure data), but in CCEM every variant only moves along the safety-progress trade-off (safe success at or below uncalibrated in 63 of 64 cells). Real-frame offsets are about 2x too strict for predicted states because they re-ground.
- Threshold-free elite rules lower collision only by stalling (safe success 0.06-0.26 vs 0.44-0.50).

## R6 (differs from Claim 6): BN adaptation fixes the model that errs optimistic

Same procedure as Dubins (`reacher_bn_adapt.py`, 2000 safe shifted frames, both BatchNorm sets), CCEM-MLP collision / safe success, 50 pairs:

| Model, condition | Unadapted | BN-adapted |
| --- | --- | --- |
| sigreg, floor α = 0.1 | 0.48 / 0.30 | 0.06 / 0.68 |
| sigreg, floor α = 0.3 | 0.38 / 0.18 | 0.14 / 0.60 |
| sigreg, white balance | 0.60 / 0.10 | 0.24 / 0.38 |
| jac, any condition | — | unchanged within noise |
| jac+adv_h, floor α = 0.4 | 0.20 / 0.10 (progress 0.28) | 0.36 / 0.14 (progress 0.61) |

- sigreg recovers to jac's level under photometric shifts, including goal reaching (raw CEM progress at floor 0.3: 0.41 → 0.94), so its "lost" state under small shifts was mostly a per-dimension offset in the projector. Lighting, exposure (little gain), occlusion and camera shift (none) are not fixed.
- jac gains nothing. For jac+adv_h, adaptation repairs goal reaching under floor shifts but blurs its sharper $h$; the projector BatchNorm carries both effects. The safe-only 10% quantile threshold stalls on Reacher; threshold 0 is the right setting.
- Opposite sign to Dubins: Reacher's unadapted $h$ errs optimistic (CCEM = raw CEM), so correcting it can only help; Dubins' errs pessimistic, and adaptation removes the conservative fallback that was the safety.

## R7 (both benchmarks): adv_h helps only where jac is hurt moderately

Why adv_h beat jac only under Reacher camera shift (`adv_align.py`, `reacher-results.md` Table 31; jac vs jac+adv_h on 400 held-out states per shift, both benchmarks, MLP real+imag margin functions):

| Shift | Change in $h$ / gap: jac → adv_h | Planning gain? |
| --- | --- | --- |
| Reacher exposure, light, white balance, floor ≤ 0.2 | 0.002-0.078 → 0.001-0.004 | no: jac already at its in-distribution CCEM level |
| Dubins diamond | 0.040 → 0.007 | no: little headroom |
| Reacher camera shift | 0.200 → 0.099 | yes (Table 30: collision halved) |
| Reacher floor 0.3 / 0.4, camera tilt, occlusion | 0.086-0.860 → 0.12-0.72 (ratio 0.84-1.41) | no: out of reach |
| Dubins purple / rotation 90° | 0.343 → 0.446 / 0.145 → 0.149 | no: out of reach |

- adv_h's robustness is local: it shrinks the change in $h$ 2-25x where the shift moves jac's $h$ by up to about 0.2 of the safe/unsafe gap, and does nothing (or worse) beyond about 0.3. A planning gain needs both headroom (jac is hurt enough to collide) and reach (the shift is small enough for adv_h). Only Reacher camera shift has both; Dubins has no shift in that band, which is why Claim 3 found no planning gain there.
- Alignment of a shift with the adversary's own directions does not predict any of this (0.17-0.43 for every real shift).

## R8: a coherent account of when robustness tools help (Reacher and Dubins)

A shift does two kinds of damage to the latent, and each tool repairs one kind under its own limits.

1. A shared, affine part: every state's latent moves by about the same per-dimension offset or rescaling. Test-time BatchNorm adaptation (label-free, unlabeled shifted frames) undoes it, and because it corrects the latent itself it also re-aligns the goal latent and the predictor's inputs. sigreg's damage under small photometric shifts is mostly this part (about 0.55 of its floor shift is one shared offset in the route decomposition, vs 0.13-0.22 for jac), so BN brings sigreg to jac's level (floor 0.3 CCEM 0.38 / 0.18 → 0.14 / 0.60; R6). jac's invariance has already removed most of it, so BN adds nothing for jac.
2. A per-state part (the blur, R4): states lose resolution near the boundary. No affine correction recovers it; only training-time robustness does, and only under three conditions:
   - Type: the adversary perturbs pixel intensities at fixed positions, so it hardens the encoder against intensity changes (colour, lighting, texture, patches). A real 3D viewpoint change (camera tilt, Dubins rotation) instead scrambles the latent's state information: a linear joint-angle readout fit on clean latents has 50-110° error under camera tilt / shift vs 1° clean, with no consistent bias, while synthetic image-plane warps of similar size and the newly exposed edge region barely matter. So viewpoint change is a large off-manifold shift, not a small warp.
   - Reach: the robustness is local. At 8/255 it shrinks the change in $h$ 2-25x where the shift moves jac's $h$ by up to about 0.2 of the safe/unsafe gap, and does nothing beyond about 0.3 (R7). Reach grows with the adversary: the whole-latent target brings floor 0.3 within reach (ratio 0.18), a varying radius (4-32/255) brings floor 0.3 (0.02), floor 0.4 (0.09) and occlusion (0.25) within reach. Camera shift sits at the edge (about 0.5 for every adversarial variant); camera tilt stays out of reach (and gets worse with the varying radius, unexplained).
   - Headroom: planning improves only where jac is hurt enough to collide. Under exposure, light, white balance and small floor shifts adv_h makes $h$ much better, but jac's CCEM is already at its in-distribution level.

Planning gains appear exactly where reach and headroom coincide: camera shift (jac+adv_h collision 0.427 → 0.207, whole latent → 0.160 with higher safe success, 150 pairs) and floor 0.3 with the whole-latent target (0.233 → 0.113 at equal safe success). The tools are complementary: adv_h repairs $h$ but not the goal latent, whose shared offset under large floor shifts breaks goal reaching; BN adaptation repairs that offset (raw CEM progress at floor 0.4: 0.28 → 0.64) but blurs adv_h's $h$, and a split planner (adapted goal cost, unadapted $h$) keeps the progress. The predictor is never the bottleneck: perturbations shrink with prediction depth, and every model's prediction error grows 30-130x under the shifts regardless of its in-distribution quality.

Why Dubins differs. Its planning shifts fail the conditions: purple is an intensity shift but out of reach for an 8/255 adversary (it moves jac's $h$ by 0.34 of the gap), rotation is a viewpoint change, and diamond is within reach but has no headroom. Dubins also draws safety in colour, so intensity perturbations overlap the safety signal and invariance to them costs the predictor (1.7-6x prediction loss vs about 1% on Reacher, where the box is invisible and every intensity change is nuisance). Dubins purple is largely affine, which is why BN adaptation repairs its per-step ranking and halves collision for the one-step CBF filter (Claim 7), while CCEM's 10-25-step plans still carry no distance information (Claim 6).

Predictions still open: the varying-radius adv_h should beat jac in CCEM under floor 0.3 / 0.4 (150-pair test running); the varying-radius whole-latent model (training) should be the best overall except under tilt; on Dubins a larger or varying radius may bring purple within reach but, with colour carrying safety, at a higher cost than on Reacher.

## Pending or not paper-ready (Reacher)

- One training seed for every Reacher model; the camera-shift result needs seeds of jac and jac+adv_h before it is a claim.
- jac+pull(2.4)+adv_h: evaluated (50 pairs: keeps floor 0.3 progress, loses most of adv_h's camera gain; no orthogonality); its 150-pair test is done and not yet written up. jac+adv_λ: eval being repaired (rendering failures), 150-pair test running. jac + whole-latent adversary: training, done about 11 AM PST Nov 10.
- Protocol differences from Dubins: 10 CEM iterations (LeWM paper) vs 30; frameskip 5, so feasibility is checked every 5 env steps; LeWM's Reacher replanning (2 replans per episode).

## Where things are (Reacher)

- Full report: `reacher-results.md` (Tables 1-30). Scripts (le-wm/scripts): `reacher_margin_imag.py`, `reacher_ood.py`, `reacher_cem.py` (CCEM, shifts, `H_OFFSET(S)`, `H_WM` split), `reacher_shift.py`, `reacher_veto_diag.py` (blur curve), `reacher_within_start.py`, `reacher_calib.py`, `reacher_calib_pred.py`, `reacher_bn_adapt.py`, `reacher_routes.py`, `reacher_outcomes.py` (outcomes + attribution), `adv_sensitivity.py`, `adv_depth_diag.py`. Results: `/data/seongbin/lewm/reacher_cem/`, logs `/data/seongbin/lewm/reacher_eval_logs/`.
