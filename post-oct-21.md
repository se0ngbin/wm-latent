# Dubins, Safety Gym and Cartpole

# Summary (2026-11-05)

What is established, with pointers to the sections below. All margin-function numbers use frames in the training orientation; all CEM OOD numbers use a goal frame in the same shifted appearance. Superseded results and paused experiments are in `archive.md`.

### Evaluation fixes

- Two evaluation bugs were found and fixed. The margin-function OOD evals fed the world model transposed frames (every benchmark; "rotate" was a mirror flip), and the Dubins CEM purple/diamond runs used a red-circle goal frame. Reruns of the old paths reproduce the old tables exactly, so the fixed tables replace them. Still on transposed frames, and moved to `archive.md` until rerun: cartpole "why viewpoint fails" and value functions, and support vs aliasing.

### Dubins: margin functions

- Across world-model seeds (the 21-model paper grid in `paper-results.tex`; blanks = not done): single-seed color results do not hold. jac+pull's purple robustness was one lucky seed (linear AUC 0.991 at seed 3072, 0.35-0.39 at seeds 1 and 2), and adv_λ and adv_h vary widely; jac+pull+adv_h is the one model whose purple ranking holds on every seed (0.986 ± 0.015 linear, 0.925 ± 0.030 GP). Rotation holds on every seed of every jac-base model. The single-seed tables below are seed 3072.
- Margin functions (Dubins 3×3 grid): sigreg fails every shift; every regularized model handles rotation and shape; color is the hard shift, where adv_λ keeps the ranking on every base (0.911-0.982 AUC). No model's color threshold survives (sign accuracy ≈ 0.5).
- Linear margin function (all 13 Dubins models, red-trained): every safety reg keeps at most 26% of its red safe/unsafe gap under purple, but on the robust models the scale-free separation survives (Table 1). A linear $h$ equals the GP margin function on red and ranks real purple frames much better (jac 0.575 -> 0.972, jac+adv_h 0.174 -> 0.991; Table 2), with higher sign accuracy under diamond and rotation; under purple both still put every state on one side of 0, and linear $h$ does not change CEM (Table 17).
- Margin-function training length (Dubins, 1000-10000 steps): rotate and shape are unaffected, but color is not. jac, jac+pull and jac+pull+adv_h lose 0.2+ AUC with longer training and the ranking among them changes with the step count; only adv_λ is stable (0.982-0.989). The spread comes from the margin function being underdetermined by red data (a 0.5% latent perturbation moves color AUC by up to 0.049). On imagined latents a 2500-step $h$ changes little (jac false-safe 0.115 -> 0.109), so it will not change CEM much.

### Dubins: planning

- Planning (main results: $h$ = GP margin function; full and encoder-only safety regs both reported on all four conditions): raw CEM collapses for every model under a consistent recolor (7.5-34% success). A safety-first soft cost cuts collision on every jac-base model on every condition (jac: 5-27%) and never helps the sigreg-base models. Encoder-only is at least as safe as full on every jac-base safety-reg model on every condition; jac+adv_h encoder-only is the safest model under purple. The label-free GP margin function matches the oracle value function. 85-100% of the remaining collisions are false-safe: $h$ on imagined latents is optimistic, more so further ahead.
- Safety regs and planning: on the same base, jac+pull+adv_h loses 6.2 ± 1.8 points of raw-CEM success against jac+pull over 3 WM seeds. The adversarial predictor term raises rollout error. Dropping it (encoder-only) helps planning, and keeps color robustness for adv_h on the jac base (jac+adv_h 0.167 -> 0.808, jac+pull+adv_h 0.782 -> 0.891) but not for adv_λ (0.40-0.44 on every base). Best safety-reg models with the soft cost: jac+adv_h encoder-only and jac+adv_λ encoder-only (red 8% collision at about 80% success, matching jac).
- Fixing the optimistic $h$ (red, replanning every 5 steps): training $h$ on predicted latents of CEM-style plans labelled in the simulator lowers collision on 6 of 8 jac-base models (jac 84.5 / 6.0); with a margin that grows with the imagined step it cuts the robust models' collision the most (jac+adv_h encoder-only 17.5 -> 6.0, jac+pull+adv_h 41.5 -> 17.5) at 5-10 points of success. No variant helps under purple: from a recolored start $h$'s ranking on imagined latents is gone within 5-10 steps on every model (AUC 0.40-0.59 over the horizon), although real purple frames keep it (0.885-0.963 on the robust models); the next bullet gives the cause.
- Why purple imagined states fail (jac+pull+adv_h encoder-only): the predictor tracks the purple world about as well as the red one (error 0.57 vs 0.53 of the latent variance at step 10), but in the purple region $h$'s safe/unsafe gap is about 30 times smaller (0.06 vs 1.73), so one step of prediction noise already erases its ranking. The defect is the size of the safety gap under an appearance shift, not predictor drift.
- Planning-side fixes for purple give no consistent gain: a linear $h$ ties the GP margin function (Table 17), and the threshold-free costs (dynamic threshold, a cost relative to the current frame) collide more under purple (54-74%) than the fixed margin (43-47.5%, Table 18).
- CCEM on dangerous goals (straight line to the goal crosses an obstacle; goals 10-25 steps ahead; Tables 22-24): a constrained elite rule (feasible = $h > 0$ on the whole plan; fallback = highest worst-case $h$) cuts collision for every jac-base model under purple and rotation, often by 20-50 points, and never helps the sigreg base. In paired tests on the same episodes no model beats jac; jac+pull, the encoder-only safety regs and jac+pull+adv_λ tie it, the full safety regs on the jac base are mostly worse. Threshold-free rules (rank by $\min_k h$, then goal) only trade red performance for a non-significant purple gain. Seed 3072; extra seeds running.
- adv_h with a linear internal head (the adversary attacks a linear readout; epoch 30, preliminary, Table 21): for jac+adv_h it removes most of the full safety reg's planning cost on red (collision with replanning 44.0-47.5 -> 16.0) and stabilizes the GP margin function's purple AUC (0.412 -> 0.876 at epoch 30); for jac+pull+adv_h encoder-only it raises collision. It does not lengthen the imagined-purple window.
- Predictor gain: every predictor expands a latent perturbation 2-3× in one step, but over 25 steps the worst case is only 17-35× and a typical direction about 1×, and the most-amplifying models plan best. A non-expansion penalty would bind but there is no sign that amplification limits planning.
- Free-running 5-step rollout loss (SLS-squared style): paused and moved to `archive.md`. At 50 epochs it keeps the imagined-state ranking on red longer, but safety-first CEM is no better on red and mixed under purple, and it degraded the sigreg encoder.

### Training and data

- World-model training: Dubins models are still improving at epoch 50 by prediction loss (validation fell 15-40% from epoch 30 to 49; it looks flat at the end only because the cosine schedule ends). Downstream metrics mostly are not: at the epoch-30/40/50 checkpoints of 11 models the headline margin-function and imagined-$h$ results hold throughout, and every jac-base model (with or without a safety reg) plus jac+pull and adv_λ is flat. Still improving at epoch 50: imagined-rollout quality of the sigreg-base models and of jac+pull+adv_h and jac+pull+adv_λ, which weakens the "safety regs plan worse on the jac+pull base" comparison until that pair is retrained longer. Cartpole has plateaued. `*_jaclam50` runs are unstable and several Point seed-2 runs stopped early.
- Data: the Dubins dataset is not expert data. 94% of frames are diffusion-policy rollouts from 39 training checkpoints with starts anywhere in the arena, 6% are the 200 MPPI demonstrations; collisions do not end episodes (16.7% failure frames), and "the last 10% of episodes" used for held-out windows is 49% MPPI. Write-up: `dubins_dataset.tex` (branch `worktree-dubins-dataset-doc`), with `point_dataset.tex` for the Safety Gym Point data.

### Open

- Dubins WM seeds 1 and 2: done for sigreg, adv_h, adv_λ, jac+pull, jac+pull+adv_h and seed 1 of jac+adv_h encoder-only and jac+pull+adv_h encoder-only; training since 2026-11-05 (original recipes, `train_seeds_dubins_rest.sh`): jac, jac+adv_h, jac+adv_λ, jac+pull+adv_λ, both adv_λ encoder-only models, and the two remaining encoder-only seed-2 runs. Every table above still uses seed 3072. jac+adv_h's color inversion in particular needs the seeds: across its own checkpoints the GP margin function's purple AUC swings (0.146, 0.906, 0.584, 0.412, 0.174 at epochs 10, 15, 20, 30, 50) while a linear $h$ stays at 0.98-0.99 from epoch 15.
- Fair replanning (goal cost at every imagined step): red done for the full models plus two encoder-only models; OOD done (all 108 cells). Not run: the other encoder-only models, and encoder-only on OOD.
- Why adding a full safety reg to jac weakens the soft cost: executed-plan traces show the extra collisions are false-safe (85-100% in all 9 models traced). A growing margin and $h$ on predicted latents help on red, not under purple (see "Fixing the optimistic $h$").
- Purple planning: the failure is $h$'s collapsed safe/unsafe gap in the recolored region (Table 14), not predictor drift, and rank-based costs with a linear $h$ did not get around it (Table 18). Next: a red-only training term that keeps that gap under generic appearance perturbations. adv_h with a linear internal head (the adversary attacks a linear readout) is training on five bases (2026-11-05).
- Rerun on correct frames: cartpole "why viewpoint fails" and value functions, support vs aliasing (both in `archive.md`), and the 3-WM-seed comparison.
- Margin-function training length: color tables use an untuned 5000 steps; pick a red-only stopping rule (held-out balanced sign accuracy on a separate validation split) or report several lengths, and check whether the same holds on cartpole and Point.

# Setup

## Regularizers

Every model is the same le-wm world model (ViT-tiny encoder $f$, transformer predictor $P$, 50 epochs, seed 3072) with one or more terms added to its training loss. The 9-model grid is base $\{$sigreg, jac, jac+pull$\}$ × safety reg $\{$none, adv_h, adv_λ$\}$.

- sigreg (every model has it): $\mathcal{L} = \mathcal{L}_\text{pred} + 0.09\,\mathcal{L}_\text{SIGReg}$, prediction MSE on the next frames plus SIGReg, which pushes the latent toward an isotropic Gaussian along random 1-D projections and stops collapse. Side effect on near-static scenes like Dubins: to reach unit variance the encoder amplifies a few pixel directions ($\|\partial f/\partial x\|_F \approx 29$, condition number $\approx 142$).
- jac (encoder, no labels): $\mathcal{L}_\text{jac} = \mathbb{E}_x\big[(\|J_f(x)\|_F - 1)^2\big]$, weight 1.0, where $f$ = ViT CLS token → projector (the latent the predictor sees) and $x$ = the ImageNet-normalized 224 px first frame of each window. $\|J_f(x)\|_F$ is estimated per sample with one Hutchinson probe, $\|J_f^\top v\|$ for $v \sim \mathcal{N}(0, I)$ (exact vector-Jacobian product). Caps and evens out the encoder's sensitivity (Dubins: norm $29 \to 1.1$, condition number $142 \to 4$), which strips appearance the prediction loss doesn't need.
- pull (predictor, no labels): swap the last context action $a$ with another batch sample's $a'$ (random permutation) and regress $\mathcal{L}_\text{pull} = \mathbb{E}\big[(\|P(z, a) - P(z, a')\| - \|a - a'\|)^2\big]$ on the last predicted step, weight 0.1, with $a$ the z-scored raw action and the context latents $z$ detached, so the gradient reaches only the predictor (incl. its output projector) and the action encoder. Keeps imagined successors of different actions apart. Only used together with jac (jac+pull).
- adv_h (safety labels, adversarial): a margin head $h$ is trained alongside on the failure labels, with its own optimizer. On 32 windows per batch, PGD (uniform random start, 2 sign steps of $0.5\,\varepsilon$, best iterate kept per window) finds a perturbation $\delta$ of the $[0,1]$ pixels with $\|\delta\|_\infty \le \varepsilon = 8/255$, shared across the window's 3 context frames, that moves $h$'s readout the most. Penalty:
  $$\mathcal{L}_\text{adv\_h} = \Big(\tfrac{h(f(x+\delta)) - h(f(x))}{g}\Big)^2 + \Big(\tfrac{h(P(f(x+\delta), a)) - h(P(f(x), a))}{g}\Big)^2,$$
  with $h$ frozen inside the penalty and $g = \bar h_\text{safe} - \bar h_\text{unsafe}$ the live in-batch gap, floored at 10% of its running average and kept with gradient so squashing the readout raises the penalty (the adversary uses it detached). On after step 2000 once the head's AUC reaches 0.95, ramped over 10k steps; BatchNorm uses clean-pass statistics. Called "L∞ safety reg" in `safety_adv_reg_report.md`.
- adv_λ (safety labels, adversarial): the same adversary and the same encoder and predictor terms, but the protected readout is the latent shift projected onto $\lambda = \bar z_\text{safe} - \bar z_\text{unsafe}$ from the current batch, in units of $\|\lambda\|$:
  $$\mathcal{L}_\text{adv\_λ} = \Big(\tfrac{(f(x+\delta) - f(x)) \cdot \hat\lambda}{\|\lambda\|}\Big)^2 + \Big(\tfrac{(P(f(x+\delta), a) - P(f(x), a)) \cdot \hat\lambda}{\|\lambda\|}\Big)^2,$$
  with the direction $\hat\lambda = \lambda/\|\lambda\|$ detached and the length $\|\lambda\|$ keeping its gradient. No head and no AUC gate. Called "λ batch-detached" in the report.

Both safety regs use the same $L_\infty$ attack ($\varepsilon = 8/255$); the names say which readout the attack is aimed at. The report's "w/o h" model, which protects the whole latent, would be adv_latent.

Where the code is: jac is `JacobianNormReg` (`le-wm/module.py:467`) and pull is `ActionSeparationReg` with `pull_target` set (`module.py:347`; the two-sided term at `:397`; without `pull_target` the class is a one-sided hinge $\mathrm{relu}(L^2\|\Delta a\|^2 - \|\Delta z\|^2)$, not used in any reported model). adv_h and adv_λ are `SafetyAdvInvarianceReg` (`attack=linf`, `target=head` vs `target=lambda, lambda_mode=batch_detached`), which is not on main: it is on branch `safety-adv-reg` and in the code snapshot `/data/seongbin/lewm/code_safeadv_92e7589/module.py:316` that trained every adv model. Flag recipes: `le-wm/scripts/train_dubins_jacadv.sh`.

Why pull targets exactly 1: never tuned. It copies jac's $L = 1$ as the predictor-side analogue (one latent unit per action standard deviation; SIGReg keeps the latent near unit scale and actions are z-scored). The free ratio without the term is about 1.7 at frameskip 1, and a ratio of norms between a 192-dimensional latent and the action is not dimensionally principled; the always-on two-sided form was chosen over hinges on the frameskip-5 collapse, but the target value was not swept.

## Setup relative to upstream LeWM

- Architecture unchanged from the imported le-wm (`config/train/model/lewm.yaml`): ViT-tiny encoder (patch 14, 12 layers, 3 heads, width 192), CLS token through a BatchNorm MLP projector, 6-layer 16-head AdaLN-zero predictor with dropout 0.1 and its own projector. Only the JEPA+GRU ablation swaps the predictor.
- Training differs from the upstream default in length only: 50 epochs instead of 100 (cosine schedule fitted to 50), frameskip 1 for Dubins and cartpole. The predictor sees 3 frames of history and is trained one step ahead with teacher forcing (4-frame windows), with no multi-step loss except the rollout-loss variants (archived, `archive.md`).
- Planning is much deeper than upstream. The upstream benchmarks (pushT, cube, reacher, tworoom) use frameskip 5, `action_block` 5, a 5-step predictor horizon and replanning every 5 steps, with a 1-frame planning history. Our Dubins protocol (sg25clean) rolls the predictor out 25 steps from 1 real frame, 5 times deeper into the free-running regime that one-step training never covers.

## Do the world models converge in 50 epochs?

Every world model trains for 50 epochs with a cosine learning-rate schedule that reaches about 0 at epoch 50, so every loss curve is flat over the last few epochs by construction (0-2% change from epoch 45 to 49 on almost every run). The useful signal is how much validation prediction loss still fell from epoch 30, when the learning rate was still about 35% of its peak. Sources: wandb histories and, for runs with wandb off, the per-epoch validation tables in `run_logs/` (they match wandb exactly where both exist).

Validation pred_loss, Dubins

| Model | Epoch 10 | Epoch 20 | Epoch 30 | Epoch 40 | Epoch 49 | Drop 30 -> 49 | Drop 40 -> 49 |
|---|---|---|---|---|---|---|---|
| sigreg (same-config rerun `sigreg_dubins_sched50`) | 0.0139 | 0.0073 | 0.0047 | 0.0035 | 0.0033 | 30.6% | 6.4% |
| jac (seed replicates `jacw1_s100` / `_s200`) | 0.0201 / 0.0202 | 0.0139 / 0.0136 | 0.0111 / 0.0097 | 0.0090 / 0.0083 | 0.0086 / 0.0081 | 22.4% / 16.2% | 5.0% / 2.5% |
| jac+pull | 0.0256 | 0.0160 | 0.0120 | 0.0101 | 0.0099 | 17.2% | 1.4% |
| adv_h | 0.0510 | 0.0164 | 0.0128 | 0.0091 | 0.0085 | 33.2% | 6.8% |
| jac+adv_h | 0.0289 | 0.0178 | 0.0134 | 0.0113 | 0.0110 | 17.8% | 2.6% |
| jac+pull+adv_h | 0.0362 | 0.0226 | 0.0167 | 0.0144 | 0.0142 | 15.3% | 1.9% |
| adv_λ | 0.0424 | 0.0147 | 0.0132 | 0.0082 | 0.0080 | 39.5% | 2.0% |
| jac+adv_λ | 0.0211 | 0.0143 | 0.0114 | 0.0096 | 0.0093 | 18.0% | 3.1% |
| jac+pull+adv_λ | 0.0263 | 0.0187 | 0.0148 | 0.0125 | 0.0123 | 16.8% | 1.6% |

- Dubins models are not converged in the strict sense: prediction loss still fell 15-40% between epochs 30 and 49 on every model (seed replicates 13-57%; encoder-only models 14-26%), and it flattens only because the learning rate runs out. A longer schedule would very likely lower prediction loss further, most for sigreg, adv_h and adv_λ. Whether that changes margin-function or planning results is untested; prediction loss is not what limits planning (Diagnostics).
- The two models the tables use for sigreg and jac (`sigreg_only_dubins`, `jacobian_w1_dubins`) kept no history; the rows above are a same-config rerun and the seed replicates.
- Cartpole (all 9 models) has plateaued: 4.5-9.3% from epoch 30, 0.2-1.9% from epoch 40. Safety Gym Point and Car runs are mostly plateaued, but several adversarial-λ runs reach their lowest validation loss at epoch 0-2 and some rise late, so prediction loss is not a reliable convergence signal for those.
- Broken runs, not used in the tables: `lewm_dubins_jaclam50_*` and `sg_jaclam50_*` (validation loss spikes to 20-25 and ends 20-50× above jac+adv_λ; best epoch around 17-27).
- Incomplete runs: several Point seed-2 runs (and one seed-1) stopped early, at epochs 2-49, and are not running; the Point seed tables need them rerun.

### Are the downstream metrics still moving?

Prediction loss is not what the conclusions rest on, so we re-measured the downstream quantities at the saved epoch-10/20/30/40/50 checkpoints of 6 Dubins models, and at epochs 30/40/50 of 5 more (the jac-base safety-reg models, full and encoder-only, and jac+pull+adv_λ) (`epoch_downstream.py`, results `epoch_downstream.results.txt`). Per checkpoint: the GP margin function (5000 steps, red uniform states, seed-0 states) and its zero-shot AUC on color (mean ± std over 3 margin-function seeds), rotate and shape (seed 0); the same seed-0 margin function on imagined latents of 600 CEM-style plans (as Table 15); and goal-distance fidelity near the goal (Spearman between latent distance to the goal frame and true $(x, y)$ distance, true distance < 0.5, 400 held-out windows × 25 steps), on real and on imagined latents. The epoch-50 rows reproduce Table 15 (e.g. sigreg 0.814 / 0.423, jac+pull 0.961 / 0.281). Caveat: an epoch-30 checkpoint is not a model trained for 30 epochs, since the learning rate is still about 35% of its peak there.

| Model | Epoch | Color AUC | Rotate AUC | Shape AUC | Imagined red AUC | Imagined red false-safe | Goal fidelity, real | Goal fidelity, imagined |
|---|---|---|---|---|---|---|---|---|
| sigreg | 30 | 0.124 ± 0.007 | 0.100 | 0.715 | 0.801 | 0.260 | 0.570 | 0.235 |
| sigreg | 40 | 0.100 ± 0.004 | 0.059 | 0.566 | 0.811 | 0.383 | 0.566 | 0.368 |
| sigreg | 50 | 0.101 ± 0.004 | 0.086 | 0.464 | 0.814 | 0.423 | 0.570 | 0.448 |
| jac | 30 | 0.553 ± 0.049 | 0.994 | 0.995 | 0.960 | 0.129 | 0.654 | 0.509 |
| jac | 40 | 0.522 ± 0.023 | 0.991 | 0.995 | 0.966 | 0.129 | 0.659 | 0.496 |
| jac | 50 | 0.558 ± 0.017 | 0.990 | 0.994 | 0.968 | 0.113 | 0.659 | 0.522 |
| jac+pull | 30 | 0.776 ± 0.062 | 0.990 | 0.994 | 0.938 | 0.290 | 0.658 | 0.386 |
| jac+pull | 40 | 0.786 ± 0.053 | 0.983 | 0.994 | 0.952 | 0.285 | 0.683 | 0.437 |
| jac+pull | 50 | 0.833 ± 0.100 | 0.979 | 0.993 | 0.961 | 0.281 | 0.686 | 0.445 |
| adv_h | 30 | 0.961 ± 0.018 | 0.995 | 0.998 | 0.662 | 0.881 | 0.518 | 0.205 |
| adv_h | 40 | 0.853 ± 0.026 | 0.995 | 0.998 | 0.669 | 0.793 | 0.524 | 0.244 |
| adv_h | 50 | 0.764 ± 0.056 | 0.995 | 0.998 | 0.661 | 0.796 | 0.526 | 0.298 |
| jac+pull+adv_h | 30 | 0.889 ± 0.040 | 0.997 | 0.997 | 0.891 | 0.384 | 0.635 | 0.237 |
| jac+pull+adv_h | 40 | 0.774 ± 0.073 | 0.996 | 0.997 | 0.907 | 0.382 | 0.651 | 0.326 |
| jac+pull+adv_h | 50 | 0.817 ± 0.096 | 0.996 | 0.997 | 0.927 | 0.351 | 0.653 | 0.375 |
| adv_λ | 30 | 0.972 ± 0.009 | 0.997 | 0.998 | 0.612 | 0.754 | 0.482 | 0.009 |
| adv_λ | 40 | 0.985 ± 0.002 | 0.998 | 0.998 | 0.593 | 0.833 | 0.486 | 0.168 |
| adv_λ | 50 | 0.985 ± 0.003 | 0.998 | 0.998 | 0.628 | 0.764 | 0.487 | 0.183 |
| jac+adv_h | 30 | 0.272 ± 0.102 | 0.997 | 0.997 | 0.934 | 0.298 | 0.659 | 0.499 |
| jac+adv_h | 40 | 0.177 ± 0.059 | 0.996 | 0.997 | 0.946 | 0.298 | 0.668 | 0.469 |
| jac+adv_h | 50 | 0.102 ± 0.043 | 0.996 | 0.997 | 0.951 | 0.266 | 0.670 | 0.504 |
| jac+adv_h, encoder-only | 30 | 0.717 ± 0.092 | 0.996 | 0.996 | 0.954 | 0.147 | 0.643 | 0.482 |
| jac+adv_h, encoder-only | 40 | 0.728 ± 0.053 | 0.995 | 0.996 | 0.963 | 0.151 | 0.653 | 0.482 |
| jac+adv_h, encoder-only | 50 | 0.808 ± 0.073 | 0.995 | 0.996 | 0.969 | 0.109 | 0.655 | 0.515 |
| jac+adv_λ | 30 | 0.851 ± 0.148 | 0.997 | 0.997 | 0.948 | 0.197 | 0.659 | 0.490 |
| jac+adv_λ | 40 | 0.970 ± 0.015 | 0.996 | 0.997 | 0.959 | 0.221 | 0.679 | 0.492 |
| jac+adv_λ | 50 | 0.970 ± 0.012 | 0.996 | 0.997 | 0.968 | 0.173 | 0.678 | 0.527 |
| jac+adv_λ, encoder-only | 30 | 0.273 ± 0.020 | 0.996 | 0.997 | 0.949 | 0.152 | 0.658 | 0.516 |
| jac+adv_λ, encoder-only | 40 | 0.514 ± 0.165 | 0.996 | 0.997 | 0.961 | 0.131 | 0.673 | 0.511 |
| jac+adv_λ, encoder-only | 50 | 0.482 ± 0.135 | 0.996 | 0.997 | 0.961 | 0.129 | 0.676 | 0.536 |
| jac+pull+adv_λ | 30 | 0.957 ± 0.044 | 0.996 | 0.997 | 0.917 | 0.356 | 0.621 | 0.355 |
| jac+pull+adv_λ | 40 | 0.957 ± 0.035 | 0.996 | 0.997 | 0.935 | 0.300 | 0.651 | 0.395 |
| jac+pull+adv_λ | 50 | 0.944 ± 0.047 | 0.995 | 0.997 | 0.949 | 0.294 | 0.653 | 0.429 |

- By base: every jac-base model, with or without a safety reg and full or encoder-only (jac, jac+adv_h, jac+adv_λ and both encoder-only versions), has flat imagined-$h$ and goal-fidelity metrics from epoch 40 to 50 (imagined red AUC within 0.01, goal fidelity within 0.04). The jac+pull models with a safety reg (jac+pull+adv_h, jac+pull+adv_λ) are still improving on imagined rollouts at epoch 50 (imagined goal fidelity +0.03-0.05 per 10 epochs), as are the sigreg-base models.
- Stable from epoch 30: rotate and shape on every regularized model, real-frame goal fidelity on every model (at most +0.03 from epoch 30 to 50), and everything for jac. The headline claims hold at every checkpoint from epoch 30: sigreg fails every shift, every regularized model handles rotate and shape, adv_λ has the best color AUC (0.972-0.985), and jac reads imagined states best.
- Still moving at epoch 50: imagined goal fidelity on sigreg (0.368 -> 0.448 from epoch 40 to 50), adv_h (0.244 -> 0.298) and jac+pull+adv_h (0.326 -> 0.375), and jac+pull+adv_h's imagined red AUC (0.891 -> 0.927 from epoch 30). These are the models with the most prediction loss left (previous table). jac, jac+pull and adv_λ are flat from epoch 40 to 50.
- Consequence for one claim: "jac+pull+adv_h loses 6.2 points of raw-CEM success against jac+pull" and "the adversarial predictor term raises rollout error" compare a model whose imagined rollouts are still improving (imagined goal fidelity 0.375 at epoch 50) with one that has flattened (0.445). Part of that gap may be the 50-epoch budget rather than the regularizer; a 100-epoch retrain of that pair would settle it.
- adv_h's color robustness falls with training (0.961 -> 0.853 -> 0.764 from epoch 30 to 50); its color number depends on the training budget as well as on the margin-function length. jac+adv_h's color inversion deepens with training (0.272 -> 0.177 -> 0.102), so the main table's 0.167 is not an artifact of the epoch-50 snapshot. jac+adv_λ settles at 0.970 from epoch 40; the encoder-only models' color AUC is noisy across margin-function seeds (± 0.07-0.17).
- sigreg is still moving, but in the direction that strengthens its conclusion: shape 0.715 -> 0.464 and imagined false-safe 0.260 -> 0.423.
- Framing: report the models as trained for an equal 50-epoch budget with the learning rate annealed to zero, not as converged.

# Dubins across world-model seeds (LeWM protocol)

The 21-model paper grid (red, purple, rotated 90°; 3 world-model seeds per model; real-frame AUC and sign accuracy, raw and safety-first CEM open loop at 50 episodes per seed, replanning every 5 to run) is in `paper-results.tex`, generated from the run logs by `le-wm/scripts/make_paper_results.py` (rerun it to refresh). The CEM for it runs from `le-wm/scripts/paper_grid_cem.sh`.

# Dubins: margin functions under OOD shifts

![Dubins base vs OOD frames](figs/ood_frames_dubins.png)

The same 3×3 grid on Dubins: base {sigreg, jac, jac+pull} × safety reg {none, adv_h, adv_λ}. Zero-shot margin function (margin_gp loss on the failure label): trained on red-obstacle frames and scored on the same states re-rendered with purple obstacles (color), the whole world rotated 90° (rotate) or diamond obstacles (shape); labels are exact under every shift. One WM per model (training seed 3072); ± = std over 3 margin-function seeds, each resampling the rendered states (train and test) and the init. Blank rows: those WMs have not been trained on Dubins.

Frames are fed in the training orientation (fixed 2026-10-28). The earlier numbers, here and in `safety_adv_reg_report.md`, came from evals that passed numpy HWC renders to `ToImage`, which transposes them, so every frame was mirrored across the diagonal and "rotate" was really a flip. They are superseded by the tables below (script `dub_fixed.py`, logs `/data/seongbin/lewm/safeadv_results/fixed_orient/`).

AUC

| Model | base | color | rotate | shape |
|---|---|---|---|---|
| sigreg | 0.984 ± 0.000 | 0.110 ± 0.019 | 0.082 ± 0.003 | 0.487 ± 0.067 |
| jac | 0.998 ± 0.000 | 0.591 ± 0.019 | 0.988 ± 0.001 | 0.995 ± 0.001 |
| jac+pull | 0.998 ± 0.000 | 0.787 ± 0.084 | 0.976 ± 0.002 | 0.995 ± 0.001 |
| adv_h | 0.999 ± 0.001 | 0.789 ± 0.074 | 0.993 ± 0.001 | 0.998 ± 0.000 |
| jac+adv_h | 0.999 ± 0.000 | 0.167 ± 0.021 | 0.995 ± 0.001 | 0.997 ± 0.000 |
| jac+pull+adv_h | 0.999 ± 0.000 | 0.782 ± 0.068 | 0.995 ± 0.001 | 0.997 ± 0.001 |
| adv_λ | 0.999 ± 0.000 | 0.982 ± 0.001 | 0.995 ± 0.003 | 0.998 ± 0.000 |
| jac+adv_λ | 0.999 ± 0.000 | 0.946 ± 0.031 | 0.994 ± 0.001 | 0.997 ± 0.001 |
| jac+pull+adv_λ | 0.999 ± 0.000 | 0.911 ± 0.067 | 0.994 ± 0.001 | 0.997 ± 0.001 |

Sign accuracy, balanced: (TPR_safe + TPR_unsafe) / 2 on the full test set, i.e. the expected accuracy on a 50/50 safe/unsafe split (h ≥ 0 = safe; 0.500 = everything called one class). The margin function itself is trained on the natural split. "base" is the in-distribution value over all 9 runs (3 axes × 3 seeds).

| Model | base | color | rotate | shape |
|---|---|---|---|---|
| sigreg | 0.939 ± 0.004 | 0.502 ± 0.002 | 0.473 ± 0.018 | 0.549 ± 0.072 |
| jac | 0.956 ± 0.010 | 0.500 ± 0.000 | 0.842 ± 0.017 | 0.911 ± 0.007 |
| jac+pull | 0.956 ± 0.008 | 0.500 ± 0.000 | 0.822 ± 0.019 | 0.909 ± 0.011 |
| adv_h | 0.977 ± 0.003 | 0.502 ± 0.001 | 0.578 ± 0.036 | 0.969 ± 0.009 |
| jac+adv_h | 0.974 ± 0.005 | 0.464 ± 0.050 | 0.912 ± 0.009 | 0.943 ± 0.011 |
| jac+pull+adv_h | 0.975 ± 0.009 | 0.500 ± 0.000 | 0.886 ± 0.014 | 0.941 ± 0.003 |
| adv_λ | 0.977 ± 0.006 | 0.500 ± 0.000 | 0.944 ± 0.027 | 0.973 ± 0.007 |
| jac+adv_λ | 0.974 ± 0.009 | 0.500 ± 0.000 | 0.929 ± 0.007 | 0.958 ± 0.008 |
| jac+pull+adv_λ | 0.972 ± 0.009 | 0.500 ± 0.000 | 0.919 ± 0.008 | 0.955 ± 0.009 |

Reading:

- sigreg fails every shift: color 0.110, a real rotation 0.082 (inverted ranking), shape 0.487. Its thresholds collapse too.
- Every other model handles rotation and shape (AUC ≥ 0.976, shape sign accuracy 0.909-0.973). Adding jac fixes adv_h's drifting rotation threshold (0.578 -> 0.912).
- Color is the hard shift. adv_λ keeps the ranking on every base (0.911-0.982); jac drops to 0.591, adv_h and the jac+pull models sit at 0.78-0.79, and jac+adv_h inverts it (0.167).
- No threshold survives color: every model is near 0.500. Predicted-safe fraction on purple: jac, jac+pull and jac+pull+adv_h call everything safe; adv_h and adv_λ call everything unsafe.
- Encoder-only ablations (adversarial penalty on the encoder term only): color AUC drops from 0.789 to 0.454 (adv_h) and from 0.982 to 0.440 (adv_λ), so the predictor term is where the color robustness comes from (see the CEM section for its planning cost).
- One WM seed per model; jac+adv_h's color inversion in particular needs a second seed before it is trusted. The earlier 3-WM-seed comparison was run on transposed frames and needs redoing.

## Does the color result depend on how long the margin function is trained?

Every margin function above is trained for a fixed 5000 AdamW steps at a constant learning rate of $3 \times 10^{-4}$ (no schedule, no validation split), constants copied from the Dreamer config and never tuned. We retrained it for 1000, 2500, 5000 and 10000 steps on the same latents, for 6 models × 3 axes × 3 margin-function seeds (`dub_fixed_margin.py` protocol; logs `/data/seongbin/lewm/safeadv_results/margin_steps/`). The 5000-step rows reproduce the tables above exactly (54 of 54 runs).

Color AUC by margin-function training steps

| Model | 1000 | 2500 | 5000 | 10000 |
|---|---|---|---|---|
| sigreg | 0.108 ± 0.023 | 0.098 ± 0.009 | 0.110 ± 0.019 | 0.094 ± 0.005 |
| jac | 0.673 ± 0.022 | 0.620 ± 0.027 | 0.591 ± 0.019 | 0.475 ± 0.043 |
| jac+pull | 0.758 ± 0.028 | 0.716 ± 0.054 | 0.787 ± 0.084 | 0.548 ± 0.144 |
| adv_h | 0.613 ± 0.067 | 0.748 ± 0.130 | 0.789 ± 0.074 | 0.740 ± 0.131 |
| jac+pull+adv_h | 0.965 ± 0.002 | 0.931 ± 0.018 | 0.782 ± 0.068 | 0.761 ± 0.030 |
| adv_λ | 0.989 ± 0.002 | 0.982 ± 0.003 | 0.982 ± 0.001 | 0.987 ± 0.002 |

In-distribution balanced sign accuracy by training steps (pooled over 3 axes × 3 seeds)

| Model | 1000 | 2500 | 5000 | 10000 |
|---|---|---|---|---|
| sigreg | 0.940 | 0.939 | 0.939 | 0.932 |
| jac | 0.965 | 0.967 | 0.956 | 0.942 |
| jac+pull | 0.970 | 0.968 | 0.956 | 0.945 |
| adv_h | 0.980 | 0.979 | 0.977 | 0.974 |
| jac+pull+adv_h | 0.979 | 0.978 | 0.975 | 0.971 |
| adv_λ | 0.979 | 0.977 | 0.977 | 0.978 |

- Rotate and shape do not depend on training length: every model stays within ±0.003 AUC from 1000 to 10000 steps, except sigreg on shape (0.789, 0.540, 0.487, 0.453).
- Color does. Longer training lowers it on jac (0.673 -> 0.475), jac+pull (0.758 -> 0.548) and jac+pull+adv_h (0.965 -> 0.761), raises it on adv_h (0.613 -> 0.789 at 5000), and leaves adv_λ flat (0.982-0.989). At 1000 steps jac+pull+adv_h nearly ties adv_λ; at 5000 it ties jac+pull and adv_h. So the color ranking among jac+pull, adv_h and jac+pull+adv_h in the main table depends on the step count; adv_λ's lead does not.
- In-distribution AUC is flat at 0.982-0.999 for every length, so it cannot pick the length. Balanced sign accuracy can: it is best at 1000-2500 steps and falls by 5000-10000 (jac 0.967 -> 0.942), so the margin function starts to overfit red after about 2500 steps. A red-only stopping rule would therefore pick 1000-2500, not 5000.
- The variance is mostly the margin function, not the latent. Feeding the encoder channels-last instead of contiguous tensors moves the latents by about 0.5% (max abs difference 0.004 on a latent scale of 0.77) and leaves in-distribution AUC unchanged (±0.0002), yet moves color AUC by up to 0.049 (jac+pull, 5000 steps, seed 0). Many margin functions fit red equally well and extrapolate to purple differently; nothing in the red data chooses between them. adv_λ is the exception (±0.001-0.003 across seeds and lengths): there the representation itself fixes how every red-fit margin function reads purple.
- Implication for the tables: report color at more than one step count, or use a red-only stopping rule (e.g. stop when held-out balanced sign accuracy stops improving, on a validation split separate from the test states), or average several margin functions. Only adv_λ's color claim is robust to this choice.

## Linear vs MLP margin function

Table 14 located the purple failure in the size of $h$'s safe/unsafe gap, not in the predictor. Two follow-ups on that, both trained on red only (OOD stays a test-only measure). First, does each safety reg keep the safe/unsafe separation along the direction a red-trained readout uses (real latents, no prediction)? Second, does a linear readout $h(z) = w^\top z + b$ transfer better than the GP margin function's 192-512-512-1 MLP? The linear margin function uses exactly the GP recipe (`train_lin_margin_dubins.py`: margin_gp loss on the failure label, 8000 uniform red states, training orientation, same steps, learning rate and batch) and is loaded by the soft-cost CEM through `_load_margin_head` (`safe_solver.py`); saved as `margin_lin_fail_red_s0.pt` next to each WM.

### Does the safety reg keep the gap?

`diag_gap_ratio_dubins.py`: 4000 uniform states (seed 1, disjoint from the margin functions' training draws) rendered with red and with purple circles. The safety direction $u$ = mean latent of safe states minus mean latent of unsafe states, fit on red and normalized. Per color, $d' = (\bar p_\text{safe} - \bar p_\text{unsafe}) / s$ with $p = u^\top z$ and $s$ the pooled within-class standard deviation of $p$ (scale-free separation); raw gap ratio = purple gap / red gap along $u$.

Table 1. Safe/unsafe separation along the red safety direction, real latents, red vs purple.

| Model | $d'$ red | $d'$ purple | $d'$ ratio | Raw gap ratio | AUC of $u^\top z$, red | AUC of $u^\top z$, purple |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sigreg | 2.62 | 0.39 | 0.150 | 0.017 | 0.985 | 0.707 |
| adv_h | 5.65 | 1.56 | 0.276 | 0.078 | 0.998 | 0.992 |
| adv_λ | 6.66 | 3.79 | 0.569 | 0.100 | 0.998 | 0.993 |
| jac | 4.19 | 2.39 | 0.572 | 0.044 | 0.995 | 0.967 |
| jac+adv_h | 4.50 | 2.33 | 0.519 | 0.199 | 0.996 | 0.986 |
| jac+adv_h, encoder-only | 4.70 | 1.71 | 0.365 | 0.039 | 0.996 | 0.905 |
| jac+adv_λ | 5.35 | 4.20 | 0.786 | 0.260 | 0.997 | 0.996 |
| jac+adv_λ, encoder-only | 5.03 | 0.48 | 0.095 | 0.055 | 0.996 | 0.735 |
| jac+pull | 3.85 | 2.86 | 0.744 | 0.159 | 0.994 | 0.988 |
| jac+pull+adv_h | 4.80 | 4.27 | 0.891 | 0.227 | 0.997 | 0.992 |
| jac+pull+adv_h, encoder-only | 4.71 | 4.45 | 0.945 | 0.195 | 0.996 | 0.996 |
| jac+pull+adv_λ | 5.40 | 4.19 | 0.776 | 0.166 | 0.997 | 0.996 |
| jac+pull+adv_λ, encoder-only | 5.31 | 0.05 | 0.010 | 0.093 | 0.996 | 0.271 |

- Every model keeps at most 26% of its red gap along $u$ under purple (raw gap ratio 0.017-0.260): the latent compresses along the safety direction.
- The within-class spread shrinks with it, so the scale-free separation survives on the robust models ($d'$ ratio 0.945 for jac+pull+adv_h encoder-only, 0.891 jac+pull+adv_h, 0.786 jac+adv_λ) and the single direction $u$ still ranks purple states at AUC 0.99+ on 7 models. A readout that only needs the ordering could work; one that needs the absolute scale (a fixed margin) cannot.
- The live-gap term in the safety reg (the $g$ floor in the adv_h / adv_λ penalty) does keep more gap when the reg also trains the predictor path: jac+adv_h 0.199 vs its encoder-only version 0.039, jac+adv_λ 0.260 vs 0.055, jac+pull+adv_λ 0.166 vs 0.093, jac+pull+adv_h 0.227 vs 0.195. It is computed on red batches, so it does not stop the shrinkage under a shift it never sees.
- jac+adv_λ encoder-only and jac+pull+adv_λ encoder-only lose the separation entirely ($d'$ ratio 0.095 and 0.010).

### Linear vs GP margin function on real frames

`eval_head_color_dubins.py`, extended to all three axes: the same 4000 seed-1 states rendered red (in-distribution), purple, diamond (labels follow the diamond shape) and rotated 90°; both margin functions of each WM scored on the same latents. GP columns match Table 9 where they overlap.

Table 2. AUC on real latents, linear vs GP margin function, both trained on red.

| Model | Red, linear | Red, GP | Purple, linear | Purple, GP | Diamond, linear | Diamond, GP | Rot 90°, linear | Rot 90°, GP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sigreg | 0.986 | 0.987 | 0.631 | 0.082 | 0.973 | 0.433 | 0.056 | 0.097 |
| adv_h | 0.998 | 0.999 | 0.991 | 0.692 | 0.998 | 0.998 | 0.990 | 0.993 |
| adv_λ | 0.999 | 0.999 | 0.992 | 0.980 | 0.998 | 0.999 | 0.997 | 0.996 |
| jac | 0.997 | 0.998 | 0.972 | 0.575 | 0.996 | 0.995 | 0.989 | 0.989 |
| jac+adv_h | 0.998 | 0.999 | 0.991 | 0.174 | 0.996 | 0.997 | 0.994 | 0.995 |
| jac+adv_h, encoder-only | 0.998 | 0.998 | 0.877 | 0.885 | 0.996 | 0.997 | 0.995 | 0.994 |
| jac+adv_λ | 0.998 | 0.999 | 0.997 | 0.981 | 0.997 | 0.998 | 0.994 | 0.995 |
| jac+adv_λ, encoder-only | 0.997 | 0.999 | 0.734 | 0.336 | 0.996 | 0.997 | 0.994 | 0.994 |
| jac+pull | 0.997 | 0.998 | 0.991 | 0.862 | 0.995 | 0.994 | 0.978 | 0.978 |
| jac+pull+adv_h | 0.998 | 0.999 | 0.993 | 0.896 | 0.997 | 0.998 | 0.994 | 0.994 |
| jac+pull+adv_h, encoder-only | 0.997 | 0.998 | 0.996 | 0.954 | 0.996 | 0.997 | 0.995 | 0.995 |
| jac+pull+adv_λ | 0.998 | 0.998 | 0.996 | 0.986 | 0.997 | 0.997 | 0.994 | 0.994 |
| jac+pull+adv_λ, encoder-only | 0.998 | 0.998 | 0.271 | 0.581 | 0.996 | 0.997 | 0.994 | 0.993 |

Table 3. Balanced sign accuracy on real latents (0.500 = every state on one side of 0), same states and heads as Table 2.

| Model | Red, linear | Red, GP | Purple, linear | Purple, GP | Diamond, linear | Diamond, GP | Rot 90°, linear | Rot 90°, GP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sigreg | 0.948 | 0.947 | 0.500 | 0.500 | 0.641 | 0.484 | 0.407 | 0.489 |
| adv_h | 0.981 | 0.981 | 0.502 | 0.502 | 0.978 | 0.966 | 0.712 | 0.566 |
| adv_λ | 0.980 | 0.982 | 0.500 | 0.500 | 0.982 | 0.981 | 0.975 | 0.960 |
| jac | 0.978 | 0.971 | 0.500 | 0.500 | 0.954 | 0.928 | 0.899 | 0.862 |
| jac+adv_h | 0.980 | 0.976 | 0.500 | 0.499 | 0.970 | 0.955 | 0.963 | 0.916 |
| jac+adv_h, encoder-only | 0.979 | 0.975 | 0.500 | 0.500 | 0.973 | 0.947 | 0.958 | 0.905 |
| jac+adv_λ | 0.982 | 0.977 | 0.500 | 0.500 | 0.976 | 0.965 | 0.962 | 0.937 |
| jac+adv_λ, encoder-only | 0.981 | 0.975 | 0.500 | 0.500 | 0.975 | 0.959 | 0.960 | 0.919 |
| jac+pull | 0.977 | 0.969 | 0.500 | 0.500 | 0.950 | 0.919 | 0.878 | 0.830 |
| jac+pull+adv_h | 0.980 | 0.975 | 0.500 | 0.500 | 0.975 | 0.957 | 0.947 | 0.898 |
| jac+pull+adv_h, encoder-only | 0.977 | 0.973 | 0.500 | 0.500 | 0.960 | 0.943 | 0.934 | 0.895 |
| jac+pull+adv_λ | 0.981 | 0.978 | 0.500 | 0.500 | 0.978 | 0.961 | 0.960 | 0.925 |
| jac+pull+adv_λ, encoder-only | 0.980 | 0.973 | 0.405 | 0.500 | 0.969 | 0.946 | 0.951 | 0.907 |

Table 4. Median $h$ on safe / unsafe states, red and purple.

| Model | Red, linear | Red, GP | Purple, linear | Purple, GP |
| --- | ---: | ---: | ---: | ---: |
| sigreg | 1.06 / −0.42 | 1.53 / −0.39 | −1.06 / −1.06 | 0.20 / 0.24 |
| adv_h | 1.22 / −0.84 | 1.91 / −0.83 | −0.84 / −0.98 | −0.52 / −0.54 |
| adv_λ | 0.77 / −1.29 | 1.12 / −1.27 | −1.00 / −1.21 | −0.64 / −0.76 |
| jac | 0.32 / −0.31 | 1.54 / −0.45 | −0.47 / −0.50 | 0.17 / 0.17 |
| jac+adv_h | 0.29 / −0.36 | 1.71 / −0.56 | −0.61 / −0.75 | 0.12 / 0.20 |
| jac+adv_h, encoder-only | 0.32 / −0.34 | 1.65 / −0.54 | −0.12 / −0.14 | 0.29 / 0.26 |
| jac+adv_λ | 0.40 / −0.43 | 1.67 / −0.58 | −0.32 / −0.54 | 0.17 / 0.05 |
| jac+adv_λ, encoder-only | 0.40 / −0.38 | 1.69 / −0.55 | 6.09 / 6.02 | 0.77 / 0.77 |
| jac+pull | 0.33 / −0.29 | 1.47 / −0.42 | −0.33 / −0.44 | 0.36 / 0.33 |
| jac+pull+adv_h | 0.31 / −0.37 | 1.65 / −0.59 | −0.19 / −0.36 | 0.34 / 0.22 |
| jac+pull+adv_h, encoder-only | 0.33 / −0.36 | 1.56 / −0.53 | −0.11 / −0.25 | 0.38 / 0.32 |
| jac+pull+adv_λ | 0.45 / −0.43 | 1.62 / −0.55 | −0.21 / −0.36 | 0.19 / 0.08 |
| jac+pull+adv_λ, encoder-only | 0.38 / −0.43 | 1.55 / −0.49 | −0.20 / −0.10 | 0.26 / 0.25 |

- Red: the two margin functions are equivalent (AUC within 0.002, sign accuracy within 0.008).
- Purple ranking: the linear margin function matches or beats the GP one on 11 of 13 models, with large gains exactly where the GP one collapses or inverts: jac 0.575 -> 0.972, jac+adv_h 0.174 -> 0.991, adv_h 0.692 -> 0.991, jac+pull 0.862 -> 0.991, jac+pull+adv_h 0.896 -> 0.993. The MLP's extra nonlinear features are what break under recolor; the linear direction survives, as Table 1's AUC of $u^\top z$ predicted. Exceptions: jac+adv_h encoder-only (tie, 0.877 vs 0.885) and jac+pull+adv_λ encoder-only (0.271 vs 0.581); both adv_λ encoder-only models stay poor with either readout, as their lost separation in Table 1 predicts.
- Purple sign: balanced sign accuracy is 0.500 for every model and both margin functions (0.405 once). Purple moves every state to one side of 0: the linear margin function mostly reads all purple states as unsafe (medians below 0), the GP one on the jac-base models as safe (medians just above 0, the dangerous side), and the linear one's safe-unsafe median distance is at most 0.22 on purple vs 0.6-0.9 on red for the jac-base models. For jac+adv_λ encoder-only the linear margin function sits at +6 on all purple states.
- Diamond and rotated 90°: equal AUC, and the linear margin function has higher sign accuracy on every jac-base model (rotated 90°: +0.03 to +0.05, e.g. jac 0.862 -> 0.899, jac+adv_h encoder-only 0.905 -> 0.958). Its red values are symmetric around 0 (about ±0.35) where the GP one's are not (about +1.6 / −0.5).
- sigreg is beyond either readout: rotated 90° is inverted for both (0.056, 0.097), the support failure of Diagnostics.

# Dubins: planning with soft-cost CEM

CEM with the margin function's unsafe region as an extra cost on the imagined rollout: $\text{cost} = \text{goal MSE} + w \sum_t \mathrm{relu}(-h(\hat z_t)) + C\,\mathbb{1}[\min_t h(\hat z_t) < 0]$. Unsafe-looking candidates rank badly instead of being removed (the earlier hard filter, `safe_solver.py` gauge map, collided more than plain CEM). Safety-first adds $C = 10^4$ (larger than any goal cost, about 130), so every candidate $h$ calls safe beats every unsafe one, while the hinge still ranks the unsafe ones; graded has $C = 0$. Code: `SoftSafeCEMSolver` in `le-wm/safe_solver.py` (`solver=soft_cem`), runs `le-wm/scripts/soft_cem*.sh`, logs `run_logs/softcem_<model>_w<w>[_c<C>]_<condition>.log`.

## Setup

- $h$ (main results): the GP margin function on failure labels, the same recipe as the margin-function OOD tables, trained on red frames in the training orientation and encoded by the frozen WM (`margin_gp_fail_red_s0.pt`). It never sees purple, diamond or rotated frames, so the OOD columns test the world model and $h$ together, zero-shot. A supervised value function fit to the oracle HJ value is the privileged comparison (Secondary results).
- Models: 15 Dubins WMs, grouped by base (sigreg, jac, jac+pull). Each safety reg (adv_h, adv_λ) on each base is run in two versions, both first-class: full (adversarial penalty on the encoder and predictor terms) and encoder-only ($\mathrm{pred\_weight} = 0$, encoder term only). Every table lists them as adjacent rows.
- Conditions: red (training appearance), purple (recolor), diamond (obstacle shape), rotated (90° view). Purple and diamond use a goal frame rendered in the same shifted appearance (`eval.py` re-renders it; fixed 2026-10-29). Before the fix, `World.evaluate` took the goal frame from the dataset, so purple and diamond runs showed a red-circle goal in a purple or diamond world; those numbers, and the earlier claim that recolor does not affect CEM, were artifacts. Control: red through the re-render path reproduces the original red result (jac+pull raw 83.0 / 27.5 vs 81.5 / 29.0, logs `softcem_jacpull_rr_*`). Rotated already rotated the goal frame.
- Protocol sg25clean: goals 25 steps ahead in an annulus around the obstacles, horizon 25, budget 50, plan executed open loop unless stated. CEM: 300 samples, 30 elites, 30 iterations, initial std 1 per action; the executed plan is the mean of the last 30 elites, not the best sample (convergence check under Diagnostics). N = 200 episodes per cell, one run per cell. Frames are in the training orientation, so the eval transpose bug does not affect these numbers. Entries are success / collision (%).

Zero-shot color AUC of the GP margin function, full vs encoder-only (fixed orientation, 3 seeds), for reference:

| Model | Margin AUC, full | Margin AUC, encoder-only |
| --- | ---: | ---: |
| adv_h | 0.789 | 0.454 |
| adv_λ | 0.982 | 0.440 |
| jac+adv_h | 0.167 | 0.808 |
| jac+adv_λ | 0.946 | 0.398 |
| jac+pull+adv_h | 0.782 | 0.891 |
| jac+pull+adv_λ | 0.911 | 0.403 |

## Main results: GP margin function

Summary:

- Raw CEM collapses for every model under recolor (7.5-34% success, 49-82% collision); diamond hurts only the sigreg base.
- The soft cost cuts collision on every jac-base model on every condition, including the OOD ones, although $h$ only saw red frames. It never helps the sigreg-base models.
- Encoder-only is at least as safe as full on every jac-base safety-reg model on every condition with the safety-first cost (red 7.5-22.5 vs 13.0-29.0, purple 21.0-29.5 vs 36.0-44.5, rotated 21.5-27.5 vs 29.0-46.0% collision). The full safety regs lose most of the soft cost's gain over jac; encoder-only recovers it open loop, but only partly with replanning every 5 steps (17.5-20.0 vs jac 8.0% on red).
- Best models: jac (safety-first 79.0 / 8.5 red, 5-27% collision on every condition) and jac+adv_h encoder-only (80.5 / 8.0 red; safest under purple, 40.0 / 21.0 safety-first and 45.5 / 20.0 graded). jac+adv_λ encoder-only matches them on red and diamond (78.5 / 7.5, 78.5 / 6.5).
- Graded ($C = 0$) and safety-first give nearly the same numbers.
- With the goal cost scored at every imagined step and replanning every 5 steps, jac safety-first reaches 84.0 / 8.0 on red and 80.0 / 6.5 on diamond, the best results so far. Under purple, frequent replanning raises collision instead (jac 26.5 -> 43.0).
- 85-100% of the remaining collisions are false-safe: the executed plan's $h$ stayed $\ge 0$ up to the crash. $h$ on imagined latents is optimistic, more so further ahead (executed-plan trace under Diagnostics).

Original CEM cost (LeWM, raw CEM): only goal distance in latent space.

$$J(a_{1:H}) = \big\|\hat z_H - z_g\big\|^2$$

- $\hat z_H$ is the imagined latent after $H = 25$ steps under action sequence $a_{1:H}$, and $z_g$ is the encoded goal frame.
- "final" scores only that last step. Our fair-replanning variant ("mean") averages over every step: $\frac{1}{H}\sum_{t=1}^{H} \|\hat z_t - z_g\|^2$.
- Safety appears nowhere. Any avoidance comes only from the world model's dynamics, which mostly mimic the expert data.

With the safety penalty (soft-cost CEM):

$$J(a_{1:H}) = \underbrace{\big\|\hat z_H - z_g\big\|^2}_{\text{goal}} \;+\; \underbrace{w \sum_{t=1}^{H} \mathrm{relu}\big(m - h(\hat z_t)\big)}_{\text{hinge}} \;+\; \underbrace{C\,\mathbb{1}\big[\min_t h(\hat z_t) < m\big]}_{\text{flat}}$$
- $h$ is the margin function: positive means safe, negative means unsafe.
- $m$ is the safety margin, 0 by default (the `unsafe_margin` knob).
- Hinge: adds a penalty proportional to how far below $m$ each imagined step falls. It is zero for any trajectory that stays predicted-safe, so it never distorts goal-seeking among safe plans. $w = 1000$ in the main results.
- Flat term: a constant hit for any violation. With $C = 10^4$ (safety-first), any predicted-safe plan beats any predicted-unsafe one, since goal costs are about 130 at most. Graded sets $C = 0$.

### Raw CEM (no safety cost)

| Model | red | purple | diamond | rotated |
| --- | ---: | ---: | ---: | ---: |
| sigreg | 62.5 / 50.0 | 33.0 / 65.0 | 42.0 / 49.0 | 26.5 / 62.5 |
| adv_h | 57.5 / 57.0 | 34.0 / 63.5 | 49.5 / 47.0 | 20.5 / 70.5 |
| adv_h, encoder-only | 60.5 / 47.5 | 30.5 / 60.5 | 55.5 / 47.0 | 30.0 / 60.5 |
| adv_λ | 51.5 / 58.0 | 26.5 / 66.5 | 42.0 / 51.0 | 17.5 / 66.0 |
| adv_λ, encoder-only | 61.0 / 44.0 | 27.0 / 79.0 | 54.5 / 42.0 | 30.5 / 58.5 |
| jac | 81.5 / 22.5 | 16.0 / 75.5 | 79.0 / 15.5 | 30.0 / 60.0 |
| jac+adv_h | 79.0 / 25.0 | 7.5 / 81.5 | 78.0 / 18.5 | 19.5 / 64.5 |
| jac+adv_h, encoder-only | 81.5 / 26.5 | 23.0 / 72.5 | 80.0 / 15.5 | 29.5 / 47.5 |
| jac+adv_λ | 81.0 / 27.5 | 26.0 / 61.0 | 80.5 / 15.0 | 36.5 / 48.5 |
| jac+adv_λ, encoder-only | 83.0 / 21.0 | 28.0 / 59.5 | 80.0 / 13.5 | 28.0 / 57.0 |
| jac+pull | 81.5 / 29.0 | 25.5 / 59.0 | 82.0 / 14.0 | 41.5 / 54.5 |
| jac+pull+adv_h | 75.5 / 29.0 | 32.0 / 65.0 | 75.5 / 18.5 | 30.0 / 53.0 |
| jac+pull+adv_h, encoder-only | 78.5 / 33.0 | 28.5 / 49.0 | 75.5 / 17.0 | 30.5 / 38.5 |
| jac+pull+adv_λ | 81.0 / 29.0 | 33.0 / 58.0 | 77.5 / 16.5 | 46.5 / 46.0 |
| jac+pull+adv_λ, encoder-only | 78.5 / 28.5 | 32.5 / 53.5 | 76.5 / 15.5 | 38.0 / 41.5 |

- Recolor is the hardest shift for planning: every model collapses (7.5-34% success, 49-82% collision); jac+adv_h is worst at 7.5 / 81.5. Diamond hurts only the sigreg-base models (8-20 points of success); the jac-base models are unaffected.
- Encoder-only vs full: better on the sigreg base (red adv_h 57.5 / 57.0 -> 60.5 / 47.5, adv_λ 51.5 / 58.0 -> 61.0 / 44.0, though adv_λ encoder-only collides more under purple, 79.0 vs 66.5) and for jac+adv_λ on red (81.0 / 27.5 -> 83.0 / 21.0); mixed elsewhere. Under rotation encoder-only lowers collision on every jac-base model except jac+adv_λ.
- jac+pull+adv_λ is the strongest raw planner overall (no success cost on red, 81.0 / 29.0 vs jac+pull 81.5 / 29.0; best raw rotation, 46.5 / 46.0). jac+adv_h is the weakest jac-base model under shifts (purple 7.5 / 81.5, rotated 19.5 / 64.5).

### Safety-first ($w = 1000$, $C = 10^4$)

| Model | red | purple | diamond | rotated |
| --- | ---: | ---: | ---: | ---: |
| sigreg | 57.5 / 49.5 | 34.0 / 66.0 | 40.5 / 50.5 | 26.0 / 64.5 |
| adv_h | 55.5 / 53.0 | 34.5 / 63.0 | 50.5 / 46.5 | 19.0 / 69.5 |
| adv_h, encoder-only | 58.5 / 47.0 | 31.0 / 60.0 | 55.5 / 46.0 | 32.0 / 59.5 |
| adv_λ | 48.0 / 59.0 | 28.0 / 66.0 | 41.5 / 52.0 | 16.0 / 66.5 |
| adv_λ, encoder-only | 64.0 / 44.0 | 28.5 / 77.5 | 55.5 / 40.5 | 30.5 / 58.5 |
| jac | 79.0 / 8.5 | 35.0 / 26.5 | 78.0 / 5.0 | 45.0 / 21.5 |
| jac+adv_h | 76.0 / 13.0 | 25.5 / 37.5 | 73.5 / 11.0 | 18.5 / 46.0 |
| jac+adv_h, encoder-only | 80.5 / 8.0 | 40.0 / 21.0 | 78.0 / 5.5 | 41.5 / 22.5 |
| jac+adv_λ | 74.0 / 17.5 | 37.5 / 36.0 | 75.5 / 14.0 | 31.0 / 39.5 |
| jac+adv_λ, encoder-only | 78.5 / 7.5 | 36.5 / 27.5 | 78.5 / 6.5 | 41.0 / 27.5 |
| jac+pull | 77.0 / 20.0 | 35.5 / 36.5 | 75.5 / 14.0 | 35.0 / 31.0 |
| jac+pull+adv_h | 70.5 / 25.0 | 39.0 / 44.5 | 68.5 / 16.0 | 34.5 / 41.0 |
| jac+pull+adv_h, encoder-only | 70.0 / 22.5 | 34.0 / 26.5 | 65.5 / 15.5 | 29.5 / 27.0 |
| jac+pull+adv_λ | 70.0 / 29.0 | 40.5 / 38.0 | 68.0 / 15.5 | 43.0 / 29.0 |
| jac+pull+adv_λ, encoder-only | 76.0 / 20.5 | 30.5 / 29.5 | 77.0 / 12.0 | 38.5 / 21.5 |

- The soft cost cuts collision on every jac-base model on every condition. jac gains the most: 5-27% collision on every condition vs 15.5-75.5% raw, keeping 78-79% success on red and diamond. Under purple: jac 16.0 / 75.5 -> 35.0 / 26.5.
- It never helps the sigreg-base models, full or encoder-only, on any condition.
- Full vs encoder-only, jac base: encoder-only collides less on every model and condition (red: jac+adv_h 13.0 -> 8.0, jac+adv_λ 17.5 -> 7.5, jac+pull+adv_h 25.0 -> 22.5, jac+pull+adv_λ 29.0 -> 20.5; purple: 37.5 -> 21.0, 36.0 -> 27.5, 44.5 -> 26.5, 38.0 -> 29.5; rotated: 46.0 -> 22.5, 39.5 -> 27.5, 41.0 -> 27.0, 29.0 -> 21.5). On the jac+pull base it costs some OOD success (purple 39.0 -> 34.0, 40.5 -> 30.5).
- jac+adv_h encoder-only and jac+adv_λ encoder-only are the strongest safety-reg models (red 7.5-8.0% collision at 78.5-80.5% success, matching jac). jac+adv_h encoder-only also keeps color robustness in its margin function (AUC 0.808); jac+adv_λ encoder-only does not (0.398).
- The full safety regs lose most of the in-distribution gain (red 13.0-29.0% collision vs jac 8.5); the diagnostics below trace this to false-safe $h$ on imagined latents.

### Graded ($w = 1000$, $C = 0$)

| Model | red | purple | diamond | rotated |
| --- | ---: | ---: | ---: | ---: |
| sigreg | 57.0 / 49.5 | 33.0 / 66.0 | 41.0 / 50.5 | 26.0 / 64.5 |
| adv_h | 54.5 / 53.0 | 34.0 / 63.5 | 50.5 / 46.5 | 18.5 / 69.5 |
| adv_h, encoder-only | 58.5 / 47.0 | 29.5 / 60.5 | 55.5 / 46.5 | 32.0 / 60.0 |
| adv_λ | 49.5 / 59.0 | 26.5 / 66.5 | 42.0 / 52.5 | 18.5 / 67.5 |
| adv_λ, encoder-only | 64.5 / 44.0 | 28.5 / 80.0 | 56.0 / 40.5 | 30.0 / 59.5 |
| jac | 79.5 / 9.0 | 37.5 / 27.5 | 78.5 / 5.0 | 45.5 / 23.0 |
| jac+adv_h | 78.0 / 12.5 | 23.5 / 43.0 | 75.5 / 11.5 | 21.5 / 47.0 |
| jac+adv_h, encoder-only | 80.0 / 9.0 | 45.5 / 20.0 | 78.0 / 6.0 | 39.0 / 24.5 |
| jac+adv_λ | 76.5 / 18.0 | 37.5 / 37.0 | 77.5 / 11.5 | 33.0 / 39.5 |
| jac+adv_λ, encoder-only | 80.0 / 8.0 | 39.0 / 28.5 | 78.5 / 7.0 | 40.0 / 25.5 |
| jac+pull | 77.0 / 19.5 | 35.0 / 34.0 | 76.5 / 13.0 | 36.5 / 32.0 |
| jac+pull+adv_h | 71.0 / 25.5 | 35.0 / 47.5 | 70.0 / 15.5 | 36.0 / 39.0 |
| jac+pull+adv_h, encoder-only | 69.5 / 23.0 | 33.0 / 27.0 | 68.5 / 13.0 | 31.5 / 27.5 |
| jac+pull+adv_λ | 73.0 / 25.0 | 41.5 / 38.5 | 69.5 / 18.0 | 45.0 / 33.0 |
| jac+pull+adv_λ, encoder-only | 77.0 / 20.0 | 31.0 / 28.0 | 77.5 / 11.5 | 38.5 / 25.0 |

- Graded and safety-first give nearly the same numbers on every condition (jac graded 79.5 / 9.0, 37.5 / 27.5, 78.5 / 5.0, 45.5 / 23.0 vs safety-first 79.0 / 8.5, 35.0 / 26.5, 78.0 / 5.0, 45.0 / 21.5): with $w = 1000$ the hinge already dominates the goal cost, so the flat $C$ adds little.
- Under shift, encoder-only is safer than full on every jac-base safety-reg model: purple collision 20.0-28.5 vs 37.0-47.5, rotated 24.5-27.5 vs 33.0-47.0.
- jac+adv_h encoder-only is the only model that beats jac under purple (45.5 / 20.0 vs 37.5 / 27.5) and is close to jac elsewhere (rotated 39.0 / 24.5 vs 45.5 / 23.0). The full jac+adv_h is the worst jac-base model under purple and rotation (43.0 and 47.0% collision).
- The sigreg-base models gain nothing on any condition.

### Replanning: goal cost at every imagined step

The open-loop runs above score only the last imagined step; with that cost, replanning pushes arrival later and lowers success (Secondary results). Here the goal cost is the latent goal distance averaged over every imagined step ($\mathrm{goal\_cost} = \mathrm{mean}$, `soft_cem_goalmean_queue.sh`), with both replanning intervals under the same cost: every 25 steps (open loop) and every 5 steps. Safety-first as above. The two encoder-only cells come from the executed-plan trace runs (tracing does not change behavior); the other encoder-only models and cells are not run. Red:

| Model | raw, every 25 | raw, every 5 | safety-first, every 25 | safety-first, every 5 |
| --- | ---: | ---: | ---: | ---: |
| sigreg | 61.0 / 52.5 | 63.0 / 55.5 | 60.0 / 52.0 | 63.0 / 55.5 |
| adv_h | 53.0 / 54.5 | 64.0 / 55.5 | 53.5 / 54.0 | 64.0 / 54.5 |
| adv_λ | 60.5 / 57.0 | 56.5 / 55.0 | 60.5 / 58.5 | 56.0 / 54.5 |
| jac | 68.0 / 52.0 | 77.5 / 48.5 | 76.0 / 22.0 | 84.0 / 8.0 |
| jac+adv_h | 51.0 / 61.5 | 56.5 / 63.5 | 57.5 / 47.5 | 60.5 / 47.5 |
| jac+adv_h, encoder-only | not run | not run | not run | 84.5 / 17.5 |
| jac+adv_λ | 70.5 / 46.5 | 82.5 / 42.5 | 71.0 / 39.0 | 80.0 / 28.5 |
| jac+adv_λ, encoder-only | not run | not run | not run | 78.5 / 20.0 |
| jac+pull | 66.0 / 46.0 | 76.0 / 43.5 | 61.5 / 38.5 | 74.0 / 19.5 |
| jac+pull+adv_h | 58.0 / 54.5 | 68.0 / 53.5 | 59.5 / 44.5 | 69.5 / 41.5 |
| jac+pull+adv_λ | 61.0 / 48.5 | 75.0 / 45.0 | 59.5 / 39.5 | 72.0 / 28.5 |

- With this cost, replanning every 5 steps helps success for every model except adv_λ (e.g. jac 68.0 -> 77.5, jac+adv_λ 70.5 -> 82.5): the earlier drop was the final-step cost, not replanning.
- Raw CEM collides about 45-65% under both intervals. The mean cost rewards getting close early, which favors cutting through obstacles (jac open loop: 81.5 / 22.5 with the final-step cost vs 68.0 / 52.0 here), so the 22.5% earlier was a side effect of the goal cost, not safety knowledge.
- Safety-first plus replanning every 5 steps gives the best result so far: jac 84.0 / 8.0, better than every other jac setting on both success and collision. jac+pull also improves (19.5% collision).
- Adding a full safety reg weakens the soft cost here too: 28.5-47.5% collision with safety-first every 5 steps for the four jac-base safety-reg models, against 8.0 (jac) and 19.5 (jac+pull). Safety-first still does nothing on the sigreg base.
- Encoder-only (two models, safety-first every 5 steps): jac+adv_h encoder-only 84.5 / 17.5 and jac+adv_λ encoder-only 78.5 / 20.0. Better than their full versions (47.5, 28.5% collision) but 2-2.5x jac's 8.0, unlike open-loop safety-first, where they match jac (8.0, 7.5).

OOD conditions, same settings:

Purple

| Model | raw, every 25 | raw, every 5 | safety-first, every 25 | safety-first, every 5 |
| --- | ---: | ---: | ---: | ---: |
| sigreg | 35.5 / 67.0 | 14.5 / 72.0 | 34.5 / 68.5 | 14.0 / 72.5 |
| adv_h | 37.0 / 66.5 | 35.0 / 60.5 | 38.5 / 63.0 | 32.0 / 59.5 |
| adv_λ | 40.5 / 56.5 | 37.5 / 62.0 | 43.5 / 60.0 | 24.5 / 65.0 |
| jac | 16.0 / 80.5 | 20.5 / 75.5 | 43.0 / 26.5 | 34.5 / 43.0 |
| jac+adv_h | 14.0 / 80.0 | 18.0 / 79.0 | 33.0 / 56.5 | 33.5 / 56.0 |
| jac+adv_λ | 20.5 / 74.5 | 26.5 / 65.5 | 35.0 / 44.5 | 26.0 / 50.0 |
| jac+pull | 30.5 / 63.0 | 33.0 / 64.0 | 43.5 / 38.0 | 18.5 / 53.5 |
| jac+pull+adv_h | 21.5 / 75.0 | 19.5 / 79.0 | 42.5 / 53.0 | 26.5 / 61.5 |
| jac+pull+adv_λ | 24.0 / 71.5 | 27.5 / 74.0 | 43.0 / 37.0 | 32.5 / 50.5 |

Diamond

| Model | raw, every 25 | raw, every 5 | safety-first, every 25 | safety-first, every 5 |
| --- | ---: | ---: | ---: | ---: |
| sigreg | 51.5 / 50.0 | 59.5 / 46.5 | 49.5 / 52.0 | 57.5 / 47.0 |
| adv_h | 52.5 / 46.0 | 57.0 / 48.0 | 51.5 / 45.5 | 59.0 / 47.0 |
| adv_λ | 48.0 / 50.0 | 60.0 / 40.5 | 49.0 / 50.5 | 64.0 / 41.0 |
| jac | 65.5 / 33.5 | 79.5 / 29.5 | 76.5 / 10.5 | 80.0 / 6.5 |
| jac+adv_h | 55.0 / 47.5 | 57.0 / 48.0 | 56.0 / 32.5 | 58.5 / 32.5 |
| jac+adv_λ | 70.0 / 30.0 | 80.5 / 23.0 | 69.0 / 25.5 | 80.5 / 15.5 |
| jac+pull | 60.5 / 34.5 | 74.5 / 26.5 | 63.0 / 24.5 | 75.5 / 13.0 |
| jac+pull+adv_h | 57.5 / 39.5 | 65.5 / 42.0 | 53.5 / 31.5 | 65.0 / 33.0 |
| jac+pull+adv_λ | 63.0 / 33.0 | 72.0 / 31.5 | 64.5 / 29.0 | 70.5 / 22.5 |

Rotated

| Model | raw, every 25 | raw, every 5 | safety-first, every 25 | safety-first, every 5 |
| --- | ---: | ---: | ---: | ---: |
| sigreg | 23.5 / 53.5 | 56.0 / 58.5 | 23.0 / 54.0 | 55.0 / 58.5 |
| adv_h | 26.5 / 59.0 | 41.5 / 57.0 | 25.0 / 59.0 | 39.5 / 56.0 |
| adv_λ | 22.5 / 62.0 | 31.5 / 60.0 | 20.5 / 61.0 | 32.0 / 59.5 |
| jac | 25.5 / 62.0 | 47.5 / 54.0 | 39.0 / 38.5 | 50.0 / 40.0 |
| jac+adv_h | 19.5 / 66.5 | 30.5 / 63.0 | 18.0 / 52.0 | 33.0 / 55.0 |
| jac+adv_λ | 34.0 / 61.0 | 51.5 / 57.5 | 31.5 / 57.5 | 33.0 / 53.0 |
| jac+pull | 35.5 / 40.0 | 53.0 / 45.5 | 36.0 / 29.5 | 52.5 / 31.5 |
| jac+pull+adv_h | 29.0 / 64.0 | 32.0 / 62.5 | 29.0 / 50.0 | 37.0 / 54.0 |
| jac+pull+adv_λ | 35.5 / 46.5 | 43.0 / 52.5 | 44.0 / 34.5 | 50.0 / 44.0 |

- Purple: replanning every 5 steps hurts the soft cost. Safety-first collision rises from every 25 to every 5 steps on every jac-base model except jac+adv_h, which is flat and already high (jac 26.5 -> 43.0, jac+adv_λ 44.5 -> 50.0, jac+pull 38.0 -> 53.5, jac+pull+adv_h 53.0 -> 61.5, jac+pull+adv_λ 37.0 -> 50.5; jac+adv_h 56.5 -> 56.0), the opposite of red. Safety-first every 25 is still the best purple setting for jac (43.0 / 26.5, vs 35.0 / 26.5 open loop with the final-step cost). Raw CEM collides 56.5-80.5% under both intervals.
- Diamond behaves like red: safety-first every 5 steps is best (jac 80.0 / 6.5, jac+adv_λ 80.5 / 15.5, jac+pull 75.5 / 13.0).
- Rotated: replanning every 5 steps raises success at similar collision for jac and jac+pull (safety-first 39.0 / 38.5 -> 50.0 / 40.0, 36.0 / 29.5 -> 52.5 / 31.5); the full safety-reg models collide more (jac+pull+adv_λ 34.5 -> 44.0, jac+pull+adv_h 50.0 -> 54.0). jac+pull is the safest model under rotation.
- As on red, the soft cost does nothing on the sigreg base, and jac+adv_h is the weakest jac-base model (safety-first every 25: 56.5, 32.5, 52.0% collision on purple, diamond, rotated).
- Not run: the encoder-only models on OOD conditions.

## Secondary results

### Supervised value function as $h$

$h$ fit by MSE regression onto the oracle HJ value $V^*$ (computed offline by a grid solver on the true dynamics, `hj_truth.npz`) at random states rendered red and encoded by the frozen WM (`train_margin_hj_lewm.py`, `margin_hj_color_red_rk0_s0.pt`; no GP, no RL). The labels are privileged. Safety-first ($w = 1000$, $C = 10^4$):

| Model | red | purple | diamond | rotated |
| --- | ---: | ---: | ---: | ---: |
| sigreg | 39.0 / 60.5 | 28.5 / 61.5 | 35.5 / 50.0 | 23.5 / 64.0 |
| adv_h | 41.0 / 59.5 | 28.0 / 62.5 | 35.0 / 55.5 | 22.5 / 64.0 |
| adv_h, encoder-only | 44.5 / 52.5 | 22.5 / 67.0 | 44.0 / 43.5 | 27.0 / 58.0 |
| adv_λ | 49.0 / 59.0 | 24.5 / 68.0 | 44.0 / 51.0 | 20.5 / 63.5 |
| adv_λ, encoder-only | 46.0 / 55.0 | 28.0 / 57.5 | 36.5 / 46.0 | 31.0 / 55.0 |
| jac | 72.0 / 11.0 | 39.0 / 15.5 | 71.5 / 5.5 | 48.0 / 13.5 |
| jac+adv_h | 68.5 / 19.5 | 34.0 / 30.0 | 65.0 / 17.5 | 27.0 / 43.0 |
| jac+adv_h, encoder-only | 76.5 / 20.0 | 41.0 / 32.5 | 71.0 / 11.5 | 42.0 / 29.0 |
| jac+adv_λ | 73.5 / 29.0 | 45.0 / 41.5 | 72.5 / 16.5 | 44.5 / 39.5 |
| jac+adv_λ, encoder-only | 71.5 / 16.5 | 35.0 / 34.5 | 70.5 / 9.5 | 36.5 / 30.0 |
| jac+pull | 75.0 / 20.0 | 38.5 / 41.0 | 73.0 / 18.0 | 35.0 / 37.5 |
| jac+pull+adv_h | 66.5 / 25.0 | 33.0 / 33.5 | 62.0 / 15.5 | 33.0 / 31.5 |
| jac+pull+adv_h, encoder-only | 69.0 / 25.0 | 34.0 / 30.0 | 66.0 / 18.0 | 34.0 / 26.0 |
| jac+pull+adv_λ | 62.5 / 29.0 | 44.0 / 28.5 | 62.5 / 16.5 | 42.5 / 23.0 |
| jac+pull+adv_λ, encoder-only | 67.0 / 25.5 | 32.0 / 24.0 | 67.0 / 15.5 | 38.5 / 20.0 |

- The label-free GP margin function matches the privileged value function in most cells and is better for most models on red (jac 79.0 / 8.5 vs 72.0 / 11.0); the value function is better for jac under purple and rotation (39.0 / 15.5, 48.0 / 13.5 vs 35.0 / 26.5, 45.0 / 21.5).
- Same pattern as the GP: gains on the jac base only.

Weight sweep, value function, red:

| Model | raw CEM | $w = 100$ | $w = 1000$ | $w = 10^4$ | safety-first |
| --- | ---: | ---: | ---: | ---: | ---: |
| sigreg | 62.5 / 50.0 | 45.0 / 58.5 | 39.5 / 60.5 | 35.5 / 61.5 | 39.0 / 60.5 |
| jac | 81.5 / 22.5 | 82.5 / 10.0 | 80.0 / 9.5 | 71.5 / 9.0 | 72.0 / 11.0 |

### Graded GP weight sweep, red

Hinge only ($C = 0$) with the GP margin function, $w = 100$ vs $1000$ (`soft_cem_weight_pair_gp.sh`):

| Model | raw CEM | $w = 100$ | $w = 1000$ |
| --- | ---: | ---: | ---: |
| adv_h | 57.5 / 57.0 | 55.0 / 53.0 | 54.5 / 53.0 |
| adv_h, encoder-only | 60.5 / 47.5 | 58.0 / 47.0 | 58.5 / 47.0 |
| adv_λ | 51.5 / 58.0 | 50.5 / 57.0 | 49.5 / 59.0 |
| adv_λ, encoder-only | 61.0 / 44.0 | 62.5 / 41.5 | 64.5 / 44.0 |
| jac+adv_h | 79.0 / 25.0 | 77.5 / 16.5 | 78.0 / 12.5 |
| jac+adv_h, encoder-only | 81.5 / 26.5 | 81.5 / 9.5 | 80.0 / 9.0 |
| jac+adv_λ | 81.0 / 27.5 | 79.0 / 17.5 | 76.5 / 18.0 |
| jac+adv_λ, encoder-only | 83.0 / 21.0 | 80.0 / 8.0 | 80.0 / 8.0 |
| jac+pull+adv_h | 75.5 / 29.0 | 72.5 / 23.0 | 71.0 / 25.5 |
| jac+pull+adv_h, encoder-only | 78.5 / 33.0 | 73.5 / 25.0 | 69.5 / 23.0 |
| jac+pull+adv_λ | 81.0 / 29.0 | 77.0 / 27.5 | 73.0 / 25.0 |
| jac+pull+adv_λ, encoder-only | 78.5 / 28.5 | 78.0 / 21.0 | 77.0 / 20.0 |

- $w = 100$ and $w = 1000$ are close; on the jac base the cost works better encoder-only on every model except jac+pull+adv_h.

### Control: does the safety gain come from what $h$ knows?

On red with jac, a margin function with no safety knowledge makes CEM worse, not safer: shuffled labels 63.5 / 39.5, random init 36.5 / 43.0, against the GP margin function 79.0 / 8.5 and raw CEM 81.5 / 22.5. The safety gain comes from what $h$ knows.

# Dubins diagnostics: why safety regs plan worse, and why CEM still collides

Diagnostics: `diag_rollout_safety.py`; ablations: `ablate_plan_worse.sh` and `seeds_jacpull_vs_adv.sh`.

- Same base, 3 WM seeds: jac+pull+adv_h loses 6.2 ± 1.8 points of raw-CEM success against jac+pull on red (75.5 / 79.0 / 78.0 vs 81.5 / 87.0 / 82.5; pooled paired 52 vs 15 episodes, $p < 10^{-4}$), with no consistent change in collision.
- Across 22 Dubins WMs, raw-CEM collision tracks goal-distance fidelity, the Spearman correlation between latent distance to the goal and true $(x, y)$ distance near the goal ($\rho = -0.84$); 25-step rollout error does not ($\rho \approx 0$). adv_h lowers that fidelity on imagined latents and raises 5-step rollout error on every seed of the jac+pull pair.
- Projecting the safety direction $\lambda$ out of the latent leaves goal fidelity unchanged on every model (λ is about 2.5% of the latent variance), so the distortion is not along the safety axis.
- Part-ablations of adv_h (sigreg base): the adversarial predictor term causes the rollout-error increase (predictor-only 0.084, full 0.082, encoder-only 0.049, no reg 0.014 at 5 steps); detaching the gap denominator restores most of the real-latent goal geometry; random noise instead of the adversary costs almost nothing. The encoder-only ablation above tests this on every safety-reg model.
- Training the value function on imagined latents (rollout_k = 8) does not restore $h$ on imagined latents and does not make the soft cost help the sigreg-base models.
- The goal marker is a nuisance for sigreg's margin function only (AUC 0.985 -> 0.901 when the marker moves; ≤ 0.017 for every other model).

Why does a safety reg weaken the soft cost on the jac base? The soft cost penalizes $h(\hat z_t) < 0$, so what matters is the sign of $h$ on imagined latents. A false-safe frame is an unsafe frame with $h \ge 0$: a colliding plan through it goes unpenalized. GP margin function on 400 held-out expert windows, real latents vs 1-25-step open-loop predictions (`diag_rollout_safety.py`, `rollout_diag_sign/`); balanced sign accuracy is on a 50/50 safe/unsafe weighting. Collision is red GP safety-first CEM (%):

| Model | Bal. sign acc, real | Bal. sign acc, imagined | False-safe, real | False-safe, imagined | False-safe, step 25 | Collision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| jac | 0.950 | 0.906 | 0.078 | 0.132 | 0.346 | 8.5 |
| jac+adv_h | 0.967 | 0.907 | 0.054 | 0.149 | 0.462 | 13.0 |
| jac+adv_h, encoder-only | 0.962 | 0.932 | 0.065 | 0.100 | 0.269 | 8.0 |
| jac+adv_λ | 0.965 | 0.909 | 0.060 | 0.156 | 0.500 | 17.5 |
| jac+adv_λ, encoder-only | 0.967 | 0.928 | 0.056 | 0.113 | 0.346 | 7.5 |
| jac+pull | 0.944 | 0.850 | 0.095 | 0.215 | 0.500 | 20.0 |
| jac+pull+adv_h | 0.966 | 0.879 | 0.057 | 0.154 | 0.308 | 25.0 |
| jac+pull+adv_h, encoder-only | 0.962 | 0.877 | 0.065 | 0.158 | 0.462 | 22.5 |
| jac+pull+adv_λ | 0.961 | 0.872 | 0.066 | 0.176 | 0.654 | 29.0 |
| jac+pull+adv_λ, encoder-only | 0.961 | 0.861 | 0.066 | 0.206 | 0.538 | 20.5 |

- On real latents every safety-reg model's $h$ is as good as or better than jac's (false-safe 0.054-0.066 vs 0.078). The errors appear only on imagined latents, so the predictor, not the encoder, is what $h$ misreads.
- On the jac base the full safety regs have the most imagined false-safes (0.149, 0.156; 0.462, 0.500 at step 25) and the most collisions (13.0, 17.5); encoder-only brings both below jac (0.100-0.113, 7.5-8.0). Spearman with collision: 0.90 over these five models, 0.73 over all ten.
- It does not explain the jac+pull family: within it the order runs the other way (Spearman -0.70, n = 5; jac+pull has the most false-safes but the fewest collisions). As a group, pull raises both false-safes and collisions.
- Caveats: one WM seed; windows follow expert actions, while CEM scores random candidates that can leave the expert distribution. The executed-plan test below checks this on the plans CEM actually runs.

## Why safety-first CEM still collides: executed-plan trace

The rollout margin test above uses expert actions. Here the evidence comes from the plans CEM actually executes. Setting: GP safety-first ($w = 1000$, $C = 10^4$), goal cost at every imagined step, replanning every 5 steps, red. At every replan we record $h$ along the chosen plan's imagined rollout, and for every episode the first env step that collides (`soft_cem.sh` with `TRACE`, `collision_cause_queue.sh`, `diag_collision_cause.py`). Replan $k$ starts after $5k$ env steps, and imagined step $j$ of its plan is env step $5k + j + 1$. Each collision is split by what the plan being executed predicted at that step: predicted unsafe ($h < 0$ at or before the collision step: the planner knowingly took a violating plan, no predicted-safe plan was found) or false-safe ($h \ge 0$ up to the collision step: the world model and margin function missed it). Tracing does not change behavior (every model with an untraced run reproduces it, e.g. jac 84.0 / 8.0, jac+adv_λ 80.0 / 28.5, jac+adv_h 60.5 / 47.5).

| Model | Collision (%) | Predicted unsafe (%) | False-safe (%) | False-safe share of collisions |
| --- | ---: | ---: | ---: | ---: |
| sigreg | 55.5 | 0.5 | 55.0 | 99% |
| adv_h | 54.5 | 1.5 | 53.0 | 97% |
| jac | 8.0 | 1.0 | 7.0 | 88% |
| jac+adv_h | 47.5 | 0.0 | 47.5 | 100% |
| jac+adv_h, encoder-only | 17.5 | 1.0 | 16.5 | 94% |
| jac+adv_λ | 28.5 | 2.5 | 26.0 | 91% |
| jac+adv_λ, encoder-only | 20.0 | 1.0 | 19.0 | 95% |
| jac+pull | 19.5 | 3.0 | 16.5 | 85% |
| jac+pull+adv_h | 41.5 | 4.5 | 37.0 | 89% |

Not run: adv_λ, the sigreg-base encoder-only models, jac+pull+adv_h encoder-only, jac+pull+adv_λ (full and encoder-only).

$h$ along the executed plans (same runs):

| Model | $h$ at crash step, median (p10-p90) | Same step, previous replan, median | Min $h$ over executed 5 steps, safe episodes, median (p10) | Crashes with $h < 0.1$ |
| --- | ---: | ---: | ---: | ---: |
| sigreg | 0.871 (0.498-1.180) | 1.089 | 0.998 (0.528) | 0.02 |
| adv_h | 0.672 (0.211-1.354) | 1.337 | 1.166 (0.684) | 0.06 |
| jac | 0.117 (0.009-0.311) | 0.394 | 0.724 (0.073) | 0.50 |
| jac+adv_h | 0.908 (0.326-1.378) | 1.091 | 0.899 (0.588) | 0.04 |
| jac+adv_h, encoder-only | 0.327 (0.086-0.683) | 0.447 | 0.977 (0.178) | 0.11 |
| jac+adv_λ | 0.239 (0.003-0.707) | 0.622 | 1.125 (0.355) | 0.25 |
| jac+adv_λ, encoder-only | 0.433 (0.032-0.888) | 0.498 | 1.148 (0.087) | 0.17 |
| jac+pull | 0.111 (0.001-0.397) | 0.399 | 0.742 (0.203) | 0.49 |
| jac+pull+adv_h | 0.404 (0.006-1.339) | 0.976 | 1.046 (0.323) | 0.22 |

- 85-100% of collisions are false-safe in every model traced. The planner almost always finds a plan it predicts safe, so search and infeasibility are not the problem. The gap between jac and the safety-reg models is almost all false-safe (7.0 vs 16.5-47.5).
- $h$ on imagined latents is optimistic, not random, for jac and jac+pull: crash steps get much lower $h$ than safe driving (0.117 vs 0.724, 0.111 vs 0.742) but stay above 0, so the soft cost never fires. For jac+adv_h it does not separate them at all: median crash-step $h$ 0.908, the same as the safe-episode minimum 0.899. The same holds on the sigreg base (sigreg 0.871 vs 0.998, adv_h 0.672 vs 1.166; 2-6% of crashes below 0.1), which explains why the soft cost never helps those models: imagined crashes look like safe driving to $h$, so the penalty never fires.
- The optimism grows with how far ahead the prediction is. One replan earlier, the same crash step had higher $h$ in every model (jac 0.394 vs 0.117, jac+adv_λ 0.622 vs 0.239); it only falls toward 0 as the car gets close. This matches the expert-window false-safe rate rising with horizon (0.10-0.16 over all steps, 0.35-0.50 at step 25).
- jac's and jac+pull's misses are mostly borderline (half of their crashes have $h < 0.1$); the safety-reg models' are more confident (4-25% below 0.1; jac+adv_h 4%). Encoder-only lowers jac+adv_h's crash-step $h$ (0.908 -> 0.327) but not jac+adv_λ's (0.239 -> 0.433); both encoder-only models collide 17.5-20.0%, between jac and the full models.

Likely causes of the optimism:

- The predictor is trained with MSE on expert data that rarely touches obstacles. Under uncertainty it predicts close to the average future, and the average expert future is safe. Contact is a thin, rare region, so imagined states rarely land in it.
- $h$ is trained on encoder latents of real frames and never sees predicted latents, so nothing anchors its sign on the states it is used on.
- CEM samples action sequences that leave the expert distribution (harder turns), and the predictor fills those in with expert-like, safe-looking futures.
- The full safety regs add to this. Their predictor term penalizes changes in what $h$ reads off the predictor output, which plausibly also dulls its response to actually approaching an obstacle (hypothesis, not isolated). The encoder-only versions recover part of it: with replanning they collide 17.5-20.0% vs jac 8.0 and the full versions 28.5-47.5.

Fixes to test, cheapest first: a safety margin $h(\hat z_t) \ge \epsilon$ (`unsafe_margin`; for jac, $\epsilon = 0.1$ covers half the crashes but would also flag about 10% of safe-episode replans); a margin that grows with the imagined step, matching the growing optimism; and training $h$ on predicted latents from CEM-style action sequences, labelled by running the same actions in the simulator (the earlier imagined-latent value function used expert windows only and the sigreg base). Tested in the next section.

## Fixing the optimistic $h$: per-step margin, $h$ on predicted latents, and recolor

Two fixes from the list above, plus two label-free variants for recolor. All CEM runs use the executed-plan trace setting (GP safety-first, $w = 1000$, $C = 10^4$, goal cost at every imagined step, replanning every 5 steps, tracing on, 200 episodes). Scripts: `fix_optimism_queue.sh`, `robust_cem_queue.sh`, `hs_cem_queue.sh`; diagnostics `eval_head_color_dubins.py`, `diag_imag_color_dubins.py`.

### What was tried

- Ramp margin: penalize $h(\hat z_t) < \epsilon_t$ with $\epsilon_t = \min(0.6,\ 0.04\,(t+1))$ for imagined step $t = 0, \dots, 24$ (`unsafe_margin` now takes a per-step list). Calibrated on the traces above: crash-step $h$ grows with how far ahead it was predicted (jac, 75th percentile: 0.17 at 1-5 steps ahead, 0.51 at 6-10, 0.8-1.0 beyond), while safe-episode plan $h$ stays flat (10th percentile 0.2-0.3). One schedule for every model, although $h$'s scale differs between models.
- $h$ on predicted latents (`train_gp_margin_imag_dubins.py`): start from one real frame of a training-split expert episode (CEM also uses a 1-frame history), sample a 25-step plan $a_t = b + 0.7\,\varepsilon_t$ in CEM's normalized action units ($b \sim \mathcal{N}(0, 1)$ per plan, $\varepsilon_t \sim \mathcal{N}(0, 1)$ per step: CEM's first iteration plus the smooth turning plans it converges to), roll it out with the predictor exactly as `JEPA.rollout`, and label each imagined step by running the same de-normalized, clipped actions through the Dubins simulator from the frame's true state (1 = inside a red circle). Trained on 6000 plans × 25 steps plus the original GP's 8000 real uniform-state latents, with the same GP loss and steps. About 15% of imagined steps are unsafe.
- Dynamic threshold (`h_quantile` $= p$; implemented as a rank normalization of $h$): at each replan, rescale $h$ over all 300 candidates × imagined steps so that its $p$-quantile maps to 0 and its median to 1; the bottom $p$ of imagined latents then read unsafe. In effect the safe/unsafe threshold is re-set every replan from the current scene instead of fixed at 0, so a shift or rescaling of $h$ under OOD (Table 9) cancels out. Only $p = 0.05$ was run.
- Short-horizon cost (`h_steps` $= n$): safety cost on the first $n$ imagined steps only.

### Does training on predicted latents fix $h$? (no planning)

Table 5. GP margin function evaluated on imagined latents of held-out CEM-style plans (red, last 10% of episodes, 1000 plans × 25 steps, simulator labels). "Old" = trained on real latents only; "new" = trained on real + predicted latents. False-safe = share of unsafe imagined steps with $h \ge 0$; false alarm = share of safe imagined steps with $h < 0$.

| Model | AUC, old | AUC, new | False-safe, old | False-safe, new | False-safe steps 21-25, old | False-safe steps 21-25, new | False alarm, old | False alarm, new |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sigreg | 0.642 | 0.822 | 0.968 | 0.302 | 0.962 | 0.447 | 0.012 | 0.181 |
| adv_h | 0.637 | 0.731 | 0.971 | 0.420 | 0.996 | 0.572 | 0.002 | 0.250 |
| jac | 0.967 | 0.979 | 0.112 | 0.074 | 0.151 | 0.113 | 0.067 | 0.060 |
| jac+adv_h | 0.960 | 0.974 | 0.228 | 0.084 | 0.296 | 0.150 | 0.021 | 0.084 |
| jac+adv_h, encoder-only | 0.966 | 0.979 | 0.118 | 0.083 | 0.169 | 0.134 | 0.065 | 0.054 |
| jac+adv_λ | 0.967 | 0.978 | 0.163 | 0.080 | 0.243 | 0.161 | 0.033 | 0.065 |
| jac+adv_λ, encoder-only | 0.964 | 0.976 | 0.145 | 0.117 | 0.204 | 0.193 | 0.062 | 0.043 |
| jac+pull | 0.963 | 0.979 | 0.240 | 0.093 | 0.379 | 0.146 | 0.037 | 0.051 |
| jac+pull+adv_h | 0.937 | 0.967 | 0.296 | 0.122 | 0.440 | 0.226 | 0.052 | 0.074 |
| jac+pull+adv_h, encoder-only | 0.948 | 0.971 | 0.232 | 0.110 | 0.376 | 0.214 | 0.059 | 0.076 |

- Training on predicted latents halves false-safe on most jac-base models (0.228 -> 0.084 for jac+adv_h, 0.296 -> 0.122 for jac+pull+adv_h), while false alarms change by −2 to +6 points. On the sigreg base the old $h$ called almost every imagined unsafe step safe (0.968, 0.971); the new one recovers part of the ranking (AUC 0.822, 0.731) at 18-25% false alarms.

### Planning on red: does each fix reduce collisions?

Table 6. CEM on red (in-distribution), success / collision (%), 200 episodes. One column per $h$ variant; blank = not run.

| Model | Old $h$ | Ramp margin | $h$ on predicted latents | Predicted latents + ramp |
| --- | ---: | ---: | ---: | ---: |
| sigreg | 63.0 / 55.5 | 61.0 / 57.5 | 53.5 / 52.5 | |
| adv_h | 64.0 / 54.5 | 64.5 / 54.0 | 61.0 / 49.5 | |
| jac | 84.0 / 8.0 | 74.5 / 9.0 | 84.5 / 6.0 | 73.5 / 7.0 |
| jac+adv_h | 60.5 / 47.5 | 61.5 / 39.5 | 61.5 / 36.5 | |
| jac+adv_h, encoder-only | 84.5 / 17.5 | 81.5 / 12.0 | 82.5 / 20.0 | 74.5 / 6.0 |
| jac+adv_λ | 80.0 / 28.5 | 77.5 / 21.5 | 81.5 / 19.5 | |
| jac+adv_λ, encoder-only | 78.5 / 20.0 | 78.0 / 7.0 | 81.0 / 19.5 | |
| jac+pull | 74.0 / 19.5 | 72.5 / 8.0 | 80.0 / 14.5 | |
| jac+pull+adv_h | 69.5 / 41.5 | 67.0 / 33.5 | 74.0 / 33.5 | 64.5 / 17.5 |
| jac+pull+adv_h, encoder-only | 76.0 / 20.5 | | 75.5 / 15.5 | 66.5 / 11.5 |

- $h$ on predicted latents lowers collision on 6 of the 8 jac-base models (by 2-11 points). Success rises or stays within 1.5 points on all of them. jac reaches 84.5 / 6.0. Exceptions: jac+adv_h encoder-only (17.5 -> 20.0) and jac+adv_λ encoder-only (20.0 -> 19.5, unchanged).
- The ramp lowers collision on 6 of the 7 jac-base models it ran on, most on jac+pull (19.5 -> 8.0) and jac+adv_λ encoder-only (20.0 -> 7.0), at 0.5-3 points of success; on jac it only costs success (84.0 -> 74.5).
- Both together give the largest drops on the robust models: jac+adv_h encoder-only 17.5 -> 6.0, jac+pull+adv_h 41.5 -> 17.5, jac+pull+adv_h encoder-only 20.5 -> 11.5, at 5-10 points of success.
- The sigreg base does not benefit from either fix (49.5-57.5% collision), consistent with its $h$ not separating imagined crashes.
- Most remaining red collisions are still false-safe (62-100% of collisions in every traced red run with the ramp or the predicted-latent $h$; jac 75-93%). Exception: adv_h with $h$ on predicted latents, where 28.5 of its 49.5 points are predicted unsafe: $h$ now sees the crash, but CEM finds no predicted-safe plan.

### Planning under recolor (purple): does any fix help?

Table 7. CEM with purple obstacles (OOD), success / collision (%), 200 episodes, for the models that are robust on real samples plus jac as reference. One column per $h$ variant; blank = not run. Old-$h$ cells for jac and jac+pull+adv_h are the untraced fair-replanning runs.

| Model | Old $h$ | $h$ on predicted latents | Predicted latents + ramp | Dynamic threshold, predicted latents ($p = 0.05$) | Dynamic threshold, old $h$, first 5 steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| jac | 34.5 / 43.0 | 29.0 / 48.5 | 21.5 / 40.0 | | 23.5 / 72.0 |
| jac+adv_h, encoder-only | 35.5 / 51.5 | 27.5 / 54.5 | 26.5 / 49.0 | | |
| jac+pull+adv_h | 26.5 / 61.5 | 30.0 / 51.0 | 32.5 / 42.0 | | |
| jac+pull+adv_h, encoder-only | 30.5 / 47.5 | 20.0 / 48.0 | 16.0 / 57.0 | 33.0 / 54.5 | 29.0 / 68.0 |

Table 8. The two label-free variants of Table 7 run on red, to check what they cost in-distribution. Success / collision (%), 200 episodes; blank = not run.

| Model | Old $h$ (from Table 6) | Dynamic threshold, predicted latents ($p = 0.05$) | Dynamic threshold, old $h$, first 5 steps |
| --- | ---: | ---: | ---: |
| jac | 84.0 / 8.0 | | 78.0 / 38.5 |
| jac+pull+adv_h | 69.5 / 41.5 | 74.5 / 39.0 | |
| jac+pull+adv_h, encoder-only | 76.0 / 20.5 | 76.5 / 20.5 | 73.0 / 33.0 |

- No fix helps under purple: every variant gives 40-72% collision at 16-35.5% success, and the robust models do not beat jac.
- The collision split flips with $h$. With the old $h$, jac+pull+adv_h encoder-only's purple collisions are 100% false-safe; with $h$ on predicted latents 95% are predicted unsafe (jac: 93%). Under purple the old $h$ reads everything safe and the new one reads nearly everything unsafe, so the safety-first cost ties all candidates and ranks on goal cost alone.
- Restricting the cost to the first 5 imagined steps is much worse even on red (jac 8.0 -> 38.5% collision): at fixed speed the car needs more than 5 steps of lookahead to see a doomed state, which a margin function (instantaneous failure) does not encode.

- The dynamic threshold costs nothing on red (76.0 / 20.5 -> 76.5 / 20.5) but does not help under purple; restricting to 5 steps hurts on both. A per-replan threshold can only re-zero $h$ where $h$ still ranks the imagined latents correctly, and from a purple start that ranking is gone after about 5 imagined steps (Table 10); a per-episode shift anchored on the real start frame and other values of $p$ were not run, since they hit the same limit.

### Why recolor fails (a): $h$ on real purple frames

Table 9. GP margin function on real latents (no prediction): 4000 uniform states rendered with purple circles (training orientation, disjoint from the training draws). AUC = ranking; balanced sign accuracy = whether $h$'s sign is right (0.500 = everything on one side of 0).

| Model | Purple AUC, old $h$ | Purple AUC, $h$ on predicted latents | Purple bal. sign acc, old | Purple bal. sign acc, new |
| --- | ---: | ---: | ---: | ---: |
| jac | 0.575 | 0.643 | 0.500 | 0.500 |
| jac+adv_h, encoder-only | 0.885 | 0.898 | 0.500 | 0.500 |
| jac+pull+adv_h | 0.896 | 0.946 | 0.500 | 0.500 |
| jac+pull+adv_h, encoder-only | 0.954 | 0.963 | 0.500 | 0.861 |

- Training on predicted latents does not cost real-sample color robustness; purple AUC rises on every model.
- On the robust models purple keeps the ranking but collapses the scale. jac+pull+adv_h encoder-only, median $h$ on safe / unsafe states: red 1.56 / −0.53, purple 0.38 / 0.32 (old $h$); 0.03 / −0.08 with $h$ on predicted latents. Hence balanced sign accuracy 0.500, and the motivation for the dynamic threshold.

### Why recolor fails (b): $h$ on imagined latents, per imagined step

Table 10. AUC of $h$ on imagined latents at each imagined step, rolled out from a red or a purple start frame (600 held-out expert states re-rendered in each color, CEM-style plans, simulator labels). GP margin function trained on real latents unless the row says otherwise. Per-step AUCs measured at 2 decimals.

| WM, $h$ | Start | Step 1 | Step 3 | Step 5 | Step 10 | Step 15 | Step 25 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| jac+pull+adv_h, encoder-only | red | 1.00 | 1.00 | 1.00 | 0.98 | 0.93 | 0.77 |
| jac+pull+adv_h, encoder-only | purple | 0.80 | 0.88 | 0.81 | 0.54 | 0.54 | 0.51 |
| jac+pull+adv_h, encoder-only, $h$ on predicted latents | purple | 0.80 | 0.82 | 0.58 | 0.42 | 0.47 | 0.50 |
| jac+pull+adv_h, encoder-only, value function | purple | 0.84 | 0.68 | 0.60 | 0.42 | 0.47 | 0.50 |
| jac+pull+adv_h | red | 1.00 | 1.00 | 1.00 | 0.96 | 0.92 | 0.80 |
| jac+pull+adv_h | purple | 0.74 | 0.58 | 0.66 | 0.55 | 0.56 | 0.57 |
| jac+adv_h, encoder-only | purple | 0.64 | 0.56 | 0.52 | 0.48 | 0.51 | 0.48 |
| jac | purple | 0.53 | 0.62 | 0.51 | 0.45 | 0.46 | 0.54 |
| coloraug_jacobian | purple | 0.97 | 0.95 | 0.60 | 0.44 | 0.48 | 0.48 |
| coloraug_jacpull | purple | 0.81 | 0.62 | 0.46 | 0.57 | 0.54 | 0.50 |

- From a red start every model keeps $h$'s ranking for 10+ imagined steps (0.92-1.00 at step 10). From a purple start it is gone by step 5-10 for every model, including the color-augmented WMs and full jac+pull+adv_h, whose predictor term targets $h$. Over all 25 steps the purple AUC is 0.40-0.59 for every model and head.

### Conclusions and next step

- The encoder is color-robust (real purple frames: 0.885-0.963 AUC, Table 9), yet $h$ on imagined purple latents is near chance after 5-10 steps. The next section locates the cause: the predictor tracks the purple world about as well as the red one, and the failure is $h$'s tiny safe/unsafe gap in the purple region (Table 14).
- Fixes to $h$ (margin, retraining, dynamic threshold, the value function) cannot help while that gap is smaller than one step of prediction noise.
- Caveats: one WM seed; dynamic-threshold runs record raw $h$ in their traces (the chosen plan is re-scored alone), so their predicted-unsafe / false-safe split uses raw $h$; the ramp schedule is not tuned per model.

## Why purple imagined states fail: the predictor tracks purple, $h$'s gap collapses

The tables above leave a gap in the explanation: $h$ reads real purple frames well (Table 9) and the predictor is accurate on red, yet $h$ on imagined purple latents is near chance. To locate the failure we rolled out the same 400 held-out states and CEM-style plans from a red and from a purple start, and also encoded the true future frames rendered in each color (`diag_purple_drift.py`, original 50-epoch models).

Table 11. Where a purple-start rollout goes, jac+pull+adv_h encoder-only. Distances are mean squared distance divided by the total variance of real red latents.

| Per imagined step | 1 | 3 | 5 | 10 | 15 | 25 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| How far recolor moves the true latent | 1.34 | 1.31 | 1.29 | 1.18 | 1.07 | 0.92 |
| Purple rollout vs true purple future | 0.07 | 0.14 | 0.19 | 0.57 | 1.16 | 1.80 |
| Red rollout vs true red future (reference) | 0.02 | 0.05 | 0.12 | 0.53 | 1.01 | 1.42 |
| Purple rollout vs true red future | 1.33 | 1.33 | 1.38 | 1.72 | 2.19 | 2.38 |
| Purple rollout vs red rollout | 1.31 | 1.29 | 1.31 | 1.68 | 2.14 | 2.39 |

Table 12. AUC of $h$ on the same states, jac+pull+adv_h encoder-only.

| Per imagined step | 1 | 3 | 5 | 10 | 15 | 25 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| True purple futures (no prediction) | 0.94 | 0.94 | 0.94 | 0.95 | 0.92 | 0.91 |
| Purple rollout | 0.80 | 0.88 | 0.78 | 0.59 | 0.50 | 0.40 |
| Red rollout | 1.00 | 1.00 | 1.00 | 0.96 | 0.93 | 0.78 |

Table 13. Same as Tables 11-12 for jac (not color-robust), selected rows.

| Per imagined step | 1 | 3 | 5 | 10 | 15 | 25 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| How far recolor moves the true latent | 1.64 | 1.61 | 1.59 | 1.47 | 1.35 | 1.19 |
| Purple rollout vs true purple future | 0.19 | 0.26 | 0.47 | 1.17 | 1.97 | 2.35 |
| Red rollout vs true red future | 0.02 | 0.05 | 0.13 | 0.46 | 0.78 | 1.24 |
| AUC of $h$, true purple futures | 0.52 | 0.53 | 0.53 | 0.55 | 0.47 | 0.49 |
| AUC of $h$, purple rollout | 0.52 | 0.50 | 0.49 | 0.43 | 0.43 | 0.46 |

- Recolor moves the latent by more than the whole spread of red latents (1.3× its variance), so purple frames live in their own region of latent space.
- The predictor rolls a purple latent forward inside that region: it stays near the true purple future and does not snap back toward red-world latents. Its error is somewhat larger than on red at first and about the same from step 10 (0.57 vs 0.53).
- Yet at step 10, with about equal prediction error, $h$ ranks the red rollout at 0.96 and the purple rollout at 0.59.
- For jac, $h$ already fails on true purple frames (0.52), so jac is not this case: its problem is the encoder, not prediction.

Table 14. $h$'s signal vs prediction noise, jac+pull+adv_h encoder-only. Gap = mean $h$ on safe minus mean $h$ on unsafe true future latents; noise = standard deviation over samples of $h$(rollout) − $h$(true future).

| Per imagined step | 1 | 3 | 5 | 10 | 15 | 25 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Red: $h$ gap | 1.73 | 1.72 | 1.71 | 1.69 | 1.56 | 1.40 |
| Red: $h$ noise from prediction | 0.03 | 0.09 | 0.13 | 0.34 | 0.52 | 0.67 |
| Red: gap / noise | 60.24 | 19.76 | 12.70 | 4.96 | 2.99 | 2.08 |
| Purple: $h$ gap | 0.06 | 0.06 | 0.06 | 0.06 | 0.05 | 0.04 |
| Purple: $h$ noise from prediction | 0.05 | 0.05 | 0.05 | 0.05 | 0.06 | 0.33 |
| Purple: gap / noise | 1.26 | 1.20 | 1.20 | 1.16 | 0.75 | 0.13 |

- In the purple region $h$ still orders states correctly but its whole safe-to-unsafe range is about 30 times smaller (0.06 vs 1.73). One step of prediction noise (0.05) is already as large as that range, so the ranking on imagined latents is near chance from the first steps; on red the 1.7-wide gap absorbs 10+ steps of noise.
- So the defect is the size of $h$'s safe/unsafe gap under an appearance shift, not predictor drift. This explains why a better predictor (the rollout loss, `archive.md`) does not help purple, and why $h$-side fixes (margin, retraining on predicted latents, dynamic threshold) cannot either.
- Candidate fixes, all trained on red data only (OOD stays a test-only measure): keep the safety gap under generic appearance perturbations (an encoder-side penalty when perturbations shrink the safe/unsafe distance along the direction $h$ reads; the saved $\lambda$-projected invariance idea is one version); or weight the prediction loss by how much an error moves $h$, so the predictor is most accurate along the directions $h$ depends on.
- Caveats: one WM seed; 400 states; a single recolor (purple).

## Would a shorter-trained margin function help CEM?

The margin-function step sweep in the Dubins section shows that 2500 steps transfers better to purple frames than 5000 for some models. To see whether it would change planning, we trained 2500-step GP margin functions with exactly the protocol of the CEM's $h$ (`train_gp_margin_dubins_steps.py`: red, uniform states, seed 0; saved as `margin_gp_fail_red_s0_st2500.pt`) and scored them, next to the CEM's own 5000-step $h$, on imagined latents of CEM-style plans (`diag_imag_steps.py`: the setup of `diag_imag_color_dubins.py`, 600 starts from the last 10% of episodes, 25-step plans $a_t = b + 0.7\,\varepsilon_t$, simulator labels, 16.7% of imagined steps unsafe).

Table 15. $h$ on imagined latents, margin function trained 2500 vs 5000 steps (one margin-function seed). False-safe = share of unsafe imagined steps with $h \ge 0$.

| Model | Red AUC, 2500 | Red AUC, 5000 | Red false-safe, 2500 | Red false-safe, 5000 | Red false-safe at step 25, 2500 | Red false-safe at step 25, 5000 | Purple AUC, 2500 | Purple AUC, 5000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sigreg | 0.813 | 0.814 | 0.353 | 0.423 | 0.509 | 0.630 | 0.517 | 0.499 |
| jac | 0.969 | 0.969 | 0.109 | 0.115 | 0.222 | 0.213 | 0.422 | 0.429 |
| jac+pull | 0.961 | 0.961 | 0.204 | 0.281 | 0.315 | 0.435 | 0.512 | 0.504 |
| adv_h | 0.645 | 0.661 | 0.789 | 0.796 | 0.981 | 0.981 | 0.419 | 0.409 |
| jac+pull+adv_h | 0.923 | 0.927 | 0.330 | 0.351 | 0.417 | 0.435 | 0.551 | 0.586 |
| adv_λ | 0.621 | 0.626 | 0.763 | 0.765 | 0.972 | 0.972 | 0.576 | 0.572 |

- Training length does not change how $h$ ranks imagined states: red AUC is equal within 0.016 on every model.
- On red, the 2500-step $h$ sits lower on unsafe imagined states, which cuts false-safe for sigreg (0.423 -> 0.353) and jac+pull (0.281 -> 0.204; at step 25 0.435 -> 0.315). jac, the best planner, does not move (0.115 -> 0.109); adv_h and adv_λ are unchanged and read imagined red states poorly either way.
- On purple the ranking is near chance at both lengths (0.41-0.59), as in Table 10 and Table 14: the better real-frame transfer of the shorter margin function does not survive prediction.
- Verdict: switching to 2500 steps will not change CEM much. The one cell worth a CEM run is jac+pull on red.
- Why sigreg and adv_h differ from Table 5's "old" column (same checkpoint and margin-function file): the start frame. Table 5 (`train_gp_margin_imag_dubins.py`) starts each rollout from the stored dataset frame, whose goal marker sits where it was during collection (jittered around $(1.3, 0)$); this table, Table 10 and Table 14 (`diag_imag_color_dubins.py`, `diag_purple_drift.py`) re-render the start state with the swm environment, which draws the marker at its default $(1.3, 0)$, as in the margin function's own training frames. The two frames differ only in the marker (about 0.3% of pixels). An A/B on identical starts, plans and labels (`imag_startframe_ab.py`, CEM's 5000-step $h$):

| Model | Start frame | Imagined AUC | False-safe |
| --- | --- | ---: | ---: |
| sigreg | re-rendered | 0.814 | 0.423 |
| sigreg | dataset frame | 0.664 | 0.967 |
| adv_h | re-rendered | 0.661 | 0.796 |
| adv_h | dataset frame | 0.584 | 0.983 |
| jac | re-rendered | 0.969 | 0.115 |
| jac | dataset frame | 0.968 | 0.115 |

  The dataset-frame rows reproduce Table 5 (sigreg 0.642 / 0.968, adv_h 0.637 / 0.971); jac is unaffected. So $h$ on sigreg and adv_h is sensitive to where the goal marker is drawn, which the real-frame check (marker nuisance, Diagnostics) had found for sigreg only (AUC 0.985 -> 0.901), not for adv_h. In CEM the marker is drawn at each episode's goal, a third placement, so neither row is exactly CEM's input for these two models; this sensitivity may be part of why the soft cost never helps the sigreg-base models (not tested). The jac-base conclusions do not depend on the choice.

## Does the predictor amplify latent errors?

A proposed predictor regularizer is a non-expansion penalty on the dynamics, $\max(0, \|\partial P/\partial z\| - 1)^2$, on the argument that with gain $L > 1$ per step a rollout error grows like $L^H$. Before adding it we measured the gain of the existing predictors (`pred_jacobian_gain.py`): 128 random windows of the Dubins dataset, 3 real context frames and real actions, rollout as in `JEPA.rollout`; per window, the largest singular value $\sigma_{\max}$ of the map from the 3-frame context window to the predicted latent $K$ steps later (power iteration, 15 steps; $Jv$ by central finite differences, $J^\top u$ by autograd), and the gain along one random direction. A gain above 1 means a perturbation grows.

Table 16. Predictor gain, median over windows (p90 for 25 steps in parentheses). "Last frame" = gain with respect to the last context frame only.

| Model | 1 step, window | 1 step, last frame | 5 steps | 10 steps | 25 steps | 25 steps, random direction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sigreg | 3.14 | 3.11 | 10.1 | 14.5 | 20.4 (42.7) | 0.66 |
| jac | 2.70 | 2.68 | 7.1 | 13.7 | 34.8 (63.4) | 1.03 |
| jac+pull | 2.02 | 2.00 | 6.2 | 12.5 | 33.4 (51.3) | 1.11 |
| adv_h | 2.55 | 2.51 | 6.9 | 13.1 | 23.3 (49.5) | 0.67 |
| jac+pull+adv_h | 2.18 | 2.16 | 6.0 | 11.3 | 24.8 (51.4) | 0.81 |
| adv_λ | 2.63 | 2.62 | 6.0 | 10.0 | 17.0 (41.8) | 0.60 |
| sigreg, rollout loss from 3 frames | 2.91 | 1.48 | 5.4 | 6.1 | 4.7 (11.4) | 0.22 |
| jac, rollout loss from 3 frames | 1.83 | 1.65 | 4.4 | 8.4 | 19.6 (25.9) | 0.61 |

- Every predictor expands at one step ($\sigma_{\max}$ 1.8-3.1), so a non-expansion hinge at 1 would be active everywhere.
- The growth is far from exponential: a one-step gain of 2-3 compounded over 25 steps would be $10^{8}$-$10^{12}$, the measured worst case is 17-35, and a typical direction is about neutral (0.60-1.11). The most-amplified directions change from step to step instead of stacking.
- Amplification does not track planning: jac and jac+pull amplify the most at 25 steps and plan best, consistent with collision tracking goal-distance fidelity rather than 25-step rollout error (Diagnostics).
- The two rollout-loss rows are the failed first version (rollout from 3 real frames), whose rollouts collapse after step 5 (`archive.md`, Table A1). Their low long-horizon gain (sigreg 4.7, random direction 0.22) is that collapse, not stability; low amplification is not good by itself. The fixed version was evaluated at 50 epochs and paused (`archive.md`).
- As written the proposal also bounds the wrong norm: it uses the Frobenius norm via the same Hutchinson estimator as jac, but the error argument needs the spectral norm, and for a near-identity 192-dimensional map $\|J\|_F \approx \sqrt{192} \approx 14$, so a Frobenius bound of 1 would force strong contraction against the prediction loss. If tried, it should be a spectral-norm hinge (power iteration) on the full context window; the rollout loss, which attacks the same compounding-error problem, did not improve planning (`archive.md`).

## Linear $h$ and threshold-free costs in CEM

The linear margin function is introduced with the margin functions ("Linear vs MLP margin function"); here it and two threshold-free costs are used as $h$ in CEM.

### Linear $h$ in CEM

Safety-first soft cost ($w = 1000$, $C = 10^4$), goal cost at every imagined step, `lin_h_cem_queue.sh`, 50 episodes per cell (the screen size); GP cells are the 200-episode runs of Tables 6-7.

Table 17. CEM, linear vs GP margin function as $h$, success / collision (%). Blank = not run.

| Model | Setting | Scenario | Linear $h$ (50 episodes) | GP $h$ (200 episodes) |
| --- | --- | --- | --- | --- |
| jac | replan every 5 | red | 86.0 / 10.0 | 84.0 / 8.0 |
| jac | replan every 5 | purple | 22.0 / 52.0 | 34.5 / 43.0 |
| jac | open loop | red | 82.0 / 14.0 | 76.0 / 22.0 |
| jac | open loop | purple | 44.0 / 24.0 | 43.0 / 26.5 |
| jac+pull+adv_h, encoder-only | replan every 5 | red | 74.0 / 20.0 | 76.0 / 20.5 |
| jac+pull+adv_h, encoder-only | replan every 5 | purple | 24.0 / 36.0 | 30.5 / 47.5 |
| jac+pull+adv_h, encoder-only | open loop | red | 60.0 / 38.0 | |
| jac+pull+adv_h, encoder-only | open loop | purple | 40.0 / 30.0 | |

- Red: a tie within the noise of 50 episodes (about ±7 points at these rates).
- Purple: no consistent gain. jac with replanning collides more (52 vs 43), jac+pull+adv_h encoder-only collides less but also succeeds less (36 vs 47.5 collision, 24 vs 30.5 success), open loop is a tie.
- This is what Tables 2-4 predict: the fixed-margin cost needs $h$'s sign, and under purple the sign is wrong for every model and both margin functions; the linear margin function fixes only the ranking. A rank-based cost (the dynamic threshold, `HQ`) is the cost that could use the recovered ranking, provided it survives prediction (Table 14 shows the gap on imagined purple latents is within one step of prediction noise for the GP $h$; not yet measured for the linear one).
- Caveats: one WM seed, one margin-function seed, 50-episode CEM cells.

### Threshold-free costs: dynamic threshold and a cost relative to the current frame

Two costs that need only $h$'s ranking, each with the GP and the linear margin function, safety cost on the first 5 imagined steps only (the window where a linear readout still ranks purple rollouts, 0.99 / 0.95 / 0.85 at steps 1 / 3 / 5, and the only steps executed with replanning every 5). Dynamic threshold: per-replan rank-normalized $h$ with $p = 0.05$, $C = 10^4$ (`HQ`). Relative cost: $w \sum_{t \le 5} \mathrm{relu}\big(h(z_0) - h(\hat z_t)\big)$ with $z_0$ the current real encoded frame, so a constant shift of $h$ cancels (`h_rel` in `safe_solver.py`, `HREL`); $C = 0$, since nearly every candidate lowers $h$ at some step. Both: $w = 1000$, replan every 5, goal cost at every imagined step (`lin_h_rank_cem_queue.sh`, `gp_h_rel_cem_queue.sh`).

Table 18. CEM with threshold-free costs, success / collision (%). GP dynamic threshold: 200 episodes; others: 50 episodes. Reference, GP $h$ with the fixed margin on all 25 steps (200 episodes): jac red 84.0 / 8.0, purple 34.5 / 43.0; jac+pull+adv_h encoder-only red 76.0 / 20.5, purple 30.5 / 47.5.

| Model | Scenario | GP $h$, dynamic threshold | Linear $h$, dynamic threshold | GP $h$, relative cost | Linear $h$, relative cost |
| --- | --- | --- | --- | --- | --- |
| jac | red | 78.0 / 38.5 | 82.0 / 32.0 | 66.0 / 34.0 | 74.0 / 24.0 |
| jac | purple | 23.5 / 72.0 | 18.0 / 58.0 | 36.0 / 54.0 | 18.0 / 72.0 |
| jac+pull+adv_h, encoder-only | red | 73.0 / 33.0 | 80.0 / 38.0 | 62.0 / 28.0 | 64.0 / 38.0 |
| jac+pull+adv_h, encoder-only | purple | 29.0 / 68.0 | 26.0 / 74.0 | 34.0 / 58.0 | 26.0 / 58.0 |

- Purple: every threshold-free variant collides more (54-74%) than the fixed margin on all 25 steps (43-47.5%). The linear margin function is no better than the GP one, although it ranks real purple frames far better (Table 2).
- Red: restricting the safety cost to 5 steps costs 16-30 points of collision against the fixed margin, as the 5-step GP runs did before.
- Recovering the ranking on real frames does not carry into planning: on imagined purple latents $h$'s variation over the executed steps is within prediction noise (Table 14), whichever readout or cost form reads it. The remaining lever is the encoder, keeping the safe/unsafe gap under appearance change with a red-only training term.

## adv_h with a linear internal head (2026-11-05, epoch 30, preliminary)

adv_h trains an internal safety head alongside the world model on the failure labels, and its adversary searches for the pixel perturbation that moves that head's output most. In the original adv_h this internal head is an MLP (192-512-512-1, the GP margin function's architecture), so the invariance it buys targets an MLP readout, while the linear margin function above reads the latent through a direction the adversary never attacked. The variant makes the internal head linear, $h_\text{adv}(z) = w^\top z + b$ (`head_layers: 0`), everything else identical: code snapshot `code_safeadv_92e7589_linhead` (the original snapshot plus that option), `train_linhead_dubins.sh`. Naming: "MLP / linear internal head" is the world model; "GP / linear margin function" is the downstream $h$ trained afterwards on the frozen latents. All four use the same GP loss. Trained so far: jac+adv_h and jac+pull+adv_h encoder-only (seed 3072, 50 epochs, at epoch 30-35 here); adv_h, jac+adv_h encoder-only and jac+pull+adv_h are queued. Each checkpoint is scored at epochs 10, 15, 20 and 30 (`epoch_head_check.sh`), the original models at the same epochs, since the adv term only reaches full strength around epoch 12.

Table 19. Purple AUC on real latents (4000 uniform states, as Table 2), by training epoch. Red, diamond and rotated 90° are 0.995-0.998 for every entry.

| Model | Internal head | Linear $h$, ep 10 | ep 15 | ep 20 | ep 30 | GP $h$, ep 10 | ep 15 | ep 20 | ep 30 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| jac+adv_h | MLP | 0.177 | 0.976 | 0.985 | 0.990 | 0.146 | 0.906 | 0.584 | 0.412 |
| jac+adv_h | linear | 0.396 | 0.945 | 0.990 | 0.993 | 0.666 | 0.916 | 0.932 | 0.876 |
| jac+pull+adv_h, encoder-only | MLP | 0.945 | 0.980 | 0.991 | 0.996 | 0.641 | 0.797 | 0.784 | 0.958 |
| jac+pull+adv_h, encoder-only | linear | 0.725 | 0.984 | 0.990 | 0.994 | 0.756 | 0.908 | 0.988 | 0.968 |

Table 20. AUC on imagined latents from a purple start, epoch 30 (as Table 10), by imagined step.

| Model | Internal head | Margin function | Step 1 | Step 3 | Step 5 | Step 10 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| jac+adv_h | MLP | linear | 0.93 | 0.78 | 0.66 | 0.57 |
| jac+adv_h | linear | linear | 0.97 | 0.79 | 0.29 | 0.31 |
| jac+adv_h | MLP | GP | 0.23 | 0.63 | 0.68 | 0.57 |
| jac+adv_h | linear | GP | 0.34 | 0.48 | 0.28 | 0.28 |
| jac+pull+adv_h, encoder-only | MLP | linear | 0.99 | 0.94 | 0.85 | 0.51 |
| jac+pull+adv_h, encoder-only | linear | linear | 0.99 | 0.90 | 0.76 | 0.46 |
| jac+pull+adv_h, encoder-only | MLP | GP | 0.80 | 0.54 | 0.49 | 0.43 |
| jac+pull+adv_h, encoder-only | linear | GP | 0.98 | 0.60 | 0.50 | 0.46 |

Table 21. Safety-first CEM ($w = 1000$, $C = 10^4$, goal cost at every imagined step), success / collision (%). Linear internal head: epoch-30 checkpoint, 50 episodes (`linhead_cem_queue.sh`). MLP internal head: the 50-epoch models; linear-$h$ cells 50 episodes (Table 17 and this screen), GP-$h$ cells 200 episodes. Blank = not run.

| Model | Setting | Scenario | Linear internal head, linear $h$ | Linear internal head, GP $h$ | MLP internal head, linear $h$ | MLP internal head, GP $h$ |
| --- | --- | --- | --- | --- | --- | --- |
| jac+adv_h | replan every 5 | red | 76.0 / 16.0 | 70.0 / 32.0 | 64.0 / 44.0 | 60.5 / 47.5 |
| jac+adv_h | replan every 5 | purple | 28.0 / 56.0 | 10.0 / 56.0 | 24.0 / 62.0 | 33.5 / 56.0 |
| jac+adv_h | open loop | red | 62.0 / 36.0 | 60.0 / 34.0 | 54.0 / 50.0 | 57.5 / 47.5 |
| jac+adv_h | open loop | purple | 30.0 / 44.0 | 24.0 / 42.0 | 18.0 / 58.0 | 33.0 / 56.5 |
| jac+pull+adv_h, encoder-only | replan every 5 | red | 84.0 / 30.0 | 70.0 / 30.0 | 74.0 / 20.0 | 76.0 / 20.5 |
| jac+pull+adv_h, encoder-only | replan every 5 | purple | 26.0 / 56.0 | 30.0 / 68.0 | 24.0 / 36.0 | 30.5 / 47.5 |
| jac+pull+adv_h, encoder-only | open loop | red | 74.0 / 40.0 | 64.0 / 46.0 | 60.0 / 38.0 | |
| jac+pull+adv_h, encoder-only | open loop | purple | 42.0 / 52.0 | 40.0 / 54.0 | 40.0 / 30.0 | |

- Real frames: the linear internal head does not change the linear margin function, which reads purple at 0.98-0.99 from epoch 15-20 with either internal head. It does stabilize the GP margin function on jac+adv_h, whose purple AUC with the MLP internal head swings and falls with training (0.906, 0.584, 0.412 at epochs 15, 20, 30) and with the linear internal head stays at 0.88-0.93. Purple sign accuracy stays 0.500 in almost every cell (exceptions: jac+adv_h, linear internal head, GP margin function 0.705 and 0.800 at epochs 15 and 30; jac+pull+adv_h encoder-only, linear internal head, linear margin function 0.903 at epoch 20).
- Imagined purple: no longer window. For jac+adv_h the linear internal head shortens it (linear margin function at step 5: 0.66 -> 0.29).
- CEM, jac+adv_h: the largest effect is on red. With replanning, collision falls from 44.0-47.5 to 16.0 with the linear margin function (32.0 with the GP one), so most of what the full safety reg cost the soft cost on the jac base (jac: 84.0 / 8.0) is recovered. Under purple, open loop improves (56.5-58.0 -> 42.0-44.0); with replanning it does not (56.0).
- CEM, jac+pull+adv_h encoder-only: more success but more collision (red with replanning 30.0 vs 20.0-20.5; purple 52.0-68.0 vs 30.0-47.5).
- Caveats: epoch-30 checkpoints against 50-epoch originals (the learning rate is still about 35% of its peak at epoch 30); 50-episode cells, about ±7 points; one WM seed and one margin-function seed. The same screen reruns on the epoch-50 checkpoints, and the other three bases are queued.

## Do CEM's elites converge? Elite spread

CEM refits the sampling distribution to the 30 elites each iteration and executes the elite mean after 30 iterations, so the plan run is never scored itself. If the elites had not converged, or split between two routes (left and right of an obstacle), their mean could drive between them. Measured with a callback on every CEM solve (`SOFTCEM_SPREAD`, `elite_spread.sh`, `diag_elite_spread.py`): jac, red, open loop, final-step goal cost, about 240 solves per run. Std is in CEM's action units (initial std 1). The runs reproduce the main results (safety-first 79.0 / 8.5; raw 79.5 / 26.0 vs 81.5 / 22.5, within noise).

| | Raw CEM | Safety-first |
| --- | ---: | ---: |
| Median elite std at iteration 0 / 10 / 20 / 29 | 0.971 / 0.607 / 0.330 / 0.182 | 0.975 / 0.603 / 0.314 / 0.172 |
| Final elite std at imagined step 1 / 5 / 15 / 25 | 0.100 / 0.130 / 0.181 / 0.208 | 0.083 / 0.113 / 0.173 / 0.204 |
| RMS distance, best elite to the executed mean, median (p90) | 0.189 (0.342) | 0.169 (0.278) |
| Share of elite variance on the top direction, median (p90) | 0.21 (0.29) | 0.21 (0.29) |
| Solves with all 30 elites predicted safe | n/a (no $h$) | 0.967 |

- The elites do not fully converge. The std shrinks about 6% per iteration and is still falling when CEM stops (0.24 -> 0.18 over the last 5 iterations). There is no stopping rule; CEM stops at about a fifth of the initial spread.
- The first actions are pinned down (std 0.08-0.10), the late ones about twice as loose (0.20-0.21): the cost depends most on early actions. Open loop executes all 25 actions including the loose tail; replanning every 5 steps executes only the tight first 5.
- The executed mean is a typical elite, not a blend of distinct plans: it sits about one elite std from the best elite (0.17-0.19), and the elites form one cloud (21% of their variance on the top direction; two routes would put most of it there).
- With safety-first, all 30 elites are predicted safe in 97% of solves, so the mean averages predicted-safe plans. Consistent with the executed-plan trace (0-4.5% of episodes crash on a plan predicted unsafe), returning the elite mean is not what causes collisions; false-safe $h$ is.
- Cheap changes if needed: more iterations, or return the best elite (or check the mean and fall back to the best elite if it is predicted unsafe). Not tested; expected to change little.

# Dubins: CCEM on dangerous goals (2026-11-06 / 07)

A new planning protocol, motivated by two problems with the one above: most sg25clean episodes do not test avoidance (at short goal distances only about half the recorded paths move toward an obstacle), and the safety-first cost is our own construction. Full grids are in `paper-results.tex` (generated by `le-wm/scripts/make_paper_results.py`); methods there.

- Episodes: start and goal from one recorded clean-success episode, goal $K$ steps later, kept only if the straight start-goal line crosses an obstacle (`eval.py` `goal_crosses_obstacle`), so heading straight at the goal collides and the recorded path is a collision-free detour. $K \in \{10, 15, 20, 25\}$ (5 has only 30 such pairs), plan horizon $K$ executed open loop, budget $2K$, 50 episodes; the same 50 episodes for every model and seed at a given $K$ (fixed eval seed), which makes paired tests possible.
- Planners: raw CEM = LeWM's CEM and final-step goal cost, unchanged. CCEM = the same CEM with a constrained elite rule: a plan is feasible if $h > 0$ on every state (current one included); if at least 30 of 300 are feasible the elites are the 30 feasible plans with the best goal cost, otherwise all feasible plans plus the infeasible ones with the highest worst-case margin $\min_k h(\hat z_k)$ (`safe_solver.py`, `ccem_fallback=minh`; SafeDreamer's rule instead ranks infeasible plans by the count of unsafe states). $h$ is the red-trained margin function (MLP or linear), trained on uniform states only, never on predicted latents.

Table 22. Goal-distance sweep, seed 3072: collision (%) as raw / CCEM-MLP / CCEM-linear. Success and red are in `paper-results.tex`.

| Model | 15 steps, purple | 15 steps, rotated 90° | 20 steps, purple | 20 steps, rotated 90° |
| --- | --- | --- | --- | --- |
| sigreg | 56 / 60 / 58 | 50 / 52 / 50 | 62 / 62 / 60 | 58 / 58 / 58 |
| adv_h | 50 / 48 / 46 | 54 / 56 / 56 | 38 / 40 / 38 | 40 / 38 / 38 |
| adv_λ | 42 / 38 / 50 | 66 / 66 / 70 | 50 / 50 / 50 | 62 / 64 / 58 |
| adv_λ, encoder-only | 54 / 50 / 54 | 74 / 76 / 76 | 56 / 56 / 58 | 44 / 44 / 46 |
| adv_h, encoder-only | 42 / 38 / 46 | 62 / 54 / 54 | 58 / 56 / 52 | 52 / 50 / 46 |
| jac | 74 / 32 / 32 | 74 / 30 / 32 | 62 / 12 / 14 | 58 / 10 / 18 |
| jac+adv_h | 82 / 62 / 50 | 78 / 62 / 62 | 70 / 50 / 34 | 56 / 52 / 50 |
| jac+adv_h, linear head | 80 / 68 / 56 | 68 / 58 / 60 | 54 / 34 / 40 | 54 / 34 / 36 |
| jac+adv_λ | 72 / 56 / 46 | 68 / 60 / 48 | 44 / 32 / 28 | 40 / 34 / 30 |
| jac+adv_λ, encoder-only | 70 / 50 / 48 | 62 / 36 / 30 | 42 / 24 / 26 | 56 / 20 / 22 |
| jac+adv_h, encoder-only | 82 / 32 / 28 | 62 / 28 / 24 | 70 / 24 / 22 | 52 / 28 / 26 |
| jac+pull | 56 / 36 / 28 | 28 / 28 / 22 | 50 / 22 / 24 | 48 / 22 / 30 |
| jac+pull+adv_h | 72 / 50 / 54 | 60 / 48 / 48 | 58 / 32 / 44 | 52 / 36 / 36 |
| jac+pull+adv_λ | 60 / 30 / 42 | 34 / 28 / 24 | 50 / 28 / 32 | 26 / 16 / 22 |
| jac+pull+adv_λ, encoder-only | 52 / 40 / 46 | 60 / 46 / 48 | 46 / 28 / 26 | 52 / 30 / 32 |
| jac+pull+adv_h, encoder-only | 76 / 52 / 42 | 50 / 42 / 40 | 38 / 24 / 22 | 38 / 32 / 28 |
| jac+pull+adv_h, linear head, encoder-only | 82 / 60 / 68 | 54 / 44 / 40 | 64 / 48 / 52 | 48 / 28 / 36 |

- Difficulty: at 10 steps raw CEM mostly succeeds (red 80-96% success); by 15-20 steps purple and rotation collapse (raw success often 10-50%, collision 40-80%).
- CCEM cuts collision for every jac-base model at every goal distance under purple and rotation (often by 20-50 points), at a small success cost, with the MLP or the linear margin function alike; it never helps the sigreg base.

Table 23. Paired tests vs jac, CCEM with the MLP margin function, per-episode collision on the same 50 episodes (exact McNemar, two-sided). Seed 3072.

| Goal, condition | jac collisions / 50 | significantly worse (p < 0.05) | tie | significantly better | ties with jac |
| --- | ---: | ---: | ---: | ---: | --- |
| 15 steps, purple | 16 | 7 | 9 | 0 | adv_h; adv_λ; adv_λ, encoder-only; adv_h, encoder-only; jac+adv_h, encoder-only; jac+pull; jac+pull+adv_h; jac+pull+adv_λ; jac+pull+adv_λ, encoder-only |
| 15 steps, rotated 90° | 15 | 8 | 8 | 0 | jac+adv_λ, encoder-only; jac+adv_h, encoder-only; jac+pull; jac+pull+adv_h; jac+pull+adv_λ; jac+pull+adv_λ, encoder-only; jac+pull+adv_h, encoder-only; jac+pull+adv_h, linear head, encoder-only |
| 20 steps, purple | 6 | 10 | 6 | 0 | jac+adv_λ, encoder-only; jac+adv_h, encoder-only; jac+pull; jac+pull+adv_λ; jac+pull+adv_λ, encoder-only; jac+pull+adv_h, encoder-only |
| 20 steps, rotated 90° | 5 | 13 | 3 | 0 | jac+adv_λ, encoder-only; jac+pull; jac+pull+adv_λ |
| 25 steps, purple | 13 | 6 | 10 | 0 | jac+adv_h; jac+adv_h, linear head; jac+adv_λ; jac+adv_λ, encoder-only; jac+adv_h, encoder-only; jac+pull; jac+pull+adv_h; jac+pull+adv_λ; jac+pull+adv_λ, encoder-only; jac+pull+adv_h, encoder-only |
| 25 steps, rotated 90° | 12 | 5 | 11 | 0 | jac+adv_h; jac+adv_h, linear head; jac+adv_λ; jac+adv_λ, encoder-only; jac+adv_h, encoder-only; jac+pull; jac+pull+adv_h; jac+pull+adv_λ; jac+pull+adv_λ, encoder-only; jac+pull+adv_h, encoder-only; jac+pull+adv_h, linear head, encoder-only |

- No model is significantly better than jac in any comparison (also with the linear margin function, checked at 25 steps). The models that tie it most consistently are jac+pull, jac+adv_h encoder-only, jac+adv_λ encoder-only and jac+pull+adv_λ (MLP margin function; with the linear one it is significantly worse under rotation at 25 steps); the full safety regs on the jac base are significantly worse in most comparisons. 96 tests in this table: a few p < 0.05 are expected by chance; the strong ones (p ≤ 0.003) survive a Holm correction.
- Seeds: these are seed 3072. The same sweep on the extra trained seeds is running (`goal_dist_sweep_seeds.sh`); seed counts per row are tracked in `paper-results.tex`.

Table 24. Threshold-free elite rules on jac+pull+adv_h (purple AUC with a linear $h$ 0.986 ± 0.015 over 3 seeds, sign accuracy 0.57), linear margin function. Safe → task: the 50 safest plans by $\min_k h$, then the 30 with the best goal cost; task → safe: the 50 best by goal cost, then the 30 safest; only the ranking of $h$ is used. Success / collision (%), pooled over the seeds with all four planners done (n = episodes).

| Goal, condition | n | raw | CCEM | safe → task | task → safe | safe → task vs CCEM (more / fewer collisions, p) |
| --- | ---: | --- | --- | --- | --- | --- |
| 10 steps, red | 150 | 93 / 11 | 89 / 11 | 77 / 24 | 83 / 12 | 25 / 5, p = 0.000 |
| 10 steps, purple | 150 | 73 / 43 | 73 / 31 | 61 / 29 | 65 / 41 | 17 / 21, p = 0.627 |
| 10 steps, rotated 90° | 150 | 66 / 42 | 69 / 33 | 61 / 29 | 66 / 31 | 14 / 20, p = 0.392 |
| 15 steps, red | 150 | 79 / 27 | 73 / 31 | 44 / 45 | 76 / 25 | 38 / 16, p = 0.004 |
| 15 steps, purple | 150 | 32 / 68 | 30 / 55 | 27 / 45 | 39 / 60 | 21 / 36, p = 0.063 |
| 15 steps, rotated 90° | 150 | 31 / 55 | 32 / 48 | 32 / 49 | 34 / 50 | 24 / 23, p = 1.000 |
| 20 steps, red | 50 | 82 / 20 | 84 / 16 | 40 / 34 | 76 / 20 | 14 / 5, p = 0.064 |
| 20 steps, purple | 50 | 18 / 58 | 34 / 44 | 22 / 36 | 16 / 44 | 6 / 10, p = 0.454 |
| 20 steps, rotated 90° | 50 | 26 / 52 | 30 / 36 | 22 / 42 | 28 / 42 | 6 / 3, p = 0.508 |
| 25 steps, red | 50 | 82 / 36 | 82 / 26 | 32 / 38 | 68 / 28 | 14 / 8, p = 0.286 |
| 25 steps, purple | 50 | 16 / 66 | 26 / 46 | 28 / 34 | 32 / 48 | 8 / 14, p = 0.286 |
| 25 steps, rotated 90° | 50 | 28 / 60 | 10 / 38 | 22 / 38 | 30 / 48 | 8 / 8, p = 1.000 |

- Safe → task trades red performance for purple: on red it is significantly worse than CCEM (more collisions and much lower success, e.g. 44% vs 73% success at 15 steps), under purple it collides less but not significantly (45 vs 55% at 15 steps, p = 0.06). Task → safe is about a tie with CCEM. Ranking without a threshold helps only where the threshold is broken; CCEM's rule (threshold first, ranking as a fallback) is the better default.

Where $h$ fails, ordering vs threshold (summary of the margin-function tables, Table 10 and Table 14 here, and `paper-results.tex`):

| | Ordering (AUC) | Threshold (sign) |
| --- | --- | --- |
| Real frames, red | right (0.99+) | right (balanced sign accuracy about 0.97) |
| Real frames, purple | right for jac-base models with a linear $h$ (about 0.95-0.99); wrong for sigreg and adv_λ encoder-only | wrong for every model (0.50) |
| Real frames, rotated 90° | right for every jac-base model (0.98+) | mostly right (0.85-0.95) |
| Imagined, red | right early, decays with depth (about 0.98 at step 10, 0.77-0.91 at step 25) | optimistic, more so with depth (false-safe 7% at real frames, 35-50% at step 25) |
| Imagined, purple | gone within about 5-10 steps for every model | wrong |

- The threshold fails along both axes, OOD and prediction depth; the ordering survives everything except imagined OOD states at depth. Rank-based rules can only help where the ordering survives (imagined states early in the plan; rotation). Not yet run: per-step normalization of $h$ across candidates (removes a depth-dependent offset shared by all candidates) and CCEM with $h$ trained on predicted latents.

# Safety Gym

![Safety Gym Point base vs OOD frames](figs/ood_frames_sg_point.png)

Goal: a second benchmark where safety is coupled to dynamics (unlike a "teleportable" agent whose safe set is just the
obstacle geometry), to re-test the appearance-shift findings from Dubins. Code and logs: branch `point-dynamics-probe`
(`safety_gym_port/`, `le-wm/scripts/sg_point_*.py`, `run_logs/sg_point/`); full tables in `sg_point_ovh_results.md`.

## Point instead of Car: safety that depends on velocity

The earlier safety-gym port used SafetyCarGoal1. The Car brakes from top speed in 0.055 (a quarter of the hazard radius
0.2) and reaches top speed in 5 steps, so its safe set is essentially the hazard geometry. The Point robot (thrust along
heading, world-frame slide joints) needs 60 steps to reach top speed, brakes in 0.238 and coasts 0.78, so turning does not
redirect its momentum.

Test (`safety_gym_port/point_doomed.py`): inject Point states near hazards, call a currently-safe state "doomed" if none of
13 escape policies avoids every hazard for 80 steps (an upper bound on true unsafety). Fraction doomed:

| distance past hazard edge | moving away | approach 0-0.01/step | 0.01-0.02 | 0.02-0.03 (top speed) |
|---|---|---|---|---|
| 0-0.05 | 0.04 | 0.20 | 0.88 | 1.00 |
| 0.05-0.10 | 0.01 | 0.01 | 0.28 | 0.87 |
| 0.10-0.15 | 0.00 | 0.00 | 0.01 | 0.57 |
| 0.15-0.20 | 0.00 | 0.00 | 0.00 | 0.28 |
| 0.20-0.30 | 0.00 | 0.00 | 0.00 | 0.04 |

At top approach speed the doomed band reaches about 0.24 past the hazard edge (the braking distance); moving away, almost
nothing is doomed. Heading barely matters, so the coupling is translational momentum.

## Problem 1: every Car dataset leaked the label

Safety-gym's `render()` draws a translucent red sphere at the agent whenever the previous step had cost > 0, which is exactly
our failure label. A red-pixel count alone predicts unsafe at AUC 0.998-1.000 on the Car training set and on every Car eval
set (color, camera rotation, shape, multicolor). All renders since then are "clean": the sphere and the lidar ring are off
(`--clean`; red-pixel AUC 0.08-0.55 on every clean set, i.e. no marker).

How much the Car models leaned on it (margin function trained with the sphere, tested on the same frames without it): the gp margin
function barely moved (zero-shot drop at most 0.017), but the jac/jac+pull latents encode the sphere along the safe/unsafe
direction (cosine ~0.8) and an unregularized margin function dropped 1.000 -> 0.84. The earlier Car color/rotation safety-reg gains
should be treated as suspect.

## Problem 2: the old protocol measured layout memorization

Hazard layouts are random every episode. The old eval (one 5000-pose collection, shuffled 80/20 frame split) lets the margin function's
train and test frames share layouts. Margin functions on clean fixedfar@128 Point models scored 0.99 in-distribution, but 0.47-0.52
on base-view frames from fresh layouts, i.e. chance. The camera-rotation axis failed for the same reason: a 90 degree turn
of this camera looks like the normal view of a different layout.

The benchmark itself was learnable. A supervised CNN on raw fixedfar@128 pixels reaches AUC 0.998-0.999 on unseen layouts
(and 0.958-0.999 on every OOD view). The bottleneck was the world model's pooled latent: a margin function on the same encoder's patch
tokens reached 0.79 where the pooled latent gave 0.35.

Why Dubins never showed this: it has one fixed obstacle layout, so the agent's pose alone determines safety. Here the hazards
change every episode, never move within one and do not affect the Point's motion, so the world-model objective has little
reason to encode them.

## Fix: overhead camera at 224 px plus a layout-disjoint eval

- Camera: safety-gym's fixedfar camera moved straight above the arena (0, 0, 4), framing [-1.8, 1.8]. The agent goes from
  about 6 px to about 143 px, and the encoder's 224 px input no longer upsamples a 128 px render. A 90 degree camera turn is
  now an exact image rotation.
- Training data: `sg_point_ovh224.h5`, 170,400 clean frames, about 850 random layouts, same trajectories and labels as the
  fixedfar set.
- How it was collected (`collect_overhead_train.sh`, branch `point-dynamics-probe`): 852 scripted 200-step episodes, one random layout each (8 hazards of radius 0.2), 4 shards with seeds 20-23. Per episode, with probability 0.6 the robot dashes at the nearest hazard and brakes at a random distance 0-0.4 from its edge, otherwise it drives at the goal at full thrust without braking and overshoots; both with action noise. Episodes are cut into 8-frame chunks and shuffled, so the file holds 21,300 "episodes" of 8 frames and no training sample is longer than 8 steps. 19.3% failure frames (no chunk was dropped: the natural rate already exceeded the 17% target), mostly the robot sitting inside a hazard (mean speed 0.20 when unsafe vs 0.54 when safe). Not expert data. Write-up: `point_dataset.tex` (branch `worktree-dubins-dataset-doc`).
- Eval: margin function (margin_gp) trained on 50k frames of the training set, scored on 5000 poses of a separate collection (every
  layout unseen). Views: base, purple hazards (color), 90 degree camera rotation (rotate), box-shaped hazards (shape), Car
  robot instead of Point (agent).

Pilot, same margin function on unseen layouts (AUC, base view): sigreg 0.362 (fixedfar) -> 0.581 (overhead); jac+pull 0.563 -> 0.836.

## Results: 9 world models on overhead@224, layout-disjoint

le-wm, 50 epochs, seed 3072. Safety-reg recipes copied from the earlier safety-gym reference runs. ± = std over 3 margin-function seeds
(training subsample and init) on one WM per model (training seed 3072). "train fit" = the margin function's AUC on its own training frames.

Frames are fed in the training orientation (fixed 2026-10-28). The earlier tables came from `sg_point_bigprobe.py`, which preprocessed the h5 and npz frames as numpy HWC (directly through `ToImage`, or with the equivalent `permute(0, 3, 2, 1)`), so every frame was transposed and "rotate" was a mirror flip, not the 90° camera turn. A rerun of that path reproduces the old tables exactly (AUC within 0.0005). Fixed outputs: `/data/seongbin/lewm/sg_point_balanced/fixed/` (`tables.md`, per-sample margins in `<model>_seed<s>.npz`). The old "train fit" column was also on transposed frames and is dropped.

AUC (base = unseen layouts)

| Model | base | color | rotate | shape | agent |
|---|---|---|---|---|---|
| sigreg | 0.560 ± 0.013 | 0.501 ± 0.005 | 0.526 ± 0.014 | 0.406 ± 0.016 | 0.559 ± 0.013 |
| jac | 0.953 ± 0.007 | 0.733 ± 0.032 | 0.948 ± 0.003 | 0.957 ± 0.008 | 0.863 ± 0.002 |
| jac+pull | 0.804 ± 0.015 | 0.644 ± 0.029 | 0.813 ± 0.011 | 0.845 ± 0.014 | 0.618 ± 0.016 |
| adv_h | 0.960 ± 0.002 | 0.609 ± 0.003 | 0.940 ± 0.011 | 0.899 ± 0.003 | 0.937 ± 0.006 |
| jac+adv_h | 0.980 ± 0.001 | 0.920 ± 0.006 | 0.976 ± 0.002 | 0.979 ± 0.001 | 0.891 ± 0.003 |
| jac+pull+adv_h | 0.914 ± 0.009 | 0.776 ± 0.038 | 0.889 ± 0.014 | 0.935 ± 0.006 | 0.782 ± 0.028 |
| adv_λ | 0.919 ± 0.009 | 0.691 ± 0.028 | 0.937 ± 0.008 | 0.927 ± 0.008 | 0.847 ± 0.022 |
| jac+adv_λ | 0.971 ± 0.000 | 0.884 ± 0.008 | 0.966 ± 0.004 | 0.971 ± 0.001 | 0.877 ± 0.002 |
| jac+pull+adv_λ | 0.968 ± 0.000 | 0.858 ± 0.009 | 0.965 ± 0.004 | 0.969 ± 0.001 | 0.897 ± 0.005 |

Sign accuracy, balanced: (TPR_safe + TPR_unsafe) / 2 on the full test set, i.e. the expected accuracy on a 50/50 safe/unsafe split (0.500 = everything called one class). The margin function is trained on the natural split.

| Model | base | color | rotate | shape | agent |
|---|---|---|---|---|---|
| sigreg | 0.475 ± 0.004 | 0.492 ± 0.007 | 0.483 ± 0.004 | 0.481 ± 0.005 | 0.473 ± 0.006 |
| jac | 0.782 ± 0.006 | 0.597 ± 0.032 | 0.704 ± 0.002 | 0.757 ± 0.014 | 0.764 ± 0.006 |
| jac+pull | 0.550 ± 0.002 | 0.544 ± 0.102 | 0.574 ± 0.006 | 0.644 ± 0.004 | 0.546 ± 0.004 |
| adv_h | 0.908 ± 0.007 | 0.500 ± 0.000 | 0.836 ± 0.014 | 0.826 ± 0.001 | 0.849 ± 0.001 |
| jac+adv_h | 0.911 ± 0.003 | 0.639 ± 0.028 | 0.839 ± 0.042 | 0.953 ± 0.003 | 0.736 ± 0.003 |
| jac+pull+adv_h | 0.574 ± 0.027 | 0.639 ± 0.080 | 0.688 ± 0.016 | 0.695 ± 0.032 | 0.575 ± 0.041 |
| adv_λ | 0.779 ± 0.002 | 0.674 ± 0.026 | 0.794 ± 0.010 | 0.783 ± 0.001 | 0.654 ± 0.001 |
| jac+adv_λ | 0.946 ± 0.004 | 0.765 ± 0.016 | 0.858 ± 0.040 | 0.948 ± 0.002 | 0.809 ± 0.017 |
| jac+pull+adv_λ | 0.931 ± 0.012 | 0.758 ± 0.006 | 0.834 ± 0.031 | 0.946 ± 0.001 | 0.841 ± 0.009 |

Reading:

- sigreg is at chance on unseen layouts and every shift (AUC 0.41-0.56, sign accuracy below 0.5); its earlier 0.78-0.82 unbalanced sign accuracy came from calling almost everything safe.
- jac+adv_h has the best ranking on base, color, rotate and shape (0.920-0.980); on the agent shift adv_h (0.937), jac+pull+adv_λ (0.897) and jac+adv_λ (0.877) do better.
- jac+adv_λ has the best thresholds (sign accuracy 0.946 base, 0.765 color); jac+adv_λ and jac+pull+adv_λ are the only models close to calibrated on every view.
- Color is still the hardest shift; adv_h ranks it poorly (0.609) and calls every purple frame unsafe.
- The action-separation pull hurts wherever it is added (jac+pull 0.804 vs jac 0.953 base; jac+pull+adv_h below jac+adv_h on every view).

## Linear vs MLP margin function (2026-11-05)

Same eval as above (layout-disjoint, correctly oriented frames, 50k training frames, 20k steps, same $\ell_{gp}$ loss), with the margin function either the 2-layer MLP above or a single linear map $h(z) = w^\top z + b$ on the frozen CLS latent. For a linear head the gradient penalty pins $\|w\|$ to 0.1, so only the direction and offset are learned. Script: `le-wm/scripts/sg_point_bigprobe.py` (`HEAD=both`), launcher `run_point_lin_eval.sh`, logs `run_logs/sg_point/bigprobe_both_ovh_seed{0,1,2}.log` (branch `point-dynamics-probe`).

All finished world models: training seeds 3072 and 1 for every recipe, plus seed 2 for jac and adv_h (the other seven seed-2 runs were stopped mid-training on 2026-10-28). Each WM is averaged over 3 margin-function seeds; ± = std over WM training seeds (WM seeds column), so these MLP rows are not the single-WM numbers in the tables above. Check: the new MLP on the seed-3072 jac WM reproduces the table above exactly (eval seed 0: base 0.962, color 0.767, rotate 0.945).

AUC (base = unseen layouts)

| Model | head | WM seeds | base | color | rotate | shape | agent |
|---|---|---|---|---|---|---|---|
| sigreg | linear | 2 | 0.582 ± 0.000 | 0.561 ± 0.051 | 0.617 ± 0.082 | 0.665 ± 0.056 | 0.587 ± 0.005 |
| sigreg | MLP | 2 | 0.603 ± 0.054 | 0.479 ± 0.022 | 0.460 ± 0.053 | 0.514 ± 0.114 | 0.605 ± 0.056 |
| jac | linear | 3 | 0.951 ± 0.018 | 0.822 ± 0.044 | 0.948 ± 0.033 | 0.957 ± 0.017 | 0.847 ± 0.043 |
| jac | MLP | 3 | 0.922 ± 0.050 | 0.779 ± 0.039 | 0.918 ± 0.060 | 0.931 ± 0.043 | 0.832 ± 0.031 |
| jac+pull | linear | 2 | 0.911 ± 0.042 | 0.794 ± 0.019 | 0.917 ± 0.047 | 0.927 ± 0.037 | 0.788 ± 0.101 |
| jac+pull | MLP | 2 | 0.853 ± 0.050 | 0.689 ± 0.045 | 0.859 ± 0.046 | 0.880 ± 0.034 | 0.738 ± 0.120 |
| adv_h | linear | 3 | 0.954 ± 0.021 | 0.719 ± 0.068 | 0.971 ± 0.006 | 0.889 ± 0.051 | 0.918 ± 0.023 |
| adv_h | MLP | 3 | 0.960 ± 0.001 | 0.712 ± 0.073 | 0.963 ± 0.019 | 0.930 ± 0.025 | 0.922 ± 0.021 |
| jac+adv_h | linear | 2 | 0.977 ± 0.003 | 0.851 ± 0.017 | 0.972 ± 0.001 | 0.977 ± 0.002 | 0.892 ± 0.006 |
| jac+adv_h | MLP | 2 | 0.978 ± 0.002 | 0.890 ± 0.029 | 0.975 ± 0.001 | 0.978 ± 0.001 | 0.889 ± 0.002 |
| jac+pull+adv_h | linear | 2 | 0.971 ± 0.006 | 0.873 ± 0.006 | 0.957 ± 0.011 | 0.975 ± 0.003 | 0.874 ± 0.022 |
| jac+pull+adv_h | MLP | 2 | 0.943 ± 0.029 | 0.851 ± 0.075 | 0.926 ± 0.038 | 0.954 ± 0.019 | 0.842 ± 0.059 |
| adv_λ | linear | 2 | 0.939 ± 0.000 | 0.627 ± 0.010 | 0.950 ± 0.002 | 0.943 ± 0.001 | 0.853 ± 0.031 |
| adv_λ | MLP | 2 | 0.906 ± 0.013 | 0.536 ± 0.155 | 0.932 ± 0.005 | 0.930 ± 0.003 | 0.817 ± 0.030 |
| jac+adv_λ | linear | 2 | 0.968 ± 0.001 | 0.782 ± 0.028 | 0.957 ± 0.006 | 0.968 ± 0.002 | 0.905 ± 0.011 |
| jac+adv_λ | MLP | 2 | 0.967 ± 0.004 | 0.863 ± 0.021 | 0.961 ± 0.005 | 0.968 ± 0.003 | 0.881 ± 0.004 |
| jac+pull+adv_λ | linear | 2 | 0.970 ± 0.002 | 0.767 ± 0.003 | 0.961 ± 0.006 | 0.970 ± 0.001 | 0.909 ± 0.002 |
| jac+pull+adv_λ | MLP | 2 | 0.968 ± 0.000 | 0.876 ± 0.018 | 0.964 ± 0.000 | 0.971 ± 0.002 | 0.899 ± 0.002 |

Sign accuracy, balanced ((TPR_safe + TPR_unsafe) / 2; 0.500 = everything called one class), plus the fraction of purple (color) frames called safe (true safe fraction 0.830).

| Model | head | base | color | rotate | shape | agent | color pred. safe |
|---|---|---|---|---|---|---|---|
| sigreg | linear | 0.563 ± 0.008 | 0.544 ± 0.053 | 0.614 ± 0.064 | 0.570 ± 0.027 | 0.561 ± 0.033 | 0.386 ± 0.000 |
| sigreg | MLP | 0.486 ± 0.012 | 0.495 ± 0.004 | 0.491 ± 0.006 | 0.488 ± 0.004 | 0.484 ± 0.013 | 0.991 ± 0.007 |
| jac | linear | 0.829 ± 0.058 | 0.579 ± 0.063 | 0.839 ± 0.094 | 0.782 ± 0.037 | 0.763 ± 0.060 | 0.132 ± 0.106 |
| jac | MLP | 0.725 ± 0.137 | 0.547 ± 0.039 | 0.706 ± 0.100 | 0.732 ± 0.078 | 0.700 ± 0.092 | 0.714 ± 0.345 |
| jac+pull | linear | 0.790 ± 0.062 | 0.524 ± 0.007 | 0.803 ± 0.087 | 0.745 ± 0.007 | 0.721 ± 0.089 | 0.041 ± 0.011 |
| jac+pull | MLP | 0.603 ± 0.052 | 0.608 ± 0.065 | 0.627 ± 0.053 | 0.646 ± 0.001 | 0.597 ± 0.051 | 0.722 ± 0.085 |
| adv_h | linear | 0.907 ± 0.010 | 0.500 ± 0.000 | 0.899 ± 0.040 | 0.803 ± 0.095 | 0.846 ± 0.012 | 0.000 ± 0.000 |
| adv_h | MLP | 0.867 ± 0.029 | 0.500 ± 0.000 | 0.864 ± 0.053 | 0.837 ± 0.022 | 0.812 ± 0.038 | 0.000 ± 0.000 |
| jac+adv_h | linear | 0.911 ± 0.011 | 0.735 ± 0.078 | 0.916 ± 0.014 | 0.951 ± 0.004 | 0.736 ± 0.001 | 0.778 ± 0.119 |
| jac+adv_h | MLP | 0.908 ± 0.003 | 0.612 ± 0.027 | 0.835 ± 0.004 | 0.953 ± 0.000 | 0.726 ± 0.010 | 0.932 ± 0.033 |
| jac+pull+adv_h | linear | 0.896 ± 0.011 | 0.702 ± 0.082 | 0.887 ± 0.011 | 0.940 ± 0.013 | 0.741 ± 0.011 | 0.455 ± 0.257 |
| jac+pull+adv_h | MLP | 0.694 ± 0.121 | 0.627 ± 0.011 | 0.758 ± 0.069 | 0.795 ± 0.100 | 0.636 ± 0.060 | 0.898 ± 0.046 |
| adv_λ | linear | 0.921 ± 0.003 | 0.508 ± 0.008 | 0.901 ± 0.026 | 0.921 ± 0.007 | 0.790 ± 0.017 | 0.016 ± 0.016 |
| adv_λ | MLP | 0.787 ± 0.008 | 0.587 ± 0.087 | 0.828 ± 0.034 | 0.839 ± 0.055 | 0.667 ± 0.013 | 0.376 ± 0.376 |
| jac+adv_λ | linear | 0.941 ± 0.001 | 0.565 ± 0.038 | 0.870 ± 0.026 | 0.936 ± 0.001 | 0.833 ± 0.010 | 0.109 ± 0.063 |
| jac+adv_λ | MLP | 0.924 ± 0.022 | 0.754 ± 0.012 | 0.838 ± 0.021 | 0.942 ± 0.005 | 0.799 ± 0.010 | 0.518 ± 0.071 |
| jac+pull+adv_λ | linear | 0.944 ± 0.000 | 0.577 ± 0.026 | 0.887 ± 0.011 | 0.940 ± 0.001 | 0.828 ± 0.009 | 0.128 ± 0.043 |
| jac+pull+adv_λ | MLP | 0.919 ± 0.012 | 0.774 ± 0.016 | 0.850 ± 0.015 | 0.948 ± 0.003 | 0.836 ± 0.005 | 0.633 ± 0.009 |

Train fit (AUC on the margin function's own training frames): linear 0.955-0.994 for every model except sigreg (0.735 ± 0.004); MLP 0.989-0.999 for all.

Reading:

- Safety is close to linearly readable from every regularized latent: on base, rotate, shape and agent the linear margin function is within 0.01 of the MLP's AUC or better (base 0.911-0.977), except adv_h shape (0.889 vs 0.930); its balanced sign accuracy on base and rotate is higher than the MLP's for every model.
- Linear helps most where the MLP was weakest (base sign accuracy jac 0.829 vs 0.725, jac+pull 0.790 vs 0.603, jac+pull+adv_h 0.896 vs 0.694, adv_λ 0.921 vs 0.787): the MLP fits the training layouts (0.99+) with features that do not carry to new layouts.
- sigreg's safety is not linear in its latent: the linear head cannot even fit the training layouts (0.735), and both heads are near chance on unseen ones.
- Color is the exception. The purple shift moves the latent along the linear safety direction, so a linear threshold calls most purple frames unsafe (color pred. safe 0.016-0.132 for jac, jac+pull, adv_λ, jac+adv_λ, jac+pull+adv_λ; 0.000 for adv_h with either head). With the adv_λ recipes the MLP keeps more of the color ranking (jac+adv_λ 0.863 vs 0.782, jac+pull+adv_λ 0.876 vs 0.767).
- jac+adv_h is the one model whose linear color threshold holds (pred. safe 0.778, balanced sign accuracy 0.735, better than its MLP's 0.612); jac+pull+adv_h has the best linear color AUC (0.873).

## Caveats and next steps

- Training seeds: 3072 and 1 are done for all 9 recipes; seed 2 is done only for jac and adv_h. The other seven seed-2 runs were stopped mid-training on 2026-10-28 and need a restart from scratch (`le-wm/scripts/run_point_ovh_seeds.sh`). The single-WM tables above use seed 3072 only; the linear-vs-MLP tables average over the finished seeds.
- Labels are position-only failure (agent inside a hazard). The velocity-dependent value (the doomed band above) is not yet
  used as a target; `point_doomed.py` can generate Monte Carlo value labels.
- Possible variant: a fixed-layout Point benchmark (Dubins-like), which would also make an exact HJ value oracle feasible.
- No Point Dreamer has been trained.
