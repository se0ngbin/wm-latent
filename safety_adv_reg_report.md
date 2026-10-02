# Safety-projected adversarial invariance (label-only, no OOD data)

Status (2026-10-02 18:40 UTC): **v1 (run A) finished and FAILED on color and shape through a BatchNorm cheat (§5). Fixed in v2; v2 run A is training, v2 run B is waiting for GPU memory.**
Branch `safety-adv-reg`, code in `le-wm/module.py` (`SafetyAdvInvarianceReg`, config key `safety_adv`). v1 = `ca9ec21`, v2 fix = `c6e93c6`.

## 1. Motivation

Earlier OOD results suggest which direction an appearance shift moves the latent matters more than how far:

- Color shift relocates the latent instead of destroying safe-set information. Retraining on the new appearance recovers the margin to .90–.99 on every encoder. Zero-shot failure is a calibration problem.
- Removing just 1–2 nuisance directions recovers color OOD AUC (baseline .43 → .67) at about 0 in-dist cost.
- The color shift is smaller than the safe/unsafe gap (robust_ratio < 1), yet 3 of 4 encoders invert, because the shift is aligned with the margin axis.
- Global invariance (color+rotation aug) bought robustness by collapsing σ_red 18.7 → 0.27, which caused the color↔rotation see-saw.

So the target is minimal invariance: the safety readout should not move under appearance change, and every other latent direction is left free.

Two constraints from the design discussion:

1. **No OOD data.** You can't learn invariance to a factor that never varies in training. The information has to come from a prior: perturbations we generate, not renders we collect.
2. **Only binary failure labels.** The safety direction has to be learned from them.

## 2. Method

**Safety readout.** A margin head h (margin_gp: zs 0.1, hinge@0 1.0, WGAN-GP 10 @ ‖∇‖=0.1; same 512×2 architecture as latent_cbf `MarginHead`) trains during WM training on detached embeddings plus the dataset's `failures` labels. It has its own AdamW (lr 3e-4), because the main optimizer only owns `model`. With the higher LR it tracks the moving latent faster than the encoder changes.

**Perturbation family g (generated, not data).** A spatially varying hue rotation about the gray axis plus a saturation scale:
- Parameterized on a 4×4 grid, bilinear-upsampled, one field per sequence and shared across context frames.
- Box: |hue| ≤ 90°, saturation ∈ [0.5, 2].
- Leaves the white background exactly unchanged and preserves mean luminance.
- Cannot erase an obstacle (chroma is only scaled).
- Verified: red → (0.5, 0, 0.5) purple at hue −60°, saturation ×0.5.

**Adversary.** 2-step sign-PGD from a random start, maximizing the penalty itself, keeping each sample's best iterate.

**Penalty**, with h's parameters frozen (`functional_call`) and clean targets detached:

```
L = mean_ctx ((h(f(g x)) − h(f(x))) / gap)²  +  ((h(pred(f(g x))) − h(pred(f(x)))) / gap)²
```

`gap` is the live in-batch safe-minus-unsafe head gap, with gradient (see §3, bug 2). The predictor term covers the 1-step rollout, because the reachability critic bootstraps through the dynamics. BatchNorm: v1 put clean and perturbed inputs in one joint train-mode batch, which the encoder exploited (§5). v2 normalizes every perturbed pass with the clean batch's statistics and never updates running stats.

**Schedule.** The head trains from step 0. The penalty is off until step ≥ 2000 and the head's bias-corrected EMA AUC ≥ .95. It then ramps linearly 0 → 1 over 10k steps and stays on (latched). PGD is skipped entirely while the gate is closed. The config `weight` is λ_max = 1.0.

**Why the circularity is not a deadlock.** The head only sees in-distribution frames and labels, and the training data never varies the nuisance, so nothing pulls w toward it. The penalty only moves the encoder's response to g. Stop-grads block both shortcuts: the head can't rotate its readout away from the perturbation, and the encoder can't shrink the gap without the live-gap normalization noticing.

## 3. Smoke-test findings (GPU probe, 200 steps, sigreg base)

The probe caught three bugs, all fixed before launch:

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | Gate would open hundreds of steps late | AUC/gap EMAs started at 0/1 with no bias correction | Adam-style debiasing |
| 2 | Once on, the penalty **eroded** the safety separation: AUC .96 → .88, gap .31 → .16 | Normalizing by a detached, lagging EMA gap made flattening h everywhere the cheapest descent direction | Normalize by the live in-batch gap with gradient, so shrinking separation raises the penalty immediately |
| 3 | `adv_gain` < 1 (0.30–0.56): PGD ended worse than its random start | 2 sign steps of 0.5·ε overshoot | Keep each sample's best iterate |

After the fixes (200 steps, gate forced open at step 30):
- Head AUC .95 → .995.
- Gap grows .35 → 1.06.
- `adv_gain` ≥ 1.03 throughout (1.03–3.45).
- Penalty .29 → .02.

**Cost:** peak memory 13.5 → 18.4 GB at `n_sub=32` (+5 GB), step time about 2×. The jac+pull base alone already peaks at 24.8 GB.

## 4. Runs

All runs: dubins, 50 epochs, seed 3072, same recipe as the matched baseline, `data=dubins_safety`.

| Run | Regularizers | Matched baseline | Status (2026-10-02 18:40 UTC) |
|---|---|---|---|
| v1 A `lewm_dubins_safeadv50` | sigreg + safety_adv (v1) | `sigreg_only_dubins` | **done** (50 epochs); OOD results in §5 |
| v1 B `lewm_dubins_jacpull_safeadv50` | jac+pull + safety_adv (v1) | `dubins_jacpull50` | stopped at about 1h: same bug as v1 A (first attempt OOM'd at launch) |
| v2 A `lewm_dubins_safeadv2_50` | sigreg + safety_adv (v2, BN fix) | `sigreg_only_dubins` | training on GPU0, started 18:27, about 6–7h |
| v2 B `lewm_dubins_jacpull_safeadv2_50` | jac+pull + safety_adv (v2) | `dubins_jacpull50` | waiting for ≥ 34 GB free on a GPU (`/data/seongbin/lewm/code_safeadv_c6e93c6/wait_launch_B.sh`, retries on OOM) |

Code snapshots: v1 `/data/seongbin/lewm/code_safeadv_ca9ec21/`, v2 `/data/seongbin/lewm/code_safeadv_c6e93c6/` (`run.sh A|B`, logs in `.run_A/`, `.run_B/`).

### v1 run A training curve (W&B, mean over ±1000 steps)

| step | ramp | head AUC | gap | penalty | adv_gain | pred_loss |
|---|---|---|---|---|---|---|
| 2000 | .05 | .990 | 1.10 | .0021 | 1.77 | .035 |
| 6000 | .40 | .999 | 1.42 | .0038 | 1.89 | .021 |
| 12000 | .98 | .999 | 1.65 | .0030 | 1.59 | .015 |
| 30000 | 1 | .999 | 1.80 | .0021 | 1.52 | .010 |
| 55000 | 1 | .9995 | 1.96 | .0029 | 1.42 | .0067 |

At the time this looked healthy: separation kept improving, the gap leveled off after 20k steps, and the penalty stayed near .003. §5 shows that the small penalty was the cheat, not invariance.

## 5. v1 results: negative, caused by a BatchNorm cheat

### Zero-shot margin_gp OOD (seed 0, `ood_margin_gp_jepa.py`, acc / AUC)

| encoder | in-dist | color (red → purple) | shape (circle → diamond) | rotate (90°) |
|---|---|---|---|---|
| baseline (sigreg) | .945 / .980 | .164 / .136 | .316 / .885 | .837 / .973 |
| jacobian | .971 / .992 | .164 / .757 | .971 / .993 | .966 / .991 |
| jac+pull | .960 / .984 | .837 / .707 | .963 / .977 | .941 / .980 |
| **v1 A: sigreg + safety_adv** | **.975 / .997** | **.164 / .401** | **.108 / .085** | **.981 / .996** |

(Acc .164 is the base rate: everything called safe. Retraining a head on the new appearance recovers v1 A fully: color .940/.982, shape .950/.994, rotate .979/.997.)

- **In-dist:** v1 A is the sharpest of the four.
- **Color:** v1 A collapses like baseline (only the ranking is less inverted, AUC .40 vs .14) and is far from jac+pull.
- **Shape:** v1 A is *worse* than baseline; even the ranking inverts (AUC .085).
- **Rotation:** v1 A is the best of all four (.981 / .996), even though rotation was outside the perturbation family.

### Diagnosis

**1. Purple was inside the training family.** The eval's purple render is exactly the global in-family transform (hue −60°, sat ×0.5) of the red render, and antialiased edges map the same way. In latent space the in-family transform lands within 1.7% of the real purple latents. So the color failure isn't out-of-family generalization.

**2. Even the training-time head fails.** Scoring the regularizer's own head (from the Lightning checkpoint) on purple: readout shift +2.8 class gaps, AUC .092. The encoder became *more* color-sensitive than baseline: the mean purple latent shift is 4.8× the safe/unsafe centroid distance, vs 2.4× for baseline and 1.3× for jac+pull.

**3. It isn't a weak adversary.** In BN train mode (batch stats), 2-step PGD finds perturbations moving the readout by .70 class gaps, and 20-step PGD finds .99. Training had logged about .05.

**4. The cause: BatchNorm batch coupling.** The projector and predictor heads use BatchNorm. v1 passed clean and perturbed frames through one joint train-mode batch. Measured on v1 A:

| batch composition | clean head gap | \|Δh\|/gap, purple-equivalent | \|Δh\|/gap, random |
|---|---|---|---|
| separate batches | .973 | .458 | .529 |
| joint clean+perturbed (as trained) | **.076** | 1.371 (≈ .11 in clean-gap units) | 1.559 (≈ .12) |
| identity control | .973 | 0.000 | — |

The encoder learned to make perturbed frames "loud" in some features. That inflates the shared BN variance and squashes *both* halves' readouts together, so the penalty looks tiny without any invariance. The loud statistics also leaked into the running stats, through the 3–4 extra train-mode forwards per step:

| model | BN0 \|running mean − clean mean\| / std | running var ratio | training-head AUC, eval-BN vs train-BN |
|---|---|---|---|
| v1 A | 2.44 | ×6.7 | .853 vs .980 |
| baseline | 0.91 | ×2.3 | — |
| jac+pull | 0.47 | ×1.5 | — |

Every downstream use (the OOD eval, the planner) runs in eval mode with running stats, so the deployed representation was not the one trained. The rotation win may be a side effect of the same distortion, so treat it as unexplained until v2 reproduces it.

### Fix (v2, `c6e93c6`)

`_CleanStatBN`:
- One clean no-grad pass captures each BN layer's batch statistics (detached).
- All perturbed passes (PGD and the graded pass) normalize with those clean stats.
- Running stats are never updated by the regularizer.

Unit-tested:
- Capture output equals train-mode BN output.
- In apply mode, a sample's embedding is independent of the rest of the batch (the loud-sample cheat is impossible).
- Running stats are untouched.
- Encoder gets gradient; head params don't.

v2 GPU smoke (200 steps, gate forced open early):
- The penalty now stays around 1.0 (≈ one class gap of readout shift) instead of dropping to .02, so it is exerting real pressure.
- Head AUC .95 → .993.
- Cheaper than v1: 15.9 GB peak (v1 18.4) and about 0.3 s/step (v1 0.48), because the doubled joint batch is gone.

## 6. Evaluation plan (v2)

**Zero-shot margin OOD.** Add both runs to `ENCODERS` in `le-wm/scripts/ood_margin_jepa_static.py:30`. Run `{color, shape, rotate} × seeds {0,1,2}`, then aggregate with `ood_seed_aggregate.py`.

| encoder | in-dist acc/AUC | color zs acc/AUC | shape zs acc/AUC | rotate zs acc/AUC |
|---|---|---|---|---|
| baseline (sigreg) | ~.94–.99 | .17/.14 (collapses) | .31/.91 (threshold breaks, ranking holds) | robust (.83–.96 acc) |
| jacobian | ~.94–.99 | .17/.67 (ranking only) | robust | robust (.83–.96 acc) |
| jac+pull | ~.94–.99 | .83/.75 (survives) | robust | robust (.83–.96 acc) |
| **v2 A: sigreg + safety_adv** | pending | pending | pending | pending |
| **v2 B: jac+pull + safety_adv** | pending | pending | pending | pending |

(Baseline rows: the corrected margin_gp study of 2026-09-09, n=3 seeds, which supersedes the earlier V*-regression numbers. Read AUC as well as sign-acc, since acc is pinned at the 17% base rate when the threshold breaks.)

**What to look at:**
1. **Color zero-shot:** does A survive where the plain baseline inverts? Caveat: purple lies inside the perturbation family, so this tests whether the prior works, not true out-of-family generalization.
2. **Rotation and shape:** these are outside the family. Any gain there would come from the latent being better structured, not from coverage.
3. **Critic zero-shot (the real target):** retrain the reachability critic on each new encoder (`wm_ddpg.py`, then `critic_ood_eval.py`). Every encoder so far, jac+pull included, collapses here on color (.225 sign). The predictor term exists to address exactly this.
4. **No collapse:** σ_red (spread along the obstacle-color direction) stays near the baseline's, unlike the global-aug run (18.7 → 0.27); in-dist AUC is preserved.
5. **Planner:** in-dist CEM success and collision under the sg25clean protocol (`raw-cem-ood-protocol`), to confirm control isn't hurt.

**What would falsify it:** A no better than baseline on color zero-shot AUC, or a critic still at the .225 base rate. Either would mean projecting onto the instantaneous margin isn't enough, and the reachability direction needs its own signal (time-to-failure labels or a longer rollout term).

## 7. Known limitations

- Only a 1-step predictor term: dataset windows are `history_size + num_preds = 4` frames, so a longer rollout term needs longer windows.
- The perturbation family covers color only, not geometry.
- n = 1 seed per run so far.
- Validation loss excludes the penalty (PGD is skipped under inference mode). Head stats are still logged.
- LeWM's projector BatchNorm has a train/eval gap even without this regularizer (baseline \|z_eval − z_train\|/\|z\| = .99). Any regularizer that adds train-mode forwards needs the same clean-stat treatment.
- v1 OOD numbers are seed 0 only. Seeds 1–2 were still running at the time of writing; v1 is superseded anyway.
