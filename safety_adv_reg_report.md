# Safety-projected adversarial invariance (label-only, no OOD data)

Status (2026-10-02): **implementation done, training in progress, no OOD results yet.**
Branch `safety-adv-reg`, code in `le-wm/module.py` (`SafetyAdvInvarianceReg`, config key `safety_adv`).

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

`gap` is the live in-batch safe-minus-unsafe head gap, with gradient (see §3, bug 2). The predictor term covers the 1-step rollout, because the reachability critic bootstraps through the dynamics. Clean and perturbed inputs go through one joint forward so the projector's BatchNorm stats are shared.

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

All runs: dubins, 50 epochs, seed 3072, same recipe as the matched baseline, `data=dubins_safety`. Code snapshot `/data/seongbin/lewm/code_safeadv_ca9ec21/` (`run.sh A|B`).

| Run | Regularizers | Matched baseline | Status (2026-10-02 00:07 UTC) |
|---|---|---|---|
| A `lewm_dubins_safeadv50` | sigreg + safety_adv | `sigreg_only_dubins` | training, epoch 3/50, 2.4 it/s, about 6h left ([W&B](https://wandb.ai/seongbin/lewm/runs/dubins_safeadv50)) |
| B `lewm_dubins_jacpull_safeadv50` | sigreg + jacobian 1.0 + action_sep pull 0.1 + safety_adv | `dubins_jacpull50` | **not started**: `wait_launch_B.sh` is waiting for ≥ 34 GB free on a GPU |

### Run A training so far (W&B, mean over ±100 steps)

| step | ramp | head AUC | gap (EMA) | gap (live) | penalty | adv_gain | pred_loss | sigreg |
|---|---|---|---|---|---|---|---|---|
| 2000 | .008 | .985 | 1.04 | 1.09 | .0049 | 1.84 | .039 | 2.08 |
| 2500 | .053 | .990 | 1.10 | 1.16 | .0014 | 1.71 | .033 | 2.01 |
| 3000 | .103 | .994 | 1.13 | 1.14 | .0019 | 1.88 | .032 | 1.85 |
| 4000 | .195 | .997 | 1.19 | 1.22 | .0048 | 1.45 | .024 | 1.74 |

Reading so far:
- The gate opened right at `min_steps`: the head was already at .985 AUC, so the step floor was the binding condition.
- Separation keeps improving under the penalty, and the adversary keeps finding perturbations.
- The penalty is small (≤ .005, in squared gap units). The perturbation family moves the safety readout by only about 7% of the class gap at this stage. If it stays this small at full ramp, λ_max = 1 may be too weak, and a λ sweep would be the next knob.
- The gap is growing slowly (1.04 → 1.19). Keep watching whether the live-gap normalization pushes safe/unsafe apart beyond normal head training.

## 5. Evaluation plan (pending: fill in when the runs finish)

**Zero-shot margin OOD.** Add both runs to `ENCODERS` in `le-wm/scripts/ood_margin_jepa_static.py:30`. Run `{color, shape, rotate} × seeds {0,1,2}`, then aggregate with `ood_seed_aggregate.py`.

| encoder | in-dist acc/AUC | color zs acc/AUC | shape zs acc/AUC | rotate zs acc/AUC |
|---|---|---|---|---|
| baseline (sigreg) | ~.94–.99 | .17/.14 (collapses) | .31/.91 (threshold breaks, ranking holds) | robust (.83–.96 acc) |
| jacobian | ~.94–.99 | .17/.67 (ranking only) | robust | robust (.83–.96 acc) |
| jac+pull | ~.94–.99 | .83/.75 (survives) | robust | robust (.83–.96 acc) |
| **A: sigreg + safety_adv** | pending | pending | pending | pending |
| **B: jac+pull + safety_adv** | pending | pending | pending | pending |

(Baseline rows: the corrected margin_gp study of 2026-09-09, n=3 seeds, which supersedes the earlier V*-regression numbers. Read AUC as well as sign-acc, since acc is pinned at the 17% base rate when the threshold breaks.)

**What to look at:**
1. **Color zero-shot:** does A survive where the plain baseline inverts? Caveat: purple lies inside the perturbation family, so this tests whether the prior works, not true out-of-family generalization.
2. **Rotation and shape:** these are outside the family. Any gain there would come from the latent being better structured, not from coverage.
3. **Critic zero-shot (the real target):** retrain the reachability critic on each new encoder (`wm_ddpg.py`, then `critic_ood_eval.py`). Every encoder so far, jac+pull included, collapses here on color (.225 sign). The predictor term exists to address exactly this.
4. **No collapse:** σ_red (spread along the obstacle-color direction) stays near the baseline's, unlike the global-aug run (18.7 → 0.27); in-dist AUC is preserved.
5. **Planner:** in-dist CEM success and collision under the sg25clean protocol (`raw-cem-ood-protocol`), to confirm control isn't hurt.

**What would falsify it:** A no better than baseline on color zero-shot AUC, or a critic still at the .225 base rate. Either would mean projecting onto the instantaneous margin isn't enough, and the reachability direction needs its own signal (time-to-failure labels or a longer rollout term).

## 6. Known limitations

- Only a 1-step predictor term: dataset windows are `history_size + num_preds = 4` frames, so a longer rollout term needs longer windows.
- The perturbation family covers color only, not geometry.
- n = 1 seed per run so far.
- Validation loss excludes the penalty (PGD is skipped under inference mode). Head stats are still logged.
