# Safety-projected adversarial invariance: report

*Updated 2026-10-05. Branch `safety-adv-reg`; a copy lives at the main checkout root.*

## TL;DR

- **What we tried:** a training-time regularizer that makes the world model's *safety readout* (a margin head learned from the binary failure labels) insensitive to generated color perturbations. No OOD data is used.
- **Result:** on a plain (sigreg-only) world model it clearly helps with color: zero-shot color AUC goes from .14 (inverted) to .63. That's about what the existing jacobian penalty already gets (.67).
- **But:** it hurts shape robustness (.91 → .61), and on top of jac+pull it adds nothing (color .75 → .68, within seed noise).
- **Neither model fixes the decision threshold.** Under purple, every state is still called safe, so a deployed safety filter would still fail without recalibration.
- **Bottom line:** so far this is not better than jac+pull. It reproduces jacobian-level color robustness by a different route, at a cost on shape.

## Results

Zero-shot margin_gp AUC: a head trained on the normal (red, circle, upright) appearance, scored on the shifted appearance. AUC .5 is chance and below .5 is inverted. Mean of 3 seeds; per-seed values in brackets.

| model | in-dist AUC | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| baseline (sigreg only) | .979 | .14 [.14 .12 .18] | .91 [.89 .93 .92] | .96 [.97 .93 .97] |
| jacobian | .992 | .67 [.76 .61 .66] | .99 | .99 |
| jac+pull | .983 | .75 [.71 .75 .80] | .98 | .98 |
| **baseline + safety reg (v2 A)** | **.996** | **.63** [.73 .45 .72] | **.61** [.63 .53 .68] | **.97** [.97 .96 .97] |
| **jac+pull + safety reg (v2 B)** | **.985** | **.68** [.75 .79 .51] | **.98** [.98 .97 .98] | *pending* |

Each new run is compared against the matched model without the regularizer (same recipe, seed, 50 epochs). Retraining a head on the shifted appearance recovers every model to ≥ .95 AUC. As in all earlier OOD results, the shift relocates the latent rather than destroying information.

### Why the table uses AUC, not accuracy

Zero-shot accuracy is misleading here. With 17% unsafe states, a head that calls *everything safe* scores ≈ .83 on color, and ≈ .89 on shape (the diamond is smaller). Both new models score exactly that: color .837/.825/.828 against an all-safe rate of .826/.825/.833. Baseline's .164 is the opposite collapse (everything called unsafe). So accuracy only tells you *which way* the threshold broke.

This also raises a doubt about an earlier finding. Jac+pull's color accuracy (.837/.824), previously read as "jac+pull survives color", sits on the same all-safe rate. A check that logs the fraction of states predicted safe is running (`/data/seongbin/lewm/safeadv_results/predsafe_check/`). Until it lands, treat jac+pull's color result as ranking-only (AUC .75), not as a working threshold.

## Interpretation

1. **The regularizer does what it targets.** On a model with no other invariance pressure, color goes from badly inverted to clearly informative. Purple lies inside the perturbation family, so this shows the prior works; it is not out-of-family generalization.
2. **It trades away shape.** Shape is outside the family. Pushing color variation off the safety readout appears to make the readout lean on features that a shape change disturbs. Rotation is unaffected.
3. **It is redundant with jac+pull.** Jac+pull already reaches ~.75 on color, and adding the regularizer doesn't move it. The two seem to buy the same thing.
4. **The threshold problem remains.** No model keeps its safe/unsafe cutoff under purple. For a deployable filter, the recolor still shifts the readout coherently in one direction.

## Caveats

- **One seed per trained model.** The 3 seeds are eval seeds (different sampled states and head initializations), not independent training runs.
- **This only tests the margin head.** The more important target, the reachability critic's zero-shot behavior, hasn't been evaluated. Every earlier model collapsed there on color.
- **BatchNorm offset in v2 A.** v2 A's first BatchNorm layer has a large running-stat offset (mean 2.2 std, variance ×9.6, vs baseline 0.9 / ×2.3). The regularizer can no longer write running stats, so this comes from the ordinary training pass. Its final-latent train/eval gap (1.05) is close to baseline's (0.99). It could still contribute to the shape drop, which is unverified. v2 B is clean (0.9 / ×2.6).

## Next steps (proposed)

1. Finish the pending cells: v2 B rotation, and the predicted-safe check for jac+pull's color threshold.
2. Ablation: same safety-projected penalty with a small-ε L∞ noise attack instead of hue, to separate "safety projection" from "color prior".
3. Critic zero-shot for v2 A/B, the real target.
4. If continuing, look at why shape degrades: whether the readout shifts toward edge/shape features, and whether the BatchNorm offset is involved.

---

## Appendix A: method

- **Safety readout.** A margin head (margin_gp loss: zs 0.1, hinge 1.0, gradient penalty 10 @ 0.1; 512×2 MLP) trains during world-model training on detached embeddings and the dataset's `failures` labels (16.7% positives), with its own optimizer.
- **Perturbation.** A spatially varying hue rotation (±90°) and saturation scale (×0.5–2) on a 4×4 grid. White stays white, and obstacles can't be erased. Found per batch by 2-step sign-PGD, keeping each sample's best iterate.
- **Penalty.** The change in the frozen head's output under the perturbation, on context frames and the 1-step prediction, divided by the in-batch safe/unsafe gap (with gradient).
- **Schedule.** The head trains from step 0. The penalty switches on at step 2000 once head AUC ≥ .95, then ramps linearly to weight 1.0 over 10k steps.
- **BatchNorm.** All perturbed passes are normalized with the clean batch's statistics and never update running stats (see Appendix B).

**Why hue rather than ordinary noise:**
- A recolor is a large, coherent pixel change, far outside a small noise ball.
- Noise-style regularizers (Gaussian pixel invariance, encoder-Lipschitz, plain jacobian) had already failed to fix color.
- A noise budget large enough to reach purple could also erase obstacles. Penalizing readout changes then teaches the encoder to ignore them.
- The cost: the prior only covers color.

## Appendix B: v1 failure (BatchNorm cheat), now fixed

v1 sent clean and perturbed frames through one shared train-mode batch. The encoder learned to make perturbed frames produce extreme values in some features. That inflated the shared BatchNorm variance and squashed both halves' readouts together: the clean safe/unsafe gap fell from .97 to .07 inside that batch. The penalty looked tiny (~.003) with no real invariance.

The skewed statistics also leaked into the running stats (variance ×6.7), so the deployed encoder differed from the trained one. v1 zero-shot AUC (seed 0): color .40, shape .085, rotation .996.

Fix (`c6e93c6`): capture clean-batch BN stats once, apply them to every perturbed pass, and never update running stats. Unit-tested: batch-composition independence, running stats untouched, gradients only to the encoder. **This applies to any le-wm regularizer that runs extra forward passes**, since the projector and predictor heads use BatchNorm.

## Appendix C: where things are

- Code: `le-wm/module.py` (`SafetyAdvInvarianceReg`, `_CleanStatBN`), config key `safety_adv`, data config `data=dubins_safety`.
- Checkpoints: `/data/seongbin/lewm/checkpoints/lewm_dubins_safeadv2_50/`, `lewm_dubins_jacpull_safeadv2_50/` (v2); `lewm_dubins_safeadv50/` (v1).
- Training logs and code snapshots: `/data/seongbin/lewm/code_safeadv_c6e93c6/` (v2), `code_safeadv_ca9ec21/` (v1). W&B project `seongbin/lewm`.
- Eval logs: `/data/seongbin/lewm/safeadv_results/{v2_ood_eval, v1_ood_eval, predsafe_check}/`. Diagnostic scripts: `.../diagnostics/`.
- Baseline numbers: `/home/seongbin/latent/run_logs/ood_gp_jepa_*_s*.log` (same eval script, `ood_margin_gp_jepa.py`).
