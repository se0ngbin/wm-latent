# Safety-projected adversarial invariance: report

*Updated 2026-10-05 14:00 UTC. Branch `safety-adv-reg`; a copy lives at the main checkout root.*

## TL;DR

- **What we tried:** a training-time regularizer that makes the world model's *safety readout* (a margin head learned from the binary failure labels) insensitive to adversarial perturbations. No OOD data is used. Two perturbation families: a structured **hue/saturation** shift (a color prior) and plain per-pixel **L∞ noise** (ε = 8/255, no prior).
- **The L∞ version is the best model we have on every axis.** Zero-shot AUC: color **.91** (jac+pull .75, baseline .14), shape **1.00**, rotation **1.00**, in-dist .998.
- **It generalizes rather than covering the test.** An 8/255 per-pixel budget cannot reach purple (red → purple needs a ~0.5 change per channel), and noise contains no shape or rotation, so all three shifts are outside the training perturbation.
- **The hue version, despite targeting color, was worse:** color .63, shape dropped to .61, and it added nothing on top of jac+pull.
- **Still unsolved: the threshold.** Under purple, no model keeps its safe/unsafe cutoff. The L∞ model calls everything *unsafe* (conservative); jac+pull and the hue models call everything *safe* (the dangerous direction). Jac+pull's earlier "color survival" was this artifact.
- **Costs and open questions:** prediction loss is higher (.0114 vs .0067 for the hue model), and planning and the reachability critic haven't been tested. Only one training seed per model.

## Results

Zero-shot margin_gp AUC: a head trained on the normal (red, circle, upright) appearance, scored on the shifted appearance. AUC .5 is chance and below .5 is inverted. Mean of 3 eval seeds; per-seed values in brackets.

| model | in-dist AUC | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| baseline (sigreg only) | .979 | .14 [.14 .12 .18] | .91 [.89 .93 .92] | .96 [.97 .93 .97] |
| jacobian | .992 | .67 [.76 .61 .66] | .99 | .99 |
| jac+pull | .983 | .75 [.71 .75 .80] | .98 | .98 |
| baseline + safety reg, hue (v2 A) | .996 | .63 [.73 .45 .72] | .61 [.63 .53 .68] | .97 [.97 .96 .97] |
| jac+pull + safety reg, hue (v2 B) | .985 | .68 [.75 .79 .51] | .98 [.98 .97 .98] | .98 [.99 .99 .98] |
| **baseline + safety reg, L∞ noise** | **.998** | **.91** [.98 .96 .80] | **1.00** [.998 .996 .998] | **1.00** [.997 .994 .996] |

Each new run is compared against the matched model without the regularizer (same recipe, seed, 50 epochs). The second head type (margin_nogp) agrees for the L∞ model: color .974/.976/.765, shape ≥ .993, rotation ≥ .982. Retraining a head on the shifted appearance recovers every model to ≥ .93 AUC; the shift relocates the latent rather than destroying information.

### Why the table uses AUC, not accuracy

Zero-shot accuracy is misleading here. With 17% unsafe states, a head that calls *everything safe* scores ≈ .83 on color (≈ .89 on shape, since the diamond is smaller), and one that calls *everything unsafe* scores ≈ .17. Every model lands on one of those two numbers under purple. To confirm which, I logged the fraction of states each zero-shot head predicts safe (`/data/seongbin/lewm/safeadv_results/predsafe_check/`):

| model | predicted safe under purple (3 seeds) | failure direction |
|---|---|---|
| baseline | 0.000 / 0.000 / 0.000 | all unsafe |
| jacobian | 0.000 / 0.002 / 0.000 | all unsafe |
| jac+pull | **1.000 / 0.929 / 1.000** | all safe |
| hue safety reg (v2 A) | 1.000 / 1.000 / 1.000 | all safe |
| hue safety reg on jac+pull (v2 B) | 0.999 / 1.000 / 0.947 | all safe |
| L∞ safety reg | ≈ 0 (accuracy .164 / .177 / .177 equals the unsafe rate; not logged directly) | all unsafe |

The true safe fraction is .83–.84. So **no model keeps a usable threshold under the color shift**. What separates them is AUC, i.e. whether the safe/unsafe *ranking* survives. This overturns the earlier claim that jac+pull "survives color": its accuracy came from calling everything safe.

## Interpretation

1. **The safety projection, not the color prior, is what helps.** Plain L∞ noise projected onto the safety readout beats the hand-designed hue family on every axis, including the color shift the hue family was built for. The hue family seems to have taught a narrow color invariance that cost shape robustness. Generic small-noise robustness *along the safety readout* transferred to all three large shifts.
2. **It goes well beyond jacobian.** Jacobian also penalizes local sensitivity, but uniformly across all latent directions, and reaches color .67. Restricting the noise penalty to the safety readout (adversarially) reaches .91. That's consistent with the "keep nuisance off the readout axis" lever from the earlier nuisance-projection analysis.
3. **The remaining failure is calibration.** Even the best ranking comes with a coherent readout offset under purple that pushes every state across the threshold. For a deployable filter, this needs recalibration or an offset-correction mechanism, not more invariance.
4. **The hue variant is not worth pursuing.** It underperforms L∞ everywhere and adds nothing on top of jac+pull.

## Caveats

- **One training seed per model.** The 3 seeds are eval seeds (sampled states and head initializations). Color seed variation is large (.80–.98 for L∞), so a second training seed is needed before calling this robust.
- **Prediction quality cost.** Final pred_loss is .0114 for L∞ vs .0067 for the hue model. Whether this hurts CEM planning hasn't been tested.
- **Only the margin head is tested.** The reachability critic's zero-shot behavior (where every earlier model collapsed on color) hasn't been evaluated.
- **BatchNorm health.** L∞ is clean: running-stat offset 0.83, variance ×1.66 (baseline 0.91 / ×2.26). The hue v2 A model had a large offset (2.2 / ×9.6) from its ordinary training pass, which may relate to its shape drop (unverified).

## Next steps (proposed)

1. **Second training seed** of the L∞ model, to confirm the color result.
2. **Planner eval** (sg25clean protocol, in-dist and under shift), to check the pred_loss cost.
3. **Critic zero-shot** for the L∞ model, the real target.
4. **L∞ on top of jac+pull**, and an ε sweep (4/255, 16/255).
5. **Calibration fix** for the threshold collapse, e.g. unsupervised re-centering of the readout per appearance.

---

## Appendix A: method

- **Safety readout.** A margin head (margin_gp loss: zs 0.1, hinge 1.0, gradient penalty 10 @ 0.1; 512×2 MLP) trains during world-model training on detached embeddings and the dataset's `failures` labels (16.7% positives), with its own optimizer.
- **Perturbation**, found per batch by 2-step sign-PGD, keeping each sample's best iterate. Two families (config `attack`):
  - `hue`: a spatially varying hue rotation (±90°) and saturation scale (×0.5–2) on a 4×4 grid. White stays white, and obstacles can't be erased.
  - `linf`: per-pixel noise |δ| ≤ ε = 8/255 in RGB, one δ per sequence shared across context frames.
- **Penalty.** The change in the frozen head's output under the perturbation, on context frames and the 1-step prediction, divided by the in-batch safe/unsafe gap (with gradient).
- **Schedule.** The head trains from step 0. The penalty switches on at step 2000 once head AUC ≥ .95, then ramps linearly to weight 1.0 over 10k steps.
- **BatchNorm.** All perturbed passes are normalized with the clean batch's statistics and never update running stats (see Appendix B).

**Why hue was tried first, and why it lost:** a recolor is a large, coherent pixel change, far outside a small noise ball, and earlier noise-style regularizers (Gaussian pixel invariance, encoder-Lipschitz, plain jacobian) hadn't fixed color. So a structured color family seemed necessary. The L∞ ablation shows that reasoning was wrong. Those earlier regularizers applied noise invariance *globally*. Applied adversarially and only along the safety readout, small noise transfers to large shifts.

## Appendix B: v1 failure (BatchNorm cheat), now fixed

v1 sent clean and perturbed frames through one shared train-mode batch. The encoder learned to make perturbed frames produce extreme values in some features. That inflated the shared BatchNorm variance and squashed both halves' readouts together: the clean safe/unsafe gap fell from .97 to .07 inside that batch. The penalty looked tiny (~.003) with no real invariance.

The skewed statistics also leaked into the running stats (variance ×6.7), so the deployed encoder differed from the trained one. v1 zero-shot AUC (seed 0): color .40, shape .085, rotation .996.

Fix (`c6e93c6`): capture clean-batch BN stats once, apply them to every perturbed pass, and never update running stats. Unit-tested: batch-composition independence, running stats untouched, gradients only to the encoder. **This applies to any le-wm regularizer that runs extra forward passes**, since the projector and predictor heads use BatchNorm.

## Appendix C: where things are

- Code: `le-wm/module.py` (`SafetyAdvInvarianceReg`, `_CleanStatBN`), config key `safety_adv`, data config `data=dubins_safety`.
- Checkpoints: `/data/seongbin/lewm/checkpoints/lewm_dubins_safeadv_linf50/` (L∞), `lewm_dubins_safeadv2_50/`, `lewm_dubins_jacpull_safeadv2_50/` (hue v2); `lewm_dubins_safeadv50/` (v1).
- Training logs and code snapshots: `/data/seongbin/lewm/code_safeadv_77856fd/` (L∞), `code_safeadv_c6e93c6/` (hue v2), `code_safeadv_ca9ec21/` (v1). W&B project `seongbin/lewm` (runs `dubins_safeadv_linf50`, `dubins_safeadv2_50`, `dubins_jacpull_safeadv2_50`).
- Eval logs: `/data/seongbin/lewm/safeadv_results/{linf_ood_eval, v2_ood_eval, v1_ood_eval, predsafe_check}/`. Diagnostic scripts: `.../diagnostics/`.
- Baseline numbers: `/home/seongbin/latent/run_logs/ood_gp_jepa_*_s*.log` (same eval script, `ood_margin_gp_jepa.py`).
