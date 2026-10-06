# Safety-projected adversarial invariance: report

*Updated 2026-10-05 22:25 UTC. All results are Dubins. Branch `safety-adv-reg`; a copy lives at the main checkout root.*

## TL;DR

- **What we tried:** a training-time regularizer that makes the world model's *safety readout* insensitive to adversarial perturbations. The readout is a margin head learned from the binary failure labels; no OOD data is used. The main variant uses plain per-pixel **L∞ noise** (ε = 8/255). Full training details are in Appendix A.
- **It's the best model we have on every shift.** Zero-shot AUC: color **0.910 ± 0.098** (jac+pull 0.753 ± 0.049, baseline 0.144 ± 0.032), shape **0.997 ± 0.001**, rotation **0.996 ± 0.002**, in-dist 0.998 ± 0.001.
- **It generalizes rather than covering the test.** An 8/255 per-pixel budget cannot reach purple (red → purple needs a ~0.5 change per channel), and noise contains no shape or rotation change, so all three shifts are outside the training perturbation.
- **Still unsolved: the threshold under recolor.** Under purple, no model keeps its safe/unsafe cutoff: on a 50/50 safe/unsafe test set every model scores 0.500–0.503 sign accuracy (L∞ included). Which way it breaks (everything safe vs everything unsafe) depends on how the head was trained, not on the encoder. Jac+pull's earlier "color survival" was this artifact. Under shape and rotation, L∞ keeps a working threshold (0.974 / 0.959 on 50/50).
- **A structured color (hue) perturbation, tried first, did worse** and is covered in Appendix B.
- **Costs and open questions:** prediction loss is higher (0.0114 vs 0.0067 for a comparable run), and planning and the reachability critic haven't been tested. One training seed per model. Running now: L∞ on top of jac+pull.

## Results

Zero-shot margin_gp head: trained on the normal (red, circle, upright) appearance, scored on the shifted appearance.
- **Sign accuracy:** does the head's sign (h ≥ 0 = safe) match the label?
- **AUC:** does the safe/unsafe ranking survive, independent of the threshold? (0.500 = chance, below = inverted.)

All values are mean ± std over 3 eval seeds; in-dist is from the color run. Eval states are uniform over the world, about 17% unsafe (circle) or 11% (diamond).

**Sign accuracy**

| model | in-dist | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| baseline (sigreg only) | 0.943 ± 0.005 | 0.170 ± 0.006 | 0.310 ± 0.025 | 0.830 ± 0.006 |
| jacobian | 0.968 ± 0.002 | 0.171 ± 0.007 | 0.968 ± 0.007 | 0.962 ± 0.004 |
| jac+pull | 0.959 ± 0.003 | 0.830 ± 0.007 | 0.957 ± 0.008 | 0.939 ± 0.003 |
| **baseline + safety reg, L∞ noise** | 0.981 ± 0.004 | 0.174 ± 0.006 | 0.980 ± 0.003 | 0.975 ± 0.004 |

**AUC**

| model | in-dist | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| baseline (sigreg only) | 0.979 ± 0.002 | 0.144 ± 0.032 | 0.913 ± 0.024 | 0.956 ± 0.023 |
| jacobian | 0.992 ± 0.001 | 0.673 ± 0.076 | 0.992 ± 0.003 | 0.989 ± 0.002 |
| jac+pull | 0.983 ± 0.001 | 0.753 ± 0.049 | 0.977 ± 0.007 | 0.978 ± 0.003 |
| **baseline + safety reg, L∞ noise** | 0.998 ± 0.001 | 0.910 ± 0.098 | 0.997 ± 0.001 | 0.996 ± 0.002 |

Each new run is compared against the matched model without the regularizer (same recipe, seed, 50 epochs). The second head type (margin_nogp) agrees for the L∞ model: color AUC 0.905 ± 0.121, shape ≥ 0.993, rotation ≥ 0.982. Retraining a head on the shifted appearance recovers every model to ≥ 0.930 AUC; the shift relocates the latent rather than destroying information.

### Reading sign accuracy under the color shift

Under purple, sign accuracy is misleading on its own. With ~17% unsafe states, a head that calls *everything safe* scores ≈ 0.830, and one that calls *everything unsafe* scores ≈ 0.170. Every model lands on one of those two values. To confirm which, I logged the fraction of states each zero-shot head predicts safe (`/data/seongbin/lewm/safeadv_results/predsafe_check/`):

| model | predicted safe under purple | true safe fraction |
|---|---|---|
| baseline (sigreg only) | 0.000 ± 0.000 | 0.830 ± 0.006 |
| jacobian | 0.001 ± 0.001 | 0.830 ± 0.006 |
| jac+pull | 0.976 ± 0.041 | 0.830 ± 0.006 |
| **baseline + safety reg, L∞ noise** | not logged; sign accuracy 0.174 ± 0.006 equals the unsafe rate, so ≈ 0.000 (measured directly in the 50/50 eval) | 0.830 ± 0.006 |

So **no model keeps a usable threshold under the color shift**. On this eval set baseline, jacobian and L∞ call everything unsafe and jac+pull calls everything safe; but the 50/50 eval below shows the *direction* flips with the head's training set (jac+pull calls everything unsafe there), so only the collapse itself is a stable finding. What separates the models on color is AUC, i.e. whether the safe/unsafe *ranking* survives. This overturns the earlier claim that jac+pull "survives color": its 0.830 sign accuracy came from calling everything safe. Under shape and rotation the L∞ threshold does survive (sign accuracy 0.980 / 0.975).

### 50/50 safe/unsafe eval

The same eval with every set exactly balanced: heads trained on 3000 safe + 3000 unsafe states, tested on 1000 + 1000, each balanced on the label it's scored against (circle for in-dist, diamond for shape zero-shot). States are rejection-sampled from the same uniform distribution, rendered once per axis and seed, and shared by all models. Here, both "all safe" and "all unsafe" score 0.500 sign accuracy, so sign accuracy reads directly. Mean ± std over 3 eval seeds (`balanced_eval/`).

**Sign accuracy (50/50: .500 = threshold gone)**

| model | in-dist | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| baseline (sigreg only) | 0.940 ± 0.003 | 0.500 ± 0.000 | 0.585 ± 0.010 | 0.500 ± 0.000 |
| jacobian | 0.963 ± 0.003 | 0.500 ± 0.000 | 0.952 ± 0.003 | 0.946 ± 0.009 |
| jac+pull | 0.940 ± 0.005 | 0.503 ± 0.005 | 0.919 ± 0.005 | 0.919 ± 0.013 |
| **baseline + safety reg, L∞ noise** | 0.980 ± 0.004 | 0.501 ± 0.001 | 0.974 ± 0.003 | 0.959 ± 0.008 |

**AUC**

| model | in-dist | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| baseline (sigreg only) | 0.985 ± 0.002 | 0.177 ± 0.032 | 0.911 ± 0.029 | 0.902 ± 0.041 |
| jacobian | 0.994 ± 0.001 | 0.711 ± 0.069 | 0.992 ± 0.001 | 0.990 ± 0.003 |
| jac+pull | 0.987 ± 0.002 | 0.747 ± 0.063 | 0.981 ± 0.001 | 0.983 ± 0.004 |
| **baseline + safety reg, L∞ noise** | 0.999 ± 0.001 | 0.927 ± 0.021 | 0.998 ± 0.000 | 0.996 ± 0.002 |

**Predicted-safe fraction (true = .500)**

| model | in-dist | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| baseline (sigreg only) | 0.497 ± 0.003 | 0.000 ± 0.000 | 0.085 ± 0.010 | 1.000 ± 0.000 |
| jacobian | 0.502 ± 0.002 | 0.000 ± 0.000 | 0.471 ± 0.002 | 0.505 ± 0.006 |
| jac+pull | 0.492 ± 0.008 | 0.003 ± 0.005 | 0.480 ± 0.012 | 0.455 ± 0.010 |
| **baseline + safety reg, L∞ noise** | 0.501 ± 0.003 | 0.001 ± 0.001 | 0.509 ± 0.005 | 0.522 ± 0.001 |

What it shows:
- **Confirms:** every model's color threshold collapses (sign accuracy 0.500–0.503). L∞ has the best ranking on every shift: color AUC 0.927 ± 0.021 (less seed-variable than in the uniform eval), shape 0.998, rotation 0.996. L∞ keeps working thresholds on shape (0.974) and rotation (0.959).
- **Corrects (direction of collapse):** jac+pull now calls almost everything *unsafe* under purple (predicted safe 0.003 ± 0.005), whereas in the uniform eval it called everything *safe* (0.976). Same encoder, different head training set. The collapse is a property of the encoder; its direction is not.
- **Corrects (baseline rotation):** baseline's rotation threshold also collapses (sign accuracy 0.500, everything called safe). In the uniform eval its 0.830 rotation sign accuracy equalled the all-safe rate, which hid this.

## Interpretation

1. **Safety-projected noise robustness transfers to large shifts.** Small adversarial pixel noise, penalized only through the safety readout, gives robustness to three shifts it never saw, including the color shift that a hand-designed color perturbation handled worse (Appendix B). Earlier *global* noise-invariance regularizers (Gaussian pixel invariance, encoder-Lipschitz) never fixed color; the difference is restricting the penalty to the safety readout and making it adversarial.
2. **It goes well beyond jacobian.** Jacobian also penalizes local sensitivity, but uniformly across all latent directions, and reaches color AUC 0.673. The safety-projected adversarial version reaches 0.910. That's consistent with the "keep nuisance off the readout axis" lever from the earlier nuisance-projection analysis.
3. **The remaining failure is calibration.** Even the best ranking comes with a coherent readout offset under purple that pushes every state across the threshold. For a deployable filter, this needs recalibration or an offset-correction mechanism, not more invariance.

## Caveats

- **One training seed per model.** The 3 seeds are eval seeds (sampled states and head initializations). Color varies a lot across them (L∞ color AUC 0.910 ± 0.098), so a second training seed is needed before calling this robust.
- **Prediction quality cost.** Final pred_loss is 0.0114 for L∞ vs 0.0067 for the hue-variant run. Whether this hurts CEM planning hasn't been tested.
- **Only the margin head is tested.** The reachability critic's zero-shot behavior (where every earlier model collapsed on color) hasn't been evaluated.
- **Eval states are uniform, not on-policy.** They cover every heading and positions deep inside obstacles; the world model trained on expert trajectories. "In-dist" means in-distribution *appearance*.

## Next steps (proposed)

1. **Running: L∞ on top of jac+pull** (`lewm_dubins_jacpull_safeadv_linf50`, started 2026-10-05 21:58 UTC, ~8–10 h; uniform and 50/50 evals start automatically when it finishes).
2. **Running: "remove h" ablation** (`lewm_dubins_linf_latent50`, started 2026-10-06 00:45 UTC): identical to the L∞ run except the penalty compares the whole latent, ‖f(x_t + δ) − f(x_t)‖² (and the same for the prediction), normalized by the squared distance between the safe and unsafe latent centroids instead of the head gap. Tests whether projecting onto the safety readout is what makes L∞ work. Evals start automatically when it finishes.
3. **Second training seed** of the L∞ model, to confirm the color result.
4. **Planner eval** (sg25clean protocol, in-dist and under shift), to check the pred_loss cost.
5. **Critic zero-shot** for the L∞ model, the real target.
6. **ε sweep** (4/255, 16/255).
7. **Calibration fix** for the threshold collapse, e.g. unsupervised re-centering of the readout per appearance.

---

## Appendix A: how the L∞ model was trained

Run `lewm_dubins_safeadv_linf50`, code `77856fd` (snapshot `/data/seongbin/lewm/code_safeadv_77856fd/`, launcher `run.sh`), W&B `seongbin/lewm/dubins_safeadv_linf50`.

### A.0 Baseline training vs. adding the safety regularizer, at a glance

The two runs share everything that defines the world model: architecture, data, base losses, optimizer, schedule, 50 epochs, seed 3072. The safety regularizer only *adds* things:

| | baseline (`sigreg_only_dubins`) | + safety regularizer, L∞ (`lewm_dubins_safeadv_linf50`) |
|---|---|---|
| Data loaded | pixels, actions, proprio, state | same, plus the per-frame `failures` label (`data=dubins_safety`) |
| Trained modules | encoder, projector, predictor, action embedder | same, plus a separate margin head h with its own optimizer (never part of the world model) |
| Loss on the world model | prediction MSE + 0.09 · SIGReg | same, plus 1.0 · ramp · (safety-readout change under worst-case 8/255 noise) |
| Extra work per step | none | head update; 1 clean + 3 perturbed + 1 graded forward over 32 windows × 3 frames |
| When the extra term is active | — | off for the first 2,000 steps, then ramped to full strength by step 12,000 |
| BatchNorm | standard | regularizer passes use clean-batch statistics and never touch running stats |
| Peak GPU memory | 13.5 GB | 15.8 GB |
| Saved world-model weights | `weights_epoch_N.pt` | same format and size (drop-in replacement); the head lives only in the Lightning checkpoint |

What changes in the result:
- **Prediction loss** ends somewhat higher (0.0114; the baseline's isn't logged to W&B, and the hue-variant run ends at 0.0067). The extra term competes a little with dynamics accuracy.
- **In-distribution safety separation** improves: margin-head AUC 0.998 ± 0.001 vs 0.979 ± 0.002, sign accuracy 0.981 vs 0.943.
- **Zero-shot robustness** improves on every shift: AUC color 0.144 → 0.910, shape 0.913 → 0.997, rotation 0.956 → 0.996. The color threshold still collapses (50/50 sign accuracy 0.501), as the baseline's does.

Nothing about how the world model is *used* changes. The planner and the margin/critic training consume the same weights in the same way; the regularizer only changes what those weights learned.

### A.1 World model and data (unchanged from the baseline)

- **Data:** `dubins_expert.h5`: 169,189 frames in 4,100 expert episodes, 128×128 RGB, resized to 224×224 and ImageNet-normalized. Windows of 4 consecutive frames, frameskip 1, 90/10 train/val split, seed 3072. Loaded with `data=dubins_safety`, which adds the per-frame `failures` column (1 = agent center inside an obstacle; 16.7% of frames) and keeps it out of z-scoring.
- **Model:** LeWM JEPA. ViT-tiny encoder (patch 14, from scratch) → CLS token → projector MLP (192 → 2048 → 192, BatchNorm) = embedding z (192-d). Autoregressive transformer predictor (depth 6, 16 heads, **causal** attention) with an action embedder, followed by `pred_proj` (MLP with BatchNorm). It always predicts the **next** frame's embedding, looking back at most 3 frames (`history_size: 3`). One 4-frame window gives three predictions trained together: frame 1 → frame 2, frames 1–2 → frame 3, frames 1–3 → frame 4.
- **Base losses:** prediction MSE over those three next-frame predictions, plus SIGReg (weight 0.09, 17 knots, 1024 projections). This is exactly the `sigreg_only_dubins` baseline recipe.
- **Optimization:** AdamW lr 5e-5, weight decay 1e-3, linear-warmup cosine schedule over 50 epochs (55,150 steps of batch 128), bf16 mixed precision, gradient clipping 1.0.

### A.2 The safety readout (trained alongside, not part of the world model)

- **Head h:** MLP 192 → 512 → 512 → 1 (Linear without bias, LayerNorm, SiLU, twice; then Linear). Same architecture as the deployed latent_cbf `MarginHead`.
- **Loss (margin_gp, the repo's instantaneous-margin loss):** 0.1 · (mean h(unsafe) − mean h(safe)) + 1.0 · (mean ReLU(h(unsafe)) + mean ReLU(−h(safe))) + 10 · gradient penalty pulling ‖∇h‖ toward 0.1 at random safe/unsafe interpolates. Positive h = safe.
- **Inputs:** every training step, the head takes one optimizer step on the *detached* embeddings of all 512 frames in the batch (128 windows × 4 frames) and their `failures` labels, in fp32.
- **Own optimizer:** AdamW lr 3e-4. The main optimizer only owns the world model, so world-model gradients never reach the head, and the head trains about 6× faster than the encoder (lr 3e-4 vs 5e-5), so it tracks the moving latent.
- **Tracked statistics:** in-batch AUC and the safe-minus-unsafe gap (mean h(safe) − mean h(unsafe)), each as a bias-corrected EMA (decay 0.99).

### A.3 The perturbation and the adversary

- **Which images get perturbed:** a training *window* is 4 consecutive frames from an expert trajectory; frames 1–3 are the *context* (the frames the predictor reads), and frames 2–4 are its next-frame targets (A.1). A batch has 128 windows. To keep the cost down, the regularizer uses only 32 of them (the first 32 in the batch, effectively random since the loader shuffles) and perturbs only their 3 context frames: 96 images per step. Frame 4 is not perturbed; the penalty instead compares the *predicted* frame-4 embedding (from all three context frames) under perturbed vs. clean context (A.4). The two shorter-history predictions (frame 2 from 1, frame 3 from 1–2) are not penalized. The head's own training still uses all 128 × 4 = 512 clean frames.
- **Perturbation:** δ added to the image in [0, 1] RGB space at 224×224, then clamped to [0, 1] and renormalized. One δ of shape 3×224×224 per window, **shared across its 3 context frames**, so a window sees a consistent "appearance" over time. Budget |δ| ≤ ε = 8/255 per pixel and channel (L∞).
- **Search (PGD):** random start δ ~ Uniform[−ε, ε]. Two sign-gradient ascent steps of size 0.5·ε on the penalty below, each projected back into the ε-box. The penalty is evaluated at the start and after each step, and **each window keeps its best of the three iterates**, so the adversary can never end up worse than its random start.
- **What it maximizes:** the same normalized readout change that the encoder minimizes (A.4), using the detached gap.

### A.4 The penalty

For each window, with clean context frames x₁..₃, perturbed frames x₁..₃ + δ, actions a, encoder f, predictor P:

    d = mean over t of ((h(f(x_t + δ)) − h(f(x_t))) / gap)²  +  ((h(P(f(x + δ), a)) − h(P(f(x), a))) / gap)²

- **First term:** the safety readout of each perturbed context frame must match the clean one.
- **Second term:** the same for the predictor's last prediction (frame 4, from all three context frames), so the robustness also has to hold through the dynamics.
- **Clean targets** h(f(x_t)) and h(P(f(x), a)) are computed once without gradient.
- **h's parameters are frozen inside the penalty** (called with detached parameters). Only the encoder, projector, predictor and action embedder receive gradients, so the head can't satisfy the penalty by flattening itself.
- **gap** = the live in-batch gap: mean h(f(x)) over safe frames minus over unsafe frames, across all 512 frames of the main forward pass, **with gradient**, floored at 0.1 × the EMA gap. Shrinking the safe/unsafe separation therefore *raises* the penalty immediately, so the encoder can't win by squashing the readout.
- **Loss added to training:** λ · ramp · mean(d) over the 32 windows, with λ = 1.0.

### A.5 BatchNorm handling

The projector and `pred_proj` contain BatchNorm. Before the adversary runs, one no-gradient clean pass records each BatchNorm layer's batch mean and variance. Every perturbed pass (PGD and the final one) is normalized with those clean statistics, and running statistics are never updated by the regularizer. Without this, the encoder can game shared batch statistics; that's exactly how the first version failed (Appendix C).

### A.6 Schedule

- Steps 0–1999: penalty off (no adversary cost); the head trains on every step.
- Gate: the penalty switches on at the first step with step ≥ 2000 **and** head AUC (EMA) ≥ 0.95. In this run it opened at **step 2000** exactly; head AUC was already ≈ 0.99.
- Ramp: ramp rises linearly 0 → 1 over 10,000 steps (full strength at step 12,000, ~22% of training), then stays at 1 for the rest of the run.
- The penalty runs only in training mode; validation loss excludes it.

### A.7 Training curve (W&B, mean over ±1000 steps)

| step | ramp | penalty (unramped) | head AUC | gap | adversary gain over random start | pred_loss |
|---|---|---|---|---|---|---|
| 2000 | 0.053 | ≈ 0.192 | 0.990 | 1.090 | 87.3× | 0.048 |
| 6000 | 0.403 | ≈ 0.013 | 0.996 | 1.469 | 18.7× | 0.038 |
| 12000 | 0.976 | ≈ 0.005 | 0.998 | 1.828 | 15.6× | 0.030 |
| 25000 | 1.000 | 0.003 | 0.998 | 2.066 | 14.6× | 0.017 |
| 40000 | 1.000 | 0.002 | 0.999 | 2.179 | 13.8× | 0.012 |
| 55000 | 1.000 | 0.002 | 0.999 | 2.217 | 13.4× | 0.011 |

- **The encoder learns the robustness quickly:** the worst-case readout change falls to ≈ 0.045 class gaps (√0.002).
- **The safety separation keeps widening:** gap 1.09 → 2.22, with head AUC 0.999 at the end.
- **The adversary stays effective:** it finds perturbations 13–87× stronger than a random one throughout, so the low penalty isn't a weak attack.
- **BatchNorm stays healthy** (checked on the final model): running-stat offset 0.83 std and variance ratio ×1.66, vs baseline 0.91 / ×2.26.

### A.8 What it does to the latent space

Same 4000 uniform Dubins states rendered red vs purple, one eval seed (`diagnostics/diag_latent_linf.py`):

| | baseline | jac+pull | L∞ safety reg |
|---|---|---|---|
| Mean latent shift under purple ÷ safe/unsafe centroid distance | 2.377 | 1.272 | **0.771** |
| Per-state shift ÷ centroid distance | 2.723 | 3.324 | **1.308** |
| cos(mean shift, safety-readout gradient) | −0.685 | −0.201 | −0.641 |
| Share of the mean shift along the readout direction | 0.469 | 0.041 | 0.411 |
| Effective latent dimensions (participation ratio, of 192) | 10.5 | 45.3 | 10.8 |
| Safe/unsafe separation d′ (fresh head) | 3.41 | 4.60 | **7.44** |

- **Mechanism: magnitude, not direction.** L∞ makes the color shift small relative to the safety separation. The safe/unsafe separation roughly doubles (d′ 3.41 → 7.44), and purple moves each latent about half as far as in baseline, so in class-separation units the push is about a third of baseline's. The shift still points along the readout about as much as baseline's. Jac+pull is the opposite: it rotates the color shift off the readout (4% along it) and spreads the latent over ~45 dimensions.
- **No collapse:** effective dimensionality is unchanged from baseline (10.8 vs 10.5).
- **Why the color threshold collapses to "all unsafe":** the residual shift points down the readout, lowering every state's margin. The run's own in-training head moves by −0.705 class gaps under purple (predicted safe 0.004), while its ranking survives (AUC 0.962 red-trained head on purple). A uniform offset like this is a calibration problem.

### A.9 Cost

- **Memory:** 15.8 GB peak at batch 128, vs 13.5 GB without the regularizer.
- **Time:** 4 h 44 min for 50 epochs (08:41–13:25 UTC on 2026-10-05) on one RTX PRO 6000, which was shared with other jobs.
- **Extra work per step:** 1 clean pass, 3 perturbed passes (2 with gradient to δ) and 1 graded pass, each over 32 windows × 3 frames.

## Appendix B: the hue (color-prior) variant

**Perturbation:** a spatially varying hue rotation (±90°) and saturation scale (×0.5–2), parameterized on a 4×4 grid and bilinear-upsampled, one field per window shared across context frames. White stays white, and obstacles can't be erased (chroma is only scaled). Everything else (head, penalty, adversary, BatchNorm handling, schedule) is the same as Appendix A (code `c6e93c6`, `attack=hue`). Unlike L∞, red → purple lies inside this family.

**Why it was tried first:** a recolor is a large, coherent pixel change, far outside a small noise ball, and earlier global noise-style regularizers hadn't fixed color. So a structured color family seemed necessary. The L∞ result shows that reasoning was wrong.

**Results** (same eval, mean ± std over 3 eval seeds):

**Sign accuracy**

| model | in-dist | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| baseline (sigreg only) | 0.943 ± 0.005 | 0.170 ± 0.006 | 0.310 ± 0.025 | 0.830 ± 0.006 |
| jacobian | 0.968 ± 0.002 | 0.171 ± 0.007 | 0.968 ± 0.007 | 0.962 ± 0.004 |
| jac+pull | 0.959 ± 0.003 | 0.830 ± 0.007 | 0.957 ± 0.008 | 0.939 ± 0.003 |
| **baseline + safety reg, hue** | 0.970 ± 0.003 | 0.830 ± 0.006 | 0.891 ± 0.005 | 0.843 ± 0.024 |
| **jac+pull + safety reg, hue** | 0.969 ± 0.004 | 0.816 ± 0.026 | 0.968 ± 0.003 | 0.892 ± 0.018 |

**AUC**

| model | in-dist | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| baseline (sigreg only) | 0.979 ± 0.002 | 0.144 ± 0.032 | 0.913 ± 0.024 | 0.956 ± 0.023 |
| jacobian | 0.992 ± 0.001 | 0.673 ± 0.076 | 0.992 ± 0.003 | 0.989 ± 0.002 |
| jac+pull | 0.983 ± 0.001 | 0.753 ± 0.049 | 0.977 ± 0.007 | 0.978 ± 0.003 |
| **baseline + safety reg, hue** | 0.996 ± 0.003 | 0.634 ± 0.158 | 0.612 ± 0.080 | 0.966 ± 0.009 |
| **jac+pull + safety reg, hue** | 0.985 ± 0.004 | 0.681 ± 0.150 | 0.977 ± 0.007 | 0.983 ± 0.005 |

50/50 eval (sign accuracy, AUC, predicted-safe fraction):

**Sign accuracy (50/50: .500 = threshold gone)**

| model | in-dist | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| baseline (sigreg only) | 0.940 ± 0.003 | 0.500 ± 0.000 | 0.585 ± 0.010 | 0.500 ± 0.000 |
| jacobian | 0.963 ± 0.003 | 0.500 ± 0.000 | 0.952 ± 0.003 | 0.946 ± 0.009 |
| jac+pull | 0.940 ± 0.005 | 0.503 ± 0.005 | 0.919 ± 0.005 | 0.919 ± 0.013 |
| **baseline + safety reg, hue** | 0.970 ± 0.005 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.629 ± 0.069 |
| **jac+pull + safety reg, hue** | 0.964 ± 0.003 | 0.645 ± 0.101 | 0.942 ± 0.005 | 0.849 ± 0.046 |

**AUC**

| model | in-dist | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| baseline (sigreg only) | 0.985 ± 0.002 | 0.177 ± 0.032 | 0.911 ± 0.029 | 0.902 ± 0.041 |
| jacobian | 0.994 ± 0.001 | 0.711 ± 0.069 | 0.992 ± 0.001 | 0.990 ± 0.003 |
| jac+pull | 0.987 ± 0.002 | 0.747 ± 0.063 | 0.981 ± 0.001 | 0.983 ± 0.004 |
| **baseline + safety reg, hue** | 0.996 ± 0.002 | 0.623 ± 0.150 | 0.631 ± 0.084 | 0.967 ± 0.013 |
| **jac+pull + safety reg, hue** | 0.993 ± 0.001 | 0.685 ± 0.086 | 0.987 ± 0.002 | 0.986 ± 0.004 |

**Predicted-safe fraction (true = .500)**

| model | in-dist | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| baseline (sigreg only) | 0.497 ± 0.003 | 0.000 ± 0.000 | 0.085 ± 0.010 | 1.000 ± 0.000 |
| jacobian | 0.502 ± 0.002 | 0.000 ± 0.000 | 0.471 ± 0.002 | 0.505 ± 0.006 |
| jac+pull | 0.492 ± 0.008 | 0.003 ± 0.005 | 0.480 ± 0.012 | 0.455 ± 0.010 |
| **baseline + safety reg, hue** | 0.507 ± 0.006 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.856 ± 0.071 |
| **jac+pull + safety reg, hue** | 0.500 ± 0.013 | 0.388 ± 0.159 | 0.504 ± 0.025 | 0.360 ± 0.051 |

Predicted-safe fraction under purple:

| model | predicted safe under purple | true safe fraction |
|---|---|---|
| baseline (sigreg only) | 0.000 ± 0.000 | 0.830 ± 0.006 |
| jacobian | 0.001 ± 0.001 | 0.830 ± 0.006 |
| jac+pull | 0.976 ± 0.041 | 0.830 ± 0.006 |
| **baseline + safety reg, hue** | 1.000 ± 0.000 | 0.830 ± 0.006 |
| **jac+pull + safety reg, hue** | 0.982 ± 0.030 | 0.830 ± 0.006 |

**Reading:**
- **On the sigreg base,** hue lifts color AUC from 0.144 to 0.634 but drops shape from 0.913 to 0.612. Its color *and* shape thresholds collapse (50/50 sign accuracy 0.500 on both, everything called safe).
- **On jac+pull** it adds nothing (color 0.753 → 0.681, within seed noise).
- **BatchNorm:** the sigreg-base hue model also has a large first-layer running-stat offset (2.2 std, variance ×9.6) from its ordinary training pass, which may relate to its shape drop (unverified).
- **Conclusion:** L∞ beats it on every axis, including color, which hue was built for.

## Appendix C: v1 failure (BatchNorm cheat), now fixed

v1 sent clean and perturbed frames through one shared train-mode batch. The encoder learned to make perturbed frames produce extreme values in some features. That inflated the shared BatchNorm variance and squashed both halves' readouts together: the clean safe/unsafe gap fell from .97 to .07 inside that batch. The penalty looked tiny (~.003) with no real invariance.

The skewed statistics also leaked into the running stats (variance ×6.7), so the deployed encoder differed from the trained one. v1 zero-shot AUC (seed 0): color .40, shape .085, rotation .996.

Fix (`c6e93c6`): capture clean-batch BN stats once, apply them to every perturbed pass, and never update running stats. Unit-tested: batch-composition independence, running stats untouched, gradients only to the encoder. **This applies to any le-wm regularizer that runs extra forward passes**, since the projector and predictor heads use BatchNorm.

## Appendix D: where things are

- **Code:** `le-wm/module.py` (`SafetyAdvInvarianceReg`, `_CleanStatBN`), config key `safety_adv` (kwargs `attack: linf | hue`, `eps_pix`, `min_steps`, `auc_gate`, `ramp_steps`, `n_sub`, `pgd_steps`), data config `data=dubins_safety`.
- **Checkpoints** (under `/data/seongbin/lewm/checkpoints/`): `lewm_dubins_safeadv_linf50/` (L∞), `lewm_dubins_jacpull_safeadv_linf50/` (jac+pull + L∞, training), `lewm_dubins_safeadv2_50/` and `lewm_dubins_jacpull_safeadv2_50/` (hue), `lewm_dubins_safeadv50/` (v1). The in-training margin head is only in the Lightning checkpoints (`~/.cache/stable-pretraining/runs/<date>/<time>/<hash>/checkpoints/`).
- **Training logs and code snapshots:** `/data/seongbin/lewm/code_safeadv_77856fd/` (L∞, `.run_L/`; jac+pull + L∞, `.run_JL/`), `code_safeadv_c6e93c6/` (hue), `code_safeadv_ca9ec21/` (v1). W&B project `seongbin/lewm`.
- **Eval logs** (under `/data/seongbin/lewm/safeadv_results/`): `linf_ood_eval/`, `jacpull_linf_ood_eval/`, `v2_ood_eval/` (hue), `v1_ood_eval/`, `predsafe_check/`, `balanced_eval/` (50/50); diagnostic scripts in `diagnostics/`. Tables are generated by `make_tables.py`.
- **Baseline numbers:** `/home/seongbin/latent/run_logs/ood_gp_jepa_*_s*.log` (same eval script, `le-wm/scripts/ood_margin_gp_jepa.py`).
