# Safety-projected adversarial invariance: report

*Updated 2026-10-21 05:30 UTC. Dubins unless marked Safety Gym; LeWM unless marked Dreamer. Branch `safety-adv-reg`; a copy lives at the main checkout root.*

## TL;DR

- **What we tried:** a training-time regularizer that makes the world model's *safety readout* insensitive to adversarial perturbations. The readout is a margin head learned from the binary failure labels; no OOD data is used. The main variant uses plain per-pixel **L∞ noise** (ε = 8/255). Full training details are in Appendix A.
- **On Dubins, jac+pull + L∞ safety reg is the best model and the only one that is stable across world-model training seeds** ([training-seed variability](#training-seed-variability)). Zero-shot color AUC (uniform eval, mean ± std over 3 WM seeds): jac+pull + L∞ safety reg **0.957 ± 0.021**, baseline + L∞ safety reg w/o h 0.852 ± 0.098, baseline + L∞ safety reg 0.643 ± 0.247, jac+pull 0.498 ± 0.234; sigreg only 0.144 (1 seed). Every model keeps shape and rotation (≥ 0.975). The 50/50 eval agrees (0.955 ± 0.017 for jac+pull + L∞ safety reg). The single-seed numbers first reported (0.910 for the L∞ safety reg, 0.753 for jac+pull) were lucky seeds. #oct-18
- **The safety projection is not what makes it work, but every other ingredient is.** The "w/o h" version penalizes the whole latent shift ‖f(x + δ) − f(x)‖², scaled by the safe/unsafe centroid distance, and never touches the head. Over 3 WM seeds it beats the head-projected version on color (0.852 ± 0.098 vs 0.643 ± 0.247). Removing any one of its other parts drops color AUC to 0.63–0.70 (one seed each): the adversarial search, the gradient through the class-separation denominator, the encoder term, or the predictor term ([ablations](#ablations-what-makes-the-no-h-regularizer-work)). The attack must also search the quantity being penalized: attacking through h while penalizing the whole latent gives 0.251. #oct-18
- **Adding the L∞ safety reg on top of jac+pull** lifts color AUC from 0.498 ± 0.234 to 0.957 ± 0.021 over 3 WM seeds. It costs more prediction loss (0.016–0.017 vs 0.012–0.014). #oct-18
- #oct-18 **On Safety Gym, jac+pull + L∞ safety reg is the best and most stable model:** over 3 WM seeds color AUC 1.000 ± 0.000, rotation 0.981 ± 0.018, and the threshold holds on every seed (sign accuracy 0.979 / 0.962). With the Dubins result, it's the same recipe that wins on both benchmarks. Separately, **batch λ with a detached direction** keeps Safety Gym thresholds on all 3 seeds (rotation 0.945 ± 0.023), but its Dubins color is seed-unstable (0.721 ± 0.434).
- **Safety Gym (new): the regularizer transfers and is even stronger there** ([section](#safety-gym-and-the-regularizer-on-dreamer)). LeWM + L∞ safety reg reaches zero-shot AUC 0.999 on color and **0.962 on a real 90° camera rotation**, where jac+pull scores 0.679 and sigreg only 0.557. It is also the **first model in the study whose safe/unsafe threshold survives a shift**: sign accuracy 0.987 on color and 0.977 on rotation, against 0.833 for calling everything safe. #oct-18 Over 3 WM seeds this holds on 2 of 3: color AUC 0.985 ± 0.024 and sign accuracy 0.957 ± 0.056, but rotation AUC 0.851 ± 0.222, because seed 2 loses most of the rotation gain (0.596). The cost is 6× the prediction loss (0.0168 vs 0.0026). On Safety Gym the penalty had to run without the AUC gate, because the sigreg-only latent doesn't generalize safety across episodes.
- **Dreamer (new): smaller gains.** The head-projected regularizer lifts Dubins color AUC from 0.638 to 0.816 and Safety Gym color from 0.961 to 0.977. Rotation stays at chance on both benchmarks and thresholds still collapse. No-h doesn't help Dubins Dreamer at all (0.626), so for Dreamer the safety projection matters, unlike LeWM on Dubins.
- **λ-projection (new):** replacing the head with one shared safety direction λ = mean(z_safe) − mean(z_unsafe), and penalizing only the perturbation's shift along it, works on Dubins when λ is a running global average (color AUC 0.936, lowest prediction cost of any safety-reg model). It fails on Safety Gym, where the base latent has no consistent linear safety axis ([section](#safety-direction-λ-projection-instead-of-the-head)).
- **Adversarial vs. random noise (new):** replacing PGD with random ±8/255 noise of the same size loses most of the gain on every benchmark and axis, with or without h: Dubins color 0.910 → 0.300 (with h) and 0.951 → 0.631 (no h); Safety Gym color 0.999 → 0.871 and rotation 0.962 → 0.729 (with h). Random noise through the head even breaks Dubins rotation (0.555 vs 0.956 for sigreg only) ([section](#adversarial-vs-random-noise-of-the-same-size)).
- **Noise size / attack strength (new):** the Dubins color result is fragile. With the head variant, ε = 4/255 gives color AUC 0.771, ε = 16/255 gives 0.305, and 8/255 with 5 PGD steps gives 0.088 (inverted), against 0.910 for the reference 8/255 with 2 steps. Reseeds of the reference are running ([section](#noise-size-and-attack-strength)).
- **Training-seed check:** every headline model was retrained with two more seeds. Single-seed Dubins color numbers swing by up to ±0.37 across training seeds, so the 3-seed results are in their own tables ([training-seed variability](#training-seed-variability)), and the comparison tables elsewhere stay single-seed. Shape and rotation are stable across seeds; color is not, except for jac+pull + L∞ safety reg. #oct-18
- **Planning (new): the safety reg doesn't improve raw CEM.** Baseline + L∞ safety reg plans slightly worse than sigreg only in-distribution (57.5–59.5% success / 53–57% collision on three training seeds vs 62.5% / 50.0%). jac and jac+pull stay best (81.5% success). Recolor doesn't affect CEM for any model; rotation hurts all ([section](#planning-raw-cem-on-dubins)).
- #oct-18 **Targeted Jacobian reg (first-order λ penalty, no adversary) doesn't reproduce the adversarial gains** ([section](#targeted-jacobian-reg-the-first-order-version-without-an-adversary)). Over 3 WM seeds: Dubins color 0.729 ± 0.164, Safety Gym color 0.880 / rotation 0.865 with thresholds lost. Prediction loss is 10–30× higher (0.13–0.32). The finite-step adversarial search is doing real work.
- **It generalizes rather than covering the test.** An 8/255 per-pixel budget cannot reach purple (red → purple needs a ~0.5 change per channel), and noise contains no shape or rotation change, so all three shifts are outside the training perturbation.
- **Encoder Jacobians (new, [section below](#encoder-jacobians-across-models)):** L∞ shrinks the Jacobian about 4× but keeps baseline's low-rank, spiky shape, and the readout's pixel sensitivity drops 8×. The Jacobian penalty instead whitens J (condition number ~4, ~85–90 effective dimensions). Local sensitivity alone doesn't predict color robustness: plain jacobian has the lowest readout sensitivity of all and only 0.673 color AUC.
- **Still unsolved on Dubins: the threshold under recolor.** Under purple, no Dubins model keeps its safe/unsafe cutoff (on Safety Gym, LeWM + L∞ does): on a 50/50 safe/unsafe test set every model scores 0.500–0.505 sign accuracy, including all three L∞ models. Which way it breaks (everything safe vs everything unsafe) varies with the encoder and the head's training set. Jac+pull's earlier "color survival" was this artifact. Under shape and rotation, every L∞ model keeps a working threshold (≥ 0.958 on 50/50).
- **A structured color (hue) perturbation, tried first, did worse** and is covered in Appendix B.
- **Costs and open questions:** prediction loss goes up: L∞ 0.0113, jac+pull + L∞ 0.0165, no-h 0.0253, vs 0.0067 for a comparable run without the regularizer. Planning and the reachability critic haven't been tested. One training seed per model.

## Results

Zero-shot margin_gp head: trained on the normal (red, circle, upright) appearance, scored on the shifted appearance.
- **Sign accuracy:** does the head's sign (h ≥ 0 = safe) match the label?
- **AUC:** does the safe/unsafe ranking survive, independent of the threshold? (0.500 = chance, below = inverted.)

All values are mean ± std over 3 eval seeds; in-dist is from the color run. Eval states are uniform over the world, about 17% unsafe (circle) or 11% (diamond).

#oct-18 **World-model training seeds.** Every comparison table in this report shows one trained world model per row (training seed 3072); its ± is the std over 3 eval seeds (test-state samples and margin-head initializations). Six models were also retrained with seeds 1 and 2. Their results, with ± across world-model training seeds, are in separate tables in [Training-seed variability](#training-seed-variability).

**Sign accuracy**

| model | in-dist | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| baseline (sigreg only) | 0.943 ± 0.005 | 0.170 ± 0.006 | 0.310 ± 0.025 | 0.830 ± 0.006 |
| jacobian | 0.968 ± 0.002 | 0.171 ± 0.007 | 0.968 ± 0.007 | 0.962 ± 0.004 |
| jac+pull | 0.959 ± 0.003 | 0.830 ± 0.007 | 0.957 ± 0.008 | 0.939 ± 0.003 |
| baseline + L∞ safety reg | 0.981 ± 0.004 | 0.174 ± 0.006 | 0.980 ± 0.003 | 0.975 ± 0.004 |
| jac+pull + L∞ safety reg | 0.973 ± 0.003 | 0.830 ± 0.006 | 0.978 ± 0.005 | 0.970 ± 0.009 |
| baseline + L∞ safety reg w/o h | 0.977 ± 0.005 | 0.170 ± 0.006 | 0.977 ± 0.003 | 0.977 ± 0.003 |

**AUC**

| model | in-dist | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| baseline (sigreg only) | 0.979 ± 0.002 | 0.144 ± 0.032 | 0.913 ± 0.024 | 0.956 ± 0.023 |
| jacobian | 0.992 ± 0.001 | 0.673 ± 0.076 | 0.992 ± 0.003 | 0.989 ± 0.002 |
| jac+pull | 0.983 ± 0.001 | 0.753 ± 0.049 | 0.977 ± 0.007 | 0.978 ± 0.003 |
| baseline + L∞ safety reg | 0.998 ± 0.001 | 0.910 ± 0.098 | 0.997 ± 0.001 | 0.996 ± 0.002 |
| jac+pull + L∞ safety reg | 0.996 ± 0.001 | 0.965 ± 0.013 | 0.994 ± 0.002 | 0.994 ± 0.002 |
| baseline + L∞ safety reg w/o h | 0.997 ± 0.001 | 0.951 ± 0.004 | 0.996 ± 0.001 | 0.997 ± 0.001 |

Each new run is compared against the matched model without the regularizer (same recipe, seed, 50 epochs). The second head type (margin_nogp) agrees for the L∞ model: color AUC 0.905 ± 0.121, shape ≥ 0.993, rotation ≥ 0.982. Retraining a head on the shifted appearance recovers every model to ≥ 0.930 AUC; the shift relocates the latent rather than destroying information.

### Reading sign accuracy under the color shift

Under purple, sign accuracy is misleading on its own. With ~17% unsafe states, a head that calls *everything safe* scores ≈ 0.830, and one that calls *everything unsafe* scores ≈ 0.170. Every model lands on one of those two values. To confirm which, I logged the fraction of states each zero-shot head predicts safe (`/data/seongbin/lewm/safeadv_results/predsafe_check/`):

| model | predicted safe under purple | true safe fraction |
|---|---|---|
| baseline (sigreg only) | 0.000 ± 0.000 | 0.830 ± 0.006 |
| jacobian | 0.001 ± 0.001 | 0.830 ± 0.006 |
| jac+pull | 0.976 ± 0.041 | 0.830 ± 0.006 |
| baseline + L∞ safety reg | not logged; sign accuracy 0.174 ± 0.006 equals the unsafe rate, so ≈ 0.000 (measured directly in the 50/50 eval) | 0.830 ± 0.006 |
| jac+pull + L∞ safety reg | not logged; sign accuracy 0.830 ± 0.006 equals the safe rate, so ≈ 1.000 | 0.830 ± 0.006 |
| baseline + L∞ safety reg w/o h | not logged; sign accuracy 0.170 ± 0.006 equals the unsafe rate, so ≈ 0.000 | 0.830 ± 0.006 |

So **no model keeps a usable threshold under the color shift**. On this eval set baseline, jacobian and L∞ call everything unsafe and jac+pull calls everything safe; but the 50/50 eval below shows the *direction* flips with the head's training set (jac+pull calls everything unsafe there), so only the collapse itself is a stable finding. What separates the models on color is AUC, i.e. whether the safe/unsafe *ranking* survives. This overturns the earlier claim that jac+pull "survives color": its 0.830 sign accuracy came from calling everything safe. Under shape and rotation the L∞ threshold does survive (sign accuracy 0.980 / 0.975).

### 50/50 safe/unsafe eval

The same eval with every set exactly balanced: heads trained on 3000 safe + 3000 unsafe states, tested on 1000 + 1000, each balanced on the label it's scored against (circle for in-dist, diamond for shape zero-shot). States are rejection-sampled from the same uniform distribution, rendered once per axis and seed, and shared by all models. Here, both "all safe" and "all unsafe" score 0.500 sign accuracy, so sign accuracy reads directly. Mean ± std over 3 eval seeds (`balanced_eval/`).

**Sign accuracy (50/50: .500 = threshold gone)**

| model | in-dist | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| baseline (sigreg only) | 0.940 ± 0.003 | 0.500 ± 0.000 | 0.585 ± 0.010 | 0.500 ± 0.000 |
| jacobian | 0.963 ± 0.003 | 0.500 ± 0.000 | 0.952 ± 0.003 | 0.946 ± 0.009 |
| jac+pull | 0.940 ± 0.005 | 0.503 ± 0.005 | 0.919 ± 0.005 | 0.919 ± 0.013 |
| baseline + L∞ safety reg | 0.980 ± 0.004 | 0.501 ± 0.001 | 0.974 ± 0.003 | 0.959 ± 0.008 |
| jac+pull + L∞ safety reg | 0.974 ± 0.004 | 0.505 ± 0.005 | 0.961 ± 0.003 | 0.958 ± 0.011 |
| baseline + L∞ safety reg w/o h | 0.977 ± 0.006 | 0.500 ± 0.000 | 0.961 ± 0.002 | 0.968 ± 0.009 |

**AUC**

| model | in-dist | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| baseline (sigreg only) | 0.985 ± 0.002 | 0.177 ± 0.032 | 0.911 ± 0.029 | 0.902 ± 0.041 |
| jacobian | 0.994 ± 0.001 | 0.711 ± 0.069 | 0.992 ± 0.001 | 0.990 ± 0.003 |
| jac+pull | 0.987 ± 0.002 | 0.747 ± 0.063 | 0.981 ± 0.001 | 0.983 ± 0.004 |
| baseline + L∞ safety reg | 0.999 ± 0.001 | 0.927 ± 0.021 | 0.998 ± 0.000 | 0.996 ± 0.002 |
| jac+pull + L∞ safety reg | 0.997 ± 0.001 | 0.958 ± 0.012 | 0.995 ± 0.001 | 0.992 ± 0.004 |
| baseline + L∞ safety reg w/o h | 0.998 ± 0.002 | 0.955 ± 0.002 | 0.997 ± 0.001 | 0.996 ± 0.001 |

**Predicted-safe fraction (true = .500)**

| model | in-dist | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| baseline (sigreg only) | 0.497 ± 0.003 | 0.000 ± 0.000 | 0.085 ± 0.010 | 1.000 ± 0.000 |
| jacobian | 0.502 ± 0.002 | 0.000 ± 0.000 | 0.471 ± 0.002 | 0.505 ± 0.006 |
| jac+pull | 0.492 ± 0.008 | 0.003 ± 0.005 | 0.480 ± 0.012 | 0.455 ± 0.010 |
| baseline + L∞ safety reg | 0.501 ± 0.003 | 0.001 ± 0.001 | 0.509 ± 0.005 | 0.522 ± 0.001 |
| jac+pull + L∞ safety reg | 0.503 ± 0.001 | 0.995 ± 0.005 | 0.514 ± 0.003 | 0.508 ± 0.004 |
| baseline + L∞ safety reg w/o h | 0.503 ± 0.002 | 0.000 ± 0.000 | 0.520 ± 0.007 | 0.510 ± 0.009 |

What it shows:
- **Confirms:** every model's color threshold collapses (sign accuracy 0.500–0.503). L∞ has the best ranking on every shift: color AUC 0.927 ± 0.021 (less seed-variable than in the uniform eval), shape 0.998, rotation 0.996. L∞ keeps working thresholds on shape (0.974) and rotation (0.959).
- **Corrects (direction of collapse):** jac+pull now calls almost everything *unsafe* under purple (predicted safe 0.003 ± 0.005), whereas in the uniform eval it called everything *safe* (0.976). Same encoder, different head training set. The collapse is a property of the encoder; its direction is not.
- **New models:** jac+pull + L∞ (color AUC 0.958 ± 0.012) and no-h (0.955 ± 0.002) both beat L∞ (0.927 ± 0.021) on color and match it elsewhere. Both still lose the color threshold, in opposite directions: jac+pull + L∞ calls almost everything *safe* (0.995), no-h calls everything *unsafe* (0.000).
- **Corrects (baseline rotation):** baseline's rotation threshold also collapses (sign accuracy 0.500, everything called safe). In the uniform eval its 0.830 rotation sign accuracy equalled the all-safe rate, which hid this.

*Per-seed values for every retrained model are in [Training-seed variability](#training-seed-variability).*

## Ablations: what makes the no-h regularizer work

Four runs (code `9dfca47`). Each is **the baseline model (sigreg only, same recipe and seed) plus the no-h regularizer with one change**: no Jacobian/pull terms, and the head is not in the penalty. They are compared with the full no-h model and with the L∞ safety reg w/ h (the head-projected version):
- **random noise:** random ±ε noise at every pixel, the same size as PGD's iterates but with no adversarial search (`pgd_steps: 0, rand_init: sign`);
- **detached denominator:** the centroid-distance normalizer is detached, so it keeps its scale but gives no reward for separating the classes (`scale_grad: false`);
- **encoder term only:** `pred_weight: 0`;
- **predictor term only:** `enc_weight: 0`.

Zero-shot AUC, mean ± std over 3 eval seeds; shape and rotation from the uniform eval:

| model | color, uniform | color, 50/50 | shape | rotation | final train pred_loss |
|---|---|---|---|---|---|
| L∞ safety reg w/o h | 0.951 ± 0.004 | 0.955 ± 0.002 | 0.996 ± 0.001 | 0.997 ± 0.001 | 0.0253 |
| w/o h, random noise (no PGD) | 0.631 ± 0.030 | 0.622 ± 0.026 | 0.997 ± 0.002 | 0.996 ± 0.001 | 0.0140 |
| w/o h, detached denominator | 0.643 ± 0.022 | 0.627 ± 0.004 | 0.991 ± 0.002 | 0.991 ± 0.005 | 0.1070 |
| w/o h, encoder term only | 0.703 ± 0.269 | 0.583 ± 0.292 | 0.997 ± 0.001 | 0.996 ± 0.001 | 0.0225 |
| w/o h, predictor term only | 0.691 ± 0.287 | 0.580 ± 0.394 | 0.996 ± 0.001 | 0.997 ± 0.001 | 0.0265 |
| w/o h penalty, attack through h #oct-18 | 0.251 ± 0.081 | 0.237 ± 0.041 | 0.996 ± 0.001 | 0.996 ± 0.001 | 0.0188 |
| *reference: L∞ safety reg w/ h* | *0.910 ± 0.098* | *0.927 ± 0.021* | *0.997 ± 0.001* | *0.996 ± 0.002* | *0.0113* |

Per eval seed (color AUC, uniform / 50/50):
- encoder only: 0.949 / 0.920, 0.416 / 0.398, 0.745 / 0.432;
- predictor only: 0.797 / 0.125, 0.366 / 0.801, 0.911 / 0.815;
- full no-h: 0.948, 0.955, 0.949 (uniform).

- **Every ingredient is needed for color; none is needed for shape and rotation.** All four ablations keep shape and rotation at ≥ 0.991 AUC, the same as the full model. On color, each one falls from 0.951 to between 0.63 and 0.70 AUC: still far above baseline (0.144), but most of the gain is lost.
- **The adversary matters.** Random noise of the same per-pixel size gives 0.631 ± 0.030. It is also the cheapest variant on prediction (pred_loss 0.0140). It doesn't separate the classes much either (centroid distance² 55 at the end, vs ~350 for the full model).
- **The separation reward matters, and without it the penalty fights prediction.** With the denominator detached, the only way to lower the penalty is real invariance. The penalty stays an order of magnitude higher (0.22 vs ~0.02), pred_loss is 4× the full model's (0.107), and color AUC is 0.643.
- **Each term alone is unstable; together they're stable.** Encoder-only and predictor-only reach about 0.70 on average, but individual eval seeds (fresh state samples and head initializations) range from 0.12 to 0.95. Whether the color ranking survives then depends on the particular head. Only with both terms is every head 0.95.
- **Caveat:** one training seed per variant. The *spread* across eval seeds is a property of the encoder; the gap between full no-h and the single-term variants could still partly be training-seed luck.

### The same ablations with h

#oct-18

The head-projected L∞ safety reg (gate 0.95, training seed 3072, code `cfa5e52`) with one change each:
- #oct-18 **random noise:** ±ε random signs, no PGD;
- #oct-18 **detached gap:** the head's safe/unsafe gap in the denominator is detached (`scale_grad: false`);
- #oct-18 **encoder term only:** `pred_weight: 0`;
- #oct-18 **predictor term only:** `enc_weight: 0`.

#oct-18 Zero-shot AUC, Dubins:

| model | color, uniform | color, 50/50 | shape | rotation | final train pred_loss |
|---|---|---|---|---|---|
| L∞ safety reg w/ h (reference) | 0.910 ± 0.098 | 0.927 ± 0.021 | 0.997 ± 0.001 | 0.996 ± 0.002 | 0.0113 |
| w/ h, random noise (no PGD) | 0.300 ± 0.024 | 0.237 ± 0.038 | 0.996 ± 0.002 | 0.555 ± 0.077 | 0.0067 |
| w/ h, detached gap | 0.778 ± 0.041 | 0.779 ± 0.024 | 0.992 ± 0.001 | 0.961 ± 0.017 | 0.0092 |
| w/ h, encoder term only | 0.277 ± 0.019 | 0.272 ± 0.021 | 0.997 ± 0.002 | 0.997 ± 0.001 | 0.0107 |
| w/ h, predictor term only | 0.536 ± 0.048 | 0.485 ± 0.067 | 0.996 ± 0.001 | 0.998 ± 0.001 | 0.0107 |

- #oct-18 **Read color against the reference's training-seed range (0.42–0.91).** Each ablation has one training seed.
- #oct-18 **Encoder-only (0.277) and random noise (0.300) fall below that range,** so dropping the predictor term or the adversary likely does hurt.
- #oct-18 **The detached gap (0.778) and predictor-only (0.536) sit inside it,** so they are not distinguishable from the full model.
- #oct-18 **Shape and rotation stay ≥ 0.992,** except random noise, which breaks rotation (0.555).
- #oct-18 **Attack through h, penalize the whole latent** (`attack_target: head`, `target: latent`, row in the table above): PGD looks for the δ that moves the safety score most, then the model is trained to keep the *whole* latent still under it. Dubins color drops to 0.251, against 0.852 ± 0.098 when the attack and the penalty both use the whole latent. On Safety Gym it gets color 0.938 / rotation 0.829 (vs 0.985 / 0.941), with pred_loss 0.081. The adversary has to search for the same quantity the penalty protects.

Jacobians, same protocol as the section below:

| model | σ1 | ‖J‖_F | cond | eff. rank | top-5 energy | ‖J‖/cdist | ‖∇h‖/gap | color AUC |
|---|---|---|---|---|---|---|---|---|
| L∞ safety reg w/o h | 1.202 | 1.869 | 96.6 | 16.5 | 0.809 | 0.100 | 0.0256 | 0.951 |
| w/o h, random noise (no PGD) | 1.373 | 2.982 | 50.7 | 22.6 | 0.670 | 0.347 | 0.0338 | 0.631 |
| w/o h, detached denominator | 1.342 | 2.026 | 451.4 | 8.1 | 0.913 | 0.105 | 0.0415 | 0.643 |
| w/o h, encoder term only | 0.694 | 1.601 | 23.4 | 34.0 | 0.578 | 0.184 | 0.0133 | 0.703 |
| w/o h, predictor term only | 1.276 | 1.993 | 100.8 | 15.0 | 0.851 | 0.151 | 0.0250 | 0.691 |

- **The local Jacobian again doesn't predict color.** Encoder-only has the smallest and best-conditioned Jacobian of the five and the least sensitive readout (0.0133), yet unstable color.
- **Detached denominator:** about as small relative to class separation as the full model (0.105), but extremely anisotropic (condition number 451, 8 effective dimensions) with the most sensitive readout. Without the separation route, the encoder squeezes sensitivity into a few directions.
- **Random noise:** pixel sensitivity relative to class separation is 3.5× the full model's (0.347). That fits a weaker regularizer that doesn't push the classes apart.

## Safety Gym, and the regularizer on Dreamer

### Safety Gym (SafetyCarGoal1, top-down), LeWM and Dreamer

Same regularizer settings as on Dubins (L∞, ε = 8/255, PGD 2 steps, 32 windows, ramp 10k, weight 1), same LeWM recipe as the Safety Gym references (sigreg, seed 3072, 50 epochs, data `sg_cargoal_safety` = the reference dataset plus its `failures` labels).

- **One deviation for LeWM: no AUC gate** (`auc_gate: 0`, so the penalty opens at step 2000). On the Safety Gym training data the sigreg-only latent doesn't generalize safety across episodes: an MLP probe fit on its latents scores 0.981 on train episodes but 0.629 on held-out ones (jac: 0.985). With the gate, the in-training head stays at AUC ~0.56 and the penalty never switches on, so the gate was removed. Once the ungated penalty engaged, the in-training head went from AUC 0.55 to 0.99 within a few thousand steps: the class-separation term made safety decodable.
- **Dreamer kept the gate.** Its own head passed 0.95 at about step 2k on Safety Gym and 5–6k on Dubins.

Eval: the layout-controlled protocol (`sg_multicolor_train.npz`; every appearance variant is a render of the same pose; margin_gp head trained on 4000 blue-hazard frames, scored on the other 1000). Color = held-out purple hazards, rotation = a real 90° rotation of the camera (not an image rotation). 16.7% of test frames are unsafe, so "everything safe" scores 0.833 sign accuracy. There is no shape axis: the only Safety Gym shape eval is in the older protocol, which has a layout confound. Mean ± std over 3 eval seeds (head initialization and sampling; the test frames are fixed, so the spread is small). Script `safeadv_results/sg_eval/sg_plain_eval.py`; it reproduces the earlier AUCs, e.g. sigreg only 0.988 / 0.596 / 0.554 vs 0.994 / 0.604 / 0.574.

**Sign accuracy**

| model | in-dist | color (blue → purple) | rotation (real 90° camera) |
|---|---|---|---|
| LeWM sigreg only | 0.970 ± 0.007 | 0.833 ± 0.000 | 0.833 ± 0.000 |
| LeWM jac | 0.992 ± 0.002 | 0.921 ± 0.002 | 0.832 ± 0.001 |
| LeWM jac+pull | 0.997 ± 0.000 | 0.943 ± 0.004 | 0.835 ± 0.002 |
| LeWM + L∞ safety reg | 0.997 ± 0.002 | 0.987 ± 0.003 | 0.977 ± 0.005 |
| LeWM + L∞ safety reg w/o h | 0.990 ± 0.007 | 0.943 ± 0.005 | 0.933 ± 0.006 |
| LeWM jac+pull + L∞ safety reg #oct-18 | 1.000 ± 0.001 | 0.973 ± 0.002 | 0.964 ± 0.003 |
| Dreamer | 0.981 ± 0.004 | 0.835 ± 0.003 | 0.833 ± 0.000 |
| Dreamer + L∞ safety reg | 0.983 ± 0.002 | 0.856 ± 0.008 | 0.833 ± 0.000 |
| Dreamer + L∞ safety reg w/o h | 0.982 ± 0.002 | 0.854 ± 0.005 | 0.833 ± 0.000 |

**AUC**

| model | in-dist | color (blue → purple) | rotation (real 90° camera) |
|---|---|---|---|
| LeWM sigreg only | 0.992 ± 0.003 | 0.595 ± 0.001 | 0.557 ± 0.006 |
| LeWM jac | 0.997 ± 0.001 | 0.992 ± 0.000 | 0.641 ± 0.006 |
| LeWM jac+pull | 0.999 ± 0.000 | 0.996 ± 0.001 | 0.679 ± 0.002 |
| LeWM + L∞ safety reg | 1.000 ± 0.000 | 0.999 ± 0.000 | 0.962 ± 0.001 |
| LeWM + L∞ safety reg w/o h | 0.999 ± 0.000 | 0.985 ± 0.001 | 0.941 ± 0.005 |
| LeWM jac+pull + L∞ safety reg #oct-18 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.985 ± 0.001 |
| Dreamer | 0.993 ± 0.001 | 0.961 ± 0.002 | 0.503 ± 0.047 |
| Dreamer + L∞ safety reg | 0.995 ± 0.001 | 0.977 ± 0.003 | 0.481 ± 0.043 |
| Dreamer + L∞ safety reg w/o h | 0.995 ± 0.001 | 0.979 ± 0.001 | 0.491 ± 0.025 |

**Predicted-safe fraction (true safe = .833)**

| model | in-dist | color (blue → purple) | rotation (real 90° camera) |
|---|---|---|---|
| LeWM sigreg only | 0.820 ± 0.005 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| LeWM jac | 0.833 ± 0.004 | 0.912 ± 0.002 | 0.999 ± 0.001 |
| LeWM jac+pull | 0.832 ± 0.000 | 0.888 ± 0.004 | 0.998 ± 0.002 |
| LeWM + L∞ safety reg | 0.830 ± 0.002 | 0.823 ± 0.006 | 0.833 ± 0.006 |
| LeWM + L∞ safety reg w/o h | 0.831 ± 0.006 | 0.890 ± 0.005 | 0.858 ± 0.014 |
| LeWM jac+pull + L∞ safety reg #oct-18 | 0.833 ± 0.001 | 0.860 ± 0.002 | 0.863 ± 0.003 |
| Dreamer | 0.828 ± 0.005 | 0.995 ± 0.002 | 1.000 ± 0.000 |
| Dreamer + L∞ safety reg | 0.832 ± 0.003 | 0.977 ± 0.007 | 1.000 ± 0.000 |
| Dreamer + L∞ safety reg w/o h | 0.832 ± 0.002 | 0.977 ± 0.005 | 1.000 ± 0.000 |

- **LeWM + L∞ safety reg is the first model in the whole study whose threshold survives a shift.** Its zero-shot head keeps 0.987 sign accuracy on color and 0.977 on rotation, predicting 0.823 / 0.833 of states safe (true: 0.833). Every reference collapses to calling nearly everything safe.
- #oct-18 **jac+pull + L∞ safety reg is the best Safety Gym model, and it is stable across training seeds.** Training seed 3072: color AUC 1.000, rotation 0.985, sign accuracy 0.973 / 0.964. Over 3 WM seeds: color 1.000 ± 0.000 and rotation 0.981 ± 0.018, with the threshold holding on every seed (sign accuracy 0.979 ± 0.010 / 0.962 ± 0.021; predicted safe 0.85–0.87 vs 0.833 true). The L∞ safety reg alone loses rotation on one of its three seeds. The cost is higher prediction loss (0.032–0.037 vs 0.020 for the L∞ safety reg alone and 0.0026 for sigreg only). It ran with the usual AUC gate, which opened, because jac+pull's latent generalizes safety across episodes.
- **It also fixes rotation**, which nothing else did: AUC 0.962 vs 0.679 for jac+pull and 0.557 for sigreg only. Color goes to 0.999.
- **No-h works here but is weaker** (color 0.985, rotation 0.941, thresholds partly hold at 0.943 / 0.933), and it is much costlier (below).
- **Dreamer gains a little on color** (0.961 → 0.977 / 0.979). Rotation doesn't move (~0.49) and thresholds still collapse.
- **Prediction cost:** final train pred_loss is 0.0168 for L∞ and 0.090 for no-h, vs 0.0026 for sigreg only (6× and 35×). Dreamer's image reconstruction loss is unchanged (32.0 / 32.8 vs 32.3 for the Dreamer baseline).

### Dubins Dreamer

The Dubins Dreamer reference is the same baseline used in earlier Dreamer comparisons (`enc_lip_sweep/baseline`). It uses the same recipe as the new runs: 40k steps, batch 32 × 16, seed 0. Eval: the same uniform-state protocol as the LeWM tables (`ood_margin_gp_dreamer.py`, ~17% unsafe for circle).

**Sign accuracy**

| model | in-dist | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| Dreamer | 0.986 ± 0.002 | 0.780 ± 0.043 | 0.936 ± 0.007 | 0.708 ± 0.007 |
| Dreamer + L∞ safety reg | 0.988 ± 0.003 | 0.793 ± 0.024 | 0.937 ± 0.011 | 0.708 ± 0.005 |
| Dreamer + L∞ safety reg w/o h | 0.985 ± 0.001 | 0.726 ± 0.074 | 0.929 ± 0.005 | 0.715 ± 0.010 |

**AUC**

| model | in-dist | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| Dreamer | 0.999 ± 0.000 | 0.638 ± 0.013 | 0.993 ± 0.001 | 0.480 ± 0.011 |
| Dreamer + L∞ safety reg | 0.999 ± 0.000 | 0.816 ± 0.017 | 0.995 ± 0.001 | 0.484 ± 0.013 |
| Dreamer + L∞ safety reg w/o h | 0.999 ± 0.000 | 0.626 ± 0.048 | 0.994 ± 0.001 | 0.494 ± 0.006 |

- **The head-projected variant helps color** (AUC 0.638 → 0.816), and nothing else moves.
- **No-h does nothing on Dubins Dreamer** (color 0.626). So for Dreamer the safety projection matters, the opposite of LeWM on Dubins.
- **Rotation stays at chance** (~0.48–0.49) for every Dreamer variant on both benchmarks. Dreamer also gets the true heading as a separate non-image input, which the perturbation never touches.
- **No usable threshold.** Dubins sign accuracy under color is about 0.78 and under rotation about 0.71, both below what calling every state safe would score (~0.83).
- Final image reconstruction loss is 3.3 (head) / 3.2 (no h). The baseline's training log wasn't kept, so there is no direct comparison.

### How the regularizer was ported to Dreamer

In `latent_cbf/src/latent_cbf/scripts/dreamer_offline.py`, flags `--sa_*` (code `e99b14c`):
- **Encoder term:** the posterior feature f(post_t). **Prediction term:** the prior feature f(prior_t), the one-step RSSM prediction from the perturbed history.
- **Readout:** for the head variant, Dreamer's own in-training `margin_gp` head with its parameters frozen in the penalty. For no-h, the whole 544-d feature, normalized by the safe/unsafe centroid distance².
- **Perturbation:** δ is one L∞ field per sequence, shared over time, applied to 8 of the 32 sequences on Dubins (×16 frames) and 16 on Safety Gym (×8 frames).
- **Same RNG seed:** clean and perturbed passes reuse it, so the RSSM's stochastic samples cancel. Checked: zero δ gives exactly zero deviation.
- **Frame 0 is excluded from label statistics,** because the loader forces its label to safe.
- **No BatchNorm handling:** Dreamer has none.
- **Corrected labels:** the Safety Gym Dreamer data stored each frame's label one step off from the loader's convention. The new runs train on a corrected copy (`sg_cargoal_dreamer_lab.h5`). The Dreamer baseline used the original file, but its eval trains fresh heads on correctly labelled eval data, so this doesn't affect its numbers.

## Safety-direction (λ) projection instead of the head

**Idea.** The head variant penalizes a perturbation's effect through the learned margin head h, which in effect reads the latent along a direction that differs per sample. This variant uses one shared **safety direction**, λ = mean(z of safe frames) − mean(z of unsafe frames). It only penalizes the component of the perturbation-induced shift along λ, i.e. it asks the shift to be **orthogonal** to the safety direction:

    d = mean_t [((f(x_t + δ) − f(x_t)) · λ_n) / ‖λ‖]² + [((P(f(x + δ), a) − P(f(x), a)) · λ_n) / ‖λ‖]²  where
    λ = mean(z of safe frames) − mean(z of unsafe frames),  λ_n = λ / ‖λ‖ (unit vector along λ)

- Each term is the move along λ in units of the safe/unsafe distance, and is zero exactly when the move is orthogonal to λ.
- The adversary is unchanged (L∞, 8/255, PGD maximizing this d).
- There is no head in the penalty and so no AUC gate: the penalty opens at step 2000 and ramps up over 10k steps.
- Code: `target: lambda`, commit `32263e1`.

Two ways to get λ:
- **batch λ:** direction and length from the current batch's class means, with gradient.
- **global λ:** direction from running averages of the safe and unsafe means over all batches (decay 0.99, detached); length ‖λ‖ from the batch, with gradient.

Zero-shot AUC; Dubins from the uniform eval, mean ± std over 3 eval seeds:

| model | Dubins color | Dubins shape | Dubins rotation | Safety Gym color | Safety Gym rotation | final pred_loss (Dubins / Safety Gym) |
|---|---|---|---|---|---|---|
| sigreg only | 0.144 ± 0.032 | 0.913 ± 0.024 | 0.956 ± 0.023 | 0.595 ± 0.001 | 0.557 ± 0.006 | — / 0.0026 |
| L∞ safety reg w/ h (per-sample direction) | 0.910 ± 0.098 | 0.997 ± 0.001 | 0.996 ± 0.002 | 0.999 ± 0.000 | 0.962 ± 0.001 | 0.0113 / 0.0168 |
| L∞ safety reg, λ-projected, batch λ | 0.639 ± 0.345 | 0.996 ± 0.002 | 0.896 ± 0.013 | 0.941 ± 0.009 | 0.855 ± 0.009 | 0.0112 / 0.0242 |
| L∞ safety reg, λ-projected, batch λ, detached direction #oct-18 | 0.948 ± 0.013 | 0.996 ± 0.001 | 0.996 ± 0.001 | 0.992 ± 0.004 | 0.962 ± 0.001 | 0.0106 / 0.0221 |
| L∞ safety reg, λ-projected, global λ | 0.936 ± 0.041 | 0.995 ± 0.002 | 0.995 ± 0.002 | 0.841 ± 0.001 | 0.445 ± 0.008 | 0.0081 / 0.2342 |
| L∞ safety reg, λ-projected, global λ, projected scale | 0.950 ± 0.046 | 0.996 ± 0.002 | 0.992 ± 0.003 | 0.775 ± 0.021 | 0.413 ± 0.001 | 0.008 / 0.013 |

Zero-shot sign accuracy (h ≥ 0 = safe), mean ± std over 3 eval seeds. Dubins from the 50/50 eval, where 0.500 means the threshold is gone; Safety Gym, where calling every state safe scores 0.833:

| model | Dubins in-dist | Dubins color | Dubins shape | Dubins rotation | Safety Gym in-dist | Safety Gym color | Safety Gym rotation |
|---|---|---|---|---|---|---|---|
| sigreg only | 0.940 ± 0.003 | 0.500 ± 0.000 | 0.585 ± 0.010 | 0.500 ± 0.000 | 0.970 ± 0.007 | 0.833 ± 0.000 | 0.833 ± 0.000 |
| L∞ safety reg w/ h (per-sample direction) | 0.980 ± 0.004 | 0.501 ± 0.001 | 0.974 ± 0.003 | 0.959 ± 0.008 | 0.997 ± 0.002 | 0.987 ± 0.003 | 0.977 ± 0.005 |
| L∞ safety reg, λ-projected, batch λ | 0.976 ± 0.003 | 0.501 ± 0.004 | 0.959 ± 0.003 | 0.627 ± 0.049 | 0.990 ± 0.005 | 0.856 ± 0.007 | 0.836 ± 0.001 |
| L∞ safety reg, λ-projected, batch λ, detached direction #oct-18 | 0.977 ± 0.007 | 0.500 ± 0.000 | 0.968 ± 0.005 | 0.951 ± 0.017 | 0.997 ± 0.002 | 0.992 ± 0.001 | 0.983 ± 0.001 |
| L∞ safety reg, λ-projected, global λ | 0.977 ± 0.002 | 0.500 ± 0.000 | 0.963 ± 0.002 | 0.941 ± 0.015 | 0.892 ± 0.006 | 0.844 ± 0.007 | 0.802 ± 0.008 |
| L∞ safety reg, λ-projected, global λ, projected scale | 0.974 ± 0.003 | 0.690 ± 0.172 | 0.960 ± 0.007 | 0.920 ± 0.018 | 0.950 ± 0.003 | 0.843 ± 0.004 | 0.829 ± 0.003 |

- **Dubins:** every λ variant loses the color threshold except global λ with projected scale, which partly keeps it (0.690). Its spread across eval seeds is large and it has one training seed. Batch λ also loses the rotation threshold (0.627).
- **Safety Gym:** all λ variants collapse toward calling everything safe (0.80–0.86, near the 0.833 all-safe rate). Only the L∞ safety reg w/ h keeps its thresholds (0.987 / 0.977).

The full uniform, 50/50 and Safety Gym tables (sign accuracy, AUC, predicted-safe fraction) include both λ rows. On the 50/50 eval, Dubins color AUC is 0.819 ± 0.025 for batch λ and 0.892 ± 0.069 for global λ.

- #oct-18 **Batch λ with a detached direction fixes batch λ** (`lambda_mode: batch_detached`, one training seed). The direction comes from the current batch's class means but carries no gradient; the scale ‖λ_batch‖ keeps its gradient. On Dubins, color AUC is 0.948 ± 0.013 (eval seeds 0.94 / 0.94 / 0.96), against 0.639 ± 0.345 for batch λ, and rotation is back to 0.996 (batch λ: 0.896). On Safety Gym it reaches color 0.992 and rotation 0.962, and it is the first λ variant whose thresholds survive (sign accuracy 0.992 / 0.983; batch λ 0.856 / 0.836). Prediction loss is low (0.0106 / 0.0221). The gradient through the direction is what broke batch λ, presumably because it let the encoder rotate the protected axis. A noisy per-batch direction is fine once it is detached. The Dubins color threshold still collapses (50/50 sign accuracy 0.500). #oct-18 Over 3 training seeds it holds on Safety Gym (color 0.993 ± 0.007, rotation 0.945 ± 0.023, thresholds kept on every seed) but not on Dubins color (0.721 ± 0.434, one seed at 0.220); see [Training-seed variability](#training-seed-variability).
- **Dubins, global λ works.** Color AUC is 0.936 (seeds 0.98 / 0.92 / 0.90), above the L∞ safety reg w/ h (0.910). Shape and rotation stay intact. It also has the lowest prediction cost of any safety-reg model (pred_loss 0.0081), and its running direction agrees with the batch direction (cos ≈ 0.89–0.94).
- **Dubins, batch λ is unstable and hurts rotation.** Color AUC by eval seed is 0.83 / 0.84 / 0.24, and rotation drops to 0.896, below sigreg only (0.956). Having gradient through a noisy per-batch direction lets the encoder move the axis it is being protected along.
- **Safety Gym: neither λ variant matches the head.**
  - **Batch λ** reaches 0.941 / 0.855, with thresholds collapsing to "all safe" (predicted safe 0.977 / 0.993).
  - **Global λ is degenerate,** as flagged during training. The sigreg-only Safety Gym latent has no consistent safety axis to start from: the batch and running directions were nearly uncorrelated (cos 0.03) before the penalty opened. The penalty therefore pinned an arbitrary direction. In-dist AUC fell to 0.945, rotation to 0.445 and pred_loss rose to 0.234.
  - **The per-sample head** is the only variant that keeps its threshold on Safety Gym (sign accuracy 0.987 / 0.977).
- **Takeaway.** One linear safety direction is enough only when the base latent already separates the classes linearly (Dubins). There the global version is as good as the head at a lower cost. Where it doesn't (Safety Gym), a learned, sample-dependent readout or the whole latent is needed.
- **Global λ with projected scale (follow-up, code `cfa5e52`, `lambda_mode: global_proj`).** In plain global λ, the scale ‖λ‖ came from the whole batch gap, so the encoder got credit for separating the classes in any direction. Here the scale is the batch gap projected onto the global direction, (λ_batch · λ̂), floored at 0.1 × the global gap length. Separation only counts along the protected axis.
  - **On Safety Gym it fixes training but not out-of-distribution performance.** Batch and global directions now agree (cos 0.87 vs 0.03–0.19), the side-trained head reaches AUC 1.000, and pred_loss is 0.013 instead of 0.234. But zero-shot is still poor: color 0.775, rotation 0.413, in-dist 0.981. So the earlier failure was not only the scale loophole: one fixed linear axis is too little protection on Safety Gym.
  - **On Dubins it is the best λ variant.** Color AUC is 0.950 (seeds 0.99 / 0.90 / 0.97; 50/50: 0.940), with the lowest prediction cost (0.008). On the 50/50 eval its color threshold partially survives (sign accuracy 0.690 ± 0.172, vs 0.500 for every other model), though inconsistently across seeds.

## Adversarial vs. random noise of the same size

Each adversarial safety-reg model was rerun with **random** noise of the same size: every pixel gets ±8/255 with a random sign (`pgd_steps: 0, rand_init: sign`). That is the same size as the PGD iterates, but with no search for the worst direction. Everything else matches the adversarial run, including the gate setting: gated on Dubins, ungated on Safety Gym. The "with h" column uses the head-projected penalty; "no h" penalizes the whole latent shift.

Zero-shot AUC, mean ± std over 3 eval seeds (Dubins: uniform eval):

| | Dubins color | Dubins shape | Dubins rotation | Safety Gym color | Safety Gym rotation |
|---|---|---|---|---|---|
| sigreg only | 0.144 ± 0.032 | 0.913 ± 0.024 | 0.956 ± 0.023 | 0.595 ± 0.001 | 0.557 ± 0.006 |
| L∞ safety reg | 0.910 ± 0.098 | 0.997 ± 0.001 | 0.996 ± 0.002 | 0.999 ± 0.000 | 0.962 ± 0.001 |
| L∞ safety reg, random noise | 0.300 ± 0.024 | 0.996 ± 0.002 | 0.555 ± 0.077 | 0.871 ± 0.002 | 0.729 ± 0.003 |
| L∞ safety reg w/o h | 0.951 ± 0.004 | 0.996 ± 0.001 | 0.997 ± 0.001 | 0.985 ± 0.001 | 0.941 ± 0.005 |
| L∞ safety reg w/o h, random noise | 0.631 ± 0.030 | 0.997 ± 0.002 | 0.996 ± 0.001 | 0.891 ± 0.011 | 0.802 ± 0.004 |
| *jac+pull (reference)* | *0.753 ± 0.049* | *0.977 ± 0.007* | *0.978 ± 0.003* | *0.996 ± 0.001* | *0.679 ± 0.002* |
| *jac+pull (reference, 3 WM seeds)* | *0.498 ± 0.234* | *0.982 ± 0.004* | *0.982 ± 0.003* | — | — |

Sign accuracy under shift for the random runs:
- **Dubins:** color 0.170 (with h) and 0.176 (no h), both the all-unsafe rate; rotation 0.822 (with h) and 0.973 (no h).
- **Safety Gym:** color 0.868 / 0.859, rotation 0.841 / 0.859. Thresholds mostly collapse toward "all safe", where the adversarial with-h model keeps 0.987 / 0.977.

The 50/50 eval agrees. Dubins color AUC is 0.237 for with h + random, vs 0.927 adversarial.

- **The adversary is what does the work, everywhere.** Swapping PGD for random noise of the same size loses most of the out-of-distribution gain in every cell: Dubins color drops by 0.61 (with h) and 0.32 (no h); Safety Gym color by about 0.1 and rotation by 0.14–0.23.
- **Random noise through the head is actively harmful on Dubins.** Rotation falls to 0.555, far below sigreg only (0.956), and color is worse than every other safety-reg variant. Penalizing the head's response to random noise makes the head locally flat in random directions without protecting the directions that matter, and it apparently distorts the latent elsewhere. Without h, random noise is merely ineffective on Dubins (rotation intact).
- **Shape is unaffected** (≥ 0.996 everywhere): shape robustness on Dubins comes from any form of this training.
- **Random noise is cheap.** Final train pred_loss is 0.007 / 0.011 (with h, Dubins / Safety Gym) and 0.035 (no h, Safety Gym), vs 0.0113 / 0.0168 adversarial with h. Much of the prediction cost of the adversarial runs is the price of the robustness.
- **Training recipe:** code `cfa5e52`. Runs `lewm_dubins_safeadv_rand50`, `sg_safeadv_rand50_g0`, `sg_lz_rand50_g0`; the Dubins no-h random run is `lewm_dubins_lz_rand50` from the ablations.

## Noise size and attack strength

The head-variant L∞ safety reg on Dubins with only the adversary changed: ε = 4/255 and 16/255 (2 PGD steps, step 0.5ε), and ε = 8/255 with 5 PGD steps (step 0.25ε, a finer and stronger search). Everything else is identical to the reference run (gate 0.95, seed 3072, 50 epochs; code `cfa5e52`, which is behaviorally identical to the reference code for this configuration). Zero-shot AUC, mean ± std over 3 eval seeds; shape and rotation from the uniform eval:

| model | color, uniform | color, 50/50 | shape | rotation | final train pred_loss |
|---|---|---|---|---|---|
| sigreg only | 0.144 ± 0.032 | 0.177 ± 0.032 | 0.913 ± 0.024 | 0.956 ± 0.023 | — |
| random ±8/255, no search | 0.300 ± 0.024 | 0.237 ± 0.038 | 0.996 ± 0.002 | 0.555 ± 0.077 | 0.0067 |
| ε = 4/255, 2 steps | 0.771 ± 0.032 | 0.782 ± 0.027 | 0.997 ± 0.001 | 0.577 ± 0.105 | 0.0124 |
| ε = 8/255, 2 steps (reference) | 0.910 ± 0.098 | 0.927 ± 0.021 | 0.997 ± 0.001 | 0.996 ± 0.002 | 0.0113 |
| ε = 8/255, 2 steps (reference, 3 WM seeds) | 0.643 ± 0.247 | 0.662 ± 0.239 | 0.994 ± 0.004 | 0.975 ± 0.035 | 0.0121 ± 0.0015 |
| ε = 16/255, 2 steps | 0.305 ± 0.042 | 0.296 ± 0.026 | 0.984 ± 0.002 | 0.910 ± 0.020 | 0.0134 |
| ε = 8/255, 5 steps | 0.088 ± 0.047 | 0.286 ± 0.043 | 0.997 ± 0.001 | 0.992 ± 0.003 | 0.0116 |

- **Color is not monotonic in the attack.** Raising ε from 8/255 to 16/255 drops color AUC from 0.910 to 0.305. A stronger search at the same ε (5 steps) drops it to 0.088, below sigreg only, i.e. the purple ranking becomes inverted. 4/255 lands in between (0.771). Per eval seed: 4/255 is 0.80 / 0.74 / 0.78, 16/255 is 0.35 / 0.30 / 0.27, and 5 steps is 0.14 / 0.06 / 0.06. So the gaps between runs are far outside the eval-seed spread.
- **Rotation moves differently.** It is poor at 4/255 (0.577) and fine with a larger ε or more steps (0.910 / 0.992). A weak attack doesn't protect rotation, while a strong one does but loses color. Only the reference setting got both.
- **Shape is robust under every setting** (≥ 0.984).
- **Thresholds:** only ε = 4/255 keeps some color threshold on the 50/50 eval (sign accuracy 0.722 vs 0.500 for collapse; predicted safe 0.454). Every other run collapses on color, and ε = 4/255 and 16/255 also collapse on rotation.
- **Implication:** the reference's color result depends sharply on the adversary settings, and also on the training seed (below).

### Training-seed variability

Six models were retrained with world-model training seeds 1 and 2; seed 3072 is the original run. For each training seed, values are first averaged over its 3 eval seeds. Then the tables show mean ± std across the 3 training seeds.

Dubins, uniform eval:

**AUC**

| model (3 WM seeds) | in-dist | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| jac+pull | 0.987 ± 0.003 | 0.498 ± 0.234 | 0.982 ± 0.004 | 0.982 ± 0.003 |
| baseline + L∞ safety reg | 0.996 ± 0.002 | 0.643 ± 0.247 | 0.994 ± 0.004 | 0.975 ± 0.035 |
| jac+pull + L∞ safety reg | 0.996 ± 0.000 | 0.957 ± 0.021 | 0.994 ± 0.000 | 0.994 ± 0.001 |
| baseline + L∞ safety reg w/o h | 0.997 ± 0.001 | 0.852 ± 0.098 | 0.996 ± 0.001 | 0.997 ± 0.000 |
| L∞ safety reg, λ-projected, global λ, projected scale | 0.997 ± 0.001 | 0.612 ± 0.373 | 0.996 ± 0.001 | 0.988 ± 0.009 |

**Sign accuracy**

| model (3 WM seeds) | in-dist | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| jac+pull | 0.962 ± 0.003 | 0.828 ± 0.004 | 0.956 ± 0.013 | 0.951 ± 0.011 |
| baseline + L∞ safety reg | 0.977 ± 0.003 | 0.331 ± 0.252 | 0.978 ± 0.002 | 0.957 ± 0.029 |
| jac+pull + L∞ safety reg | 0.975 ± 0.002 | 0.830 ± 0.000 | 0.978 ± 0.001 | 0.971 ± 0.001 |
| baseline + L∞ safety reg w/o h | 0.978 ± 0.001 | 0.389 ± 0.382 | 0.978 ± 0.002 | 0.977 ± 0.002 |
| L∞ safety reg, λ-projected, global λ, projected scale | 0.977 ± 0.001 | 0.513 ± 0.352 | 0.976 ± 0.001 | 0.927 ± 0.057 |

Dubins, 50/50 eval:

**AUC**

| model (3 WM seeds) | in-dist | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| jac+pull | 0.990 ± 0.003 | 0.448 ± 0.266 | 0.985 ± 0.004 | 0.985 ± 0.002 |
| baseline + L∞ safety reg | 0.997 ± 0.002 | 0.662 ± 0.239 | 0.996 ± 0.003 | 0.974 ± 0.036 |
| jac+pull + L∞ safety reg | 0.997 ± 0.000 | 0.955 ± 0.017 | 0.995 ± 0.000 | 0.993 ± 0.001 |
| baseline + L∞ safety reg w/o h | 0.998 ± 0.000 | 0.849 ± 0.101 | 0.997 ± 0.000 | 0.996 ± 0.000 |
| L∞ safety reg, λ-projected, global λ, projected scale | 0.998 ± 0.000 | 0.574 ± 0.371 | 0.996 ± 0.001 | 0.983 ± 0.015 |

**Sign accuracy (.500 = threshold gone)**

| model (3 WM seeds) | in-dist | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| jac+pull | 0.950 ± 0.008 | 0.477 ± 0.030 | 0.924 ± 0.017 | 0.933 ± 0.012 |
| baseline + L∞ safety reg | 0.976 ± 0.003 | 0.498 ± 0.005 | 0.968 ± 0.009 | 0.927 ± 0.051 |
| jac+pull + L∞ safety reg | 0.974 ± 0.001 | 0.604 ± 0.176 | 0.964 ± 0.002 | 0.959 ± 0.002 |
| baseline + L∞ safety reg w/o h | 0.977 ± 0.001 | 0.497 ± 0.009 | 0.963 ± 0.002 | 0.965 ± 0.007 |
| L∞ safety reg, λ-projected, global λ, projected scale | 0.976 ± 0.002 | 0.564 ± 0.109 | 0.964 ± 0.004 | 0.842 ± 0.170 |

**Predicted-safe fraction (true = .500)**

| model (3 WM seeds) | in-dist | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| jac+pull | 0.498 ± 0.005 | 0.343 ± 0.513 | 0.474 ± 0.047 | 0.484 ± 0.026 |
| baseline + L∞ safety reg | 0.502 ± 0.001 | 0.212 ± 0.366 | 0.511 ± 0.005 | 0.554 ± 0.054 |
| jac+pull + L∞ safety reg | 0.503 ± 0.000 | 0.781 ± 0.376 | 0.511 ± 0.003 | 0.509 ± 0.002 |
| baseline + L∞ safety reg w/o h | 0.504 ± 0.002 | 0.336 ± 0.571 | 0.519 ± 0.002 | 0.514 ± 0.014 |
| L∞ safety reg, λ-projected, global λ, projected scale | 0.504 ± 0.002 | 0.194 ± 0.332 | 0.511 ± 0.007 | 0.628 ± 0.198 |

Per training seed, Dubins color AUC (uniform):

| model | seed 3072 | seed 1 | seed 2 |
|---|---|---|---|
| jac+pull | 0.753 | 0.292 | 0.451 |
| baseline + L∞ safety reg | 0.910 | 0.597 | 0.422 |
| jac+pull + L∞ safety reg | 0.965 | 0.974 | 0.933 |
| baseline + L∞ safety reg w/o h | 0.951 | 0.755 | 0.849 |
| L∞ safety reg, λ-projected, global λ, projected scale | 0.950 | 0.675 | 0.211 |

Safety Gym, LeWM (jac+pull's seed-3072 pred_loss was not logged):

| model | metric | seed 3072 | seed 1 | seed 2 | 3 WM seeds |
|---|---|---|---|---|---|
| jac+pull | in-dist AUC | 0.999 | 0.998 | 0.996 | 0.998 ± 0.002 |
| jac+pull | color AUC | 0.996 | 0.994 | 0.987 | 0.993 ± 0.005 |
| jac+pull | color sign accuracy | 0.943 | 0.964 | 0.930 | 0.946 ± 0.017 |
| jac+pull | rotation AUC | 0.679 | 0.593 | 0.598 | 0.623 ± 0.048 |
| jac+pull | rotation sign accuracy | 0.835 | 0.836 | 0.833 | 0.835 ± 0.002 |
| jac+pull | final train pred_loss | — | 0.0090 | 0.0111 | — |
| L∞ safety reg | in-dist AUC | 1.000 | 1.000 | 0.993 | 0.998 ± 0.004 |
| L∞ safety reg | color AUC | 0.999 | 0.998 | 0.957 | 0.985 ± 0.024 |
| L∞ safety reg | color sign accuracy | 0.987 | 0.991 | 0.892 | 0.957 ± 0.056 |
| L∞ safety reg | rotation AUC | 0.962 | 0.995 | 0.596 | 0.851 ± 0.222 |
| L∞ safety reg | rotation sign accuracy | 0.977 | 0.992 | 0.831 | 0.933 ± 0.089 |
| L∞ safety reg | final train pred_loss | 0.0168 | 0.0228 | 0.0196 | 0.0197 ± 0.0030 |
| jac+pull + L∞ safety reg | in-dist AUC | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 |
| jac+pull + L∞ safety reg | color AUC | 1.000 | 1.000 | 0.999 | 1.000 ± 0.000 |
| jac+pull + L∞ safety reg | color sign accuracy | 0.973 | 0.990 | 0.973 | 0.979 ± 0.010 |
| jac+pull + L∞ safety reg | rotation AUC | 0.985 | 0.962 | 0.997 | 0.981 ± 0.018 |
| jac+pull + L∞ safety reg | rotation sign accuracy | 0.964 | 0.941 | 0.982 | 0.962 ± 0.021 |
| jac+pull + L∞ safety reg | final train pred_loss | 0.0328 | 0.0369 | 0.0322 | 0.0340 ± 0.0026 |

- **On Safety Gym, adding the L∞ safety reg to jac+pull fixes camera rotation on every seed.** Rotation AUC goes from 0.623 ± 0.048 to 0.981 ± 0.018, and rotation sign accuracy from 0.835 (calling everything safe) to 0.962 ± 0.021. Color is already high for jac+pull (AUC 0.993); the safety reg raises its sign accuracy from 0.946 to 0.979. jac+pull's final prediction loss is 0.009–0.011, against 0.032–0.037 with the safety reg.

#oct-18 L∞ safety reg, λ-projected, batch λ with detached direction, both benchmarks:

| metric | seed 3072 | seed 1 | seed 2 | 3 WM seeds |
|---|---|---|---|---|
| Dubins color AUC (uniform) | 0.948 | 0.220 | 0.995 | 0.721 ± 0.434 |
| Dubins color AUC (50/50) | 0.953 | 0.262 | 0.994 | 0.736 ± 0.411 |
| Dubins shape AUC | 0.996 | 0.996 | 0.997 | 0.996 ± 0.000 |
| Dubins rotation AUC | 0.996 | 0.990 | 0.993 | 0.993 ± 0.003 |
| Dubins rotation sign accuracy (50/50) | 0.951 | 0.840 | 0.932 | 0.908 ± 0.060 |
| Safety Gym color AUC | 0.992 | 0.986 | 1.000 | 0.993 ± 0.007 |
| Safety Gym color sign accuracy | 0.992 | 0.988 | 0.997 | 0.992 ± 0.005 |
| Safety Gym rotation AUC | 0.962 | 0.954 | 0.919 | 0.945 ± 0.023 |
| Safety Gym rotation sign accuracy | 0.983 | 0.959 | 0.963 | 0.968 ± 0.013 |
| final train pred_loss (Dubins / Safety Gym) | 0.0106 / 0.0221 | 0.0101 / 0.0291 | 0.0141 / 0.0270 | |

- #oct-18 **Detached batch λ is seed-stable on Safety Gym but not on Dubins color.** On Safety Gym all three seeds keep both thresholds (sign accuracy ≥ 0.959; predicted safe 0.83–0.86 vs 0.833 true), with color AUC 0.993 ± 0.007 and rotation 0.945 ± 0.023. That's close to jac+pull + L∞ safety reg (1.000 / 0.981) at lower prediction cost (0.022–0.029 vs 0.032–0.037), and on the sigreg base. On Dubins, color swings from 0.22 to 0.995 across seeds, like the other λ variants. Shape and rotation are stable.

Raw CEM (sg25clean), baseline + L∞ safety reg, success % / collision %, mean ± std over 3 training seeds:

| red (in-dist) | purple | diamond | rotated |
|---|---|---|---|
| 58.2 ± 1.2 / 54.8 ± 2.0 | 58.2 ± 1.2 / 54.7 ± 2.1 | 58.2 ± 1.2 / 44.7 ± 1.3 | 27.3 ± 7.0 / 66.0 ± 4.5 |

- **jac+pull + L∞ safety reg is the only model whose Dubins color result is stable across seeds** (0.933–0.974, 0.957 ± 0.021), and it is the best. It is also the most stable on the 50/50 eval (0.955 ± 0.017) and keeps shape and rotation at 0.994.
- **The other single-seed color numbers were optimistic.** jac+pull's 0.753 becomes 0.498 ± 0.234, baseline + L∞ safety reg's 0.910 becomes 0.643 ± 0.247, and projected-scale λ's 0.950 becomes 0.612 ± 0.373 (one seed at 0.211). w/o h holds up best of the sigreg-base models: 0.852 ± 0.098.
- **Shape and rotation are stable for every model** (≥ 0.975 mean), except that the head version's rotation dips on seed 2 (0.935).
- **Thresholds:** on the Dubins 50/50 eval, color sign accuracy stays at collapse for every model's 3-seed mean (0.48–0.60). jac+pull + L∞ safety reg is the highest (0.604 ± 0.176).
- **Safety Gym:** the L∞ safety reg's threshold survival holds on 2 of 3 seeds. Seed 2 is weaker on color (sign accuracy 0.892) and loses most of its rotation gain (AUC 0.596, sign accuracy 0.831, near all-safe).
- **Prediction loss across seeds:** baseline + L∞ safety reg 0.0121 ± 0.0015; w/o h 0.0288 ± 0.0043; jac+pull + L∞ safety reg 0.0165, 0.0158, 0.0167; Safety Gym L∞ 0.0197 ± 0.0030.

## Targeted Jacobian reg: the first-order version without an adversary

#oct-18 To first order, the worst ε-perturbation in an L∞ ball moves the safety coordinate λ_n · f(x) by ε‖J_fᵀλ_n‖₁ (L1 is the dual norm of L∞). So the L∞ λ-projected reg is approximately a one-row Jacobian penalty. This run penalizes that quantity directly, with no PGD:

    d = mean over t of (ε ‖∇ₓ(λ_n · f(x_t))‖₁ / s)²  +  (ε ‖∇ₓ(λ_n · P(f(x), a))‖₁ / s)²

- λ_n: the detached global direction, from running averages of the safe and unsafe means.
- s: the projected scale (the batch gap along λ_n, with gradient), as in global λ with projected scale.
- ε = 8/255, so d is on the same scale as the adversarial penalty; weight 1, opens at step 2000, ramps over 10k steps.
- The prediction term uses one gradient over all three context frames, since δ is shared across frames.
- #oct-18 Baseline (sigreg only), Dubins and Safety Gym, training seeds 3072 / 1 / 2. Code `attack: jac` (commit `f38454c`; the double backward runs on the math attention backend).

| metric | seed 3072 | seed 1 | seed 2 | 3 WM seeds |
|---|---|---|---|---|
| Dubins in-dist AUC | 0.988 | 0.986 | 0.993 | 0.989 ± 0.004 |
| Dubins color AUC (uniform) | 0.542 | 0.801 | 0.845 | 0.729 ± 0.164 |
| Dubins color AUC (50/50) | 0.551 | 0.799 | 0.877 | 0.743 ± 0.170 |
| Dubins shape AUC | 0.984 | 0.979 | 0.991 | 0.985 ± 0.006 |
| Dubins rotation AUC | 0.988 | 0.984 | 0.991 | 0.988 ± 0.004 |
| Dubins color sign accuracy (50/50) | 0.531 | 0.712 | 0.500 | 0.581 ± 0.114 |
| Safety Gym in-dist AUC | 0.996 | 0.986 | 0.904 | 0.962 ± 0.050 |
| Safety Gym color AUC | 0.805 | 0.884 | 0.952 | 0.880 ± 0.074 |
| Safety Gym rotation AUC | 0.826 | 0.928 | 0.839 | 0.865 ± 0.056 |
| Safety Gym color / rotation sign accuracy | 0.864 / 0.907 | 0.919 / 0.916 | 0.167 / 0.167 | 0.650 ± 0.419 / 0.663 ± 0.430 |
| final train pred_loss (Dubins / Safety Gym) | 0.198 / 0.207 | 0.126 / 0.248 | 0.320 / 0.019 | |

#oct-18 For comparison, over 3 WM seeds: L∞ safety reg w/ h gets Dubins color 0.643 ± 0.247 and Safety Gym 0.985 / 0.851; jac+pull + L∞ safety reg gets 0.957 ± 0.021 and 1.000 / 0.981; global λ with projected scale (adversarial) gets Dubins color 0.612 ± 0.373. Adversarial runs end at pred_loss ≈ 0.01–0.04.

- #oct-18 **The first-order penalty does not reproduce the adversarial result.** On Safety Gym it is clearly worse than any adversarial L∞ model: color 0.880 vs 0.985–1.000, rotation 0.865 vs 0.851–0.981, and the thresholds are not kept. On Dubins its color AUC (0.729 ± 0.164) is less seed-variable than the adversarial λ variants but still below jac+pull + L∞ safety reg.
- #oct-18 **Driving the gradient norm down directly wrecks prediction.** Final pred_loss is 0.13–0.32 in five of six runs, 10–30× the adversarial versions. The one cheap run (Safety Gym seed 2, 0.019) is degenerate: its in-dist AUC drops to 0.904 and its zero-shot head calls everything unsafe under both shifts.
- #oct-18 **Together with the random-noise result, the finite-step search matters.** A small worst-case step in pixel space is not well summarized by the local gradient. Penalizing the realized shift at the PGD point gives better robustness at far lower prediction cost than penalizing the gradient everywhere.

## Planning: raw CEM on Dubins

Does the safety reg change how well the world model plans? Raw CEM, with no safety filter, on the model-differentiating sg25clean protocol:
- goals 25 steps ahead, placed in an annulus around the obstacles, so the greedy path crosses them;
- horizon 25, budget 50, N = 200 episodes per cell.

Same protocol as the existing sigreg only / jac / jac+pull numbers (`le-wm/scripts/raw_cem_ood.sh`). Each cell is success % / collision %. "rotated" rotates the whole world by 90°. One CEM evaluation per cell; training seed 3072 unless stated. Logs `run_logs/rawcem_sg25_<tag>_<condition>.log`; script `safeadv_results/cem/run_cem.sh`.

| model | red (in-dist) | purple | diamond | rotated |
|---|---|---|---|---|
| sigreg only | 62.5 / 50.0 | 62.0 / 50.0 | 62.5 / 38.5 | 26.5 / 62.5 |
| jacobian | 81.5 / 22.5 | 81.5 / 22.5 | 81.5 / 14.5 | 30.0 / 60.0 |
| jac+pull | 81.5 / 29.0 | 82.0 / 27.5 | 81.5 / 15.5 | 41.5 / 54.5 |
| baseline + L∞ safety reg | 57.5 / 57.0 | 57.5 / 57.0 | 57.5 / 46.0 | 20.5 / 70.5 |
| jac+pull + L∞ safety reg | 75.5 / 29.0 | 75.5 / 28.5 | 75.5 / 16.5 | 30.0 / 53.0 |
| baseline + L∞ safety reg w/o h | 59.0 / 51.5 | 60.0 / 52.0 | 59.0 / 39.5 | 21.0 / 64.0 |

- **The L∞ safety reg doesn't make planning better, and on the sigreg base it makes it slightly worse.** Baseline + L∞ safety reg gets 57.5% success and 57.0% collision in-distribution, against 62.5% / 50.0% for sigreg only. Its two reseeds are in the same range (57.5–59.5% / 53–54.5%), so unlike the color AUC this holds across training seeds. Without h it is about the same (59.0% / 51.5%).
- **Planning quality comes from the Jacobian reg, not the safety reg.** jac and jac+pull remain the best planners (81.5% success, 22.5–29% collision). Adding the L∞ safety reg on top of jac+pull keeps the low collision rate (29.0%) but lowers success (75.5% vs 81.5%).
- **Recolor doesn't affect raw CEM for any model.** Red and purple are within a point everywhere. The goal image is recolored too, and the planner only compares current and goal latents, so a consistent appearance shift cancels. Diamond lowers collision for everyone, because the obstacle footprint itself changes. A 90° rotation hurts every model (20–42% success here), and no safety-reg variant protects planning under rotation.
- **Implication:** the safety reg's gains are in the *safety readout* (margin-head ranking and thresholds), not in planning. A safety filter on top of CEM is where it could pay off; that hasn't been tested.
- **All other models** (the other training seeds, the w/o h ablations, random noise, attack sweep, λ variants, hue) are in [Appendix E](#appendix-e-full-cem-table).

## Encoder Jacobians across models

Exact encoder Jacobian J = ∂z/∂x (192 × 150528, built from 192 VJPs) on the same 16 dataset frames as the earlier jacobian-mechanism analysis (`le-wm/scripts/jac_singular.py`, rng seed 0). Per-frame values are averaged. Latent scales differ across models, so the last two columns are scale-free:
- **‖J‖/cdist:** ‖J‖_F divided by the safe/unsafe latent centroid distance, from 3000 labelled dataset frames. It measures how far pixel noise moves the latent, in class-separation units.
- **‖∇h‖/gap:** pixel-gradient norm of a fresh margin_gp head (trained on 4000 red uniform states, the OOD-eval protocol), in units of that head's safe/unsafe gap. It measures how fast the safety readout responds to a pixel change.

Script `diagnostics/jac_compare.py`, output `jac_compare.out`.

| model | σ1 | ‖J‖_F | cond (σ1/σ51) | eff. rank | top-5 energy | ‖J‖/cdist | ‖∇h‖/gap | color AUC (uniform) |
|---|---|---|---|---|---|---|---|---|
| baseline (sigreg only) | 7.830 | 11.146 | 130.4 | 13.2 | 0.885 | 1.538 | 0.2079 | 0.144 |
| jacobian | 0.310 | 1.008 | 4.0 | 86.4 | 0.303 | 0.239 | 0.0096 | 0.673 |
| jac+pull | 0.304 | 1.008 | 3.8 | 90.7 | 0.278 | 0.263 | 0.0112 | 0.753 |
| baseline + L∞ safety reg | 2.073 | 2.663 | 122.0 | 11.8 | 0.917 | 0.299 | 0.0266 | 0.910 |
| jac+pull + L∞ safety reg | 0.313 | 1.008 | 4.2 | 83.6 | 0.303 | 0.235 | 0.0113 | 0.965 |
| baseline + L∞ safety reg w/o h | 1.202 | 1.869 | 96.6 | 16.5 | 0.809 | 0.100 | 0.0256 | 0.951 |

- **Two different routes to low sensitivity.** The Jacobian penalty pins ‖J‖_F at ≈ 1.008 and *whitens* J: condition number ~4, ~85–90 effective dimensions, top-5 directions hold ~30% of the energy. L∞ shrinks J (‖J‖_F 11.1 → 2.7, σ1 7.8 → 2.1) but keeps baseline's *shape*: condition number 122, ~12 effective dimensions, 92% of the energy in the top 5. Relative to class separation, L∞ is 5× less pixel-sensitive than baseline (1.538 → 0.299), and its readout is 8× less sensitive (0.208 → 0.027).
- **No-h gets there by spreading the classes apart.** It has the smallest ‖J‖/cdist of all (0.100) because its centroid distance is ~2.6× baseline's (implied 18.7 vs 7.3), while its raw ‖J‖_F of 1.87 is in between. The centroid-distance denominator rewards pushing the classes apart, and in training it grew steadily (in-batch squared distance ~116 → 352). That is a legitimate way to lower the penalty, but it may also explain the higher pred_loss (0.0253).
- **On a jac+pull base, adding L∞ is invisible in J.** Every column of jac+pull + L∞ matches jac+pull (‖∇h‖/gap 0.0113 vs 0.0112), yet color AUC goes 0.753 → 0.965. So the L∞ gain on top of jac+pull is not a local first-order effect at these frames. The PGD perturbation is a finite step (8/255 on every pixel), and purple is a large shift, so whatever changed lives in the non-local geometry.
- **Local sensitivity does not rank color robustness.** Plain jacobian has the lowest readout sensitivity of all (0.0096) but only 0.673 color AUC, while L∞ is 3× more sensitive and reaches 0.910. This matches the earlier jacobian-mechanism finding that the benefit is not global smoothness.

## Interpretation

1. **Adversarial noise invariance transfers to large shifts, and the safety projection isn't needed, but the rest is.** Small adversarial pixel noise gives robustness to three shifts it never saw, including the color shift that a hand-designed color perturbation handled worse (Appendix B). The no-h ablation penalizes the whole latent instead of the readout and is at least as good (color 0.852 ± 0.098 vs 0.643 ± 0.247 over 3 WM seeds). The most reliable model combines the regularizer with jac+pull (0.957 ± 0.021). #oct-18 The follow-up ablations show that color needs all of the following:
   - a worst-case search: random noise of the same size gives 0.631, consistent with earlier *global* random-noise invariance regularizers never fixing color;
   - the gradient through the class-separation denominator: detached gives 0.643, and prediction suffers;
   - both the encoder and the predictor term: either alone is unstable across heads.

   Shape and rotation, by contrast, are easy: any of these variants gets them.
2. **It goes well beyond jacobian, by a different mechanism.** Jacobian whitens the local Jacobian and reaches color AUC 0.673. The L∞ variants keep or even sharpen the anisotropic Jacobian and reach 0.91–0.97. The two combine: jac+pull + L∞ is the best color model.
3. **The remaining failure is calibration.** Even the best ranking comes with a coherent readout offset under purple that pushes every state across the threshold, and every L∞ variant has it. For a deployable filter, this needs recalibration or an offset-correction mechanism, not more invariance. On Safety Gym, LeWM + L∞ is the exception: its threshold holds under both shifts. That suggests the offset depends on the domain, perhaps on how strongly appearance and safety are entangled at the input (which is high on Dubins), rather than being inherent to the method.

## Caveats

- **Training seeds.** Six models have 3 world-model training seeds (see [Training-seed variability](#training-seed-variability)); everything else, including all ablations, the attack sweep, the λ variants except projected scale, and the Dreamer runs, has one. Dubins color varies by up to ±0.37 across training seeds, so single-seed color differences of that size are not conclusive. Shape and rotation are stable across seeds. #oct-18
- **Prediction quality cost.** Final train pred_loss: L∞ 0.0113, jac+pull + L∞ 0.0165, no-h 0.0253, vs 0.0067 for the hue-variant run. Whether this hurts CEM planning hasn't been tested, and it matters most for no-h.
- **Only the margin head is tested.** The reachability critic's zero-shot behavior (where every earlier model collapsed on color) hasn't been evaluated.
- **Eval states are uniform, not on-policy.** They cover every heading and positions deep inside obstacles; the world model trained on expert trajectories. "In-dist" means in-distribution *appearance*.
- **Safety Gym LeWM schedule differs** (no AUC gate). Its baseline comparison is still like-for-like in recipe, but the penalty ran from step 2k regardless of head quality.
- **Safety Gym eval uses one fixed 1000-frame test set** from a single layout-controlled collection, so the ± only reflects head seeds.
- **Jacobians are local**, measured at 16 dataset frames. They describe first-order sensitivity, not the finite red → purple shift.

## Next steps (proposed)

1. **Second training seed** of L∞, jac+pull + L∞ and no-h, to rank them on color.
2. **Second training seed for the single-term variants** (encoder only, predictor only), to check whether their eval-seed instability is a property of the variant.
3. **Safety Gym follow-ups:** a second training seed of LeWM + L∞ (the threshold survival is the headline result), the same model with the gate (if a jac base makes the head learnable), and the hazard-shape axis in a layout-controlled collection.
4. **Dreamer:** why rotation doesn't move. A candidate is the non-image heading input, which the perturbation never reaches.
5. **Planner eval** (sg25clean protocol, in-dist and under shift), to check the pred_loss cost, especially no-h's.
6. **Critic zero-shot** for the L∞ models, the real target.
7. **ε sweep** (4/255, 16/255).
8. **Calibration fix** for the threshold collapse, e.g. unsupervised re-centering of the readout per appearance.

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
| Mean latent shift under purple ÷ safe/unsafe centroid distance | 2.377 | 1.272 | 0.771 |
| Per-state shift ÷ centroid distance | 2.723 | 3.324 | 1.308 |
| cos(mean shift, safety-readout gradient) | −0.685 | −0.201 | −0.641 |
| Share of the mean shift along the readout direction | 0.469 | 0.041 | 0.411 |
| Effective latent dimensions (participation ratio, of 192) | 10.5 | 45.3 | 10.8 |
| Safe/unsafe separation d′ (fresh head) | 3.41 | 4.60 | 7.44 |

- **Mechanism: magnitude, not direction.** L∞ makes the color shift small relative to the safety separation. The safe/unsafe separation roughly doubles (d′ 3.41 → 7.44), and purple moves each latent about half as far as in baseline, so in class-separation units the push is about a third of baseline's. The shift still points along the readout about as much as baseline's. Jac+pull is the opposite: it rotates the color shift off the readout (4% along it) and spreads the latent over ~45 dimensions.
- **No collapse:** effective dimensionality is unchanged from baseline (10.8 vs 10.5).
- **Why the color threshold collapses to "all unsafe":** the residual shift points down the readout, lowering every state's margin. The run's own in-training head moves by −0.705 class gaps under purple (predicted safe 0.004), while its ranking survives (AUC 0.962 red-trained head on purple). A uniform offset like this is a calibration problem.

### A.9 Cost

- **Memory:** 15.8 GB peak at batch 128, vs 13.5 GB without the regularizer.
- **Time:** 4 h 44 min for 50 epochs (08:41–13:25 UTC on 2026-10-05) on one RTX PRO 6000, which was shared with other jobs.
- **Extra work per step:** 1 clean pass, 3 perturbed passes (2 with gradient to δ) and 1 graded pass, each over 32 windows × 3 frames.

### A.10 Ablation: remove h (penalize the whole latent)

Run `lewm_dubins_linf_latent50`, code `dab24f2` (`target: latent`), launched 2026-10-06 00:45 UTC. Everything is identical to the L∞ run (ε, PGD, 32 windows, BatchNorm handling, schedule, and the head, which is still trained because it drives the AUC gate) except the quantity being protected. The head no longer appears in the penalty:

For each window, with clean context frames x₁..₃, perturbed frames x₁..₃ + δ, actions a, encoder f, predictor P:

    d = mean over t of ‖f(x_t + δ) − f(x_t)‖² / C  +  ‖P(f(x + δ), a) − P(f(x), a)‖² / C

- **First term:** the whole latent of each perturbed context frame must match the clean one. ‖·‖² is the squared Euclidean length over the 192 latent dimensions; it plays the role of the squared scalar difference in A.4.
- **Second term:** the same for the predictor's last prediction.
- **Clean targets** f(x_t) and P(f(x), a) are computed once without gradient.
- **C** = ‖μ_safe − μ_unsafe‖², the squared distance between the in-batch safe and unsafe latent centroids, **with gradient**. It replaces A.4's gap.
- **Loss added to training:** λ · ramp · mean(d) over the 32 windows, with λ = 1.0.

The adversary maximizes the same d. An earlier draft normalized by the latent spread instead; that let the regularizer win by squashing the latent, so it was replaced by the centroid distance before this run.

- **Result:** matches or beats the head-projected L∞ model on every axis (color AUC 0.951 ± 0.004 vs 0.910 ± 0.098; shape and rotation ≥ 0.996). Projecting onto the safety readout is not what makes the L∞ regularizer work.
- **Side effects:** the in-batch centroid distance² grew from ~116 to 352 over training (the denominator rewards separating the classes). Final train pred_loss is 0.0253, more than twice L∞'s 0.0113. The adversary's gain over a random perturbation is ×3.8 at the end, vs ×13 for L∞, because whole-latent displacement is harder to concentrate than readout displacement.

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
| baseline + hue safety reg | 0.970 ± 0.003 | 0.830 ± 0.006 | 0.891 ± 0.005 | 0.843 ± 0.024 |
| jac+pull + hue safety reg | 0.969 ± 0.004 | 0.816 ± 0.026 | 0.968 ± 0.003 | 0.892 ± 0.018 |

**AUC**

| model | in-dist | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| baseline (sigreg only) | 0.979 ± 0.002 | 0.144 ± 0.032 | 0.913 ± 0.024 | 0.956 ± 0.023 |
| jacobian | 0.992 ± 0.001 | 0.673 ± 0.076 | 0.992 ± 0.003 | 0.989 ± 0.002 |
| jac+pull | 0.983 ± 0.001 | 0.753 ± 0.049 | 0.977 ± 0.007 | 0.978 ± 0.003 |
| baseline + hue safety reg | 0.996 ± 0.003 | 0.634 ± 0.158 | 0.612 ± 0.080 | 0.966 ± 0.009 |
| jac+pull + hue safety reg | 0.985 ± 0.004 | 0.681 ± 0.150 | 0.977 ± 0.007 | 0.983 ± 0.005 |

50/50 eval (sign accuracy, AUC, predicted-safe fraction):

**Sign accuracy (50/50: .500 = threshold gone)**

| model | in-dist | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| baseline (sigreg only) | 0.940 ± 0.003 | 0.500 ± 0.000 | 0.585 ± 0.010 | 0.500 ± 0.000 |
| jacobian | 0.963 ± 0.003 | 0.500 ± 0.000 | 0.952 ± 0.003 | 0.946 ± 0.009 |
| jac+pull | 0.940 ± 0.005 | 0.503 ± 0.005 | 0.919 ± 0.005 | 0.919 ± 0.013 |
| baseline + hue safety reg | 0.970 ± 0.005 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.629 ± 0.069 |
| jac+pull + hue safety reg | 0.964 ± 0.003 | 0.645 ± 0.101 | 0.942 ± 0.005 | 0.849 ± 0.046 |

**AUC**

| model | in-dist | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| baseline (sigreg only) | 0.985 ± 0.002 | 0.177 ± 0.032 | 0.911 ± 0.029 | 0.902 ± 0.041 |
| jacobian | 0.994 ± 0.001 | 0.711 ± 0.069 | 0.992 ± 0.001 | 0.990 ± 0.003 |
| jac+pull | 0.987 ± 0.002 | 0.747 ± 0.063 | 0.981 ± 0.001 | 0.983 ± 0.004 |
| baseline + hue safety reg | 0.996 ± 0.002 | 0.623 ± 0.150 | 0.631 ± 0.084 | 0.967 ± 0.013 |
| jac+pull + hue safety reg | 0.993 ± 0.001 | 0.685 ± 0.086 | 0.987 ± 0.002 | 0.986 ± 0.004 |

**Predicted-safe fraction (true = .500)**

| model | in-dist | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| baseline (sigreg only) | 0.497 ± 0.003 | 0.000 ± 0.000 | 0.085 ± 0.010 | 1.000 ± 0.000 |
| jacobian | 0.502 ± 0.002 | 0.000 ± 0.000 | 0.471 ± 0.002 | 0.505 ± 0.006 |
| jac+pull | 0.492 ± 0.008 | 0.003 ± 0.005 | 0.480 ± 0.012 | 0.455 ± 0.010 |
| baseline + hue safety reg | 0.507 ± 0.006 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.856 ± 0.071 |
| jac+pull + hue safety reg | 0.500 ± 0.013 | 0.388 ± 0.159 | 0.504 ± 0.025 | 0.360 ± 0.051 |

Predicted-safe fraction under purple:

| model | predicted safe under purple | true safe fraction |
|---|---|---|
| baseline (sigreg only) | 0.000 ± 0.000 | 0.830 ± 0.006 |
| jacobian | 0.001 ± 0.001 | 0.830 ± 0.006 |
| jac+pull | 0.976 ± 0.041 | 0.830 ± 0.006 |
| baseline + hue safety reg | 1.000 ± 0.000 | 0.830 ± 0.006 |
| jac+pull + hue safety reg | 0.982 ± 0.030 | 0.830 ± 0.006 |

**Reading:**
- **On the sigreg base,** hue lifts color AUC from 0.144 to 0.634 but drops shape from 0.913 to 0.612. Its color *and* shape thresholds collapse (50/50 sign accuracy 0.500 on both, everything called safe).
- **On jac+pull** it adds nothing (color 0.753 → 0.681, within seed noise).
- **BatchNorm:** the sigreg-base hue model also has a large first-layer running-stat offset (2.2 std, variance ×9.6) from its ordinary training pass, which may relate to its shape drop (unverified).
- **Conclusion:** L∞ beats it on every axis, including color, which hue was built for.

- **Encoder Jacobian** (same protocol as [Encoder Jacobians across models](#encoder-jacobians-across-models)): baseline + hue safety reg has σ1 7.860, ‖J‖_F 13.405, condition number 91.8, effective rank 16.4, ‖J‖/cdist 4.394 and ‖∇h‖/gap 1.1858. jac+pull + hue safety reg has 0.316 / 1.010 / 4.1 / 90.0 / 0.435 / 0.0244. On the sigreg base, hue training made the readout about 6× *more* pixel-sensitive than baseline, which fits its weak shape result: training against a structured color field doesn't buy generic noise robustness.

## Appendix C: v1 failure (BatchNorm cheat), now fixed

v1 sent clean and perturbed frames through one shared train-mode batch. The encoder learned to make perturbed frames produce extreme values in some features. That inflated the shared BatchNorm variance and squashed both halves' readouts together: the clean safe/unsafe gap fell from .97 to .07 inside that batch. The penalty looked tiny (~.003) with no real invariance.

The skewed statistics also leaked into the running stats (variance ×6.7), so the deployed encoder differed from the trained one. v1 zero-shot AUC (seed 0): color .40, shape .085, rotation .996.

Fix (`c6e93c6`): capture clean-batch BN stats once, apply them to every perturbed pass, and never update running stats. Unit-tested: batch-composition independence, running stats untouched, gradients only to the encoder. **This applies to any le-wm regularizer that runs extra forward passes**, since the projector and predictor heads use BatchNorm.

## Appendix D: where things are

- **Code:** `le-wm/module.py` (`SafetyAdvInvarianceReg`, `_CleanStatBN`), config key `safety_adv` (kwargs `attack: linf | hue`, `eps_pix`, `min_steps`, `auc_gate`, `ramp_steps`, `n_sub`, `pgd_steps`), data config `data=dubins_safety`.
- **Checkpoints** (under `/data/seongbin/lewm/checkpoints/`): `lewm_dubins_safeadv_linf50/` (L∞), `lewm_dubins_jacpull_safeadv_linf50/` (jac+pull + L∞), `lewm_dubins_linf_latent50/` (no h; code `dab24f2`, `target: latent`), `lewm_dubins_lz_{rand,det,enc,pred}50/` (no-h ablations; code `9dfca47`), `lewm_dubins_lam_{batch,global}50/`, `sg_lam_{batch,global}50/` (λ-projected, code `32263e1`), `lewm_dubins_lam_gproj50/`, `sg_lam_gproj50/`, `lewm_dubins_safeadv_rand50/`, `sg_safeadv_rand50_g0/`, `sg_lz_rand50_g0/` (projected-scale λ and random-noise controls, code `cfa5e52`; evals in `lambda_eval/`), `lewm_dubins_safeadv_{eps4,eps16,pgd5}/` (attack sweep, code `cfa5e52`), `sg_safeadv_linf50_g0/` and `sg_linf_latent50_g0/` (Safety Gym, code `b9d34e2`, launchers `/data/seongbin/lewm/code_safeadv_b9d34e2/run_sg{L,LZ}g0.sh`). Dreamer: `/data/seongbin/dreamer/dreamer/safeadv/{dubins,sg}_sa_{head,latent}/rssm_ckpt.pt` (code snapshot `/data/seongbin/dreamer/code_safeadv_e99b14c/`, logs `safeadv/*.log`), `lewm_dubins_safeadv2_50/` and `lewm_dubins_jacpull_safeadv2_50/` (hue), `lewm_dubins_safeadv50/` (v1). The in-training margin head is only in the Lightning checkpoints (`~/.cache/stable-pretraining/runs/<date>/<time>/<hash>/checkpoints/`).
- **Training logs and code snapshots:** `/data/seongbin/lewm/code_safeadv_77856fd/` (L∞, `.run_L/`; jac+pull + L∞, `.run_JL/`), `code_safeadv_dab24f2/` (no h, `.run_LZ/`), `code_safeadv_9dfca47/` (ablations, `run_<k>.sh`, `.run_<k>/`), `code_safeadv_c6e93c6/` (hue), `code_safeadv_ca9ec21/` (v1). W&B project `seongbin/lewm`.
- **Eval logs** (under `/data/seongbin/lewm/safeadv_results/`): `linf_ood_eval/`, `jacpull_linf_ood_eval/`, `linf_latent_ood_eval/` (no h), `ablation_eval/` (no-h ablations, incl. `jac_ablation.out`), `sg_eval/` (Safety Gym eval script, logs, `tables.py` for the Safety Gym and Dreamer tables), `dreamer/` (Dubins Dreamer eval logs, Safety Gym label fix `fix_sg_labels.py`), `v2_ood_eval/` (hue), `v1_ood_eval/`, `predsafe_check/`, `balanced_eval/` (50/50); diagnostic scripts in `diagnostics/` (Jacobian comparison: `jac_compare.py` / `.out` / `.npy`); targeted Jacobian reg runs `lewm_dubins_jaclam50_s*`, `sg_jaclam50_s*` (code `f38454c`). Tables are generated by `make_tables.py`.
- **Baseline numbers:** `/home/seongbin/latent/run_logs/ood_gp_jepa_*_s*.log` (same eval script, `le-wm/scripts/ood_margin_gp_jepa.py`).

## Appendix E: full CEM table

All Dubins LeWM models under raw CEM, same protocol as [Planning: raw CEM on Dubins](#planning-raw-cem-on-dubins): success % / collision %, N = 200 per cell, training seed 3072 unless stated.

| model | red (in-dist) | purple | diamond | rotated |
|---|---|---|---|---|
| sigreg only | 62.5 / 50.0 | 62.0 / 50.0 | 62.5 / 38.5 | 26.5 / 62.5 |
| jacobian | 81.5 / 22.5 | 81.5 / 22.5 | 81.5 / 14.5 | 30.0 / 60.0 |
| jac+pull | 81.5 / 29.0 | 82.0 / 27.5 | 81.5 / 15.5 | 41.5 / 54.5 |
| baseline + L∞ safety reg (seed 3072) | 57.5 / 57.0 | 57.5 / 57.0 | 57.5 / 46.0 | 20.5 / 70.5 |
| baseline + L∞ safety reg (seed 1) | 57.5 / 54.5 | 57.5 / 54.0 | 57.5 / 44.5 | 27.0 / 66.0 |
| baseline + L∞ safety reg (seed 2) | 59.5 / 53.0 | 59.5 / 53.0 | 59.5 / 43.5 | 34.5 / 61.5 |
| jac+pull + L∞ safety reg | 75.5 / 29.0 | 75.5 / 28.5 | 75.5 / 16.5 | 30.0 / 53.0 |
| baseline + L∞ safety reg w/o h | 59.0 / 51.5 | 60.0 / 52.0 | 59.0 / 39.5 | 21.0 / 64.0 |
| w/o h, random noise | 57.0 / 50.5 | 57.0 / 50.0 | 57.0 / 34.5 | 16.5 / 58.5 |
| w/o h, detached denominator | 24.0 / 77.0 | 26.0 / 76.5 | 24.5 / 70.0 | 12.5 / 56.5 |
| w/o h, encoder term only | 54.5 / 45.0 | 54.5 / 46.0 | 54.5 / 31.0 | 16.0 / 49.5 |
| w/o h, predictor term only | 45.0 / 58.0 | 45.0 / 58.0 | 45.0 / 51.0 | 10.0 / 69.0 |
| L∞ safety reg, random noise | 64.5 / 37.5 | 65.5 / 37.5 | 65.0 / 28.5 | 27.0 / 63.5 |
| L∞ safety reg, ε = 4/255 | 55.0 / 51.0 | 55.0 / 51.0 | 55.0 / 40.5 | 23.0 / 64.0 |
| L∞ safety reg, ε = 16/255 | 55.0 / 54.5 | 54.5 / 54.5 | 55.5 / 45.0 | 15.5 / 65.5 |
| L∞ safety reg, 5 PGD steps | 58.0 / 51.5 | 58.0 / 51.5 | 58.0 / 44.0 | 25.5 / 67.5 |
| L∞ safety reg, λ-projected, batch λ | 49.5 / 58.0 | 49.5 / 58.0 | 49.5 / 49.5 | 34.5 / 59.5 |
| L∞ safety reg, λ-projected, global λ | 52.5 / 49.0 | 52.5 / 48.5 | 52.5 / 40.0 | 24.0 / 66.0 |
| L∞ safety reg, λ-projected, global λ, projected scale | 54.5 / 50.0 | 55.0 / 50.0 | 54.5 / 42.0 | 26.5 / 61.5 |
| baseline + hue safety reg | 65.0 / 49.0 | 65.0 / 49.0 | 65.0 / 40.0 | 22.5 / 71.0 |
| jac+pull + hue safety reg | 77.0 / 34.0 | 77.0 / 34.0 | 76.5 / 21.0 | 42.0 / 48.5 |

- **Prediction cost shows up as planning cost.** The w/o h detached-denominator run, whose prediction loss was 4× the others', plans worst: 24% success and 77% collision. The predictor-only run is also poor (45% / 58%).
- **Random-noise variants plan relatively well.** They are the cheapest on prediction loss; with h, random noise gets 64.5% / 37.5%, better than any adversarial safety-reg model on the sigreg base.
- **λ variants and the attack sweep** sit at or below the main L∞ model (49.5–58% success). The hue models plan like their bases: 65% on sigreg, 77% on jac+pull.
- **Rotation:** the best rotated success among all models is jac+pull and jac+pull + hue safety reg (41.5–42%).
