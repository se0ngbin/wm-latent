# Safety-projected adversarial invariance: report

*Updated 2026-10-08 07:30 UTC. Dubins unless marked Safety Gym; LeWM unless marked Dreamer. Branch `safety-adv-reg`; a copy lives at the main checkout root.*

## TL;DR

- **What we tried:** a training-time regularizer that makes the world model's *safety readout* insensitive to adversarial perturbations. The readout is a margin head learned from the binary failure labels; no OOD data is used. The main variant uses plain per-pixel **L∞ noise** (ε = 8/255). Full training details are in Appendix A.
- **Adversarial L∞ noise invariance gives the best models we have on every shift.** Zero-shot color AUC (uniform eval): L∞ **0.910 ± 0.098**, jac+pull + L∞ **0.965 ± 0.013**, L∞ without the head **0.951 ± 0.004**. For comparison, jac+pull scores 0.753 ± 0.049 and baseline 0.144 ± 0.032. All three L∞ models are ≥ 0.994 on shape and rotation. The 50/50 eval agrees (color 0.927 / 0.958 / 0.955).
- **The safety projection is not what makes it work, but every other ingredient is (new).** The "remove h" ablation penalizes the whole latent shift ‖f(x + δ) − f(x)‖², scaled by the safe/unsafe centroid distance, and never touches the head. It matches or beats the head-projected version on every axis, and is far less seed-variable on color (± 0.004 vs ± 0.098). Removing any one of its other parts drops color AUC from 0.951 to 0.63–0.70 (baseline: 0.144): the adversarial search, the gradient through the class-separation denominator, the encoder term, or the predictor term ([ablations](#ablations-what-makes-the-no-h-regularizer-work)). Shape and rotation stay ≥ 0.991 in every ablation.
- **Adding L∞ on top of jac+pull** lifts jac+pull's color AUC 0.753 → 0.965 and gives the best color number overall. It costs more prediction loss (0.0165).
- **Safety Gym (new): the regularizer transfers and is even stronger there** ([section](#safety-gym-and-the-regularizer-on-dreamer)). LeWM + L∞ safety reg reaches zero-shot AUC 0.999 on color and **0.962 on a real 90° camera rotation**, where jac+pull scores 0.679 and sigreg only 0.557. It is also the **first model in the study whose safe/unsafe threshold survives a shift**: sign accuracy 0.987 on color and 0.977 on rotation, against 0.833 for calling everything safe. The cost is 6× the prediction loss (0.0168 vs 0.0026). On Safety Gym the penalty had to run without the AUC gate, because the sigreg-only latent doesn't generalize safety across episodes.
- **Dreamer (new): smaller gains.** The head-projected regularizer lifts Dubins color AUC from 0.638 to 0.816 and Safety Gym color from 0.961 to 0.977. Rotation stays at chance on both benchmarks and thresholds still collapse. No-h doesn't help Dubins Dreamer at all (0.626), so for Dreamer the safety projection matters, unlike LeWM on Dubins.
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

**Sign accuracy**

| model | in-dist | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| baseline (sigreg only) | 0.943 ± 0.005 | 0.170 ± 0.006 | 0.310 ± 0.025 | 0.830 ± 0.006 |
| jacobian | 0.968 ± 0.002 | 0.171 ± 0.007 | 0.968 ± 0.007 | 0.962 ± 0.004 |
| jac+pull | 0.959 ± 0.003 | 0.830 ± 0.007 | 0.957 ± 0.008 | 0.939 ± 0.003 |
| **baseline + safety reg, L∞ noise** | 0.981 ± 0.004 | 0.174 ± 0.006 | 0.980 ± 0.003 | 0.975 ± 0.004 |
| **jac+pull + safety reg, L∞ noise** | 0.973 ± 0.003 | 0.830 ± 0.006 | 0.978 ± 0.005 | 0.970 ± 0.009 |
| baseline + L∞ noise, no h (whole latent) | 0.977 ± 0.005 | 0.170 ± 0.006 | 0.977 ± 0.003 | 0.977 ± 0.003 |

**AUC**

| model | in-dist | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| baseline (sigreg only) | 0.979 ± 0.002 | 0.144 ± 0.032 | 0.913 ± 0.024 | 0.956 ± 0.023 |
| jacobian | 0.992 ± 0.001 | 0.673 ± 0.076 | 0.992 ± 0.003 | 0.989 ± 0.002 |
| jac+pull | 0.983 ± 0.001 | 0.753 ± 0.049 | 0.977 ± 0.007 | 0.978 ± 0.003 |
| **baseline + safety reg, L∞ noise** | 0.998 ± 0.001 | 0.910 ± 0.098 | 0.997 ± 0.001 | 0.996 ± 0.002 |
| **jac+pull + safety reg, L∞ noise** | 0.996 ± 0.001 | 0.965 ± 0.013 | 0.994 ± 0.002 | 0.994 ± 0.002 |
| baseline + L∞ noise, no h (whole latent) | 0.997 ± 0.001 | 0.951 ± 0.004 | 0.996 ± 0.001 | 0.997 ± 0.001 |

Each new run is compared against the matched model without the regularizer (same recipe, seed, 50 epochs). The second head type (margin_nogp) agrees for the L∞ model: color AUC 0.905 ± 0.121, shape ≥ 0.993, rotation ≥ 0.982. Retraining a head on the shifted appearance recovers every model to ≥ 0.930 AUC; the shift relocates the latent rather than destroying information.

### Reading sign accuracy under the color shift

Under purple, sign accuracy is misleading on its own. With ~17% unsafe states, a head that calls *everything safe* scores ≈ 0.830, and one that calls *everything unsafe* scores ≈ 0.170. Every model lands on one of those two values. To confirm which, I logged the fraction of states each zero-shot head predicts safe (`/data/seongbin/lewm/safeadv_results/predsafe_check/`):

| model | predicted safe under purple | true safe fraction |
|---|---|---|
| baseline (sigreg only) | 0.000 ± 0.000 | 0.830 ± 0.006 |
| jacobian | 0.001 ± 0.001 | 0.830 ± 0.006 |
| jac+pull | 0.976 ± 0.041 | 0.830 ± 0.006 |
| **baseline + safety reg, L∞ noise** | not logged; sign accuracy 0.174 ± 0.006 equals the unsafe rate, so ≈ 0.000 (measured directly in the 50/50 eval) | 0.830 ± 0.006 |
| **jac+pull + safety reg, L∞ noise** | not logged; sign accuracy 0.830 ± 0.006 equals the safe rate, so ≈ 1.000 | 0.830 ± 0.006 |
| baseline + L∞ noise, no h (whole latent) | not logged; sign accuracy 0.170 ± 0.006 equals the unsafe rate, so ≈ 0.000 | 0.830 ± 0.006 |

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
| **jac+pull + safety reg, L∞ noise** | 0.974 ± 0.004 | 0.505 ± 0.005 | 0.961 ± 0.003 | 0.958 ± 0.011 |
| baseline + L∞ noise, no h (whole latent) | 0.977 ± 0.006 | 0.500 ± 0.000 | 0.961 ± 0.002 | 0.968 ± 0.009 |

**AUC**

| model | in-dist | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| baseline (sigreg only) | 0.985 ± 0.002 | 0.177 ± 0.032 | 0.911 ± 0.029 | 0.902 ± 0.041 |
| jacobian | 0.994 ± 0.001 | 0.711 ± 0.069 | 0.992 ± 0.001 | 0.990 ± 0.003 |
| jac+pull | 0.987 ± 0.002 | 0.747 ± 0.063 | 0.981 ± 0.001 | 0.983 ± 0.004 |
| **baseline + safety reg, L∞ noise** | 0.999 ± 0.001 | 0.927 ± 0.021 | 0.998 ± 0.000 | 0.996 ± 0.002 |
| **jac+pull + safety reg, L∞ noise** | 0.997 ± 0.001 | 0.958 ± 0.012 | 0.995 ± 0.001 | 0.992 ± 0.004 |
| baseline + L∞ noise, no h (whole latent) | 0.998 ± 0.002 | 0.955 ± 0.002 | 0.997 ± 0.001 | 0.996 ± 0.001 |

**Predicted-safe fraction (true = .500)**

| model | in-dist | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| baseline (sigreg only) | 0.497 ± 0.003 | 0.000 ± 0.000 | 0.085 ± 0.010 | 1.000 ± 0.000 |
| jacobian | 0.502 ± 0.002 | 0.000 ± 0.000 | 0.471 ± 0.002 | 0.505 ± 0.006 |
| jac+pull | 0.492 ± 0.008 | 0.003 ± 0.005 | 0.480 ± 0.012 | 0.455 ± 0.010 |
| **baseline + safety reg, L∞ noise** | 0.501 ± 0.003 | 0.001 ± 0.001 | 0.509 ± 0.005 | 0.522 ± 0.001 |
| **jac+pull + safety reg, L∞ noise** | 0.503 ± 0.001 | 0.995 ± 0.005 | 0.514 ± 0.003 | 0.508 ± 0.004 |
| baseline + L∞ noise, no h (whole latent) | 0.503 ± 0.002 | 0.000 ± 0.000 | 0.520 ± 0.007 | 0.510 ± 0.009 |

What it shows:
- **Confirms:** every model's color threshold collapses (sign accuracy 0.500–0.503). L∞ has the best ranking on every shift: color AUC 0.927 ± 0.021 (less seed-variable than in the uniform eval), shape 0.998, rotation 0.996. L∞ keeps working thresholds on shape (0.974) and rotation (0.959).
- **Corrects (direction of collapse):** jac+pull now calls almost everything *unsafe* under purple (predicted safe 0.003 ± 0.005), whereas in the uniform eval it called everything *safe* (0.976). Same encoder, different head training set. The collapse is a property of the encoder; its direction is not.
- **New models:** jac+pull + L∞ (color AUC 0.958 ± 0.012) and no-h (0.955 ± 0.002) both beat L∞ (0.927 ± 0.021) on color and match it elsewhere. Both still lose the color threshold, in opposite directions: jac+pull + L∞ calls almost everything *safe* (0.995), no-h calls everything *unsafe* (0.000).
- **Corrects (baseline rotation):** baseline's rotation threshold also collapses (sign accuracy 0.500, everything called safe). In the uniform eval its 0.830 rotation sign accuracy equalled the all-safe rate, which hid this.

## Ablations: what makes the no-h regularizer work

Four runs (code `9dfca47`). Each is **the baseline model (sigreg only, same recipe and seed) plus the no-h regularizer with one change**: no Jacobian/pull terms, and the head is not in the penalty. They are compared with the full no-h model and with the unregularized baseline:
- **random noise:** random ±ε noise at every pixel, the same size as PGD's iterates but with no adversarial search (`pgd_steps: 0, rand_init: sign`);
- **detached denominator:** the centroid-distance normalizer is detached, so it keeps its scale but gives no reward for separating the classes (`scale_grad: false`);
- **encoder term only:** `pred_weight: 0`;
- **predictor term only:** `enc_weight: 0`.

Zero-shot AUC, mean ± std over 3 eval seeds; shape and rotation from the uniform eval:

| model | color, uniform | color, 50/50 | shape | rotation | final train pred_loss |
|---|---|---|---|---|---|
| **no h, full** | **0.951 ± 0.004** | **0.955 ± 0.002** | 0.996 ± 0.001 | 0.997 ± 0.001 | 0.0253 |
| random noise (no PGD) | 0.631 ± 0.030 | 0.622 ± 0.026 | 0.997 ± 0.002 | 0.996 ± 0.001 | **0.0140** |
| detached denominator | 0.643 ± 0.022 | 0.627 ± 0.004 | 0.991 ± 0.002 | 0.991 ± 0.005 | 0.1070 |
| encoder term only | 0.703 ± 0.269 | 0.583 ± 0.292 | 0.997 ± 0.001 | 0.996 ± 0.001 | 0.0225 |
| predictor term only | 0.691 ± 0.287 | 0.580 ± 0.394 | 0.996 ± 0.001 | 0.997 ± 0.001 | 0.0265 |
| *reference: sigreg only* | *0.144 ± 0.032* | *0.177 ± 0.032* | *0.913 ± 0.024* | *0.956 ± 0.023* | *not logged* |

Per eval seed (color AUC, uniform / 50/50):
- encoder only: 0.949 / 0.920, 0.416 / 0.398, 0.745 / 0.432;
- predictor only: 0.797 / 0.125, 0.366 / 0.801, 0.911 / 0.815;
- full no-h: 0.948, 0.955, 0.949 (uniform).

- **Every ingredient is needed for color; none is needed for shape and rotation.** All four ablations keep shape and rotation at ≥ 0.991 AUC, the same as the full model. On color, each one falls from 0.951 to between 0.63 and 0.70 AUC: still far above baseline (0.144), but most of the gain is lost.
- **The adversary matters.** Random noise of the same per-pixel size gives 0.631 ± 0.030. It is also the cheapest variant on prediction (pred_loss 0.0140). It doesn't separate the classes much either (centroid distance² 55 at the end, vs ~350 for the full model).
- **The separation reward matters, and without it the penalty fights prediction.** With the denominator detached, the only way to lower the penalty is real invariance. The penalty stays an order of magnitude higher (0.22 vs ~0.02), pred_loss is 4× the full model's (0.107), and color AUC is 0.643.
- **Each term alone is unstable; together they're stable.** Encoder-only and predictor-only reach about 0.70 on average, but individual eval seeds (fresh state samples and head initializations) range from 0.12 to 0.95. Whether the color ranking survives then depends on the particular head. Only with both terms is every head 0.95.
- **Caveat:** one training seed per variant. The *spread* across eval seeds is a property of the encoder; the gap between full no-h and the single-term variants could still partly be training-seed luck.

Jacobians, same protocol as the section below:

| model | σ1 | ‖J‖_F | cond | eff. rank | top-5 energy | ‖J‖/cdist | ‖∇h‖/gap | color AUC |
|---|---|---|---|---|---|---|---|---|
| no h, full | 1.202 | 1.869 | 96.6 | 16.5 | 0.809 | 0.100 | 0.0256 | 0.951 |
| random noise | 1.373 | 2.982 | 50.7 | 22.6 | 0.670 | 0.347 | 0.0338 | 0.631 |
| detached denominator | 1.342 | 2.026 | 451.4 | 8.1 | 0.913 | 0.105 | 0.0415 | 0.643 |
| encoder term only | 0.694 | 1.601 | 23.4 | 34.0 | 0.578 | 0.184 | 0.0133 | 0.703 |
| predictor term only | 1.276 | 1.993 | 100.8 | 15.0 | 0.851 | 0.151 | 0.0250 | 0.691 |

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
| **LeWM + safety reg (L∞)** | 0.997 ± 0.002 | 0.987 ± 0.003 | 0.977 ± 0.005 |
| **LeWM + safety reg (L∞, no h)** | 0.990 ± 0.007 | 0.943 ± 0.005 | 0.933 ± 0.006 |
| Dreamer | 0.981 ± 0.004 | 0.835 ± 0.003 | 0.833 ± 0.000 |
| **Dreamer + safety reg (L∞)** | 0.983 ± 0.002 | 0.856 ± 0.008 | 0.833 ± 0.000 |
| **Dreamer + safety reg (L∞, no h)** | 0.982 ± 0.002 | 0.854 ± 0.005 | 0.833 ± 0.000 |

**AUC**

| model | in-dist | color (blue → purple) | rotation (real 90° camera) |
|---|---|---|---|
| LeWM sigreg only | 0.992 ± 0.003 | 0.595 ± 0.001 | 0.557 ± 0.006 |
| LeWM jac | 0.997 ± 0.001 | 0.992 ± 0.000 | 0.641 ± 0.006 |
| LeWM jac+pull | 0.999 ± 0.000 | 0.996 ± 0.001 | 0.679 ± 0.002 |
| **LeWM + safety reg (L∞)** | 1.000 ± 0.000 | 0.999 ± 0.000 | 0.962 ± 0.001 |
| **LeWM + safety reg (L∞, no h)** | 0.999 ± 0.000 | 0.985 ± 0.001 | 0.941 ± 0.005 |
| Dreamer | 0.993 ± 0.001 | 0.961 ± 0.002 | 0.503 ± 0.047 |
| **Dreamer + safety reg (L∞)** | 0.995 ± 0.001 | 0.977 ± 0.003 | 0.481 ± 0.043 |
| **Dreamer + safety reg (L∞, no h)** | 0.995 ± 0.001 | 0.979 ± 0.001 | 0.491 ± 0.025 |

**Predicted-safe fraction (true safe = .833)**

| model | in-dist | color (blue → purple) | rotation (real 90° camera) |
|---|---|---|---|
| LeWM sigreg only | 0.820 ± 0.005 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| LeWM jac | 0.833 ± 0.004 | 0.912 ± 0.002 | 0.999 ± 0.001 |
| LeWM jac+pull | 0.832 ± 0.000 | 0.888 ± 0.004 | 0.998 ± 0.002 |
| **LeWM + safety reg (L∞)** | 0.830 ± 0.002 | 0.823 ± 0.006 | 0.833 ± 0.006 |
| **LeWM + safety reg (L∞, no h)** | 0.831 ± 0.006 | 0.890 ± 0.005 | 0.858 ± 0.014 |
| Dreamer | 0.828 ± 0.005 | 0.995 ± 0.002 | 1.000 ± 0.000 |
| **Dreamer + safety reg (L∞)** | 0.832 ± 0.003 | 0.977 ± 0.007 | 1.000 ± 0.000 |
| **Dreamer + safety reg (L∞, no h)** | 0.832 ± 0.002 | 0.977 ± 0.005 | 1.000 ± 0.000 |

- **LeWM + safety reg (L∞) is the first model in the whole study whose threshold survives a shift.** Its zero-shot head keeps 0.987 sign accuracy on color and 0.977 on rotation, predicting 0.823 / 0.833 of states safe (true: 0.833). Every reference collapses to calling nearly everything safe.
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
| **Dreamer + safety reg (L∞)** | 0.988 ± 0.003 | 0.793 ± 0.024 | 0.937 ± 0.011 | 0.708 ± 0.005 |
| **Dreamer + safety reg (L∞, no h)** | 0.985 ± 0.001 | 0.726 ± 0.074 | 0.929 ± 0.005 | 0.715 ± 0.010 |

**AUC**

| model | in-dist | color (red → purple) | shape (circle → diamond) | rotation (90°) |
|---|---|---|---|---|
| Dreamer | 0.999 ± 0.000 | 0.638 ± 0.013 | 0.993 ± 0.001 | 0.480 ± 0.011 |
| **Dreamer + safety reg (L∞)** | 0.999 ± 0.000 | 0.816 ± 0.017 | 0.995 ± 0.001 | 0.484 ± 0.013 |
| **Dreamer + safety reg (L∞, no h)** | 0.999 ± 0.000 | 0.626 ± 0.048 | 0.994 ± 0.001 | 0.494 ± 0.006 |

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
| baseline + safety reg, hue | 7.860 | 13.405 | 91.8 | 16.4 | 0.827 | 4.394 | 1.1858 | 0.634 |
| jac+pull + safety reg, hue | 0.316 | 1.010 | 4.1 | 90.0 | 0.298 | 0.435 | 0.0244 | 0.681 |
| **baseline + safety reg, L∞** | 2.073 | 2.663 | 122.0 | 11.8 | 0.917 | 0.299 | 0.0266 | 0.910 |
| **jac+pull + safety reg, L∞** | 0.313 | 1.008 | 4.2 | 83.6 | 0.303 | 0.235 | 0.0113 | 0.965 |
| baseline + L∞, no h | 1.202 | 1.869 | 96.6 | 16.5 | 0.809 | 0.100 | 0.0256 | 0.951 |

- **Two different routes to low sensitivity.** The Jacobian penalty pins ‖J‖_F at ≈ 1.008 and *whitens* J: condition number ~4, ~85–90 effective dimensions, top-5 directions hold ~30% of the energy. L∞ shrinks J (‖J‖_F 11.1 → 2.7, σ1 7.8 → 2.1) but keeps baseline's *shape*: condition number 122, ~12 effective dimensions, 92% of the energy in the top 5. Relative to class separation, L∞ is 5× less pixel-sensitive than baseline (1.538 → 0.299), and its readout is 8× less sensitive (0.208 → 0.027).
- **No-h gets there by spreading the classes apart.** It has the smallest ‖J‖/cdist of all (0.100) because its centroid distance is ~2.6× baseline's (implied 18.7 vs 7.3), while its raw ‖J‖_F of 1.87 is in between. The centroid-distance denominator rewards pushing the classes apart, and in training it grew steadily (in-batch squared distance ~116 → 352). That is a legitimate way to lower the penalty, but it may also explain the higher pred_loss (0.0253).
- **On a jac+pull base, adding L∞ is invisible in J.** Every column of jac+pull + L∞ matches jac+pull (‖∇h‖/gap 0.0113 vs 0.0112), yet color AUC goes 0.753 → 0.965. So the L∞ gain on top of jac+pull is not a local first-order effect at these frames. The PGD perturbation is a finite step (8/255 on every pixel), and purple is a large shift, so whatever changed lives in the non-local geometry.
- **Local sensitivity does not rank color robustness.** Plain jacobian has the lowest readout sensitivity of all (0.0096) but only 0.673 color AUC, while L∞ is 3× more sensitive and reaches 0.910. This matches the earlier jacobian-mechanism finding that the benefit is not global smoothness.
- **Hue made the readout *more* pixel-sensitive** on the sigreg base (‖∇h‖/gap 1.19, ~6× baseline; ‖J‖/cdist 4.39). This fits its weak shape result (Appendix B). Training against a structured color field doesn't buy generic noise robustness.

## Interpretation

1. **Adversarial noise invariance transfers to large shifts, and the safety projection isn't needed, but the rest is.** Small adversarial pixel noise gives robustness to three shifts it never saw, including the color shift that a hand-designed color perturbation handled worse (Appendix B). The no-h ablation penalizes the whole latent instead of the readout and is at least as good (color 0.951 ± 0.004 vs 0.910 ± 0.098). The follow-up ablations show that color needs all of the following:
   - a worst-case search: random noise of the same size gives 0.631, consistent with earlier *global* random-noise invariance regularizers never fixing color;
   - the gradient through the class-separation denominator: detached gives 0.643, and prediction suffers;
   - both the encoder and the predictor term: either alone is unstable across heads.

   Shape and rotation, by contrast, are easy: any of these variants gets them.
2. **It goes well beyond jacobian, by a different mechanism.** Jacobian whitens the local Jacobian and reaches color AUC 0.673. The L∞ variants keep or even sharpen the anisotropic Jacobian and reach 0.91–0.97. The two combine: jac+pull + L∞ is the best color model.
3. **The remaining failure is calibration.** Even the best ranking comes with a coherent readout offset under purple that pushes every state across the threshold, and every L∞ variant has it. For a deployable filter, this needs recalibration or an offset-correction mechanism, not more invariance. On Safety Gym, LeWM + L∞ is the exception: its threshold holds under both shifts. That suggests the offset depends on the domain, perhaps on how strongly appearance and safety are entangled at the input (which is high on Dubins), rather than being inherent to the method.

## Caveats

- **One training seed per model.** The 3 seeds are eval seeds (sampled states and head initializations). L∞'s color result varies a lot across them (0.910 ± 0.098), and the ordering among the three L∞ variants on color (0.910 / 0.951 / 0.965) is within what one training seed can move. A second training seed is needed before ranking them.
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

### A.10 Ablation: remove h (penalize the whole latent)

Run `lewm_dubins_linf_latent50`, code `dab24f2` (`target: latent`), launched 2026-10-06 00:45 UTC. Everything is identical to the L∞ run (ε, PGD, 32 windows, BatchNorm handling, schedule, and the head, which is still trained because it drives the AUC gate) except the quantity being protected. The head no longer appears in the penalty:

d = mean_t ‖f(x_t + δ) − f(x_t)‖² / C + ‖P(f(x + δ), a) − P(f(x), a)‖² / C,  C = ‖μ_safe − μ_unsafe‖² (in-batch latent centroids, with gradient)

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
- **Checkpoints** (under `/data/seongbin/lewm/checkpoints/`): `lewm_dubins_safeadv_linf50/` (L∞), `lewm_dubins_jacpull_safeadv_linf50/` (jac+pull + L∞), `lewm_dubins_linf_latent50/` (no h; code `dab24f2`, `target: latent`), `lewm_dubins_lz_{rand,det,enc,pred}50/` (no-h ablations; code `9dfca47`), `sg_safeadv_linf50_g0/` and `sg_linf_latent50_g0/` (Safety Gym, code `b9d34e2`, launchers `/data/seongbin/lewm/code_safeadv_b9d34e2/run_sg{L,LZ}g0.sh`). Dreamer: `/data/seongbin/dreamer/dreamer/safeadv/{dubins,sg}_sa_{head,latent}/rssm_ckpt.pt` (code snapshot `/data/seongbin/dreamer/code_safeadv_e99b14c/`, logs `safeadv/*.log`), `lewm_dubins_safeadv2_50/` and `lewm_dubins_jacpull_safeadv2_50/` (hue), `lewm_dubins_safeadv50/` (v1). The in-training margin head is only in the Lightning checkpoints (`~/.cache/stable-pretraining/runs/<date>/<time>/<hash>/checkpoints/`).
- **Training logs and code snapshots:** `/data/seongbin/lewm/code_safeadv_77856fd/` (L∞, `.run_L/`; jac+pull + L∞, `.run_JL/`), `code_safeadv_dab24f2/` (no h, `.run_LZ/`), `code_safeadv_9dfca47/` (ablations, `run_<k>.sh`, `.run_<k>/`), `code_safeadv_c6e93c6/` (hue), `code_safeadv_ca9ec21/` (v1). W&B project `seongbin/lewm`.
- **Eval logs** (under `/data/seongbin/lewm/safeadv_results/`): `linf_ood_eval/`, `jacpull_linf_ood_eval/`, `linf_latent_ood_eval/` (no h), `ablation_eval/` (no-h ablations, incl. `jac_ablation.out`), `sg_eval/` (Safety Gym eval script, logs, `tables.py` for the Safety Gym and Dreamer tables), `dreamer/` (Dubins Dreamer eval logs, Safety Gym label fix `fix_sg_labels.py`), `v2_ood_eval/` (hue), `v1_ood_eval/`, `predsafe_check/`, `balanced_eval/` (50/50); diagnostic scripts in `diagnostics/` (Jacobian comparison: `jac_compare.py` / `.out` / `.npy`). Tables are generated by `make_tables.py`.
- **Baseline numbers:** `/home/seongbin/latent/run_logs/ood_gp_jepa_*_s*.log` (same eval script, `le-wm/scripts/ood_margin_gp_jepa.py`).
