# Dubins, Safety Gym and Cartpole

# Regularizers

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

# Dubins

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

## Caveats and next steps

- One training seed per model. Single-seed safety-reg results have been lucky before (Dubins). Seeds 1 and 2 of all 9
  recipes are training (`le-wm/scripts/run_point_ovh_seeds.sh`); the same eval runs automatically when they finish.
- Labels are position-only failure (agent inside a hazard). The velocity-dependent value (the doomed band above) is not yet
  used as a target; `point_doomed.py` can generate Monte Carlo value labels.
- Possible variant: a fixed-layout Point benchmark (Dubins-like), which would also make an exact HJ value oracle feasible.
- No Point Dreamer has been trained.

# Cartpole

![Cartpole base vs OOD frames](figs/ood_frames_cartpole.png)

Goal: a benchmark where safety is coupled to dynamics and the safe set can be computed exactly. A cart can be inside the
track with the pole upright and still be lost, because it is moving too fast to stop. Classic Gym CartPole-v1 (Euler step,
τ = 0.02 s, ±10 N push); failure = |x| > 2.4 or |θ| > 12°. Code and logs: branch `worktree-cartpole-port` (`cartpole_port/`,
`run_logs/cartpole_eval/`); eval outputs in `/data/seongbin/lewm/cartpole_eval/`.

## Setup

- Rendering: frames are drawn from the state (`render.py`, 128 px) rather than with gym's renderer, whose anti-aliased
  edges would keep the old color after a recolor. Every frame is a function of the state alone, so each OOD frame is the
  same state as its base frame and labels are exact under every shift. The pole is drawn 2.5 m long (physics keeps 1.0 m);
  otherwise a 12° lean is about 4 px wide.
- Value oracle (`value_dp.py`): safety value iteration on the env's own step, V(s) = min(ℓ(s), max_a V(f(s, a))), on a 61⁴
  grid. A state is doomed if ℓ ≥ 0 (not failed yet) but V < 0 (no action sequence avoids failure). No state with V < -0.02
  is saved by any tested policy; 91% of states with V > 0.02 survive 500 steps under the greedy-V policy.
- Training data: `cartpole_expert.h5`, 170,040 frames in 5,175 episodes. Policy = greedy-V with random actions at a
  per-episode rate ε ~ U(0, 0.6); half the episodes start near upright, half anywhere not yet failed (including doomed
  starts); episodes end 8 frames after failure. Filtered like Dubins to 17.0% failed frames (plus 15.7% doomed).
  Rollouts: `figs/cartpole_rollouts/cartpole_{greedy,noisy_eps0.4,doomed_start}.gif`.
- Eval: margin_gp head (Dubins loss, 50/50 safe/failed batches) on the frozen encoder, trained on the failure label on base
  frames of 70% of episodes, scored on the other 30% (51,282 frames) re-rendered under each shift. Views: base, purple pole
  (color; tan in training), 0.8× zoom out (zoom_out), 0.5 m horizontal pan at the zoom_out scale (pan_x), 0.3 m vertical pan
  (pan_y). Samples: `figs/cartpole_rollouts/cartpole_ood_samples.png`. Zooming in or panning sideways at native scale was
  dropped: the native view only just fits a pole leaning out at the wall, so it would crop the pole in 24-40% of failed
  frames.

## Results: 9 world models

le-wm, 50 epochs, seed 3072, same recipes and code snapshots as the Dubins runs (the adv_h gate opened: in-training head AUC
0.991). ± = std over 3 margin-function seeds (init only; the episode split is fixed) on one WM per model.

Frames are fed in the training orientation (fixed 2026-10-28). The earlier cartpole numbers came from `eval_cartpole.encode_pixels`, which passes numpy HWC frames to `ToImage` and so transposes them (the cart lying on its side); a rerun of that path reproduces the old table (AUC within 0.001, sign accuracy within 0.004). Scripts `cp_bal/margin_ood_bal_fixed.py`, outputs `/data/seongbin/lewm/cartpole_eval_balanced/fixed_orient/`.

AUC

| Model | base | color | zoom_out | pan_x | pan_y |
|---|---|---|---|---|---|
| sigreg | 1.000 ± 0.000 | 0.853 ± 0.006 | 0.984 ± 0.000 | 0.978 ± 0.001 | 0.971 ± 0.001 |
| jac | 1.000 ± 0.000 | 0.978 ± 0.001 | 0.977 ± 0.002 | 0.971 ± 0.002 | 0.928 ± 0.000 |
| jac+pull | 1.000 ± 0.000 | 0.971 ± 0.001 | 0.994 ± 0.000 | 0.986 ± 0.000 | 0.938 ± 0.001 |
| adv_h | 1.000 ± 0.000 | 0.983 ± 0.001 | 0.970 ± 0.000 | 0.964 ± 0.001 | 0.865 ± 0.006 |
| jac+adv_h | 1.000 ± 0.000 | 0.944 ± 0.001 | 0.991 ± 0.001 | 0.993 ± 0.000 | 0.939 ± 0.001 |
| jac+pull+adv_h | 1.000 ± 0.000 | 0.855 ± 0.001 | 0.989 ± 0.001 | 0.991 ± 0.000 | 0.950 ± 0.003 |
| adv_λ | 1.000 ± 0.000 | 0.983 ± 0.000 | 0.985 ± 0.000 | 0.977 ± 0.001 | 0.874 ± 0.006 |
| jac+adv_λ | 1.000 ± 0.000 | 0.973 ± 0.000 | 0.979 ± 0.001 | 0.988 ± 0.000 | 0.953 ± 0.001 |
| jac+pull+adv_λ | 0.999 ± 0.000 | 0.967 ± 0.001 | 0.980 ± 0.000 | 0.985 ± 0.001 | 0.944 ± 0.002 |

Sign accuracy, balanced: (TPR_safe + TPR_unsafe) / 2 on the full test set, i.e. the expected accuracy on a 50/50 safe/failed split (0.500 = everything called one class). The margin function is trained on the natural split.

| Model | base | color | zoom_out | pan_x | pan_y |
|---|---|---|---|---|---|
| sigreg | 0.992 ± 0.001 | 0.658 ± 0.001 | 0.877 ± 0.009 | 0.873 ± 0.009 | 0.880 ± 0.002 |
| jac | 0.990 ± 0.001 | 0.914 ± 0.003 | 0.864 ± 0.009 | 0.847 ± 0.011 | 0.679 ± 0.008 |
| jac+pull | 0.992 ± 0.001 | 0.780 ± 0.006 | 0.894 ± 0.006 | 0.905 ± 0.005 | 0.801 ± 0.007 |
| adv_h | 0.992 ± 0.001 | 0.802 ± 0.003 | 0.889 ± 0.001 | 0.894 ± 0.001 | 0.747 ± 0.003 |
| jac+adv_h | 0.992 ± 0.001 | 0.650 ± 0.002 | 0.846 ± 0.005 | 0.820 ± 0.005 | 0.768 ± 0.003 |
| jac+pull+adv_h | 0.992 ± 0.002 | 0.648 ± 0.001 | 0.827 ± 0.004 | 0.829 ± 0.003 | 0.821 ± 0.002 |
| adv_λ | 0.993 ± 0.000 | 0.724 ± 0.010 | 0.872 ± 0.003 | 0.894 ± 0.006 | 0.748 ± 0.001 |
| jac+adv_λ | 0.989 ± 0.001 | 0.651 ± 0.001 | 0.827 ± 0.009 | 0.819 ± 0.010 | 0.808 ± 0.008 |
| jac+pull+adv_λ | 0.988 ± 0.001 | 0.669 ± 0.010 | 0.830 ± 0.002 | 0.842 ± 0.003 | 0.829 ± 0.007 |

Reading:

- Viewpoint is mostly easy for every model (AUC 0.865-0.994). The earlier "viewpoint breaks every model" came from the transpose.
- Color: sigreg is the weakest (0.853); jac, adv_h and adv_λ reach 0.978-0.983; jac+pull+adv_h is low (0.855).
- Thresholds: OOD sign errors almost all lean toward calling states safe (balanced predicted-safe fraction > 0.5). On color only jac keeps its threshold (0.914); sigreg calls 83% of frames safe despite AUC 0.853. adv_h is the only model calibrated (predicted-safe about 0.50) on all three viewpoint shifts.
- The sections below (why viewpoint fails, value functions, support vs aliasing) still use the transposed frames and need rerunning.

## Why viewpoint fails: the latent moves, the information stays

Test episodes split in half: heads trained on one half, scored on the other, either zero-shot (trained on base frames) or
retrained on the shifted frames (`shift_diag.py`). "latent move" = mean ‖z_shift(s) - z(s)‖ over paired states divided by
the mean distance between two random states. Four of the models.

| Model | shift | latent move | margin zero-shot | margin retrained | supervised V zero-shot | supervised V retrained |
|---|---|---|---|---|---|---|
| sigreg | zoom_out | 1.372 | 0.612 ± 0.006 | 0.995 ± 0.001 | 0.599 ± 0.019 | 0.943 ± 0.001 |
| jac+pull | zoom_out | 0.919 | 0.699 ± 0.003 | 0.984 ± 0.000 | 0.604 ± 0.002 | 0.936 ± 0.004 |
| jac+pull+adv_λ | zoom_out | 1.310 | 0.575 ± 0.006 | 0.971 ± 0.004 | 0.647 ± 0.026 | 0.930 ± 0.001 |
| jac+adv_h | zoom_out | 0.951 | 0.587 ± 0.013 | 0.981 ± 0.003 | 0.570 ± 0.034 | 0.933 ± 0.001 |
| sigreg | color | 0.597 | 0.637 ± 0.012 | 0.991 ± 0.000 | 0.703 ± 0.004 | 0.934 ± 0.003 |
| jac+pull | color | 0.598 | 0.780 ± 0.005 | 0.987 ± 0.000 | 0.663 ± 0.010 | 0.931 ± 0.002 |
| jac+pull+adv_λ | color | 0.340 | 0.904 ± 0.004 | 0.981 ± 0.001 | 0.893 ± 0.003 | 0.930 ± 0.001 |
| jac+adv_h | color | 0.441 | 0.777 ± 0.014 | 0.984 ± 0.004 | 0.830 ± 0.004 | 0.933 ± 0.002 |

- Nothing is lost: a margin retrained on zoomed frames is as good as in-distribution (0.971-0.995).
- A 0.8× zoom moves the latent about as far as the distance between two random states (0.92-1.37), roughly twice the color
  shift. Every zoomed frame lands where the head never saw data. A likely reason, not tested: in training only the cart and
  pole move, while zoom also moves and rescales the walls, track and background, which never changed.
- adv_λ's color invariance is specific to color: it has the smallest color move (0.340) but a zoom move as large
  as sigreg's (1.310).

## Value functions: critic and supervised

The margin is "currently safe"; doomed states need a separate value function. Two heads, both on the last three latents
[z_{t-2}, z_{t-1}, z_t] (one frame carries no velocity):

- critic: trained by RL on the learned margin, the discounted safety Bellman backup, with no value labels. This version is
  offline fitted Q-iteration on dataset transitions; it will be replaced by the Dubins setup (avoid-DDPG in WM imagination,
  `wm_ddpg.py`).
- supervised value function: regression on the oracle V, a diagnostic for whether the information is in the latent at all.

Scored by doomed AUC: among not-failed frames, doomed vs safe. The same critic on the true state reaches 0.985 ± 0.004.

Critic doomed AUC

| Model | base | color | zoom_out | pan_x | pan_y |
|---|---|---|---|---|---|
| sigreg | 0.953 ± 0.002 | 0.785 ± 0.014 | 0.681 ± 0.048 | 0.689 ± 0.040 | 0.770 ± 0.014 |
| jac | 0.962 ± 0.002 | 0.797 ± 0.008 | 0.699 ± 0.015 | 0.735 ± 0.020 | 0.618 ± 0.024 |
| jac+pull | 0.962 ± 0.002 | 0.850 ± 0.011 | 0.617 ± 0.006 | 0.666 ± 0.013 | 0.670 ± 0.033 |
| adv_h | 0.940 ± 0.001 | 0.788 ± 0.023 | 0.740 ± 0.005 | 0.591 ± 0.014 | 0.732 ± 0.022 |
| jac+adv_h | 0.952 ± 0.004 | 0.893 ± 0.007 | 0.692 ± 0.060 | 0.687 ± 0.079 | 0.770 ± 0.035 |
| jac+pull+adv_h | 0.948 ± 0.002 | 0.863 ± 0.004 | 0.669 ± 0.078 | 0.692 ± 0.057 | 0.748 ± 0.031 |
| adv_λ | 0.939 ± 0.003 | 0.843 ± 0.020 | 0.646 ± 0.019 | 0.512 ± 0.017 | 0.705 ± 0.019 |
| jac+adv_λ | 0.947 ± 0.003 | 0.913 ± 0.004 | 0.563 ± 0.057 | 0.587 ± 0.052 | 0.589 ± 0.016 |
| jac+pull+adv_λ | 0.958 ± 0.002 | 0.947 ± 0.003 | 0.642 ± 0.007 | 0.584 ± 0.019 | 0.626 ± 0.009 |

Supervised value function doomed AUC

| Model | base | color | zoom_out | pan_x | pan_y |
|---|---|---|---|---|---|
| sigreg | 0.953 ± 0.001 | 0.772 ± 0.022 | 0.625 ± 0.020 | 0.616 ± 0.017 | 0.662 ± 0.023 |
| jac | 0.949 ± 0.001 | 0.701 ± 0.020 | 0.663 ± 0.017 | 0.730 ± 0.017 | 0.603 ± 0.015 |
| jac+pull | 0.950 ± 0.002 | 0.750 ± 0.008 | 0.672 ± 0.002 | 0.702 ± 0.008 | 0.703 ± 0.028 |
| adv_h | 0.946 ± 0.001 | 0.772 ± 0.015 | 0.672 ± 0.005 | 0.559 ± 0.009 | 0.790 ± 0.005 |
| jac+adv_h | 0.946 ± 0.001 | 0.863 ± 0.006 | 0.665 ± 0.027 | 0.625 ± 0.042 | 0.793 ± 0.019 |
| jac+pull+adv_h | 0.947 ± 0.002 | 0.814 ± 0.010 | 0.629 ± 0.016 | 0.638 ± 0.011 | 0.693 ± 0.035 |
| adv_λ | 0.943 ± 0.002 | 0.834 ± 0.009 | 0.605 ± 0.001 | 0.501 ± 0.004 | 0.682 ± 0.007 |
| jac+adv_λ | 0.942 ± 0.003 | 0.878 ± 0.007 | 0.556 ± 0.011 | 0.551 ± 0.038 | 0.556 ± 0.005 |
| jac+pull+adv_λ | 0.944 ± 0.001 | 0.924 ± 0.003 | 0.666 ± 0.006 | 0.632 ± 0.022 | 0.617 ± 0.016 |

Reading:

- In distribution the critic finds doomed states without doomed labels (0.939-0.962), as well as the supervised value function
  (0.942-0.953). The margin alone ranks doomed below safe at only about 0.70, so the separate value function is needed.
- Under the shifts the critic follows the margin's pattern: adv_λ best on color (jac+pull+adv_λ 0.947), every
  model weak on viewpoint (0.512-0.770). The supervised value function drops just as much, so the failure is in the latent, not the critic.

## Caveats and next steps

- One training seed per model. Single-seed color results on Dubins did not hold over 3 seeds, so the color ranking here
  needs reseeds.
- The critic above is the offline version; the Dubins-style imagination critic is next.
- The oracle is slightly optimistic near V = 0 (9% of V > 0.02 states fail under the greedy policy). Doomed labels are clean.
- A viewpoint fix probably needs geometric training (zoom/pan augmentation, or an adversary over small camera shifts)
  rather than appearance invariance.

# Support vs. aliasing: where does an OOD shift send the latent?

An OOD shift can fail a frozen safety head in two different ways:
- Off support: the shifted latent leaves the region the training latents occupy. The head's value there is
  extrapolation. A support- or uncertainty-based OOD detector can catch it, and a head retrained on the new
  appearance can recover.
- Aliasing: the shifted latent stays inside the training support but lands among training latents of the other class.
  The frame looks like a different, real state. No support-based detector can flag it.

Method (`ood_diag/support_alias.py` on branch `worktree-support-aliasing`; WM frozen; same paired renders, margin_gp
head and train/test split as each benchmark's main eval; one head seed):
- Support: mean distance to the 10 nearest training latents. "Off support" = above the 99th percentile of the same
  distance for native test frames, so about 0.01 off support by construction in distribution.
- Sign flips: the head's sign on the shifted frame differs from its sign on the native frame of the same state.
- kNN-neighbourhood AUC: AUC of "fraction of safe training neighbours", a head-free read of whether a frame's
  neighbourhood carries its class. Unsafe aliasing rate: unsafe states whose 10 nearest training latents are mostly safe.
- Point: "native" is the base view on unseen layouts (the eval set); the layout table compares those frames with
  held-out episodes of the training layouts.

## Dubins

Off-support fraction (kNN distance above the 99th percentile of native test frames)

| model | color | rotate | shape |
|---|---|---|---|
| sigreg | 1.000 | 1.000 | 0.997 |
| jac | 0.000 | 0.063 | 0.003 |
| jac+pull | 0.000 | 0.073 | 0.004 |
| adv_h | 0.999 | 0.244 | 0.011 |
| jac+pull+adv_h | 0.000 | 0.068 | 0.009 |
| adv_λ | 1.000 | 0.928 | 0.009 |

Sign flips vs the native frame: inside support / off support

| model | color | rotate | shape |
|---|---|---|---|
| sigreg | 0.000 / 0.814 | 0.000 / 0.186 | 0.000 / 0.608 |
| jac | 0.839 / 0.000 | 0.036 / 0.005 | 0.045 / 0.001 |
| jac+pull | 0.163 / 0.000 | 0.061 / 0.013 | 0.054 / 0.001 |
| adv_h | 0.001 / 0.825 | 0.003 / 0.016 | 0.052 / 0.004 |
| jac+pull+adv_h | 0.164 / 0.000 | 0.017 / 0.004 | 0.058 / 0.002 |
| adv_λ | 0.000 / 0.823 | 0.000 / 0.022 | 0.057 / 0.000 |

kNN-neighbourhood AUC (fraction of safe training neighbours as the score)

| model | native | color | rotate | shape |
|---|---|---|---|---|
| sigreg | 0.962 | 0.500 | 0.646 | 0.844 |
| jac | 0.983 | 0.662 | 0.990 | 0.987 |
| jac+pull | 0.974 | 0.500 | 0.980 | 0.955 |
| adv_h | 0.997 | 0.572 | 0.996 | 0.997 |
| jac+pull+adv_h | 0.994 | 0.500 | 0.990 | 0.979 |
| adv_λ | 0.996 | 0.493 | 0.997 | 0.994 |

Unsafe states whose nearest training latents are mostly safe

| model | native | color | rotate | shape |
|---|---|---|---|---|
| sigreg | 0.217 | 0.000 | 0.994 | 0.000 |
| jac | 0.165 | 1.000 | 0.193 | 0.155 |
| jac+pull | 0.275 | 1.000 | 0.217 | 0.311 |
| adv_h | 0.031 | 0.000 | 0.080 | 0.087 |
| jac+pull+adv_h | 0.125 | 1.000 | 0.107 | 0.183 |
| adv_λ | 0.037 | 0.000 | 0.128 | 0.114 |


## Cartpole

Off-support fraction (kNN distance above the 99th percentile of native test frames)

| model | purple | zoom_out | pan_x | pan_y |
|---|---|---|---|---|
| sigreg | 0.828 | 1.000 | 1.000 | 1.000 |
| jac | 0.887 | 1.000 | 1.000 | 1.000 |
| jac+pull | 0.892 | 1.000 | 1.000 | 1.000 |
| adv_h | 0.430 | 0.947 | 1.000 | 0.970 |
| jac+adv_h | 0.731 | 1.000 | 1.000 | 1.000 |
| jac+pull+adv_h | 0.692 | 1.000 | 1.000 | 1.000 |
| adv_λ | 0.383 | 0.774 | 0.802 | 0.937 |
| jac+adv_λ | 0.355 | 1.000 | 1.000 | 1.000 |
| jac+pull+adv_λ | 0.311 | 1.000 | 1.000 | 1.000 |

Sign flips vs the native frame: inside support / off support

| model | purple | zoom_out | pan_x | pan_y |
|---|---|---|---|---|
| sigreg | 0.039 / 0.309 | 0.000 / 0.649 | 0.000 / 0.639 | 0.000 / 0.715 |
| jac | 0.012 / 0.493 | 0.000 / 0.563 | 0.000 / 0.541 | 0.000 / 0.360 |
| jac+pull | 0.008 / 0.268 | 0.000 / 0.503 | 0.000 / 0.473 | 0.000 / 0.362 |
| adv_h | 0.149 / 0.127 | 0.001 / 0.305 | 0.000 / 0.482 | 0.000 / 0.248 |
| jac+adv_h | 0.034 / 0.122 | 0.000 / 0.590 | 0.000 / 0.394 | 0.000 / 0.336 |
| jac+pull+adv_h | 0.036 / 0.140 | 0.000 / 0.345 | 0.000 / 0.295 | 0.000 / 0.474 |
| adv_λ | 0.095 / 0.140 | 0.061 / 0.333 | 0.080 / 0.347 | 0.024 / 0.277 |
| jac+adv_λ | 0.039 / 0.079 | 0.000 / 0.804 | 0.000 / 0.791 | 0.000 / 0.452 |
| jac+pull+adv_λ | 0.080 / 0.073 | 0.000 / 0.774 | 0.000 / 0.787 | 0.000 / 0.750 |

kNN-neighbourhood AUC (fraction of safe training neighbours as the score)

| model | native | purple | zoom_out | pan_x | pan_y |
|---|---|---|---|---|---|
| sigreg | 0.988 | 0.585 | 0.650 | 0.655 | 0.771 |
| jac | 0.996 | 0.707 | 0.747 | 0.751 | 0.745 |
| jac+pull | 0.995 | 0.730 | 0.673 | 0.688 | 0.739 |
| adv_h | 0.986 | 0.649 | 0.753 | 0.615 | 0.774 |
| jac+adv_h | 0.994 | 0.705 | 0.569 | 0.606 | 0.668 |
| jac+pull+adv_h | 0.993 | 0.737 | 0.599 | 0.639 | 0.759 |
| adv_λ | 0.988 | 0.767 | 0.710 | 0.672 | 0.579 |
| jac+adv_λ | 0.986 | 0.871 | 0.509 | 0.501 | 0.587 |
| jac+pull+adv_λ | 0.985 | 0.831 | 0.527 | 0.503 | 0.602 |

Unsafe states whose nearest training latents are mostly safe

| model | native | purple | zoom_out | pan_x | pan_y |
|---|---|---|---|---|---|
| sigreg | 0.145 | 0.670 | 0.703 | 0.686 | 0.531 |
| jac | 0.152 | 0.567 | 0.176 | 0.167 | 0.529 |
| jac+pull | 0.141 | 0.626 | 0.596 | 0.572 | 0.584 |
| adv_h | 0.231 | 0.699 | 0.635 | 0.724 | 0.363 |
| jac+adv_h | 0.182 | 0.673 | 0.648 | 0.713 | 0.727 |
| jac+pull+adv_h | 0.186 | 0.612 | 0.626 | 0.603 | 0.390 |
| adv_λ | 0.255 | 0.583 | 0.622 | 0.702 | 0.738 |
| jac+adv_λ | 0.266 | 0.455 | 0.495 | 0.547 | 0.561 |
| jac+pull+adv_λ | 0.267 | 0.571 | 0.639 | 0.583 | 0.391 |


## Point (overhead@224)

Layout shift (base view, unseen layouts vs held-out frames of training layouts)

| model | off support | distance ratio | head sign error | kNN class mismatch (in support) |
|---|---|---|---|---|
| sigreg | 0.967 | 49.77 | 0.209 | 0.518 |
| jac | 0.076 | 1.25 | 0.114 | 0.104 |
| jac+pull | 0.110 | 1.36 | 0.159 | 0.166 |
| adv_h | 0.717 | 26.10 | 0.076 | 0.093 |
| jac+adv_h | 0.049 | 1.14 | 0.049 | 0.071 |
| jac+pull+adv_h | 0.112 | 1.43 | 0.152 | 0.154 |
| adv_λ | 0.543 | 325.29 | 0.122 | 0.141 |
| jac+adv_λ | 0.023 | 1.11 | 0.083 | 0.090 |
| jac+pull+adv_λ | 0.026 | 1.12 | 0.072 | 0.084 |

Off-support fraction (kNN distance above the 99th percentile of native test frames)

| model | color | rotate | shape | agent |
|---|---|---|---|---|
| sigreg | 0.973 | 0.001 | 0.000 | 0.010 |
| jac | 0.036 | 0.006 | 0.008 | 0.027 |
| jac+pull | 0.066 | 0.011 | 0.032 | 0.019 |
| adv_h | 0.000 | 0.007 | 0.307 | 0.009 |
| jac+adv_h | 0.017 | 0.002 | 0.012 | 0.301 |
| jac+pull+adv_h | 0.000 | 0.003 | 0.002 | 0.000 |
| adv_λ | 0.041 | 0.003 | 0.002 | 0.010 |
| jac+adv_λ | 0.268 | 0.006 | 0.015 | 0.939 |
| jac+pull+adv_λ | 0.304 | 0.008 | 0.015 | 0.970 |

Sign flips vs the native frame: inside support / off support

| model | color | rotate | shape | agent |
|---|---|---|---|---|
| sigreg | 0.000 / 0.046 | 0.060 / 0.000 | 0.048 / 0.000 | 0.005 / 0.000 |
| jac | 0.545 / 0.009 | 0.051 / 0.001 | 0.022 / 0.000 | 0.101 / 0.018 |
| jac+pull | 0.099 / 0.000 | 0.035 / 0.003 | 0.035 / 0.000 | 0.043 / 0.000 |
| adv_h | 0.869 / 0.000 | 0.052 / 0.000 | 0.128 / 0.000 | 0.063 / 0.000 |
| jac+adv_h | 0.110 / 0.002 | 0.100 / 0.000 | 0.030 / 0.001 | 0.058 / 0.041 |
| jac+pull+adv_h | 0.081 / 0.000 | 0.067 / 0.001 | 0.031 / 0.000 | 0.056 / 0.000 |
| adv_λ | 0.170 / 0.000 | 0.056 / 0.000 | 0.009 / 0.000 | 0.053 / 0.000 |
| jac+adv_λ | 0.258 / 0.038 | 0.087 / 0.001 | 0.047 / 0.001 | 0.005 / 0.152 |
| jac+pull+adv_λ | 0.150 / 0.023 | 0.097 / 0.003 | 0.036 / 0.001 | 0.001 / 0.173 |

kNN-neighbourhood AUC (fraction of safe training neighbours as the score)

| model | native | color | rotate | shape | agent |
|---|---|---|---|---|---|
| sigreg | 0.502 | 0.410 | 0.470 | 0.334 | 0.473 |
| jac | 0.908 | 0.679 | 0.829 | 0.887 | 0.722 |
| jac+pull | 0.692 | 0.642 | 0.727 | 0.719 | 0.494 |
| adv_h | 0.929 | 0.528 | 0.881 | 0.391 | 0.853 |
| jac+adv_h | 0.967 | 0.867 | 0.924 | 0.970 | 0.791 |
| jac+pull+adv_h | 0.708 | 0.556 | 0.754 | 0.808 | 0.601 |
| adv_λ | 0.738 | 0.343 | 0.878 | 0.775 | 0.515 |
| jac+adv_λ | 0.953 | 0.771 | 0.905 | 0.962 | 0.799 |
| jac+pull+adv_λ | 0.940 | 0.791 | 0.920 | 0.960 | 0.720 |

Unsafe states whose nearest training latents are mostly safe

| model | native | color | rotate | shape | agent |
|---|---|---|---|---|---|
| sigreg | 0.818 | 0.825 | 0.979 | 0.975 | 0.819 |
| jac | 0.578 | 0.151 | 0.696 | 0.709 | 0.925 |
| jac+pull | 0.948 | 0.568 | 0.819 | 0.875 | 0.984 |
| adv_h | 0.282 | 0.000 | 0.321 | 0.804 | 0.408 |
| jac+adv_h | 0.218 | 0.513 | 0.398 | 0.098 | 0.638 |
| jac+pull+adv_h | 0.869 | 0.904 | 0.799 | 0.788 | 0.953 |
| adv_λ | 0.642 | 0.835 | 0.160 | 0.547 | 0.958 |
| jac+adv_λ | 0.321 | 0.134 | 0.401 | 0.062 | 0.551 |
| jac+pull+adv_λ | 0.291 | 0.286 | 0.389 | 0.066 | 0.626 |


## Reading

- On Dubins color, the failure type splits by base, not by safety reg. The sigreg-base models (sigreg, adv_h,
  adv_λ) send 100% of purple frames off support, and every sign flip happens there. The jac-base models
  (jac, jac+pull, jac+pull+adv_h) keep 100% in support, and every unsafe purple state sits among mostly safe
  training latents (1.000): the purple obstacle encodes like free space. That is aliasing.
- Neighbourhood aliasing is not the same as information loss. jac+pull+adv_h has kNN AUC 0.500 on purple, but its head
  still ranks at 0.965 (main table). The color shift moves latents along a low-variance direction that Euclidean
  neighbours ignore and the head reads. Whether the information survives still needs the retrain test. Earlier
  results: jac retrained on purple recovers (0.969); jac+pull's retrain was unstable (0.48 ± 0.10).
- Dubins rotation and shape: the jac-base latents stay in support with 2-7% flips. sigreg leaves the support on
  all three shifts. On rotation its head still ranks at 0.956 anyway, so extrapolation can go well.
- Cartpole viewpoint is pure extrapolation. Every model sends 77-100% of zoomed or panned frames off support, at
  several times the normal kNN distance. That fits "the latent moves, the information stays" in the cartpole section:
  heads retrained on the shifted view recover, as expected when the latent leaves the support instead of aliasing. On the
  purple pole the adv_λ models stay closest (31-38% off support) and keep the best neighbourhood AUC
  (0.77-0.87). That matches adv_λ being the color winner there.
- Point, unseen layouts: the sigreg-base models put new layouts off support (sigreg 97%, adv_h 72%, adv_λ 54%). The jac
  base keeps them in support (2-11%), and those are the models whose margin functions generalize. Under color, failures happen in
  support: adv_h flips 87% of frames without leaving the support, jac 55%. The Car-robot (agent) shift sends the
  jac+adv_λ and jac+pull+adv_λ latents off support (94-97%).
- Net: the Jacobian penalty keeps OOD frames on the training manifold, so its remaining failures are aliasing, which a
  support-based OOD detector (e.g. uncertainty-aware latent safety filters) cannot see. The sigreg-base models fail by
  leaving the manifold, which such a detector would catch.

## Caveats

- One WM training seed and one head seed per row. Dubins has only the six trained recipes.
- Distances are Euclidean on the raw 192-d projector output, k = 10. The 99th-percentile threshold is relative to each
  model's own latent scale.
- Dubins latent neighbourhoods are not pose-local even in distribution: the nearest training latent's state is a median
  0.8-1.3 away (position plus heading on the unit circle), against 0.25-0.28 on cartpole. So the Dubins kNN numbers
  describe neighbourhoods in the latent's own geometry, not pose retrieval.
- Point layout distance ratios (up to 325×) partly reflect that training layouts recur across episodes, so held-out
  training frames have same-layout neighbours.

# Planning: soft-cost CEM (Dubins)

CEM with the margin function's unsafe region as an extra cost on the imagined rollout: $\text{cost} = \text{goal MSE} + w \sum_t \mathrm{relu}(-h(\hat z_t)) + C\,\mathbb{1}[\min_t h(\hat z_t) < 0]$. Unsafe-looking candidates rank badly instead of being removed (the earlier hard filter, `safe_solver.py` gauge map, collided more than plain CEM). The graded version has $C = 0$; the lexicographic version adds $C = 10^4$ (larger than any goal cost, about 130), so every candidate $h$ calls safe beats every unsafe one, while the hinge still ranks the unsafe ones. Code: `SoftSafeCEMSolver` in `le-wm/safe_solver.py` (`solver=soft_cem`), runs `le-wm/scripts/soft_cem*.sh`, logs `run_logs/softcem_<model>_w<w>[_c<C>]_<condition>.log`.

- $h$ here is a supervised value function, not a margin function: MSE regression onto the oracle HJ value $V^*$ (computed offline by a grid solver on the true dynamics, `hj_truth.npz`) at random states rendered red and encoded by the frozen WM (`train_margin_hj_lewm.py`, `margin_hj_color_red_rk0_s0.pt`; no GP, no RL). The labels are privileged; a GP margin function on failure labels (the one behind the OOD tables) is the label-free variant; both are compared below. Frames are in the training orientation, so the eval transpose bug does not affect these numbers.
- Protocol sg25clean: goals 25 steps ahead in an annulus around the obstacles, horizon 25, budget 50. N = 200 episodes per cell, one run per cell. Success % / collision %.

Weight sweep (red)

| model | raw CEM | w = 100 | w = 1000 | w = 10000 | safety-first |
|---|---|---|---|---|---|
| sigreg | 62.5 / 50.0 | 45.0 / 58.5 | 39.5 / 60.5 | 35.5 / 61.5 | 39.0 / 60.5 |
| jac | 81.5 / 22.5 | 82.5 / 10.0 | 80.0 / 9.5 | 71.5 / 9.0 | 72.0 / 11.0 |

Purple and diamond use a goal frame rendered in the same shifted appearance (`eval.py` re-renders it; fixed 2026-10-29). Before the fix, `World.evaluate` took the goal frame from the dataset, so purple and diamond runs showed a red-circle goal in a purple or diamond world; those numbers, and the earlier claim that recolor does not affect CEM, were artifacts and are replaced below. Control: red through the re-render path reproduces the original red result (jac+pull raw 83.0 / 27.5 vs 81.5 / 29.0, logs `softcem_jacpull_rr_*`). Red and rotated cells are unchanged (no mismatch: red has none, rotation already rotated the goal frame).

Raw CEM

| model | red | purple | diamond | rotated |
|---|---|---|---|---|
| sigreg | 62.5 / 50.0 | 33.0 / 65.0 | 42.0 / 49.0 | 26.5 / 62.5 |
| jac | 81.5 / 22.5 | 16.0 / 75.5 | 79.0 / 15.5 | 30.0 / 60.0 |
| jac+pull | 81.5 / 29.0 | 25.5 / 59.0 | 82.0 / 14.0 | 41.5 / 54.5 |
| adv_h | 57.5 / 57.0 | 34.0 / 63.5 | 49.5 / 47.0 | 20.5 / 70.5 |
| jac+adv_h | 79.0 / 25.0 | 7.5 / 81.5 | 78.0 / 18.5 | 19.5 / 64.5 |
| jac+pull+adv_h | 75.5 / 29.0 | 32.0 / 65.0 | 75.5 / 18.5 | 30.0 / 53.0 |
| adv_λ | 51.5 / 58.0 | 26.5 / 66.5 | 42.0 / 51.0 | 17.5 / 66.0 |
| jac+adv_λ | 81.0 / 27.5 | 26.0 / 61.0 | 80.5 / 15.0 | 36.5 / 48.5 |
| jac+pull+adv_λ | 81.0 / 29.0 | 33.0 / 58.0 | 77.5 / 16.5 | 46.5 / 46.0 |

Safety-first ($w = 1000$, $C = 10^4$), $h$ = supervised value function (oracle $V^*$)

| model | red | purple | diamond | rotated |
|---|---|---|---|---|
| sigreg | 39.0 / 60.5 | 28.5 / 61.5 | 35.5 / 50.0 | 23.5 / 64.0 |
| jac | 72.0 / 11.0 | 39.0 / 15.5 | 71.5 / 5.5 | 48.0 / 13.5 |
| jac+pull | 75.0 / 20.0 | 38.5 / 41.0 | 73.0 / 18.0 | 35.0 / 37.5 |
| adv_h | 41.0 / 59.5 | 28.0 / 62.5 | 35.0 / 55.5 | 22.5 / 64.0 |
| jac+adv_h | 68.5 / 19.5 | 34.0 / 30.0 | 65.0 / 17.5 | 27.0 / 43.0 |
| jac+pull+adv_h | 66.5 / 25.0 | 33.0 / 33.5 | 62.0 / 15.5 | 33.0 / 31.5 |
| adv_λ | 49.0 / 59.0 | 24.5 / 68.0 | 44.0 / 51.0 | 20.5 / 63.5 |
| jac+adv_λ | 73.5 / 29.0 | 45.0 / 41.5 | 72.5 / 16.5 | 44.5 / 39.5 |
| jac+pull+adv_λ | 62.5 / 29.0 | 44.0 / 28.5 | 62.5 / 16.5 | 42.5 / 23.0 |

Safety-first ($w = 1000$, $C = 10^4$), $h$ = GP margin function (failure labels; the same recipe as the OOD tables, trained on red frames in the training orientation, `margin_gp_fail_red_s0.pt`)

| model | red | purple | diamond | rotated |
|---|---|---|---|---|
| sigreg | 57.5 / 49.5 | 34.0 / 66.0 | 40.5 / 50.5 | 26.0 / 64.5 |
| jac | 79.0 / 8.5 | 35.0 / 26.5 | 78.0 / 5.0 | 45.0 / 21.5 |
| jac+pull | 77.0 / 20.0 | 35.5 / 36.5 | 75.5 / 14.0 | 35.0 / 31.0 |
| adv_h | 55.5 / 53.0 | 34.5 / 63.0 | 50.5 / 46.5 | 19.0 / 69.5 |
| jac+adv_h | 76.0 / 13.0 | 25.5 / 37.5 | 73.5 / 11.0 | 18.5 / 46.0 |
| jac+pull+adv_h | 70.5 / 25.0 | 39.0 / 44.5 | 68.5 / 16.0 | 34.5 / 41.0 |
| adv_λ | 48.0 / 59.0 | 28.0 / 66.0 | 41.5 / 52.0 | 16.0 / 66.5 |
| jac+adv_λ | 74.0 / 17.5 | 37.5 / 36.0 | 75.5 / 14.0 | 31.0 / 39.5 |
| jac+pull+adv_λ | 70.0 / 29.0 | 40.5 / 38.0 | 68.0 / 15.5 | 43.0 / 29.0 |

- Recolor is the hardest shift for planning: with a consistent purple scene every model collapses under raw CEM (7.5-34% success, 58-82% collision; jac+adv_h is worst at 7.5 / 81.5). Diamond hurts only the sigreg-base models (8-20 points of success); the jac-base models are unaffected.
- The soft cost rescues the jac-base models under purple even though $h$ was trained on red frames: jac 16.0 / 75.5 -> 39.0 / 15.5 (value function) and 35.0 / 26.5 (GP); jac+pull and jac+pull+adv_h drop to 33.5-44.5% collision. It never helps the sigreg-base models (sigreg, adv_h, adv_λ) on any condition.
- jac gains the most overall: collision 5-27% on every condition, versus 15.5-75.5% raw. With the GP margin function it keeps 78-79% success on red and diamond.
- The label-free GP margin function matches the privileged value function in most cells; the value function is better for jac under purple and rotation.
- New models: jac+pull+adv_λ is the strongest raw planner overall (no success cost on red, 81.0 / 29.0 vs jac+pull 81.5 / 29.0; best raw rotation, 46.5 / 46.0), and its soft cost works under the shifts (purple 44.0 / 28.5, rotated 42.5 / 23.0) but not in-distribution (red 62.5-70.0 / 29.0). jac+adv_h is the weakest jac-base model under shifts (raw purple 7.5 / 81.5, rotated 19.5 / 64.5). Adding either safety reg to jac removes most of the soft cost's in-distribution safety gain (jac 8.5-11.0% collision on red vs 13.0-29.0% for jac+adv_h, jac+adv_λ); the per-step diagnostics (h on imagined latents, rollout error, goal geometry) do not explain this, so how h ranks CEM's own candidates is the open question.
- Encoder-only safety regs (pred_weight = 0, sigreg base so far): raw CEM improves (adv_h 57.5 / 57.0 -> 60.5 / 47.5, adv_λ 51.5 / 58.0 -> 61.0 / 44.0 on red) but color robustness is lost (margin AUC 0.79 / 0.98 -> 0.45 / 0.44; adv_λ enc-only raw purple collision 79.0 vs 66.5). The predictor term trades planning for color robustness; jac+pull+adv_h encoder-only is still training.
- Replanning every 5 steps instead of executing the 25-step plan open loop (red, partial): raw CEM drops for every model (jac+pull 81.5 / 29.0 -> 60.5 / 46.0), likely because the goal cost scores only the final predicted step and each replan pushes arrival later. Within that setting the fair-pair gap reverses (jac+pull+adv_h 62.5 / 36.5 vs jac+pull 60.5 / 46.0), and the GP soft cost is far more effective (jac+pull 66.5 / 11.5; jac+pull+adv_h 64.5 / 26.0).
- Control on red (jac): a margin function with no safety knowledge makes CEM worse, not safer (shuffled labels 63.5 / 39.5, random init 36.5 / 43.0, vs GP 79.0 / 8.5 and raw 81.5 / 22.5), so the safety gain comes from what $h$ knows.

Why the safety regs plan worse (diagnostics `diag_rollout_safety.py`, ablations `ablate_plan_worse.sh`, `seeds_jacpull_vs_adv.sh`):

- Same base, 3 WM seeds: jac+pull+adv_h loses 6.2 ± 1.8 points of raw-CEM success against jac+pull on red (75.5 / 79.0 / 78.0 vs 81.5 / 87.0 / 82.5; pooled paired 52 vs 15 episodes, $p < 10^{-4}$), with no consistent change in collision.
- Across 22 Dubins WMs, raw-CEM collision tracks goal-distance fidelity, the Spearman correlation between latent distance to the goal and true $(x, y)$ distance near the goal ($\rho = -0.84$); 25-step rollout error does not ($\rho \approx 0$). adv_h lowers that fidelity on imagined latents and raises 5-step rollout error on every seed of the jac+pull pair.
- Projecting the safety direction $\lambda$ out of the latent leaves goal fidelity unchanged on every model (λ is about 2.5% of the latent variance), so the distortion is not along the safety axis.
- Part-ablations of adv_h (sigreg base): the adversarial predictor term causes the rollout-error increase (predictor-only 0.084, full 0.082, encoder-only 0.049, no reg 0.014 at 5 steps); detaching the gap denominator restores most of the real-latent goal geometry; random noise instead of the adversary costs almost nothing. Encoder-only versions of all safety-reg WMs are training to test the fix.
- Training the value function on imagined latents (rollout_k = 8) does not restore $h$ on imagined latents and does not make the soft cost help the sigreg-base models.
- The goal marker is a nuisance for sigreg's margin function only (AUC 0.985 -> 0.901 when the marker moves; ≤ 0.017 for every other model).


That changes my recommendation significantly. If your unsafe-state sink is already accurate on real observations, but imagined trajectories don't enter it when they should, I'd focus on the predictor's ability to preserve safety-relevant geometry—not replace the sink or immediately switch world models.

My leading hypothesis is that your predictor minimizes an average latent prediction loss, while safety depends on getting a relatively small set of critical transitions right. The model can look good on ordinary transitions yet systematically underpredict transitions into unsafe states.

## 1. First, identify which failure you have

Let $z_t=E(o_t)$ be the latent embedding of the real observation, and let $\hat z_{t+k}$ be the imagined latent after $k$ predicted transitions. Let $S(z)$ denote your sink's unsafe score, with larger values meaning more unsafe.

Compare these four quantities on the same held-out trajectories:

| Test | What it diagnoses |
| --- | --- |
| $S(z_t)$ on real observations | Whether the sink correctly recognizes unsafe states |
| $S(\hat z_{t+1})$ versus $S(z_{t+1})$ | Whether the predictor misses unsafe transitions immediately |
| $S(\hat z_{t+k})$ versus $S(z_{t+k})$ | Whether safety errors accumulate over imagined rollouts |
| Predicted safety score versus actual future violation | Whether the model preserves the safety signal needed for planning |

In particular, separate transitions that enter an unsafe region from transitions that remain inside it. A predictor can learn to represent already-unsafe states correctly but still fail to predict the transition that crosses the boundary.

## 2. My first intervention: safety-aware multi-step training

Keep your existing LeWM architecture and sink fixed for the first experiment. Add a loss that explicitly penalizes discrepancies in safety predictions over multi-step rollouts.

$$
\begin{aligned}
\mathcal L ={}& \mathcal L_{\text{LeWM}}\\
&+\lambda_s\sum_{k=1}^{K}w_k
\ell_s\big(S(\hat z_{t+k}),S(z_{t+k})\big).
\end{aligned}
$$

Here:

- $K$ is your rollout horizon.
- $w_k$ weights different prediction horizons.
- $\ell_s$ measures the mismatch between predicted and target safety scores.
- $\lambda_s$ controls the safety objective's strength.

If the sink is a hard classifier rather than a smooth score, use a classification loss on the sink's unsafe/unsafe-boundary labels, or distill its output into a differentiable safety head. Don't assume that gradients through a hard threshold will be useful.

Importantly, if the sink is frozen, the safety loss must still provide a gradient path through the predicted latent $\hat z$. If the sink is not differentiable, train a differentiable surrogate or supervise a separate safety head.

### A useful refinement: emphasize boundary-crossing transitions

Uniform safety loss may mostly teach the model to predict common safe states. Oversample or upweight transitions that approach or cross the unsafe boundary.

For example, if a trajectory goes from safe to unsafe between $t$ and $t+1$, penalize predictions that remain confidently safe. This directly targets the failure that matters to your planner.

## 3. Change CEM's objective so it cannot exploit predictor errors

Even after improving prediction, use an explicit safety cost in candidate evaluation:

$$
J(a_{1:H}) =
J_{\text{task}}(a_{1:H})
+\eta\sum_{k=1}^{H} c_{\text{unsafe}}(\hat z_{t+k}).
$$

A few important details:

- Penalize predicted unsafe states at every imagined step, not only at the final state.
- If the sink score is probabilistic, penalize predicted risk rather than only thresholded unsafe labels.
- Evaluate whether the model underestimates risk near the boundary, where false negatives are especially costly.
- Keep a safety margin around the sink boundary to account for model uncertainty.

A soft penalty alone does not guarantee safety. If violations are costly, you should also consider a conservative fallback action or a verified safety filter.

## 4. The most informative ablation

I'd run these four variants first:

| Variant | Dynamics model | Safety objective | Purpose |
| --- | --- | --- | --- |
| A | Original LeWM | Existing sink penalty | Baseline |
| B | Original LeWM | Stronger boundary-weighted penalty | Test whether CEM is ignoring the sink |
| C | Multi-step-trained LeWM | Existing sink penalty | Test whether imagined-state drift is the cause |
| D | Multi-step + safety-aware training | Boundary-weighted penalty | Test the combined solution |

Measure unsafe-state recall in imagined rollouts, false-negative rate near the boundary, actual constraint violations in the simulator, and task reward. Do not select the best model based only on latent prediction error.

If B helps a lot but C does not, your predictor may already encode enough information and the issue may be the planner's objective. If C helps substantially, the dynamics predictor is likely failing to preserve safety-relevant information. If neither helps, investigate whether the latent space is geometrically suitable for the sink and whether the model sees enough boundary-crossing examples.
