# Dubins, Safety Gym and Cartpole

# Summary (2026-11-01)

What is established, with pointers to the sections below. All margin-function numbers use frames in the training orientation; all CEM OOD numbers use a goal frame in the same shifted appearance.

- Two evaluation bugs were found and fixed. The margin-function OOD evals fed the world model transposed frames (every benchmark; "rotate" was a mirror flip), and the Dubins CEM purple/diamond runs used a red-circle goal frame. Reruns of the old paths reproduce the old tables exactly, so the fixed tables replace them. Still on transposed frames, and hidden below: cartpole "why viewpoint fails" and value functions, and support vs aliasing.
- Margin functions (Dubins 3×3 grid): sigreg fails every shift; every regularized model handles rotation and shape; color is the hard shift, where adv_λ keeps the ranking on every base (0.911-0.982 AUC). No model's color threshold survives (sign accuracy ≈ 0.5).
- Planning (main results: $h$ = GP margin function; full and encoder-only safety regs both reported on all four conditions): raw CEM collapses for every model under a consistent recolor (7.5-34% success). A safety-first soft cost cuts collision on every jac-base model on every condition (jac: 5-27%) and never helps the sigreg-base models. Encoder-only is at least as safe as full on every jac-base safety-reg model on every condition; jac+adv_h encoder-only is the safest model under purple. The label-free GP margin function matches the oracle value function. 85-100% of the remaining collisions are false-safe: $h$ on imagined latents is optimistic, more so further ahead.
- Fixing the optimistic $h$ (red, replanning every 5 steps): training $h$ on predicted latents of CEM-style plans labelled in the simulator lowers collision on 6 of 8 jac-base models (jac 84.5 / 6.0); with a margin that grows with the imagined step it cuts the robust models' collision the most (jac+adv_h encoder-only 17.5 -> 6.0, jac+pull+adv_h 41.5 -> 17.5) at 5-10 points of success. No variant helps under purple: from a recolored start the predictor loses $h$'s ranking within 5-10 imagined steps on every model (AUC 0.40-0.59 over the horizon), although real purple frames keep it (0.885-0.963 on the robust models).
- Why purple imagined states fail (jac+pull+adv_h encoder-only): the predictor tracks the purple world about as well as the red one (error 0.57 vs 0.53 of the latent variance at step 10), but in the purple region $h$'s safe/unsafe gap is about 30 times smaller (0.06 vs 1.73), so one step of prediction noise already erases its ranking. The defect is the size of the safety gap under an appearance shift, not predictor drift.
- Free-running 5-step rollout loss (SLS-squared style; 15-epoch screens): imagined-state ranking on red improves by 10-18 AUC points at steps 10-25, but CEM ties on red (modest gain with replanning for one model) and raw CEM under purple gets worse in all 4 comparisons. 50-epoch versions pending.
- Safety regs and planning: on the same base, jac+pull+adv_h loses 6.2 ± 1.8 points of raw-CEM success against jac+pull over 3 WM seeds. The adversarial predictor term raises rollout error. Dropping it (encoder-only) helps planning, and keeps color robustness for adv_h on the jac base (jac+adv_h 0.167 -> 0.808, jac+pull+adv_h 0.782 -> 0.891) but not for adv_λ (0.40-0.44 on every base). Best safety-reg models with the soft cost: jac+adv_h encoder-only and jac+adv_λ encoder-only (red 8% collision at about 80% success, matching jac).

Open:

- One WM seed per model except the jac+pull pair; jac+adv_h's color inversion (0.167) in particular needs a second seed.
- Fair replanning (goal cost at every imagined step): red done for the full models plus two encoder-only models; OOD done (all 108 cells). Not run: the other encoder-only models, and encoder-only on OOD.
- Why adding a full safety reg to jac weakens the soft cost: executed-plan traces show the extra collisions are false-safe (85-100% in all 9 models traced). A growing margin and $h$ on predicted latents help on red, not under purple (see "Fixing the optimistic $h$").
- Purple planning: the failure is $h$'s collapsed safe/unsafe gap in the recolored region (Table 15), not predictor drift. Next: a red-only training term that keeps that gap under generic appearance perturbations.
- Rollout loss: 50-epoch sigreg, jac, jac+pull, jac+pull+adv_h encoder-only training (branch `rollout-loss`), then the per-step test and the CEM screen; keep it only if red with replanning beats the original by about 10 points and purple is not worse (confirm key cells at 200 episodes).
- WM seeds 1 and 2 (original training) for sigreg, jac+adv_h encoder-only and jac+pull+adv_h encoder-only are training (`train_seeds_dubins.sh`).
- Rerun on correct frames: the hidden cartpole and support-vs-aliasing sections, and the 3-WM-seed comparison.

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
<!--
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
-->
<!--
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
-->
# Planning: soft-cost CEM (Dubins)

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

### Replanning every 5 steps, final-step goal cost (superseded)

Same sg25clean goals and 25-step horizon, but CEM replans every 5 steps (`REC=5`) with the original final-step goal cost. Superseded by the every-step goal cost above. Purple with a consistent goal frame:

| Model | red, raw | red, GP safety-first | purple, raw | purple, GP safety-first |
| --- | ---: | ---: | ---: | ---: |
| sigreg | 47.5 / 61.5 | 45.5 / 61.5 | 18.0 / 68.0 | 17.5 / 69.5 |
| adv_h | 48.5 / 55.5 | 50.0 / 55.0 | 29.0 / 66.0 | 29.5 / 67.0 |
| adv_λ | 33.0 / 70.5 | 39.5 / 66.5 | 20.5 / 68.5 | 22.0 / 63.5 |
| jac | 43.5 / 56.5 | 60.5 / 11.0 | 32.5 / 66.5 | 27.5 / 40.0 |
| jac+adv_h | 52.5 / 43.0 | 55.0 / 27.5 | 19.5 / 73.0 | 23.5 / 49.0 |
| jac+adv_λ | 57.5 / 39.5 | 57.5 / 16.5 | 29.5 / 54.0 | 26.5 / 47.0 |
| jac+pull | 60.5 / 46.0 | 66.5 / 11.5 | 26.0 / 57.0 | 21.5 / 48.5 |
| jac+pull+adv_h | 62.5 / 36.5 | 64.5 / 26.0 | 26.5 / 70.5 | 37.0 / 44.5 |
| jac+pull+adv_λ | 56.0 / 37.0 | 58.5 / 23.5 | 27.0 / 68.5 | 22.0 / 44.0 |

- Frequent replanning lowers raw CEM for every model (e.g. jac 81.5 / 22.5 -> 43.5 / 56.5 on red), because the goal cost scores only the final predicted step and each replan pushes arrival later.
- Within this setting the GP safety-first cost cuts collision on every jac-base model (red: jac 56.5 -> 11.0, jac+pull 46.0 -> 11.5) and still does nothing on the sigreg base.

## Diagnostics: why safety regs plan worse, and why CEM still collides

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

### Why safety-first CEM still collides: executed-plan trace

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

### Fixing the optimistic $h$: per-step margin, $h$ on predicted latents, and recolor

Two fixes from the list above, plus two label-free variants for recolor. All CEM runs use the executed-plan trace setting (GP safety-first, $w = 1000$, $C = 10^4$, goal cost at every imagined step, replanning every 5 steps, tracing on, 200 episodes). Scripts: `fix_optimism_queue.sh`, `robust_cem_queue.sh`, `hs_cem_queue.sh`; diagnostics `eval_head_color_dubins.py`, `diag_imag_color_dubins.py`.

#### What was tried

- Ramp margin: penalize $h(\hat z_t) < \epsilon_t$ with $\epsilon_t = \min(0.6,\ 0.04\,(t+1))$ for imagined step $t = 0, \dots, 24$ (`unsafe_margin` now takes a per-step list). Calibrated on the traces above: crash-step $h$ grows with how far ahead it was predicted (jac, 75th percentile: 0.17 at 1-5 steps ahead, 0.51 at 6-10, 0.8-1.0 beyond), while safe-episode plan $h$ stays flat (10th percentile 0.2-0.3). One schedule for every model, although $h$'s scale differs between models.
- $h$ on predicted latents (`train_gp_margin_imag_dubins.py`): start from one real frame of a training-split expert episode (CEM also uses a 1-frame history), sample a 25-step plan $a_t = b + 0.7\,\varepsilon_t$ in CEM's normalized action units ($b \sim \mathcal{N}(0, 1)$ per plan, $\varepsilon_t \sim \mathcal{N}(0, 1)$ per step: CEM's first iteration plus the smooth turning plans it converges to), roll it out with the predictor exactly as `JEPA.rollout`, and label each imagined step by running the same de-normalized, clipped actions through the Dubins simulator from the frame's true state (1 = inside a red circle). Trained on 6000 plans × 25 steps plus the original GP's 8000 real uniform-state latents, with the same GP loss and steps. About 15% of imagined steps are unsafe.
- Dynamic threshold (`h_quantile` $= p$; implemented as a rank normalization of $h$): at each replan, rescale $h$ over all 300 candidates × imagined steps so that its $p$-quantile maps to 0 and its median to 1; the bottom $p$ of imagined latents then read unsafe. In effect the safe/unsafe threshold is re-set every replan from the current scene instead of fixed at 0, so a shift or rescaling of $h$ under OOD (Table 5) cancels out. Only $p = 0.05$ was run.
- Short-horizon cost (`h_steps` $= n$): safety cost on the first $n$ imagined steps only.

#### Does training on predicted latents fix $h$? (no planning)

Table 1. GP margin function evaluated on imagined latents of held-out CEM-style plans (red, last 10% of episodes, 1000 plans × 25 steps, simulator labels). "Old" = trained on real latents only; "new" = trained on real + predicted latents. False-safe = share of unsafe imagined steps with $h \ge 0$; false alarm = share of safe imagined steps with $h < 0$.

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

#### Planning on red: does each fix reduce collisions?

Table 2. CEM on red (in-distribution), success / collision (%), 200 episodes. One column per $h$ variant; blank = not run.

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

#### Planning under recolor (purple): does any fix help?

Table 3. CEM with purple obstacles (OOD), success / collision (%), 200 episodes, for the models that are robust on real samples plus jac as reference. One column per $h$ variant; blank = not run. Old-$h$ cells for jac and jac+pull+adv_h are the untraced fair-replanning runs.

| Model | Old $h$ | $h$ on predicted latents | Predicted latents + ramp | Dynamic threshold, predicted latents ($p = 0.05$) | Dynamic threshold, old $h$, first 5 steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| jac | 34.5 / 43.0 | 29.0 / 48.5 | 21.5 / 40.0 | | 23.5 / 72.0 |
| jac+adv_h, encoder-only | 35.5 / 51.5 | 27.5 / 54.5 | 26.5 / 49.0 | | |
| jac+pull+adv_h | 26.5 / 61.5 | 30.0 / 51.0 | 32.5 / 42.0 | | |
| jac+pull+adv_h, encoder-only | 30.5 / 47.5 | 20.0 / 48.0 | 16.0 / 57.0 | 33.0 / 54.5 | 29.0 / 68.0 |

Table 4. The two label-free variants of Table 3 run on red, to check what they cost in-distribution. Success / collision (%), 200 episodes; blank = not run.

| Model | Old $h$ (from Table 2) | Dynamic threshold, predicted latents ($p = 0.05$) | Dynamic threshold, old $h$, first 5 steps |
| --- | ---: | ---: | ---: |
| jac | 84.0 / 8.0 | | 78.0 / 38.5 |
| jac+pull+adv_h | 69.5 / 41.5 | 74.5 / 39.0 | |
| jac+pull+adv_h, encoder-only | 76.0 / 20.5 | 76.5 / 20.5 | 73.0 / 33.0 |

- No fix helps under purple: every variant gives 40-72% collision at 16-35.5% success, and the robust models do not beat jac.
- The collision split flips with $h$. With the old $h$, jac+pull+adv_h encoder-only's purple collisions are 100% false-safe; with $h$ on predicted latents 95% are predicted unsafe (jac: 93%). Under purple the old $h$ reads everything safe and the new one reads nearly everything unsafe, so the safety-first cost ties all candidates and ranks on goal cost alone.
- Restricting the cost to the first 5 imagined steps is much worse even on red (jac 8.0 -> 38.5% collision): at fixed speed the car needs more than 5 steps of lookahead to see a doomed state, which a margin function (instantaneous failure) does not encode.

- The dynamic threshold costs nothing on red (76.0 / 20.5 -> 76.5 / 20.5) but does not help under purple; restricting to 5 steps hurts on both. A per-replan threshold can only re-zero $h$ where $h$ still ranks the imagined latents correctly, and from a purple start that ranking is gone after about 5 imagined steps (Table 6); a per-episode shift anchored on the real start frame and other values of $p$ were not run, since they hit the same limit.

#### Why recolor fails (a): $h$ on real purple frames

Table 5. GP margin function on real latents (no prediction): 4000 uniform states rendered with purple circles (training orientation, disjoint from the training draws). AUC = ranking; balanced sign accuracy = whether $h$'s sign is right (0.500 = everything on one side of 0).

| Model | Purple AUC, old $h$ | Purple AUC, $h$ on predicted latents | Purple bal. sign acc, old | Purple bal. sign acc, new |
| --- | ---: | ---: | ---: | ---: |
| jac | 0.575 | 0.643 | 0.500 | 0.500 |
| jac+adv_h, encoder-only | 0.885 | 0.898 | 0.500 | 0.500 |
| jac+pull+adv_h | 0.896 | 0.946 | 0.500 | 0.500 |
| jac+pull+adv_h, encoder-only | 0.954 | 0.963 | 0.500 | 0.861 |

- Training on predicted latents does not cost real-sample color robustness; purple AUC rises on every model.
- On the robust models purple keeps the ranking but collapses the scale. jac+pull+adv_h encoder-only, median $h$ on safe / unsafe states: red 1.56 / −0.53, purple 0.38 / 0.32 (old $h$); 0.03 / −0.08 with $h$ on predicted latents. Hence balanced sign accuracy 0.500, and the motivation for the dynamic threshold.

#### Why recolor fails (b): $h$ on imagined latents, per imagined step

Table 6. AUC of $h$ on imagined latents at each imagined step, rolled out from a red or a purple start frame (600 held-out expert states re-rendered in each color, CEM-style plans, simulator labels). GP margin function trained on real latents unless the row says otherwise. Per-step AUCs measured at 2 decimals.

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

#### Conclusions and next step

- The encoder is color-robust (real purple frames: 0.885-0.963 AUC, Table 5), and $h$ on imagined purple latents is near chance after 5-10 steps. At the time we read this as the predictor failing to carry the safety information forward. The two sections below revise that: the predictor tracks the purple world about as well as the red one, and the failure is $h$'s tiny safe/unsafe gap in the purple region (Table 15).
- Fixes to $h$ (margin, retraining, dynamic threshold, the value function) cannot help while that gap is smaller than one step of prediction noise.
- Caveats: one WM seed; dynamic-threshold runs record raw $h$ in their traces (the chosen plan is re-scored alone), so their predicted-unsafe / false-safe split uses raw $h$; the ramp schedule is not tuned per model.

### Predictor training: free-running rollout loss (SLS-squared style)

Question: does training the predictor on its own multi-step rollouts make CEM plan better, in-distribution and under OOD shifts? Code on branch `rollout-loss` (`train.py`, `rollout_steps = K`); scripts `train_rollout5_dubins.sh`, `roll5b_check.sh` (per-step test of Table 6), `roll5b_quick_cem.sh`, `roll5b50_eval_queue.sh`.

#### How le-wm and SLS-squared train the predictor

- le-wm (all Dubins WMs so far): windows of 4 frames, one forward pass of the causal transformer on the 3 real context latents, MSE of the predictions at all 3 positions to the next real latents. So every context length (1, 2, 3 frames) is trained, but only one step ahead from real latents; the predictor never sees its own outputs.
- SLS-squared (`SLS-squared/<env>/train/mlpdyn_train.py`): an MLP predictor on a single Markov state $[z_t, z_t - z_{t-1}]$, unrolled 5 steps on its own predictions, MSE averaged over the 5 steps. This replaces the one-step loss (step 0 is the one-step prediction from a real state). Plus SIGReg and a trajectory-straightening loss; model error is calibrated afterwards with conformal prediction.
- Ours ("rollout loss", $K = 5$): replaces the le-wm loss with the mean MSE over 5 free-running steps; the regularizers keep their original 4-frame window. Everything else (seed, 50 epochs, regularizer weights) as the original model.

#### First attempt: rollout from 3 real frames (failed)

The first version started each rollout from 3 real latents and scored only the last position. CEM never trains on this case: its rollouts start from 1 frame and grow to 3, so predictions from 1-2 frames of context lost their training signal. After 50 epochs the rollouts collapse after step 5 (imagined latents converge to one region; median $h$ on safe and unsafe states both 0.70), and planning breaks.

Table 7. Per-step AUC of $h$ on imagined latents from a red start (as Table 6), jac, original vs first rollout-loss version (both 50 epochs).

| jac | Step 1 | Step 3 | Step 5 | Step 10 | Step 15 | Step 25 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 1.00 | 1.00 | 1.00 | 0.98 | 0.93 | 0.88 |
| rollout loss from 3 frames | 0.98 | 0.91 | 0.88 | 0.43 | 0.42 | 0.51 |

Table 8. CEM, replanning every 5 steps, goal cost at every imagined step, success / collision (%). Original: 200 episodes; first rollout-loss version: 100 episodes.

| Model | Planner | Red | Purple | Diamond | Rotated 90° |
| --- | --- | ---: | ---: | ---: | ---: |
| jac, original | raw | 77.5 / 48.5 | 20.5 / 75.5 | 79.5 / 29.5 | 47.5 / 54.0 |
| jac, rollout loss from 3 frames | raw | 19.0 / 61.0 | 15.0 / 73.0 | 23.0 / 53.0 | 33.0 / 59.0 |
| jac, original | safety-first | 84.0 / 8.0 | 34.5 / 43.0 | 80.0 / 6.5 | 50.0 / 40.0 |
| jac, rollout loss from 3 frames | safety-first | 17.0 / 63.0 | 15.0 / 73.0 | 23.0 / 53.0 | 33.0 / 58.0 |
| sigreg, original | raw | 63.0 / 55.5 | 14.5 / 72.0 | 59.5 / 46.5 | 56.0 / 58.5 |
| sigreg, rollout loss from 3 frames | raw | 26.0 / 70.0 | 18.0 / 59.0 | 37.0 / 51.0 | 33.0 / 58.0 |

#### Fixed version: rollout from 1 real frame

The fixed version starts every training rollout from one real frame and lets the context grow to 3, exactly as CEM does, so every context length is trained. To get an answer fast, jac and jac+pull+adv_h encoder-only were trained for 15 epochs (learning-rate schedule fitted to 15) and compared with the original models' checkpoints at a similar number of epochs.

Table 9. Per-step AUC of $h$ on imagined latents (as Table 6), red and purple starts, at matched training length. The original jac only kept every 10th checkpoint.

| Model | Epoch | Start | Step 1 | Step 3 | Step 5 | Step 10 | Step 15 | Step 25 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| jac, original | 10 | red | 0.99 | 0.99 | 0.98 | 0.82 | 0.66 | 0.55 |
| jac, rollout loss | 10 | red | 0.99 | 0.99 | 0.98 | 0.93 | 0.83 | 0.75 |
| jac, rollout loss | 15 | red | 0.99 | 0.99 | 0.99 | 0.95 | 0.86 | 0.72 |
| jac+pull+adv_h, encoder-only, original | 15 | red | 0.99 | 0.99 | 0.99 | 0.85 | 0.68 | 0.55 |
| jac+pull+adv_h, encoder-only, rollout loss | 15 | red | 0.99 | 0.99 | 0.99 | 0.95 | 0.82 | 0.73 |
| jac+pull+adv_h, encoder-only, original | 15 | purple | 0.81 | 0.57 | 0.54 | 0.52 | 0.52 | 0.47 |
| jac+pull+adv_h, encoder-only, rollout loss | 15 | purple | 0.52 | 0.64 | 0.62 | 0.52 | 0.49 | 0.48 |
| jac+pull+adv_h, encoder-only, original | 50 | purple | 0.80 | 0.88 | 0.81 | 0.54 | 0.54 | 0.51 |

- On red the rollout loss keeps the ranking much longer at equal training: +10-18 AUC points at steps 10-25 on both models.
- On purple it makes no net difference at epoch 15; both are at chance from step 10. (jac is near chance on purple for every version, since it is not color-robust.)

Table 10. Quick CEM screen, 50 episodes, success / collision (%), goal cost at every imagined step. jac: original epoch 10 vs rollout loss epoch 15.

| Setting | Original, epoch 10 | Rollout loss, epoch 15 |
| --- | ---: | ---: |
| Red, open loop, raw | 58.0 / 50.0 | 56.0 / 48.0 |
| Red, open loop, safety-first | 42.0 / 50.0 | 46.0 / 54.0 |
| Purple, open loop, raw | 22.0 / 68.0 | 6.0 / 78.0 |
| Purple, open loop, safety-first | 14.0 / 62.0 | 32.0 / 50.0 |
| Red, replan every 5, raw | 64.0 / 52.0 | 56.0 / 52.0 |
| Red, replan every 5, safety-first | 52.0 / 48.0 | 50.0 / 46.0 |
| Purple, replan every 5, raw | 16.0 / 56.0 | 14.0 / 76.0 |
| Purple, replan every 5, safety-first | 24.0 / 56.0 | 20.0 / 46.0 |

Table 11. Same screen for jac+pull+adv_h encoder-only, original vs rollout loss, both at epoch 15.

| Setting | Original, epoch 15 | Rollout loss, epoch 15 |
| --- | ---: | ---: |
| Red, open loop, raw | 46.0 / 48.0 | 46.0 / 44.0 |
| Red, open loop, safety-first | 40.0 / 54.0 | 44.0 / 56.0 |
| Purple, open loop, raw | 28.0 / 58.0 | 4.0 / 82.0 |
| Purple, open loop, safety-first | 32.0 / 42.0 | 26.0 / 60.0 |
| Red, replan every 5, raw | 48.0 / 54.0 | 62.0 / 46.0 |
| Red, replan every 5, safety-first | 52.0 / 44.0 | 60.0 / 32.0 |
| Purple, replan every 5, raw | 22.0 / 68.0 | 14.0 / 78.0 |
| Purple, replan every 5, safety-first | 10.0 / 64.0 | 24.0 / 66.0 |

- At 50 episodes a gap needs to be about 15 points to be clear. Fully trained models plan much better than these 10-15-epoch checkpoints (jac at 50 epochs: red open loop safety-first 76.0 / 22.0, 200 episodes), so these tables compare training recipes at equal length, not final models.
- Red: a tie open loop; with replanning, modestly better for jac+pull+adv_h encoder-only (8-14 more points of success, 8-12 fewer of collision) and a tie for jac.
- Purple, raw CEM: worse with the rollout loss in all 4 comparisons (success 22 -> 6 and 28 -> 4 open loop). Purple, safety-first: mixed.
- Pending: the same per-step test and CEM screen at 50 epochs for sigreg, jac, jac+pull and jac+pull+adv_h encoder-only (`roll5b50_eval_queue.sh`).

### Why purple imagined states fail: the predictor tracks purple, $h$'s gap collapses

The tables above leave a gap in the explanation: $h$ reads real purple frames well (Table 5) and the predictor is accurate on red, yet $h$ on imagined purple latents is near chance. To locate the failure we rolled out the same 400 held-out states and CEM-style plans from a red and from a purple start, and also encoded the true future frames rendered in each color (`diag_purple_drift.py`, original 50-epoch models).

Table 12. Where a purple-start rollout goes, jac+pull+adv_h encoder-only. Distances are mean squared distance divided by the total variance of real red latents.

| Per imagined step | 1 | 3 | 5 | 10 | 15 | 25 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| How far recolor moves the true latent | 1.34 | 1.31 | 1.29 | 1.18 | 1.07 | 0.92 |
| Purple rollout vs true purple future | 0.07 | 0.14 | 0.19 | 0.57 | 1.16 | 1.80 |
| Red rollout vs true red future (reference) | 0.02 | 0.05 | 0.12 | 0.53 | 1.01 | 1.42 |
| Purple rollout vs true red future | 1.33 | 1.33 | 1.38 | 1.72 | 2.19 | 2.38 |
| Purple rollout vs red rollout | 1.31 | 1.29 | 1.31 | 1.68 | 2.14 | 2.39 |

Table 13. AUC of $h$ on the same states, jac+pull+adv_h encoder-only.

| Per imagined step | 1 | 3 | 5 | 10 | 15 | 25 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| True purple futures (no prediction) | 0.94 | 0.94 | 0.94 | 0.95 | 0.92 | 0.91 |
| Purple rollout | 0.80 | 0.88 | 0.78 | 0.59 | 0.50 | 0.40 |
| Red rollout | 1.00 | 1.00 | 1.00 | 0.96 | 0.93 | 0.78 |

Table 14. Same as Tables 12-13 for jac (not color-robust), selected rows.

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

Table 15. $h$'s signal vs prediction noise, jac+pull+adv_h encoder-only. Gap = mean $h$ on safe minus mean $h$ on unsafe true future latents; noise = standard deviation over samples of $h$(rollout) − $h$(true future).

| Per imagined step | 1 | 3 | 5 | 10 | 15 | 25 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Red: $h$ gap | 1.73 | 1.72 | 1.71 | 1.69 | 1.56 | 1.40 |
| Red: $h$ noise from prediction | 0.03 | 0.09 | 0.13 | 0.34 | 0.52 | 0.67 |
| Red: gap / noise | 60.24 | 19.76 | 12.70 | 4.96 | 2.99 | 2.08 |
| Purple: $h$ gap | 0.06 | 0.06 | 0.06 | 0.06 | 0.05 | 0.04 |
| Purple: $h$ noise from prediction | 0.05 | 0.05 | 0.05 | 0.05 | 0.06 | 0.33 |
| Purple: gap / noise | 1.26 | 1.20 | 1.20 | 1.16 | 0.75 | 0.13 |

- In the purple region $h$ still orders states correctly but its whole safe-to-unsafe range is about 30 times smaller (0.06 vs 1.73). One step of prediction noise (0.05) is already as large as that range, so the ranking on imagined latents is near chance from the first steps; on red the 1.7-wide gap absorbs 10+ steps of noise.
- So the defect is the size of $h$'s safe/unsafe gap under an appearance shift, not predictor drift. This explains why a better predictor (the rollout loss) does not help purple, and why $h$-side fixes (margin, retraining on predicted latents, dynamic threshold) cannot either.
- Candidate fixes, all trained on red data only (OOD stays a test-only measure): keep the safety gap under generic appearance perturbations (an encoder-side penalty when perturbations shrink the safe/unsafe distance along the direction $h$ reads; the saved $\lambda$-projected invariance idea is one version); or weight the prediction loss by how much an error moves $h$, so the predictor is most accurate along the directions $h$ depends on.
- Caveats: one WM seed; 400 states; a single recolor (purple).

### Do CEM's elites converge? Elite spread

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
