# Dubins, Safety Gym and Cartpole

# Dubins

The same 3×3 grid on Dubins for reference: base {sigreg, + jacobian, + jacobian + pull} × safety reg {none, L∞ safety
reg, λ batch-detached}. Zero-shot margin_gp: the head is trained on red-obstacle frames and scored on the same states
re-rendered with purple obstacles (color), a 90° rotation (rotate) or diamond obstacles (shape); labels are exact under
every shift. WM seed 3072, ± = std over 3 eval seeds. Numbers are from `safety_adv_reg_report.md` (logs under
`/data/seongbin/lewm/safeadv_results/`). Blank rows: those WMs have not been trained on Dubins.

AUC

| Model | base | color | rotate | shape |
|---|---|---|---|---|
| sigreg | 0.979 ± 0.002 | 0.144 ± 0.032 | 0.956 ± 0.023 | 0.913 ± 0.024 |
| jacobian | 0.992 ± 0.001 | 0.673 ± 0.076 | 0.989 ± 0.002 | 0.992 ± 0.003 |
| jac+pull | 0.983 ± 0.001 | 0.753 ± 0.049 | 0.978 ± 0.003 | 0.977 ± 0.007 |
| L∞ safety reg | 0.998 ± 0.001 | 0.910 ± 0.098 | 0.996 ± 0.002 | 0.997 ± 0.001 |
| jacobian + L∞ | | | | |
| jac+pull + L∞ | 0.996 ± 0.001 | 0.965 ± 0.013 | 0.994 ± 0.002 | 0.994 ± 0.002 |
| λ batch-detached | 0.997 ± 0.001 | 0.948 ± 0.013 | 0.996 ± 0.001 | 0.996 ± 0.001 |
| jacobian + λ | | | | |
| jac+pull + λ | | | | |

Sign accuracy (h >= 0 = safe; the test set is 0.830 safe, so 0.830 = everything called safe, 0.170 = everything unsafe)

| Model | base | color | rotate | shape |
|---|---|---|---|---|
| sigreg | 0.943 ± 0.005 | 0.170 ± 0.006 | 0.830 ± 0.006 | 0.310 ± 0.025 |
| jacobian | 0.968 ± 0.002 | 0.171 ± 0.007 | 0.962 ± 0.004 | 0.968 ± 0.007 |
| jac+pull | 0.959 ± 0.003 | 0.830 ± 0.007 | 0.939 ± 0.003 | 0.957 ± 0.008 |
| L∞ safety reg | 0.981 ± 0.004 | 0.174 ± 0.006 | 0.975 ± 0.004 | 0.980 ± 0.003 |
| jacobian + L∞ | | | | |
| jac+pull + L∞ | 0.973 ± 0.003 | 0.830 ± 0.006 | 0.970 ± 0.009 | 0.978 ± 0.005 |
| λ batch-detached | 0.975 ± 0.005 | 0.170 ± 0.006 | 0.973 ± 0.005 | 0.973 ± 0.005 |
| jacobian + λ | | | | |
| jac+pull + λ | | | | |

Reading:

- Color is the only hard shift on Dubins. Shape and rotation stay at 0.91-0.997 AUC for every model.
- No threshold survives color: every model calls purple frames all unsafe (0.170) or all safe (0.830).
- Over 3 WM seeds the single-seed color AUCs above don't hold: jac+pull 0.498 ± 0.234, L∞ 0.643 ± 0.247,
  λ batch-detached 0.721 ± 0.434. Only jac+pull + L∞ is stable (0.957 ± 0.021).

# Safety Gym

Goal: a second benchmark where safety is coupled to dynamics (unlike a "teleportable" agent whose safe set is just the
obstacle geometry), to re-test the appearance-shift findings from Dubins. Code and logs: branch `point-dynamics-probe`
(`safety_gym_port/`, `le-wm/scripts/sg_point_*.py`, `run_logs/sg_point/`); full tables in `sg_point_ovh_results.md`.

## Point instead of Car: safety that depends on velocity

The earlier safety-gym port used SafetyCarGoal1. The Car brakes from top speed in 0.055 (a quarter of the hazard radius
0.2) and reaches top speed in 5 steps, so its safe set is essentially the hazard geometry. The Point robot (thrust along
heading, world-frame slide joints) needs 60 steps to reach top speed, brakes in 0.238 and coasts 0.78, so turning does not
redirect its momentum.

Probe (`safety_gym_port/point_doomed.py`): inject Point states near hazards, call a currently-safe state "doomed" if none of
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

How much the Car models leaned on it (probe trained with the sphere, tested on the same frames without it): the gp margin
probe barely moved (zero-shot drop at most 0.017), but the jacobian/jac+pull latents encode the sphere along the safe/unsafe
direction (cosine ~0.8) and an unregularized probe dropped 1.000 -> 0.84. The earlier Car color/rotation safety-reg gains
should be treated as suspect.

## Problem 2: the old protocol measured layout memorization

Hazard layouts are random every episode. The old eval (one 5000-pose collection, shuffled 80/20 frame split) lets the probe's
train and test frames share layouts. Probes on clean fixedfar@128 Point models scored 0.99 in-distribution, but 0.47-0.52
on base-view frames from fresh layouts, i.e. chance. The camera-rotation axis failed for the same reason: a 90 degree turn
of this camera looks like the normal view of a different layout.

The benchmark itself was learnable. A supervised CNN on raw fixedfar@128 pixels reaches AUC 0.998-0.999 on unseen layouts
(and 0.958-0.999 on every OOD view). The bottleneck was the world model's pooled latent: a probe on the same encoder's patch
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
- Eval: margin_gp probe trained on 50k frames of the training set, scored on 5000 poses of a separate collection (every
  layout unseen). Views: base, purple hazards (color), 90 degree camera rotation (rotate), box-shaped hazards (shape), Car
  robot instead of Point (agent).

Pilot, same probe on unseen layouts (AUC, base view): sigreg 0.362 (fixedfar) -> 0.581 (overhead); jac+pull 0.563 -> 0.836.

## Results: 9 world models on overhead@224, layout-disjoint

le-wm, 50 epochs, seed 3072. Safety-reg recipes copied from the earlier safety-gym reference runs. ± = std over 3 eval seeds
(probe subsample and init); one training seed per model. "train fit" = probe AUC on its own training frames.

AUC

| Model | train fit | base | color | rotate | shape | agent |
|---|---|---|---|---|---|---|
| sigreg | 0.989 ± 0.001 | 0.581 ± 0.023 | 0.464 ± 0.013 | 0.447 ± 0.020 | 0.514 ± 0.043 | 0.554 ± 0.020 |
| jacobian | 1.000 ± 0.000 | 0.954 ± 0.006 | 0.725 ± 0.014 | 0.932 ± 0.009 | 0.952 ± 0.007 | 0.824 ± 0.010 |
| jac+pull | 0.999 ± 0.000 | 0.836 ± 0.015 | 0.660 ± 0.035 | 0.740 ± 0.022 | 0.865 ± 0.010 | 0.673 ± 0.045 |
| L∞ safety reg | 0.996 ± 0.000 | 0.948 ± 0.003 | 0.746 ± 0.008 | 0.909 ± 0.009 | 0.619 ± 0.015 | 0.873 ± 0.008 |
| jacobian + L∞ | 0.998 ± 0.000 | 0.985 ± 0.001 | 0.937 ± 0.006 | 0.972 ± 0.003 | 0.984 ± 0.001 | 0.917 ± 0.000 |
| jac+pull + L∞ | 0.999 ± 0.000 | 0.939 ± 0.007 | 0.753 ± 0.032 | 0.821 ± 0.012 | 0.951 ± 0.004 | 0.852 ± 0.007 |
| λ batch-detached | 0.989 ± 0.001 | 0.885 ± 0.010 | 0.501 ± 0.009 | 0.899 ± 0.003 | 0.895 ± 0.009 | 0.757 ± 0.016 |
| jacobian + λ | 0.994 ± 0.001 | 0.969 ± 0.003 | 0.898 ± 0.024 | 0.961 ± 0.003 | 0.972 ± 0.001 | 0.892 ± 0.007 |
| jac+pull + λ | 0.995 ± 0.001 | 0.971 ± 0.003 | 0.894 ± 0.013 | 0.955 ± 0.004 | 0.974 ± 0.002 | 0.867 ± 0.005 |

Sign accuracy (h >= 0 = safe; the test set is 0.830 safe, so 0.830 = everything called safe, 0.170 = everything unsafe)

| Model | base | color | rotate | shape | agent |
|---|---|---|---|---|---|
| sigreg | 0.776 ± 0.009 | 0.820 ± 0.003 | 0.794 ± 0.019 | 0.808 ± 0.008 | 0.777 ± 0.007 |
| jacobian | 0.896 ± 0.001 | 0.468 ± 0.037 | 0.866 ± 0.001 | 0.890 ± 0.005 | 0.814 ± 0.009 |
| jac+pull | 0.847 ± 0.011 | 0.757 ± 0.012 | 0.839 ± 0.010 | 0.861 ± 0.013 | 0.813 ± 0.002 |
| L∞ safety reg | 0.931 ± 0.004 | 0.170 ± 0.000 | 0.918 ± 0.004 | 0.835 ± 0.003 | 0.884 ± 0.004 |
| jacobian + L∞ | 0.956 ± 0.001 | 0.879 ± 0.005 | 0.907 ± 0.003 | 0.949 ± 0.002 | 0.866 ± 0.002 |
| jac+pull + L∞ | 0.852 ± 0.007 | 0.809 ± 0.004 | 0.852 ± 0.008 | 0.879 ± 0.013 | 0.812 ± 0.011 |
| λ batch-detached | 0.879 ± 0.000 | 0.710 ± 0.010 | 0.888 ± 0.012 | 0.889 ± 0.005 | 0.841 ± 0.004 |
| jacobian + λ | 0.934 ± 0.009 | 0.618 ± 0.023 | 0.907 ± 0.013 | 0.937 ± 0.004 | 0.815 ± 0.006 |
| jac+pull + λ | 0.940 ± 0.005 | 0.752 ± 0.022 | 0.905 ± 0.007 | 0.937 ± 0.003 | 0.808 ± 0.004 |

Reading:

- Jacobian + L∞ safety reg is best on every axis (AUC 0.917-0.985) and the only model whose threshold holds on all five
  views (predicted-safe fraction 0.79-0.86 against a true 0.83).
- The safety regularizers help on this benchmark: sigreg 0.581 base -> 0.948 with L∞ and 0.885 with λ. Combined with the
  Jacobian term they form the top group (jacobian + λ and jac+pull + λ: about 0.97 base, 0.90 color).
- Color is the hardest shift and breaks thresholds even when the ranking holds: L∞ alone calls everything unsafe on purple
  hazards, and jacobian alone calls 70% unsafe.
- The action-separation pull hurts here (jac+pull 0.836 vs jacobian 0.954 base).
- sigreg stays near chance; the camera change does not rescue an unregularized latent.
- All probes fit their own training frames (0.989-1.000), so the gaps above are generalization to new layouts, not under-fitting.

## Caveats and next steps

- One training seed per model. Single-seed safety-reg results have been lucky before (Dubins). Seeds 1 and 2 of all 9
  recipes are training (`le-wm/scripts/run_point_ovh_seeds.sh`); the same eval runs automatically when they finish.
- Labels are position-only failure (agent inside a hazard). The velocity-dependent value (the doomed band above) is not yet
  used as a target; `point_doomed.py` can generate Monte Carlo value labels.
- Possible variant: a fixed-layout Point benchmark (Dubins-like), which would also make an exact HJ value oracle feasible.
- No Point Dreamer has been trained.

# Cartpole

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

le-wm, 50 epochs, seed 3072, same recipes and code snapshots as the Dubins runs (the L∞ gate opened: in-training head AUC
0.991). ± = std over 3 head seeds; one training seed per model.

AUC

| Model | base | color | zoom_out | pan_x | pan_y |
|---|---|---|---|---|---|
| sigreg | 0.991 ± 0.001 | 0.656 ± 0.003 | 0.619 ± 0.010 | 0.620 ± 0.010 | 0.746 ± 0.014 |
| jacobian | 0.992 ± 0.002 | 0.737 ± 0.006 | 0.727 ± 0.007 | 0.754 ± 0.013 | 0.758 ± 0.016 |
| jac+pull | 0.992 ± 0.002 | 0.789 ± 0.010 | 0.708 ± 0.014 | 0.700 ± 0.014 | 0.796 ± 0.011 |
| L∞ safety reg | 0.981 ± 0.002 | 0.665 ± 0.002 | 0.800 ± 0.006 | 0.601 ± 0.020 | 0.769 ± 0.007 |
| jacobian + L∞ | 0.993 ± 0.002 | 0.753 ± 0.014 | 0.592 ± 0.013 | 0.636 ± 0.002 | 0.777 ± 0.012 |
| jac+pull + L∞ | 0.991 ± 0.000 | 0.750 ± 0.023 | 0.642 ± 0.015 | 0.723 ± 0.016 | 0.801 ± 0.003 |
| λ batch-detached | 0.984 ± 0.001 | 0.836 ± 0.008 | 0.707 ± 0.011 | 0.668 ± 0.004 | 0.699 ± 0.008 |
| jacobian + λ | 0.980 ± 0.005 | 0.891 ± 0.009 | 0.530 ± 0.018 | 0.539 ± 0.011 | 0.690 ± 0.009 |
| jac+pull + λ | 0.986 ± 0.001 | 0.894 ± 0.011 | 0.596 ± 0.007 | 0.569 ± 0.006 | 0.551 ± 0.010 |

Sign accuracy (h >= 0 = safe; the test set is 0.832 not failed, so 0.832 = everything called safe, 0.168 = everything unsafe)

| Model | base | color | zoom_out | pan_x | pan_y |
|---|---|---|---|---|---|
| sigreg | 0.953 ± 0.005 | 0.629 ± 0.031 | 0.311 ± 0.019 | 0.334 ± 0.018 | 0.292 ± 0.092 |
| jacobian | 0.949 ± 0.003 | 0.494 ± 0.027 | 0.427 ± 0.026 | 0.452 ± 0.029 | 0.633 ± 0.029 |
| jac+pull | 0.955 ± 0.010 | 0.726 ± 0.026 | 0.466 ± 0.013 | 0.499 ± 0.014 | 0.611 ± 0.033 |
| L∞ safety reg | 0.939 ± 0.019 | 0.739 ± 0.037 | 0.718 ± 0.013 | 0.427 ± 0.044 | 0.713 ± 0.001 |
| jacobian + L∞ | 0.953 ± 0.010 | 0.844 ± 0.017 | 0.440 ± 0.026 | 0.608 ± 0.008 | 0.696 ± 0.022 |
| jac+pull + L∞ | 0.950 ± 0.008 | 0.819 ± 0.010 | 0.605 ± 0.056 | 0.671 ± 0.048 | 0.512 ± 0.002 |
| λ batch-detached | 0.923 ± 0.003 | 0.744 ± 0.008 | 0.613 ± 0.017 | 0.590 ± 0.028 | 0.697 ± 0.012 |
| jacobian + λ | 0.930 ± 0.009 | 0.858 ± 0.004 | 0.227 ± 0.046 | 0.236 ± 0.059 | 0.591 ± 0.034 |
| jac+pull + λ | 0.924 ± 0.024 | 0.856 ± 0.014 | 0.192 ± 0.011 | 0.170 ± 0.006 | 0.224 ± 0.019 |

Reading:

- Base is easy for every model (AUC 0.980-0.993).
- Color: λ batch-detached is the best safety reg on every base (0.836 / 0.891 / 0.894). The Jacobian term helps on its own
  (sigreg 0.656 -> jacobian 0.737, jac+pull 0.789). L∞ barely moves the margin (0.665 / 0.753 / 0.750). The thresholds of
  jacobian + λ, jac+pull + λ and jacobian + L∞ roughly hold (sign accuracy 0.844-0.858 against 0.832 for calling
  everything safe); the others don't.
- Viewpoint breaks every model (0.530-0.801), and the ranking flips: jacobian + λ and jac+pull + λ, best on color, are the
  worst on zoom_out and pan_x (0.530-0.596) and call nearly everything unsafe (sign accuracy 0.170-0.236).
- No model is good on all three views. The best per view are L∞ on zoom_out (0.800), jacobian on pan_x (0.754) and
  jac+pull + L∞ on pan_y (0.801). The Jacobian term helps a little without a safety reg (jacobian 0.727 / 0.754 / 0.758 vs
  sigreg 0.619 / 0.620 / 0.746).

## Why viewpoint fails: the latent moves, the information stays

Test episodes split in half: heads trained on one half, scored on the other, either zero-shot (trained on base frames) or
retrained on the shifted frames (`shift_diag.py`). "latent move" = mean ‖z_shift(s) - z(s)‖ over paired states divided by
the mean distance between two random states. Four of the models.

| Model | shift | latent move | margin zero-shot | margin retrained | probe zero-shot | probe retrained |
|---|---|---|---|---|---|---|
| sigreg | zoom_out | 1.372 | 0.612 ± 0.006 | 0.995 ± 0.001 | 0.599 ± 0.019 | 0.943 ± 0.001 |
| jac+pull | zoom_out | 0.919 | 0.699 ± 0.003 | 0.984 ± 0.000 | 0.604 ± 0.002 | 0.936 ± 0.004 |
| jac+pull + λ | zoom_out | 1.310 | 0.575 ± 0.006 | 0.971 ± 0.004 | 0.647 ± 0.026 | 0.930 ± 0.001 |
| jacobian + L∞ | zoom_out | 0.951 | 0.587 ± 0.013 | 0.981 ± 0.003 | 0.570 ± 0.034 | 0.933 ± 0.001 |
| sigreg | color | 0.597 | 0.637 ± 0.012 | 0.991 ± 0.000 | 0.703 ± 0.004 | 0.934 ± 0.003 |
| jac+pull | color | 0.598 | 0.780 ± 0.005 | 0.987 ± 0.000 | 0.663 ± 0.010 | 0.931 ± 0.002 |
| jac+pull + λ | color | 0.340 | 0.904 ± 0.004 | 0.981 ± 0.001 | 0.893 ± 0.003 | 0.930 ± 0.001 |
| jacobian + L∞ | color | 0.441 | 0.777 ± 0.014 | 0.984 ± 0.004 | 0.830 ± 0.004 | 0.933 ± 0.002 |

- Nothing is lost: a margin retrained on zoomed frames is as good as in-distribution (0.971-0.995).
- A 0.8× zoom moves the latent about as far as the distance between two random states (0.92-1.37), roughly twice the color
  shift. Every zoomed frame lands where the head never saw data. A likely reason, not tested: in training only the cart and
  pole move, while zoom also moves and rescales the walls, track and background, which never changed.
- λ batch-detached's color invariance is specific to color: it has the smallest color move (0.340) but a zoom move as large
  as sigreg's (1.310).

## Value function (critic) and probe

The margin is "currently safe"; doomed states need a separate value function. Two heads, both on the last three latents
[z_{t-2}, z_{t-1}, z_t] (one frame carries no velocity):

- critic: trained by RL on the learned margin, the discounted safety Bellman backup, with no value labels. This version is
  offline fitted Q-iteration on dataset transitions; it will be replaced by the Dubins setup (avoid-DDPG in WM imagination,
  `wm_ddpg.py`).
- probe: supervised regression on the oracle V, a diagnostic for whether the information is in the latent at all.

Scored by doomed AUC: among not-failed frames, doomed vs safe. The same critic on the true state reaches 0.985 ± 0.004.

Critic doomed AUC

| Model | base | color | zoom_out | pan_x | pan_y |
|---|---|---|---|---|---|
| sigreg | 0.953 ± 0.002 | 0.785 ± 0.014 | 0.681 ± 0.048 | 0.689 ± 0.040 | 0.770 ± 0.014 |
| jacobian | 0.962 ± 0.002 | 0.797 ± 0.008 | 0.699 ± 0.015 | 0.735 ± 0.020 | 0.618 ± 0.024 |
| jac+pull | 0.962 ± 0.002 | 0.850 ± 0.011 | 0.617 ± 0.006 | 0.666 ± 0.013 | 0.670 ± 0.033 |
| L∞ safety reg | 0.940 ± 0.001 | 0.788 ± 0.023 | 0.740 ± 0.005 | 0.591 ± 0.014 | 0.732 ± 0.022 |
| jacobian + L∞ | 0.952 ± 0.004 | 0.893 ± 0.007 | 0.692 ± 0.060 | 0.687 ± 0.079 | 0.770 ± 0.035 |
| jac+pull + L∞ | 0.948 ± 0.002 | 0.863 ± 0.004 | 0.669 ± 0.078 | 0.692 ± 0.057 | 0.748 ± 0.031 |
| λ batch-detached | 0.939 ± 0.003 | 0.843 ± 0.020 | 0.646 ± 0.019 | 0.512 ± 0.017 | 0.705 ± 0.019 |
| jacobian + λ | 0.947 ± 0.003 | 0.913 ± 0.004 | 0.563 ± 0.057 | 0.587 ± 0.052 | 0.589 ± 0.016 |
| jac+pull + λ | 0.958 ± 0.002 | 0.947 ± 0.003 | 0.642 ± 0.007 | 0.584 ± 0.019 | 0.626 ± 0.009 |

Probe doomed AUC

| Model | base | color | zoom_out | pan_x | pan_y |
|---|---|---|---|---|---|
| sigreg | 0.953 ± 0.001 | 0.772 ± 0.022 | 0.625 ± 0.020 | 0.616 ± 0.017 | 0.662 ± 0.023 |
| jacobian | 0.949 ± 0.001 | 0.701 ± 0.020 | 0.663 ± 0.017 | 0.730 ± 0.017 | 0.603 ± 0.015 |
| jac+pull | 0.950 ± 0.002 | 0.750 ± 0.008 | 0.672 ± 0.002 | 0.702 ± 0.008 | 0.703 ± 0.028 |
| L∞ safety reg | 0.946 ± 0.001 | 0.772 ± 0.015 | 0.672 ± 0.005 | 0.559 ± 0.009 | 0.790 ± 0.005 |
| jacobian + L∞ | 0.946 ± 0.001 | 0.863 ± 0.006 | 0.665 ± 0.027 | 0.625 ± 0.042 | 0.793 ± 0.019 |
| jac+pull + L∞ | 0.947 ± 0.002 | 0.814 ± 0.010 | 0.629 ± 0.016 | 0.638 ± 0.011 | 0.693 ± 0.035 |
| λ batch-detached | 0.943 ± 0.002 | 0.834 ± 0.009 | 0.605 ± 0.001 | 0.501 ± 0.004 | 0.682 ± 0.007 |
| jacobian + λ | 0.942 ± 0.003 | 0.878 ± 0.007 | 0.556 ± 0.011 | 0.551 ± 0.038 | 0.556 ± 0.005 |
| jac+pull + λ | 0.944 ± 0.001 | 0.924 ± 0.003 | 0.666 ± 0.006 | 0.632 ± 0.022 | 0.617 ± 0.016 |

Reading:

- In distribution the critic finds doomed states without doomed labels (0.939-0.962), as well as the supervised probe
  (0.942-0.953). The margin alone ranks doomed below safe at only about 0.70, so the separate value function is needed.
- Under the shifts the critic follows the margin's pattern: λ batch-detached best on color (jac+pull + λ 0.947), every
  model weak on viewpoint (0.512-0.770). The probe drops just as much, so the failure is in the latent, not the critic.

## Caveats and next steps

- One training seed per model. Single-seed color results on Dubins did not hold over 3 seeds, so the color ranking here
  needs reseeds.
- The critic above is the offline version; the Dubins-style imagination critic is next.
- The oracle is slightly optimistic near V = 0 (9% of V > 0.02 states fail under the greedy policy). Doomed labels are clean.
- A viewpoint fix probably needs geometric training (zoom/pan augmentation, or an adversary over small camera shifts)
  rather than appearance invariance.
