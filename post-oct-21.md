# Safety Gym and Cartpole

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

(To be written by the cartpole agent.)
