# Clean SafetyPointGoal1: margin OOD for 9 world models (2026-10-24)

> Superseded by `sg_point_ovh_results.md`. This protocol shares hazard layouts between the probe's train and test frames
> (layout memorization); on unseen layouts these fixedfar@128 models are at chance.

Why: every earlier safety-gym pixel set (Car) contained safety-gym's red cost-indicator sphere, drawn at the agent
whenever the last step had cost > 0, i.e. exactly our failure label (red-pixel AUC .998-1.000 on training and every
eval set). These runs use clean renders (no sphere, no lidar ring) for both training and test, and the Point robot,
whose safety depends on velocity (braking distance 0.24 > hazard radius 0.2; Car brakes in 0.055).

- Training data: `sg_point_clean.h5`, 170k frames, 19.3% unsafe (red-pixel AUC .405).
- WMs: le-wm, code snapshot safety-adv-reg 92e7589 (`le-wm/scripts/run_point.sh`), seed 3072, 50 epochs. Safety-reg
  recipes copied from the Car-SG reference runs; ramp reached 1.0 in all six regularized runs (gate opened for the .95-gated ones).
- Eval: `sg_point_multi_eval.npz` (`safety_gym_port/collect_point_multi.py`): 5000 Point poses, each rendered clean as
  base / purple hazards (color) / real 90° camera rotation (rotate) / box hazards (shape) / Car robot (agent).
  Red-pixel AUC .31-.55 on every view. Protocol of `sg_plain_eval.py` (`le-wm/scripts/sg_point_eval.py`): 80/20 frame
  split, margin_gp head on base train frames, scored on the 1000 test frames. ± = std over 3 eval seeds (1 training seed).

## Zero-shot margin AUC

| Model | in-dist | color | rotate | shape | agent |
|---|---|---|---|---|---|
| sigreg | 0.993 ± 0.000 | 0.398 ± 0.025 | 0.564 ± 0.006 | 0.875 ± 0.006 | 0.991 ± 0.001 |
| jacobian | 0.993 ± 0.000 | 0.932 ± 0.010 | 0.551 ± 0.007 | 0.990 ± 0.000 | 0.989 ± 0.000 |
| jac+pull | 0.992 ± 0.000 | 0.986 ± 0.001 | 0.385 ± 0.005 | 0.990 ± 0.000 | 0.990 ± 0.000 |
| L∞ safety reg | 0.994 ± 0.000 | 0.509 ± 0.012 | 0.593 ± 0.007 | 0.869 ± 0.004 | 0.964 ± 0.001 |
| jacobian + L∞ | 0.992 ± 0.001 | 0.972 ± 0.002 | 0.337 ± 0.006 | 0.987 ± 0.000 | 0.988 ± 0.000 |
| jac+pull + L∞ | 0.992 ± 0.001 | 0.984 ± 0.001 | 0.395 ± 0.011 | 0.989 ± 0.000 | 0.991 ± 0.001 |
| λ batch-detached | 0.992 ± 0.000 | 0.572 ± 0.029 | 0.483 ± 0.018 | 0.865 ± 0.002 | 0.983 ± 0.002 |
| jacobian + λ | 0.993 ± 0.000 | 0.820 ± 0.004 | 0.457 ± 0.004 | 0.991 ± 0.000 | 0.871 ± 0.007 |
| jac+pull + λ | 0.987 ± 0.001 | 0.846 ± 0.003 | 0.414 ± 0.010 | 0.986 ± 0.001 | 0.882 ± 0.004 |

## Sign accuracy (h ≥ 0 = safe; test set is 0.817 safe, so 0.817 = everything called safe)

| Model | in-dist | color | rotate | shape | agent |
|---|---|---|---|---|---|
| sigreg | 0.985 ± 0.002 | 0.817 ± 0.000 | 0.817 ± 0.000 | 0.817 ± 0.000 | 0.974 ± 0.002 |
| jacobian | 0.983 ± 0.001 | 0.817 ± 0.000 | 0.817 ± 0.000 | 0.889 ± 0.000 | 0.943 ± 0.007 |
| jac+pull | 0.984 ± 0.002 | 0.826 ± 0.013 | 0.817 ± 0.000 | 0.915 ± 0.000 | 0.944 ± 0.008 |
| L∞ safety reg | 0.984 ± 0.001 | 0.817 ± 0.000 | 0.822 ± 0.007 | 0.813 ± 0.003 | 0.925 ± 0.002 |
| jacobian + L∞ | 0.981 ± 0.002 | 0.864 ± 0.000 | 0.817 ± 0.000 | 0.864 ± 0.000 | 0.938 ± 0.003 |
| jac+pull + L∞ | 0.979 ± 0.003 | 0.817 ± 0.000 | 0.817 ± 0.000 | 0.899 ± 0.023 | 0.978 ± 0.003 |
| λ batch-detached | 0.973 ± 0.002 | 0.812 ± 0.017 | 0.817 ± 0.000 | 0.871 ± 0.000 | 0.934 ± 0.016 |
| jacobian + λ | 0.976 ± 0.002 | 0.886 ± 0.002 | 0.803 ± 0.002 | 0.969 ± 0.001 | 0.944 ± 0.001 |
| jac+pull + λ | 0.973 ± 0.002 | 0.918 ± 0.000 | 0.786 ± 0.003 | 0.967 ± 0.004 | 0.942 ± 0.002 |

## Reading

- Color: the Jacobian family carries it (jacobian .932, jac+pull .986); sigreg is below chance (.398, inverted, as on
  Dubins). The safety regs alone do not help: L∞ .509, λ .572. Adding L∞ to the Jacobian family keeps it (.972/.984);
  adding λ lowers it (.820/.846).
- Rotation fails for every model (.34-.59); the Jacobian family is below chance (.34-.41).
- Shape is easy for the Jacobian family (.99), .87 otherwise. Agent swap (Point→Car) is easy (.96-.99) except jac/jac+pull + λ (.87-.88).
- Thresholds collapse: color/rotate predict almost everything safe for nearly all models. The λ+Jacobian models keep the
  most (color sign acc .886/.918, shape .969/.967).
- Versus the leaky Car SG numbers: there, L∞ alone gave color .999 / rotate .962 and jac+pull + L∞ rotate .981 with thresholds
  kept. On clean Point, L∞ alone is ~chance on color and nothing survives rotation. The red sphere is colour- and
  rotation-invariant, so those Car gains are suspect. Not yet separated from the Car→Point agent change: that needs a clean Car retrain.
- One training seed per model.
