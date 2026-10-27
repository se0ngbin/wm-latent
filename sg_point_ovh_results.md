# Overhead@224 SafetyPointGoal1: layout-disjoint margin OOD for 9 world models (2026-10-27)

Supersedes `sg_point_results.md`, whose protocol (sg_plain_eval: shuffled frame split of one 5000-pose collection) let the
probe's train and test frames share hazard layouts. On unseen layouts every fixedfar@128 model was at chance (.36-.63),
although a supervised CNN on the same pixels reaches .998-.999: the pooled WM latent was the bottleneck, not the benchmark.

- Training data: `sg_point_ovh224.h5`, clean renders (no cost sphere, no lidar ring) from an overhead camera at 224 px
  (fixedfar moved to (0,0,4), straight down, frames ±1.8; agent ~143 px vs 6 px). Same seeds, trajectories and labels as
  `sg_point_clean.h5`; 170,400 frames, ~850 random layouts.
- WMs: le-wm snapshot 92e7589, seed 3072, 50 epochs, same 9 recipes (`le-wm/scripts/run_point.sh`, suffix `_ovh`).
- Eval (`le-wm/scripts/sg_point_bigprobe.py`): margin_gp head trained on 50k frames of the TRAINING set, scored on all
  5000 poses of `sg_point_ovh224_multi_eval.npz`, a separate collection, so every layout is unseen. Views: base, purple
  hazards (color), 90° camera rotation (rotate; an exact image rotation for this camera), box hazards (shape), Car robot (agent).
  ± = std over 3 eval seeds (probe subsample + init); one training seed per model. "train fit" = probe AUC on its own training frames.

## AUC

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

## Sign accuracy (eval set is 0.830 safe, so 0.830 = everything called safe, 0.170 = everything called unsafe)

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

## Reading

- jacobian + L∞ safety reg is best on every axis (AUC .917-.985) and the only model whose threshold holds on all five
  views (predicted-safe .79-.86 vs true .83).
- The safety regularizers help on this benchmark: sigreg .581 base -> L∞ .948 / λ .885. Combined with the Jacobian term
  they form the top group (jacobian + λ, jac+pull + λ: ~.97 base, ~.90 color).
- Color is the hardest shift and breaks thresholds: L∞ alone calls everything unsafe on purple (sign acc .170); jacobian
  calls 70% unsafe.
- The action-separation pull hurts here (jac+pull .836 vs jacobian .954 base).
- sigreg stays near chance: the camera change does not rescue an unregularized latent.
- Overhead vs fixedfar, same probe on unseen layouts: jac+pull .563 -> .836 base, sigreg .362 -> .581.
- One training seed per model; seeds 1-2 queued.
