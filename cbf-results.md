# Dubins LeWM: Dreamer-framework CBF with different margin functions

Pipeline (same for every row): frozen LeWM (epoch 50, seed 3072) → margin function loaded into the adapter's `margin_gp` slot → reachability critic trained with imagination DDPG (`wm_ddpg.py --wm_backend lewm`, gp mode, 10 rounds × 40k steps, seed 0) → CBF-filtered diffusion policy eval (`collect_trajs.py --filter_mode cbf`, 1000 trajectories, 2 obstacles). Collision = fraction of episodes that ever enter an obstacle; success = goal reached. Single critic seed and single WM seed per row.

## Margin functions

| Tag | Script | Head | Training states | Label | Steps |
|---|---|---|---|---|---|
| buffer | `latent_cbf/.../train_margin_lewm.py` | MLP | Dreamer buffer frames (169,189 states, 4,100 trajectories) | buffer `failure` flag | 40,000 (batch 32 × 4 frames, unbalanced) |
| gpfail | `le-wm/scripts/train_gp_margin_dubins.py` | MLP | 8,000 uniform $(x, y, \theta)$ states, rendered red circles (6,000 train) | inside-obstacle (exact geometry) | 5,000 (balanced 256 + 256) |
| gpimag | `le-wm/scripts/train_gp_margin_imag_dubins.py` (branch `worktree-imag-margin`) | MLP | gpfail's 8,000 uniform states + 150k predicted latents (6,000 CEM-style 25-step plans from real expert frames) | inside-obstacle at the simulated true state | 5,000 |
| lin | `le-wm/scripts/train_lin_margin_dubins.py` (branch `worktree-imag-margin`) | linear $h(z) = w^\top z + b$ | same as gpfail (uniform only) | same as gpfail | 5,000 |

All use the same `margin_gp` loss (zero-sum 0.1, ReLU 1.0, gradient penalty 10 with target norm 0.1), AdamW 3e-4, on the frozen CLS-projector embedding; $h \ge 0$ = safe.

## Margin-function accuracy on the critic's input distribution

Scored on 19,200 Dreamer-buffer frames encoded through the adapter (82.2% safe). False-safe = fraction of unsafe frames with $h \ge 0$; false-alarm = fraction of safe frames with $h < 0$.

| WM | Margin | AUC | False-safe | False-alarm |
|---|---|---|---|---|
| sigreg | buffer | 0.983 | 0.073 | – |
| sigreg | gpfail | 0.781 | 0.927 | 0.007 |
| sigreg | gpimag | 0.938 | 0.086 | 0.172 |
| sigreg | lin | 0.750 | 0.908 | 0.028 |
| jac | buffer | 0.999 | 0.014 | – |
| jac | gpfail | 0.997 | 0.059 | 0.018 |
| jac | gpimag | 0.998 | 0.037 | 0.018 |
| jac | lin | 0.996 | 0.043 | 0.023 |
| jac+pull | buffer | 0.996 | 0.082 | – |
| jac+pull | gpfail | 0.996 | 0.069 | 0.016 |
| jac+pull | gpimag | 0.998 | 0.018 | 0.021 |
| jac+pull | lin | 0.997 | 0.049 | 0.021 |
| jac+pull+adv_h | gpfail | 0.998 | 0.041 | 0.011 |
| jac+pull+adv_h | gpimag | 0.998 | 0.005 | 0.045 |
| jac+pull+adv_h | lin | 0.998 | 0.024 | 0.019 |

On imagined latents (25-step predictor rollouts of CEM-style plans, held-out episodes; from the gpimag training logs), gpfail → gpimag false-safe: sigreg 0.968 → 0.302, jac+pull 0.240 → 0.093, jac+pull+adv_h 0.296 → 0.122.

## CBF results

| WM | Margin | Collision | Success | Filtered actions / episode |
|---|---|---|---|---|
| (no filter) | – | ≈ 0.54 | ≈ 0.87 | – |
| sigreg | buffer | 0.706 | 0.428 | – |
| sigreg | gpfail | 0.557 | 0.764 | 8.1 |
| sigreg | gpimag | pending | pending | pending |
| sigreg | lin | pending | pending | pending |
| jac | buffer | 0.071 | 0.794 | – |
| jac | gpfail | 0.065 | 0.761 | 13.5 |
| jac | gpimag | pending | pending | pending |
| jac | lin | pending | pending | pending |
| jac+pull | buffer | 0.016 | 0.821 | 9.2 |
| jac+pull | gpfail | 0.024 | 0.819 | 11.8 |
| jac+pull | gpimag | pending | pending | pending |
| jac+pull | lin | pending | pending | pending |
| jac+pull+adv_h | gpfail | pending | pending | pending |
| jac+pull+adv_h | gpimag | pending | pending | pending |
| jac+pull+adv_h | lin | pending | pending | pending |

The no-filter row is the nominal diffusion policy run against a Dreamer RSSM (`nom_baseline.h5`), so it is approximate for the LeWM rows. The jac buffer row is `lewm_cbf_gp_jacobian_w1_dubins_seed0.h5` (an earlier jac buffer run gave 0.075 / 0.761).

- gpfail gives the same CBF as the buffer margin on jac and jac+pull (within about a point on collision and success).
- On sigreg, gpfail is effectively an always-safe margin (false-safe 0.927), so its CBF behaves like no filter; the buffer margin is accurate but its CBF collides more than no filter.

## Files

- Critics: `/data/seongbin/dreamer/lewm/<wm>/seed0/` (buffer), `seed0_gpfail/`, `seed0_gpimag/`, `seed0_lin/`, each under `PyHJ/gp/epoch_id_9/policy.pth`.
- Trajectories: `/data/seongbin/dreamer/trajs/lewm_cbf_gp_<tag>_<wm>_seed0.h5` (buffer rows: `lewm_cbf_gp_<wm>[_seed0].h5`).
- Logs: `run_logs/lewm_cbf_gpfail/` (sigreg / jac / jac+pull gpfail), `run_logs/lewm_cbf_heads/` (everything else).
- Driver: `latent_cbf/scripts/cbf_margin_heads.sh wm:tag ...`; loader: `LEWMWorldModel.load_margin` in `latent_cbf/src/latent_cbf/adapters/lewm_adapter.py`.
- WM checkpoints: sigreg = `sigreg_only_dubins`, jac = `jacobian_w1_dubins`, jac+pull = `lewm_dubins_jacpull50`, jac+pull+adv_h = `lewm_dubins_jacpull_safeadv_linf50`.
