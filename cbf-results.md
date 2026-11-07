# Dubins LeWM: Dreamer-framework CBF vs margin function

Every row: frozen LeWM (epoch 50, WM seed 3072) → margin function in the adapter's `margin_gp` slot → reachability critic (`wm_ddpg.py --wm_backend lewm`, gp mode, 10 × 40k steps, seed 0) → CBF-filtered diffusion policy (`collect_trajs.py --filter_mode cbf`, 1000 trajectories). Collision = episode ever inside an obstacle. Single seed per row.

## Margin functions

All use the `margin_gp` loss (zero-sum 0.1, ReLU 1.0, gradient penalty 10 at norm 0.1), AdamW 3e-4, on the frozen CLS-projector embedding; $h \ge 0$ = safe.

| Tag | Head | Training states | Label | Steps |
|---|---|---|---|---|
| buffer | MLP | 169,189 Dreamer-buffer frames (`train_margin_lewm.py`) | buffer `failure` flag | 40,000, unbalanced |
| gpfail | MLP | 8,000 uniform $(x, y, \theta)$ renders (`train_gp_margin_dubins.py`) | inside obstacle | 5,000, balanced |
| gpimag | MLP | gpfail states + 150k predicted latents of 25-step CEM-style plans (`train_gp_margin_imag_dubins.py`) | inside obstacle at simulated state | 5,000 |
| lin | linear $w^\top z + b$ | gpfail states only (`train_lin_margin_dubins.py`); no imagined linear head exists | inside obstacle | 5,000 |

jac+pull+adv_h (`lewm_dubins_jacpull_safeadv_linf50`) was trained with an MLP adversary head (192-512-512-1, LayerNorm + SiLU, same architecture as the margin MLP). So its gpfail / gpimag rows match the adversary's head class and its lin row does not. The linear-adversary variant (`..._linhead50`, `head_layers:0`) is not trained yet.

## Margin accuracy on the critic's inputs

19,200 Dreamer-buffer frames through the adapter (82.2% safe). FS = false-safe (unsafe with $h \ge 0$), FA = false-alarm (safe with $h < 0$).

| WM | buffer AUC / FS | gpfail AUC / FS / FA | gpimag AUC / FS / FA | lin AUC / FS / FA |
|---|---|---|---|---|
| sigreg | 0.983 / 0.073 | 0.781 / 0.927 / 0.007 | 0.938 / 0.086 / 0.172 | 0.750 / 0.908 / 0.028 |
| jac | 0.999 / 0.014 | 0.997 / 0.059 / 0.018 | 0.998 / 0.037 / 0.018 | 0.996 / 0.043 / 0.023 |
| jac+pull | 0.996 / 0.082 | 0.996 / 0.069 / 0.016 | 0.998 / 0.018 / 0.021 | 0.997 / 0.049 / 0.021 |
| jac+pull+adv_h | – | 0.998 / 0.041 / 0.011 | 0.998 / 0.005 / 0.045 | 0.998 / 0.024 / 0.019 |

On imagined latents (held-out 25-step rollouts), gpfail → gpimag FS: sigreg 0.968 → 0.302, jac+pull 0.240 → 0.093, jac+pull+adv_h 0.296 → 0.122.

## CBF: collision / success

| WM | buffer | gpfail | gpimag | lin |
|---|---|---|---|---|
| sigreg | 0.706 / 0.428 | 0.557 / 0.764 | 0.710 / 0.368 | 0.617 / 0.739 |
| jac | 0.071 / 0.794 | 0.065 / 0.761 | 0.006 / 0.785 | 0.101 / 0.160 |
| jac+pull | 0.016 / 0.821 | 0.024 / 0.819 | 0.008 / 0.794 | 0.111 / 0.521 |
| jac+pull+adv_h | – | 0.100 / 0.733 | 0.021 / 0.489 | 0.289 / 0.190 |

No filter ≈ 0.54 / 0.87 (nominal policy on a Dreamer RSSM, `nom_baseline.h5`; approximate for LeWM).

- gpfail matches buffer on jac and jac+pull (within about a point).
- gpimag (uniform + imagined states) is the best margin function on jac and jac+pull: collision 0.6% / 0.8% at the same success as buffer/gpfail. Consistent with its lower false-safe rate on imagined latents (single seed).
- Linear margins collide more and succeed far less (jac 0.160, jac+pull+adv_h 0.190 success): the filter both lets the car into obstacles and blocks the goal.
- jac+pull+adv_h is worse than jac+pull with every margin (gpfail 10.0% vs 2.4%, gpimag 2.1% vs 0.8%, with lower success). The adversarial safety reg does not help the CBF in-distribution.
- sigreg stays at or above no-filter collision with every margin function.
- OOD (purple obstacles, diamond obstacles; same red-trained critics) for gpfail / gpimag / lin: running, expected about 3–4 pm PST Nov 7. Earlier buffer-margin OOD run: purple ≈ no filter for every WM (0.52–0.55 collision); diamond jac 0.032, jac+pull 0.039, sigreg 0.432.
- sigreg gpfail is effectively always-safe (FS 0.927), so its CBF ≈ no filter; the accurate buffer margin collides more than no filter.

## Files

Critics `/data/seongbin/dreamer/lewm/<wm>/seed0[_<tag>]/PyHJ/gp/epoch_id_9/policy.pth`; trajectories `/data/seongbin/dreamer/trajs/lewm_cbf_gp_[<tag>_]<wm>_seed0.h5`; logs `run_logs/lewm_cbf_gpfail/`, `run_logs/lewm_cbf_heads/`; driver `latent_cbf/scripts/cbf_margin_heads.sh wm:tag ...`; loader `LEWMWorldModel.load_margin`. WMs: sigreg = `sigreg_only_dubins`, jac = `jacobian_w1_dubins`, jac+pull = `lewm_dubins_jacpull50`, jac+pull+adv_h = `lewm_dubins_jacpull_safeadv_linf50`.
