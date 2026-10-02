# AC-MTM (action-contrastive JEPA) — OOD safety-representation study

Alternative anti-collapse from `action-contrastive-jepa/` (arXiv 2608.17542): SIGReg OFF, replaced by a
training-only inverse-dynamics head `(z_t, z_{t+1}) → a_t` classified among in-batch action blocks
(`-‖pred−tgt‖²/τ`, τ=0.1, weight 0.30). Ported into our `le-wm` fork as `loss.regularizers.action_nce`
(`ActionNCEReg`, `le-wm/module.py`) so every existing eval script loads the checkpoints unchanged.

Scope (selected 2026-10-01): **T1a** encoder mechanism probe + **T1b** margin_gp OOD, dubins + safety-gym.
Deferred: raw CEM sg25clean, fs=5 action collapse, CBF in-loop, critic OOD.

## Hypotheses
1. SIGReg's variance-cheating is the dubins "disease" that jacobian treats → does AC-MTM come out
   well-conditioned / appearance-robust *without* the jacobian penalty?
2. Risk: AC-MTM underweights variables actions don't move (it loses PushT). Obstacles/hazards are static →
   they may drop out of the latent → watch the **in-dist** margin AUC (esp. safety-gym, random layouts).

## Setup
- Arms: `acmtm`, `acmtm+jac` (jacobian w1.0, target_L 1.0) vs existing `sigreg_only_dubins` /
  `jacobian_w1_dubins` / `lewm_dubins_jacpull50` and `sg_baseline` / `sg_jacobian` / `sg_jacpull`.
- Recipe matches the references: 50 epochs, fs=1, bs 128, seed 3072 first, then seeds 1, 2.
- Train: `le-wm/scripts/train_acmtm.sh` (2 lanes, 1 run/GPU). Eval: `le-wm/scripts/eval_acmtm.sh`
  (auto-starts on each epoch-50 ckpt): `jac_singular.py acmtm_dubins`, `dubins_mechanism.py`,
  `ood_margin_gp_jepa.py {color,shape,rotate} × seeds 0–2`, `sg_mechanism.py`, `sg_robust_margin.py` (SG_SEED 0,1).
  Reference encoders re-run at probe seed 0 as a reproduction check.
- Logs: `run_logs/acmtm/`.

## Data notes
- dubins action: continuous 1-D turn rate in [−2, 2], essentially no duplicates → contrastive task well-posed.
- safety-gym action: 2-D, saturated at 1.0 on 63% / 37% of samples → many exact-duplicate in-batch
  negatives (false negatives; acknowledged in the AC-MTM README). Kept the method as published.
- Loss scale: with τ=0.1 a constant prediction scores ≈14 and a perfect inverse ≈4.1 on dubins
  (density of near-identical continuous actions), so loss > log N (5.95) is not by itself collapse.

## Training status (seed 3072, dubins, ~1 h in)
| run | epoch | inverse loss | inverse acc (chance 0.26%) | pred loss |
|---|---|---|---|---|
| dubins_acmtm | 14/50 | ~7.9 | 0.6% | 0.07 |
| dubins_acmtm_jac | 9/50 | ~4.8 | 1.5% | 0.005 |

No collapse in either (inverse head above chance). The jacobian arm makes the inverse task *easier*.
Pred losses are not comparable across arms (latent scale is free without SIGReg; the jacobian hinge pins it).

## Reference mechanism rows (dubins, `dubins_mechanism.py`, uniform-random states)
| encoder | eff_rank σ² | eff_rank σ (report def) | ‖J‖_F | s1 share | color disp/σ | shape disp/σ | rotate disp/σ |
|---|---|---|---|---|---|---|---|
| baseline | 5.5 | 14.1 | 17.05 | .315 | 1.41 | 0.73 | 1.18 |
| jacobian | 3.5 | 13.5 | 7.97 | .458 | 0.79 | 0.18 | 1.03 |
| jac+pull | 3.5 | 20.6 | 4.27 | .474 | 0.73 | 0.13 | 1.02 |

‖J‖ and color displacement order as in the report. eff_rank does NOT reproduce the report's
"~3 vs ~33": the report computes it on 16 dataset frames (`jac_singular.py`), these use uniform-random
states. `jac_singular.py acmtm_dubins` is in the eval driver for report-comparable conditioning.

## Results
*(pending — filled in as the eval driver lands)*
