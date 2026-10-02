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

## Results — WM seed 3072 (seeds 1, 2 still running)

All 4 seed-3072 WMs trained cleanly (no collapse). Reference encoders reproduce at probe seed 0
(e.g. baseline color zs_auc .136 vs worklog .144±.03; shape .885 vs .913; rotate .973 vs .956).

### ⚠ The zero-shot sign accuracy is mostly a base-rate artifact
The margin test split is ~16.3% unsafe. zs_acc = **.837 is "everything safe"** and **.164 is "everything
unsafe"**, and those exact values recur to 3 decimals across unrelated encoders (baseline, jac+pull,
acmtm, acmtm+jac). So for color/rotate zero-shot the sign threshold is degenerate and **only AUC carries
information**. This also hits the existing worklog: jac+pull color "acc .830 — the pull term carries the
sign" (post-sept-3 margin_gp table) matches the all-safe classifier. Worth re-checking with the fraction
predicted safe (or balanced acc) before relying on any zero-shot sign claim. Below: AUC only.

### T1b — dubins margin_gp AUC (gp head; acmtm arms = mean±sd over probe seeds 0–2, refs = seed 0)
| encoder | in-dist | color zs → retrained | shape zs → retrained | rotate zs → retrained |
|---|---|---|---|---|
| baseline (sigreg) | .980 | .136 → .962 | .885 → .985 | .973 → .988 |
| jacobian | .992 | .757 → .992 | .993 → .988 | .991 → .998 |
| jac+pull | .984 | .707 → .988 | .977 → .968 | .980 → .997 |
| **acmtm** | .961±.009 | .456±.025 → .962 | .650±.006 → .901 | **.478±.003** → .952 |
| **acmtm+jac** | .977±.008 | .556±.128 → .983 | .965±.011 → .965 | .959±.004 → .975 |

### T1b — safety-gym layout-controlled margin AUC (plain readout; acmtm arms = probe seeds 0/1)
| encoder | in-dist blue | zs purple | zs rotate (90° cam) |
|---|---|---|---|
| baseline | .994 | .604 | .564 |
| jacobian | .996 | .991 | .641 |
| jac+pull | .999 | .995 | .692 |
| **acmtm** | .998 / .998 | **.911 / .896** | .510 / .496 |
| **acmtm+jac** | .997 / .994 | .889 / .882 | **.901 / .884** |

(aug readout: acmtm+jac purple .947/.926, rotate .925/.911. Previous best sg rotation anywhere was the
D_R projection on jac+pull, .772.)

### T1a — encoder mechanism
dubins, report-comparable (`jac_singular.py`, 16 dataset frames):

| encoder | σ1 | ‖J‖_F | top-1 energy | #σ>0.1σ1 | eff_rank (Σσ)²/Σσ² |
|---|---|---|---|---|---|
| baseline | 7.83 | 11.02 | .505 | 11 | 13.1 |
| jacobian | 0.31 | 1.01 | .095 | 91 | 86.5 |
| jac+pull | 0.30 | 1.01 | .091 | 95 | 90.8 |
| **acmtm** | **23.5** | **36.4** | .417 | 14 | 22.6 |
| **acmtm+jac** | 0.20 | 0.99 | .041 | 146 | 124.5 |

dubins shift displacement (`dubins_mechanism.py`, disp/σ): acmtm color **2.44** / shape .53 / rotate 1.47
(baseline 1.41/.73/1.18); acmtm+jac 1.43/.13/1.17 (jacobian .79/.18/1.03).

safety-gym (`sg_mechanism.py`): acmtm eff_rank(σ²) 49.4, ‖J‖ 8.08, color disp/σ .93, rotate .91;
acmtm+jac 45.1, 0.89, .75, .90 (baseline 2.9, 7.59, .70, .90; jacobian 16.4, 1.10, .28, 1.12).

### Verdicts
1. **Hypothesis 1 (SIGReg is the disease) — refuted on dubins.** Dropping SIGReg for AC-MTM does *not*
   produce a well-conditioned encoder: AC-MTM is **3× more input-sensitive** than SIGReg (‖J‖ 36 vs 11, σ1
   23.5 vs 7.8) and amplifies the color shift more (disp/σ 2.44 vs 1.41). The hypersensitivity is not
   specific to SIGReg — an action-identifying objective also rewards amplifying a few pixel directions.
   The jacobian penalty still fixes it (acmtm+jac ‖J‖ 0.99, best-spread spectrum of any encoder).
2. **AC-MTM alone is worse than SIGReg on dubins OOD, and breaks rotation.** Rotation zs .478 ≈ chance
   (SIGReg baseline .973) — the same failure as Dreamer (.480). Shape drops .885 → .650. Color improves
   over baseline (.136 → .456) but stays well below jacobian (.757). Retrained AUC stays high (.90–.96):
   information is present, only the zero-shot transfer fails — same calibration-failure pattern as before.
3. **Hypothesis 2 (static obstacles dropped) — not realized.** In-dist AUC .961 dubins / .998 sg; safety-gym
   with random hazard layouts is the case most at risk and it is at ceiling.
4. **Safety-gym tells a different story.** AC-MTM alone fixes most of color (purple .60 → **.90**) with
   *no* jacobian and despite a *larger* color displacement (.93 vs .70) — another case of distance not
   predicting zero-shot (cf. the latent-shift probe, "direction predicts, not distance"). And
   **acmtm+jac is the first encoder to solve the 90° camera rotation on sg** (.89, vs jacobian .64,
   jac+pull .69, best prior .77) — the rotation axis that every other lever left near chance.
5. Net: AC-MTM is **not a substitute** for the jacobian penalty (it needs it to be robust on dubins), but
   **AC-MTM + jacobian is the strongest sg encoder measured** (purple .89, rotate .89) while matching
   jacobian on dubins shape/rotate. Its one weak cell is dubins color (.556, high seed variance ±.128).

### Open
- WM seeds 1/2 (evals running: `run_logs/acmtm/eval_driver_s12.log`) — needed before trusting the sg
  rotation gain, which is a single WM.
- Why does AC-MTM break dubins rotation but not sg rotation? (dubins `np.rot90` rotates the whole world
  incl. agent heading; sg rotates the camera only.)
- Downstream (deferred T1c/T2): does acmtm+jac's sg robustness carry to closed-loop CEM / the CBF filter?
