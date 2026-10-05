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

### WM seed 1 (2026-10-02) — two headlines soften
| | dubins color zs | dubins shape zs | dubins rotate zs | sg purple | sg rotate |
|---|---|---|---|---|---|
| acmtm s3072 | .456 | .650 | .478 | .904 | .503 |
| acmtm s1 | .533±.098 | .836±.016 | **.766±.011** | .886 | .578 |
| acmtm+jac s3072 | .556 | .965 | .959 | .886 | .893 |
| acmtm+jac s1 | .492±.076 | .969±.001 | .951±.006 | .917 | **.736** |

- AC-MTM's dubins rotation break (.48) is **seed-dependent**: seed 1 gets .77 (still below SIGReg's .97).
- acmtm+jac sg rotation drops .89 → .74 — still above jacobian (.64) / jac+pull (.69) at the plain readout,
  but no longer a step change; ≈ the prior best (D_R proj, .77). sg purple (~.89–.92) and dubins
  shape/rotate for acmtm+jac are stable across WM seeds.
- Mechanism s1: acmtm ‖J‖ 26 (dubins_mechanism; SIGReg 17.1 on the same probe); acmtm+jac 1.53 (dubins), 0.87 (sg).

## Final — 3 WM seeds (3072, 1, 2), 2026-10-05

Each cell = mean ± sd over the 3 WMs of the per-WM AUC (per WM: dubins = mean over probe seeds 0–2,
sg = mean over probe seeds 0–1). References are single-WM (seed 3072), as in the earlier tables.

| encoder | dubins color zs | dubins shape zs | dubins rotate zs | sg in-dist | sg purple zs | sg rotate zs |
|---|---|---|---|---|---|---|
| baseline (sigreg) | .136 | .885 | .973 | .994 | .604 | .564 |
| jacobian | .757 | .993 | .991 | .996 | **.991** | .641 |
| jac+pull | .707 | .977 | .980 | .999 | **.995** | .692 |
| **acmtm** | .498 ± .032 | .760 ± .080 | .638 ± .120 | .996 | .895 ± .007 | .549 ± .033 |
| **acmtm+jac** | .562 ± .059 | .965 ± .003 | .929 ± .036 | .997 | .865 ± .054 | **.827 ± .067** |

Per WM seed (3072 / 1 / 2): acmtm dubins rotate .478 / .766 / .670; acmtm+jac sg rotate .893 / .736 / .852;
acmtm+jac sg purple .886 / .917 / .791.

Mechanism, 3-WM range: acmtm dubins ‖J‖_F 25.8–30.8 (dubins_mechanism; SIGReg 17.1), sg ‖J‖ 8.1–8.7
(SIGReg 7.6); acmtm+jac dubins 0.97–1.53, sg 0.84–0.89.

### Final verdicts (supersede the seed-3072 ones above where they differ)
1. **AC-MTM is not a substitute for the jacobian penalty** — holds on all 3 WMs. It is more
   input-sensitive than SIGReg on dubins every seed, worse than SIGReg on dubins shape (.76 vs .89) and
   rotation (.64 vs .97), and well below jacobian on dubins color (.50 vs .76). The seed-3072 rotation
   collapse to chance (.48) was the worst seed, not the norm; the gap to SIGReg is real on every seed.
2. **On sg, AC-MTM alone fixes most of color** (purple .60 → .90, tight across seeds) without any jacobian
   — but it does **not** beat jacobian / jac+pull (.99), which were already near ceiling.
3. **acmtm+jac's sg rotation gain is real but smaller than the first seed suggested:** .83 ± .07 vs
   jacobian .64 / jac+pull .69 (+.14 over the best plain-readout reference; above the prior best D_R
   projection .77). Caveat: the references are single-WM.
4. **Correction to the seed-3072 headline:** acmtm+jac is **not** the best sg encoder overall. It trades
   color for rotation — purple .87 ± .05 is *below* jacobian's .99 — so it is the best sg encoder on
   **camera rotation only**.
5. Static obstacles are kept (in-dist ≥ .94 dubins / ≥ .99 sg on every WM). The zero-shot sign-acc
   base-rate artifact stands (use AUC; re-check the worklog's jac+pull "sign" claim).

### Open
- Reference encoders are single-WM; seed them (≥3) before claiming the sg rotation delta.
- Why does AC-MTM break dubins rotation but not sg rotation? (dubins `np.rot90` rotates the whole world
  incl. agent heading; sg rotates the camera only.)
- Downstream (deferred T1c/T2): does acmtm+jac's sg robustness carry to closed-loop CEM / the CBF filter?
