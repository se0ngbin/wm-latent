# Two-way regularization — experiment summary

**Question:** split world-model regularization by role — *invariant* encoder
(robust to nuisance shift) vs *separating* predictor (different actions → different
next latents) — and see if it makes LeWM more robust / controllable for downstream
planning & safety.

**Protocol (Dubins):** sg25 clean-annulus goals (sub-goal 25 steps ahead, kept only
if in a 0.05–0.5u annulus around an obstacle AND from a clean-success trajectory),
25-env-step plans, budget 50, 200 episodes, eval seed 42. fs=1: horizon=25/ab=1/rec=25.
fs=5: horizon=5/ab=5/rec=5. Train 50 epochs (Dubins) / 10 (pushT). Metrics: success
(reach goal within 0.2u by step ≤50), collision (ever enter an obstacle disk).

**One-sentence verdict:** regs are targeted fixes for diagnosable pathologies, not
default tonics — jacobian fixes fake-variance in sparse scenes (Dubins ✅, pushT ⚪
no disease), pull fixes frameskip action-collapse (fs=5 ✅, fs=1 ⚪ nothing to fix,
pushT ❌ wrong assumption).

---

## A. Encoder regularizers — Dubins fs=1
Baseline (sigreg only): **62.5% success / 50.0% collision**

| Reg (+ sigreg)                             | Weight | Success | Collision | Verdict |
| ------------------------------------------ | ------ | ------- | --------- | ------- |
| **JacobianNormReg**                        | 1.0    | 81.5    | 22.5      | ✅ Big win: +19 succ, collision halved. n=3 reproducible (83.0±1.8, coll 22.5 every seed) |
| JacobianNormReg                            | 0.1    | 70.0    | 35.5      | 🟡 Helps; only reg that improves with weight |
| InvarianceReg (Gaussian noise σ0.1)        | 0.1    | 70.0    | 44.0      | 🟡 Mild help |
| AugInv: color jitter only                  | 0.1    | 69.5    | 40.5      | 🟡 Mild help (jitter is safe) |
| AugInv: bright/contrast + blur + noise     | 0.1    | 62.0    | 44.0      | ⚪ Neutral |
| AugInv: noise + blur + bright/contrast     | 1.0    | 57.0    | 54.0      | ❌ Hurt (high weight) |
| AugInv: noise only                         | 1.0    | 56.0    | 50.0      | ❌ Hurt (high weight) |
| AugInv: color jitter only                  | 1.0    | 44.5    | 55.0      | ❌ Hurt (high weight) |
| AugInv: noise + blur                       | 1.0    | 41.5    | 60.0      | ❌ Hurt (high weight) |
| AugInv: full color augs (gray/randconv/cut)| 0.1    | 15.5    | 62.5      | ❌ Catastrophic: color aliasing kills goal marker |

**Takeaway:** only two-sided jacobian (‖J‖→1, invariant *and* separable) wins, and
it's the only reg that improves with weight. One-sided regs (min sensitivity) rely on
SIGReg to prevent collapse, which can't see local aliasing → they over-smooth at high
weight. Color is semantic in this env → destroying it collapses planning.

## B. Predictor reg (ActionSeparationReg) — Dubins fs=1
Baseline: 62.5 / 50.0. All forms w=0.1.

| Form            | Ratio at eq.    | pred_loss | Success | Collision | Verdict |
| --------------- | --------------- | --------- | ------- | --------- | ------- |
| floor L=1       | 1.68 (inactive) | 0.0051    | 63.0    | 49.5      | ⚪ Neutral |
| band [1.0, 1.5] | 1.14            | 0.0037    | 61.5    | 48.5      | ⚪ Neutral |
| pull to 1       | 1.01            | 0.0037    | 61.5    | 54.0      | ⚪ Neutral |

All neutral — **expected**: fs=1 doesn't collapse, so the constraint never binds
(nothing to fix). Not a failure. (Band's ceiling improved pred_loss at no cost — a
free stabilizer.)

## C. Predictor reg — Dubins fs=5
Collapsed baseline: **30.0 / 61.5**. All forms w=0.1.

| Form            | Ratio at eq.  | Success | Collision | Verdict |
| --------------- | ------------- | ------- | --------- | ------- |
| **pull to 1**   | 0.998         | 66.0    | 52.5      | ✅ Best (but high-variance across seeds: 46–66, mean 57.5) |
| floor L=1       | 1.24          | 60.5    | 56.0      | ✅ Helps |
| band [1.0, 1.5] | 1.09 (active) | 54.5    | 60.5      | 🟡 Helps least |

Here the constraint binds (frameskip induces action-collapse; lower hinge active at
eq.). All three beat collapse; pull best. **"Beats collapse" robust across seeds;
"doubles baseline" was a lucky seed.**

## D. Combined (two-way) — Dubins
| What                              | Success | Collision | Verdict |
| --------------------------------- | ------- | --------- | ------- |
| floor + full-color AugInv (fs=1)  | 39.5    | 58.5      | ❌ Inherited AugInv damage |
| jacobianW1.0 + pull               | —       | —         | ⬜ Never ran (the "best model" the ablations point to) |

## E. PushT transfer (own protocol, 10-epoch ckpts)
Baseline: **91.5 @ offset25 / 68.0 @ offset40**

| Model                        | off25 | off40 | Verdict |
| ---------------------------- | ----- | ----- | ------- |
| noise invarianceW0.1         | 90.0  | 70.0  | ⚪ Neutral (epoch-matched, cleanest datapoint) |
| jacobianW0.1 (legacy 26-ep⚠) | 91.0  | 64.5  | ⚪ ~Neutral; off40 dip is a schedule confound, not comparable |
| jacobianW1.0                 | 86.0  | 63.0  | ⚪ Neutral-to-mild-neg |
| jitterW0.1                   | 88.0  | 65.0  | ⚪ Neutral |
| jacobianW1.0 + pullW0.1      | 81.0  | 61.0  | ❌ Hurt |
| jitterW0.1 + pullW0.1        | 80.0  | 55.5  | ❌ Hurt |
| pullW0.1                     | 77.0  | 57.5  | ❌ Most harmful |

**Every reg neutral-to-harmful on pushT.** Encoder regs neutral (rich scene → no
fake-variance disease); pull hurts (uniform-authority assumption false in contact-rich
dynamics — authority is bimodal in-contact vs free). Ordering horizon-invariant.

## F. Robustness / methodology
- Seeds (n=3, training seeds 3072/100/200) on the two headline claims:
  - jacobianW1.0 Dubins fs=1: 81.5 / 85.0 / 82.5 succ (mean **83.0±1.8**), collision
    **22.5 all three** → rock solid.
  - fs=5 pull: 66.0 / 60.5 / 46.0 succ (mean **57.5**, 20-pt spread) → real but noisy;
    likely a genuine collapse bifurcation.
- Built the sg25-clean-annulus goal filter so goals actually test avoidance (turned a
  cosmetic 94% eval into the differentiating 62.5% protocol).
- PushT horizon calibration (offset 25→75, baseline 91.5→68→36→16) to find a protocol
  with headroom.

## Known gaps (disclose)
1. Everything except the two seeded claims is **n=1** (seed 3072).
2. **Single horizon** on Dubins (all sg25) — untested whether conclusions widen at
   longer horizons where rollout error compounds.
3. **jacobianW1.0 + pull never combined** (the "best model" the ablations point to).
4. **Safety / latent-CBF — the original goal — never connected.** We measured a
   planning proxy (success/collision), not CBF margins.

## Suggested next steps
1. fs=5 jacobianW1.0 + pull, stacked (encoder fix + collapse fix in the one regime
   with both pathologies).
2. Connect jacobian to the actual latent-CBF safety metric (does the smoother latent
   improve learned/HJ margins, not just the planner proxy?).
3. 2–3 more seeds on fs=5 pull to pin the effect size.
