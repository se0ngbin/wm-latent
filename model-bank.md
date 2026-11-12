# Model bank: Reacher and Dubins

Status 2026-11-12 18:30 UTC (10:30 AM PST). Paper model set: baseline, jac, jac+advh, jac+advz, jac+advhe, jac+advze (advz = whole-latent adversary, advhe / advze = varying radius $\varepsilon \sim$ log-uniform $[4/255, 32/255]$). Checkpoints under `/data/seongbin/lewm/checkpoints/`; world-model (WM) seed 3072 = the dir name, other seeds = `<dir>_s<k>`.

## Reacher

| Model | Checkpoint | WM seeds | Margin-function seeds / tests |
|---|---|---|---|
| baseline | `rc_baseline` | 1 | Zero-shot margin function: 3 margin seeds (reacher-results Table 38). CEM: early 50-pair floor / light sweep, 1 margin seed; not in the 150-pair tables (its margin function is all-safe under every shift, so CCEM = raw CEM). Shift detectors (Tables 33 / 34). |
| jac | `rc_jacobian` | 2 (3072, s1); s2 training | CEM: 3 margin seeds × 150 pairs (Table 36). Zero-shot: 3 margin seeds (Table 38). WM seed 1 in Table 39. |
| jac+advh | `rc_jac_safeadv_linf` | 1 | CEM: 3 margin seeds × 150 pairs + pooled McNemar vs jac (Tables 36 / 37). Zero-shot: 3 margin seeds (Table 38). |
| jac+advz | `rc_jac_safeadv_latent` | 1 | CEM: 3 margin seeds + McNemar (Tables 36 / 37). Zero-shot: 3 margin seeds (Table 38). |
| jac+advhe | `rc_jac_safeadv_linf_epsmix` | 2 (3072, s2); s1 training (epoch 5) | CEM: 3 margin seeds on seed 3072 (Tables 36 / 37). Zero-shot: 3 margin seeds (Table 38). WM seed 2 in Table 39: floor 0.3 and camera shift (MLP) replicate, floor 0.4 does not. |
| jac+advze | `rc_jac_safeadv_latent_epsmix` | 1 | 1 margin seed only. CEM 150 pairs + McNemar vs jac (Table 35); reach (`adv_align.py`); per-step veto diagnostic under camera tilt (`veto_tilt_perstep.py`). Not in the margin-seed tables (36-38). |

## Dubins

| Model | Checkpoint | WM seeds | Margin-function seeds / tests |
|---|---|---|---|
| baseline | `sigreg_only_dubins` | 3 (3072, s1, s2) | Seeded over WMs. |
| jac | `jacobian_w1_dubins` | 3 used (3072, s1, `lewm_dubins_jacw1_s100`); spare `lewm_dubins_jacw1_s200` | Seeded over WMs. |
| jac+advh | `lewm_dubins_jac_safeadv_linf50` | 1 in results; s1 finished 2026-11-12 17:29 UTC (not evaluated); s2 training | No margin-seed CEM. Zero-shot margin function: 3 margin seeds (post-oct-21 3×3 grid). CEM, 1 margin seed: goal-distance sweep + per-episode McNemar vs jac, 50 episodes (post-oct-21 Tables 22 / 23). Colour sweep n = 1. |
| jac+advz | — | 0 | Not trained. |
| jac+advhe | — | 0 | Not trained. |
| jac+advze | — | 0 | Not trained. |

- `lewm_dubins_linf_latent50` (3 seeds) is whole-latent adversary on the sigreg base, not jac+advz. `lewm_dubins_safeadv_eps4` / `eps16` are fixed-radius attack sweeps on sigreg, not varying radius.
- The 3-seed adversarial Dubins model is jac+pull+advh (`lewm_dubins_jacpull_safeadv_linf50`, s1, s2).

## Gaps (margin-function seeds for CEM)

- Reacher jac+advze: CEM and zero-shot are 1 margin seed.
- Reacher baseline: no 150-pair CEM (low priority, CCEM = raw CEM under shift).
- Dubins jac+advh: 1 margin seed in CEM; alternatively evaluate WM seeds s1 (done) and s2 (training).
