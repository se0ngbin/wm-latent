# Reacher results

DMC Reacher with SLS-squared's invisible joint-box constraint: a state is unsafe iff $\mathrm{wrap}(q_1) \in [0, \pi]$ and $q_2 \in [-2.88, -2.45]$. Data: LeWM's `reacher_random` (10k episodes x 201 frames, 3.25% unsafe frames), rewritten uncompressed with a `failures` key to `/data/seongbin/lewm/datasets/reacher.h5`. Episodes < 9000 train the margin functions; episodes >= 9000 are held out for every evaluation.

## Models

| Name | Recipe | Epochs |
|---|---|---|
| sigreg | `rc_baseline`, LeWM loss (SIGReg), seed 3072 | 10 |
| jac | `rc_jacobian`, LeWM loss + encoder-Jacobian reg (weight 1, target $L = 1$), seed 3072 | 10 |
| official | LeWM's released Reacher checkpoint (`lewm_reacher`), converted | 50 |

All world models use frameskip 5, history 3, one predicted step, as LeWM's Reacher config. Single world-model seed; every number below is one run (no std). jac+pull and the three jac+adv_h variants are paused mid-training and not evaluated.

## Margin functions

Two margin functions per world model, both trained with the margin_gp loss on balanced safe/unsafe batches: an MLP (192-512-512-1) and a linear $h(z) = w^\top z + b$. Real-only: 25k unsafe + 25k safe encoded frames, 20k steps (`reacher_margin.py`). Real+imag: the same real frames plus predicted latents of 8000 CEM-style 5-block plans ($a = b + 0.7\epsilon$ in z-scored action space, half the starts within 25 steps of the box), labelled by replaying the plan in the simulator, 3k steps (`reacher_margin_imag.py`). The real+imag margin functions are the ones used for OOD and CEM.

Table 1. Held-out margin function accuracy. Real = 5000 frames of held-out episodes at the natural ratio (3.5% unsafe); imagined = 5000 predicted latents of 1000 held-out plans (20.7% unsafe). Bal acc = (TPR_safe + TPR_unsafe) / 2. Real-only in parentheses.

| WM | Margin fn | Real AUC | Real bal acc | Imag AUC | Imag bal acc | Imag false-safe |
|---|---|---|---|---|---|---|
| sigreg | MLP | 0.999 (0.999) | 0.997 (0.998) | 0.981 (0.981) | 0.956 (0.946) | 0.043 (0.071) |
| sigreg | linear | 0.998 (0.998) | 0.979 (0.977) | 0.979 (0.981) | 0.945 (0.955) | 0.028 (0.034) |
| jac | MLP | 0.999 (0.999) | 0.996 (0.999) | 0.981 (0.982) | 0.959 (0.945) | 0.034 (0.074) |
| jac | linear | 0.997 (0.998) | 0.979 (0.979) | 0.980 (0.981) | 0.948 (0.952) | 0.029 (0.034) |
| official | MLP | 0.999 (0.999) | 0.997 (0.998) | 0.981 (0.981) | 0.954 (0.945) | 0.047 (0.074) |
| official | linear | 0.998 (0.998) | 0.980 (0.979) | 0.979 (0.981) | 0.948 (0.951) | 0.037 (0.041) |

- The box is read near-perfectly from every world model. Adding predicted latents roughly halves the MLP's false-safe rate on imagined states and changes nothing else.

## Zero-shot appearance OOD (margin functions)

5000 held-out states rendered under each view (`reacher_ood.py`); the box is invisible, so labels do not change. Views: arm recoloured purple (the arm is thin, so the mean pixel change is only 1.0, but the recolour renders correctly), floor checker blue to brown (mean pixel change 71.7), light intensity 0.7 to 0.3 (16.4).

Table 2. AUC / bal acc of the real+imag margin functions. Bal acc 0.500 = every state called safe.

| WM | Margin fn | Base | Arm colour | Floor colour | Light |
|---|---|---|---|---|---|
| sigreg | MLP | 1.000 / 0.997 | 0.977 / 0.500 | 0.496 / 0.500 | 0.488 / 0.500 |
| sigreg | linear | 0.999 / 0.991 | 0.956 / 0.690 | 0.579 / 0.500 | 0.354 / 0.500 |
| jac | MLP | 1.000 / 0.997 | 0.991 / 0.761 | 0.566 / 0.500 | 0.999 / 0.997 |
| jac | linear | 0.999 / 0.989 | 0.997 / 0.984 | 0.699 / 0.500 | 0.999 / 0.989 |
| official | MLP | 1.000 / 0.997 | 0.971 / 0.500 | 0.441 / 0.500 | 0.637 / 0.500 |
| official | linear | 0.999 / 0.993 | 0.980 / 0.733 | 0.439 / 0.500 | 0.484 / 0.500 |

Table 3. Floor-colour sweep: floor $= (1 - \alpha)\,$default $+ \alpha\,$brown. Bal acc (AUC) of the real+imag margin functions; mean pixel change per $\alpha$ in the header. $\alpha = 0.5$-$1$ is all-safe (bal acc 0.500) for every row.

| WM | Margin fn | $\alpha$ = 0 | 0.1 (7.1) | 0.2 (14.5) | 0.3 (21.4) | 0.4 (28.7) |
|---|---|---|---|---|---|---|
| sigreg | MLP | 0.997 (1.000) | 0.500 (0.990) | 0.500 (0.493) | 0.500 (0.499) | 0.500 (0.406) |
| sigreg | linear | 0.991 (0.999) | 0.500 (0.986) | 0.500 (0.335) | 0.500 (0.283) | 0.500 (0.290) |
| jac | MLP | 0.997 (1.000) | 0.997 (1.000) | 0.997 (0.999) | 0.500 (0.873) | 0.500 (0.667) |
| jac | linear | 0.989 (0.999) | 0.992 (0.999) | 0.990 (0.999) | 0.578 (0.952) | 0.500 (0.402) |
| official | MLP | 0.997 (1.000) | 0.689 (0.998) | 0.500 (0.865) | 0.500 (0.604) | 0.500 (0.597) |
| official | linear | 0.993 (0.999) | 0.971 (0.999) | 0.500 (0.815) | 0.500 (0.383) | 0.500 (0.327) |

- jac is exact up to $\alpha = 0.2$; sigreg already calls everything safe at $\alpha = 0.1$, the official model partly. Each model first loses the threshold (AUC still high, all safe), then the ranking. Real-only margin functions break at the same $\alpha$ (logs in `reacher_eval_logs/floor_sweep/`).

## CEM

Protocol as LeWM's Reacher eval (`config/eval/reacher.yaml`): action block 5, horizon 5 blocks, replan every 5 blocks, 300 samples, 30 elites, budget 50 steps, goal = the dataset frame 25 steps after the start, z-scored actions, 1-frame history, solver and pair seed 42; 10 CEM iterations (LeWM paper, non-PushT). Pairs: 50 from held-out episodes whose endpoints are safe but whose recorded path enters the box (3554 candidates), so the recorded route is unsafe. Success = both joints within 0.05 of the goal; collision = any step in the box before success. CCEM = LeWM's CEM with the constrained elite rule `minh` (as Dubins): a plan is feasible iff $h > 0$ on every state (current one included); if at least 30 are feasible the elites are the 30 feasible plans with the best goal cost, otherwise all feasible plans plus the infeasible ones with the highest $\min_k h(\hat z_k)$. With $n = 50$ each rate has a standard error of about 0.07.

Table 4. In distribution: success / collision / safe success.

| WM | Raw CEM | CCEM, MLP | CCEM, linear | Safety-first, MLP (legacy) |
|---|---|---|---|---|
| sigreg | 0.640 / 0.540 / 0.340 | 0.600 / 0.040 / 0.600 | 0.480 / 0.040 / 0.480 | 0.620 / 0.040 / 0.580 |
| jac | 0.760 / 0.580 / 0.360 | 0.600 / 0.060 / 0.580 | 0.460 / 0.060 / 0.440 | 0.640 / 0.060 / 0.620 |
| official | 0.600 / 0.580 / 0.300 | 0.580 / 0.020 / 0.580 | 0.520 / 0.020 / 0.520 | 0.600 / 0.020 / 0.600 |

Table 5. CEM under appearance shift (start and goal frames rendered under it): collision (success).

| WM | Shift | Raw CEM | CCEM, MLP | CCEM, linear |
|---|---|---|---|---|
| sigreg | floor 0.1 | 0.500 (0.380) | 0.480 (0.380) | 0.480 (0.380) |
| sigreg | floor 0.2 | 0.460 (0.280) | 0.460 (0.280) | 0.460 (0.280) |
| sigreg | floor 0.3 | 0.380 (0.220) | 0.380 (0.220) | 0.380 (0.220) |
| sigreg | light 0.3 | 0.420 (0.160) | 0.420 (0.160) | 0.420 (0.160) |
| sigreg | arm purple | 0.460 (0.360) | 0.320 (0.400) | 0.280 (0.420) |
| jac | floor 0.1 | 0.640 (0.600) | 0.060 (0.660) | 0.060 (0.540) |
| jac | floor 0.2 | 0.480 (0.720) | 0.080 (0.720) | 0.080 (0.500) |
| jac | floor 0.3 | 0.340 (0.640) | 0.140 (0.500) | 0.100 (0.480) |
| jac | light 0.3 | 0.560 (0.580) | 0.060 (0.580) | 0.060 (0.520) |
| jac | arm purple | 0.620 (0.360) | 0.080 (0.300) | 0.060 (0.440) |
| official | floor 0.1 | 0.480 (0.840) | 0.060 (0.620) | 0.040 (0.600) |
| official | floor 0.2 | 0.300 (0.380) | 0.280 (0.400) | 0.280 (0.400) |
| official | floor 0.3 | 0.360 (0.300) | 0.360 (0.300) | 0.360 (0.300) |
| official | light 0.3 | 0.380 (0.320) | 0.380 (0.320) | 0.380 (0.320) |
| official | arm purple | 0.460 (0.360) | 0.420 (0.340) | 0.400 (0.260) |

- When the margin function calls every state safe, every plan is feasible and CCEM is exactly raw CEM (identical rows). sigreg reaches that at floor 0.1 and light 0.3, the official model at floor 0.2 and light 0.3; jac never does in this range.
- Arm purple (the smallest pixel change): CCEM partly works for sigreg (collision 0.460 to 0.280-0.320) and fully for jac (0.620 to 0.060-0.080); the official model barely improves (0.460 to 0.400-0.420). Raw success drops to 0.360 for all three, so the recolour also hurts goal reaching, and jac's CCEM success is low here (0.300-0.440).
- The shift also breaks goal reaching for sigreg and the official model (raw success 0.640 to 0.160-0.380); jac's raw success stays at 0.580-0.720.
- Unlike Dubins, where no model beat jac in CCEM but jac's margin function OOD advantage did not separate planning, here jac separates clearly: at floor 0.1, CCEM collision 0.060 vs sigreg 0.480.

## Why jac survives

`reacher_floor_diag.py`: 1000 held-out states (500 unsafe), each with the frame 5 env steps later and one random 5-block plan replayed in the simulator for labels; real+imag MLP margin function.

- NSR (nuisance-to-signal ratio) $= \mathbb{E}\lVert z_\alpha(s) - z_0(s)\rVert / \mathbb{E}\lVert z_0(s') - z_0(s)\rVert$, with $s'$ 5 env steps after $s$: the latent shift from the floor change, in units of real motion.
- $\Delta h$/gap $= \mathbb{E}[h(z_\alpha) - h(z_0)] / (\bar h_{\mathrm{safe}} - \bar h_{\mathrm{unsafe}})$: the threshold shift in units of the class gap (positive = toward safe).
- rel$_k = \mathbb{E}\lVert \hat z^\alpha_k - \hat z^0_k\rVert / \mathbb{E}\lVert z_\alpha - z_0\rVert$: how much of the OOD offset is left after $k$ predicted blocks of the same plan (below 1 = the predictor pulls the rollout back toward the base-appearance one).

Table 6. Floor-colour diagnostic. Imag bal acc = the margin function on $\hat z^\alpha_k$ at $k = 1$ and $k = 5$ (base: about 0.950 for every model).

| WM | $\alpha$ | NSR | $\Delta h$/gap | Real AUC | Real bal acc | rel$_1$ | rel$_5$ | Imag bal acc $k=1$ | $k=5$ |
|---|---|---|---|---|---|---|---|---|---|
| sigreg | 0.1 | 1.616 | +0.410 | 0.994 | 0.500 | 0.91 | 1.00 | 0.500 | 0.518 |
| sigreg | 0.2 | 2.723 | +0.518 | 0.467 | 0.500 | 0.71 | 0.69 | 0.500 | 0.541 |
| sigreg | 0.3 | 2.605 | +0.487 | 0.479 | 0.500 | 0.75 | 0.76 | 0.500 | 0.539 |
| jac | 0.1 | 0.058 | +0.001 | 1.000 | 0.998 | 0.56 | 0.47 | 0.950 | 0.947 |
| jac | 0.2 | 0.186 | +0.011 | 1.000 | 0.997 | 0.59 | 0.49 | 0.943 | 0.946 |
| jac | 0.3 | 1.268 | +0.293 | 0.830 | 0.500 | 0.92 | 0.92 | 0.557 | 0.708 |
| jac | 0.5 | 1.896 | +0.476 | 0.555 | 0.500 | 0.95 | 0.98 | 0.500 | 0.575 |
| official | 0.1 | 0.375 | +0.064 | 1.000 | 0.687 | 0.69 | 0.45 | 0.905 | 0.935 |
| official | 0.2 | 1.687 | +0.510 | 0.855 | 0.500 | 0.95 | 0.90 | 0.500 | 0.554 |
| official | 0.3 | 2.053 | +0.537 | 0.614 | 0.500 | 0.91 | 0.83 | 0.500 | 0.551 |

- Nuisance rejection in the encoder. At $\alpha = 0.1$ sigreg's floor shift is 1.6x the latent change of 5 steps of real motion; jac's is 0.06x (28x smaller), the official model's 0.38x. At $\alpha = 0.2$: jac 0.19, official 1.69, sigreg 2.72. jac's robustness is a region, not a slope: NSR jumps from 0.19 to 1.27 between $\alpha = 0.2$ and $0.3$, where its static margin function breaks.
- The shift moves $h$ toward safe for every model ($\Delta h$/gap 0.3-0.5 once it breaks), which is why every failure is all-safe (false-safe) rather than all-unsafe.
- The predictor re-grounds small offsets. For jac at $\alpha \le 0.2$ and the official model at 0.1, half of the remaining OOD offset is gone after 5 predicted blocks (rel$_5$ 0.45-0.49); sigreg's offset does not shrink at $\alpha = 0.1$ (rel$_5$ = 1.00).
- jac at $\alpha = 0.3$: on the real frame the margin function calls everything safe, but on predicted latents it recovers with depth (imag bal acc 0.557 at $k = 1$ to 0.708 at $k = 5$; sigreg and official stay at about 0.5). CCEM requires $h > 0$ on every predicted state, so those deeper states still veto unsafe plans: collision 0.340 raw to 0.100-0.140 with CCEM. Hypothesis, not tested: jac's predictor output lies closer to the training manifold than its OOD encoder output.

Why does jac's CCEM work at floor 0.3, where its static sign acc is 0.500? Sign acc scores single real frames at threshold 0; CCEM vetoes a plan if $h < 0$ on any of its predicted states. Three measurements (same diagnostic, real+imag MLP margin function):

- On real frames nothing is vetoed: at $\alpha = 0.3$ even the deepest quarter of unsafe states reads safe for every model.
- On predicted states the threshold shift shrinks with depth for every model ($\Delta h$/gap at $k = 1 \to 5$: sigreg 0.440 to 0.205, official 0.459 to 0.188, jac 0.263 to 0.126), and jac starts with the smallest shift and the best ranking (imag AUC 0.726 at $k = 1$ vs sigreg 0.352, official 0.616). So jac's deeper predicted states of box-entering plans go below 0; the others' mostly do not.
- The CCEM logs agree: per replan, the margin function marks 21-26% of jac's candidates infeasible at floor 0.3 (30-34% in distribution, 39-44% under arm purple), but 0-0.2% for sigreg and the official model.

Table 6b. Plan-level veto, as CCEM uses it: fraction of plans with $\min_k h(\hat z_k) < 0$ over the 5 predicted blocks, among plans that do / do not enter the box. Static bal acc = Table 6.

| WM | $\alpha$ | Static bal acc | Unsafe plans vetoed | Safe plans vetoed |
|---|---|---|---|---|
| sigreg | 0 | 0.997 | 0.961 | 0.070 |
| sigreg | 0.1 | 0.500 | 0.028 | 0.015 |
| sigreg | 0.3 | 0.500 | 0.069 | 0.023 |
| jac | 0 | 0.998 | 0.961 | 0.070 |
| jac | 0.1 | 0.998 | 0.964 | 0.077 |
| jac | 0.3 | 0.500 | 0.357 | 0.031 |
| official | 0 | 0.998 | 0.951 | 0.070 |
| official | 0.1 | 0.687 | 0.900 | 0.039 |
| official | 0.3 | 0.500 | 0.077 | 0.021 |

- Plan-level veto predicts CCEM; static sign acc does not. Official at $\alpha = 0.1$ has sign acc 0.687 but vetoes 90% of unsafe plans (CCEM collision 0.040-0.060); jac at $\alpha = 0.3$ has sign acc 0.500 but vetoes 36% of unsafe plans at a 3% false-veto rate, against 7% for sigreg and 8% for official. CEM re-samples around the feasible elites for 10 iterations and replans, so a precise partial veto still steers most episodes off the box (collision 0.340 raw to 0.100-0.140).

## Prediction depth (in distribution)

On Dubins, $h$ on predicted latents became more optimistic further into the rollout. `reacher_depth_diag.py` checks this on Reacher: 1000 held-out starts (half enter the box within 75 env steps), one CEM-style plan each, rolled out for 15 predicted blocks (75 env steps; the CEM horizon is 5 blocks), labels from replaying the plan in the simulator (13-15% unsafe per step). Base appearance.

Table 7. False-safe, AUC and mean $h$ on unsafe states, averaged over predicted blocks 1-5 | 6-10 | 11-15. The real+imag margin functions saw predicted latents of blocks 1-5 only.

| WM | Margin fn | Trained on | False-safe | AUC | Mean $h$ on unsafe |
|---|---|---|---|---|---|
| sigreg | MLP | real-only | 0.096 / 0.154 / 0.148 | 0.986 / 0.964 / 0.943 | -0.182 / -0.017 / 0.035 |
| sigreg | MLP | real+imag | 0.066 / 0.129 / 0.131 | 0.984 / 0.961 / 0.940 | -0.247 / -0.140 / -0.073 |
| sigreg | linear | real-only | 0.050 / 0.108 / 0.120 | 0.984 / 0.963 / 0.946 | -0.326 / -0.309 / -0.310 |
| sigreg | linear | real+imag | 0.025 / 0.087 / 0.104 | 0.984 / 0.955 / 0.942 | -0.523 / -0.495 / -0.510 |
| jac | MLP | real-only | 0.100 / 0.160 / 0.151 | 0.986 / 0.963 / 0.945 | -0.156 / 0.006 / 0.065 |
| jac | MLP | real+imag | 0.059 / 0.120 / 0.123 | 0.984 / 0.962 / 0.944 | -0.271 / -0.164 / -0.091 |
| jac | linear | real-only | 0.037 / 0.102 / 0.109 | 0.984 / 0.959 / 0.941 | -0.426 / -0.397 / -0.396 |
| jac | linear | real+imag | 0.024 / 0.088 / 0.101 | 0.984 / 0.955 / 0.938 | -0.599 / -0.564 / -0.576 |
| official | MLP | real-only | 0.097 / 0.175 / 0.175 | 0.985 / 0.958 / 0.931 | -0.137 / 0.054 / 0.110 |
| official | MLP | real+imag | 0.073 / 0.132 / 0.139 | 0.983 / 0.960 / 0.938 | -0.181 / -0.068 / 0.006 |
| official | linear | real-only | 0.043 / 0.133 / 0.128 | 0.984 / 0.957 / 0.937 | -0.318 / -0.279 / -0.278 |
| official | linear | real+imag | 0.036 / 0.124 / 0.127 | 0.984 / 0.952 / 0.935 | -0.433 / -0.390 / -0.401 |

- The Dubins effect is present but much weaker. False-safe roughly doubles from blocks 1-5 to 6-15 (for example jac real+imag MLP 0.059 to 0.12), but stays at 0.10-0.18; it then plateaus.
- Two mechanisms. The MLP margin function drifts optimistic: mean $h$ on unsafe predicted states rises toward 0 with depth and crosses it for the real-only MLP (sigreg -0.182 to 0.035, official -0.137 to 0.110), as on Dubins. The linear margin function does not drift (mean $h$ on unsafe flat); its false-safe rise comes from the ranking blurring (AUC 0.984 to 0.94).
- Training on predicted latents of blocks 1-5 lowers false-safe at every depth, also beyond the depths it saw, and slows the MLP's drift, without removing it.
- The three world models behave the same here; jac's advantage is OOD only.
- Within the CEM horizon (blocks 1-5) false-safe is 0.024-0.073 for the real+imag margin functions, so depth is not what limits the in-distribution CEM results.

## Files

Code on branch `worktree-reacher-imag-margin`: `le-wm/scripts/reacher_margin.py`, `reacher_margin_imag.py`, `reacher_ood.py` (`FLOOR_SWEEP`), `reacher_cem.py` (`CCEM=minh`, `FLOOR`, `LIGHT`, `ARM`), `reacher_floor_diag.py`, `reacher_depth_diag.py`; `le-wm/safe_solver.py` (CCEM, from `imag-margin`). Margin functions: `/data/seongbin/lewm/reacher_heads/<wm>/` (real-only), `/data/seongbin/lewm/reacher_heads_imag/<wm>/` (real+imag). CEM trajectories: `/data/seongbin/lewm/reacher_cem/`. Logs: `/data/seongbin/lewm/reacher_eval_logs/`.
