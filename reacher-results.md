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
| sigreg | floor 0.4 | 0.420 (0.180) | 0.420 (0.180) | 0.420 (0.180) |
| sigreg | floor 0.5 | 0.400 (0.120) | 0.400 (0.120) | 0.400 (0.120) |
| sigreg | floor 0.75 | 0.320 (0.060) | 0.320 (0.060) | 0.320 (0.060) |
| sigreg | floor 1 | 0.180 (0.100) | 0.180 (0.100) | 0.180 (0.100) |
| sigreg | light 0.3 | 0.420 (0.160) | 0.420 (0.160) | 0.420 (0.160) |
| sigreg | arm purple | 0.460 (0.360) | 0.320 (0.400) | 0.280 (0.420) |
| jac | floor 0.1 | 0.640 (0.600) | 0.060 (0.660) | 0.060 (0.540) |
| jac | floor 0.2 | 0.480 (0.720) | 0.080 (0.720) | 0.080 (0.500) |
| jac | floor 0.3 | 0.340 (0.640) | 0.140 (0.500) | 0.100 (0.480) |
| jac | floor 0.4 | 0.240 (0.320) | 0.240 (0.340) | 0.220 (0.340) |
| jac | floor 0.5 | 0.380 (0.220) | 0.380 (0.220) | 0.380 (0.220) |
| jac | floor 0.75 | 0.480 (0.160) | 0.480 (0.160) | 0.480 (0.160) |
| jac | floor 1 | 0.440 (0.120) | 0.440 (0.120) | 0.440 (0.120) |
| jac | light 0.3 | 0.560 (0.580) | 0.060 (0.580) | 0.060 (0.520) |
| jac | arm purple | 0.620 (0.360) | 0.080 (0.300) | 0.060 (0.440) |
| official | floor 0.1 | 0.480 (0.840) | 0.060 (0.620) | 0.040 (0.600) |
| official | floor 0.2 | 0.300 (0.380) | 0.280 (0.400) | 0.280 (0.400) |
| official | floor 0.3 | 0.360 (0.300) | 0.360 (0.300) | 0.360 (0.300) |
| official | floor 0.4 | 0.400 (0.260) | 0.400 (0.260) | 0.400 (0.260) |
| official | floor 0.5 | 0.440 (0.280) | 0.440 (0.280) | 0.420 (0.280) |
| official | floor 0.75 | 0.260 (0.140) | 0.260 (0.140) | 0.260 (0.140) |
| official | floor 1 | 0.240 (0.080) | 0.240 (0.080) | 0.240 (0.080) |
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
| sigreg | 0.4 | 0.500 | 0.072 | 0.021 |
| sigreg | 0.5 | 0.500 | 0.023 | 0.010 |
| sigreg | 0.75 | 0.500 | 0.008 | 0.003 |
| sigreg | 1 | 0.500 | 0.000 | 0.000 |
| jac | 0 | 0.998 | 0.961 | 0.070 |
| jac | 0.1 | 0.998 | 0.964 | 0.077 |
| jac | 0.3 | 0.500 | 0.357 | 0.031 |
| jac | 0.4 | 0.500 | 0.224 | 0.039 |
| jac | 0.5 | 0.500 | 0.121 | 0.025 |
| jac | 0.75 | 0.500 | 0.000 | 0.000 |
| jac | 1 | 0.500 | 0.000 | 0.000 |
| official | 0 | 0.998 | 0.951 | 0.070 |
| official | 0.1 | 0.687 | 0.900 | 0.039 |
| official | 0.3 | 0.500 | 0.077 | 0.021 |
| official | 0.4 | 0.500 | 0.021 | 0.011 |
| official | 0.5 | 0.500 | 0.023 | 0.010 |
| official | 0.75 | 0.500 | 0.003 | 0.002 |
| official | 1 | 0.500 | 0.003 | 0.002 |

- Not luck: the veto decays smoothly with $\alpha$ (jac 0.961 / 0.357 / 0.224 / 0.121 / 0 at $\alpha$ = 0 / 0.3 / 0.4 / 0.5 / 0.75), and CCEM's benefit ends between 0.3 and 0.4: at 0.4 a 22% veto no longer lowers collision (0.220-0.240 vs raw 0.240), and from 0.5 on CCEM is identical to raw CEM for every model (Table 5). Goal reaching also collapses past 0.3 (jac raw success 0.640 at 0.3, 0.320 at 0.4, 0.120 at 1); jac's low raw collision at 0.4 reflects less movement, not avoidance. Beyond $\alpha \approx 0.5$ no model plans (success 0.060-0.280).
- Plan-level veto predicts CCEM; static sign acc does not. Official at $\alpha = 0.1$ has sign acc 0.687 but vetoes 90% of unsafe plans (CCEM collision 0.040-0.060); jac at $\alpha = 0.3$ has sign acc 0.500 but vetoes 36% of unsafe plans at a 3% false-veto rate, against 7% for sigreg and 8% for official. CEM re-samples around the feasible elites for 10 iterations and replans, so a precise partial veto still steers most episodes off the box (collision 0.340 raw to 0.100-0.140).

### Re-grounding: toward known states, but the right ones?

`reacher_nn_diag.py`: bank = encoder latents of 20k training frames (base appearance) with their joint angles; 1000 held-out states (500 unsafe) at floor $\alpha$, one random 5-block plan each. For the encoder latent (step 0) and each predicted latent (steps 1-5): kNN = median distance to the 10 nearest bank latents relative to the same at $\alpha = 0$; in-support = fraction within the $\alpha = 0$ q99 distance; NN error = median joint error (max over $q_1, q_2$, wrapped, rad) between the true state (simulated for steps $\ge 1$) and the nearest bank latent's state. In distribution the NN error is 0.02 at step 0 and 0.20 at step 5 (the bank's resolution plus prediction error). Low kNN with high NN error = aliasing (in support, wrong state).

Table 6c. kNN / in-support / NN error at step 0 (encoder) and step 5 (predicted).

| WM | $\alpha$ | Step 0 | Step 5 |
|---|---|---|---|
| sigreg | 0.1 | 5.74 / 0.00 / 2.56 | 1.10 / 0.88 / 2.16 |
| sigreg | 0.2 | 8.86 / 0.00 / 1.85 | 1.52 / 0.48 / 1.65 |
| sigreg | 0.3 | 8.67 / 0.00 / 1.74 | 2.51 / 0.35 / 1.55 |
| sigreg | 0.5 | 8.76 / 0.00 / 1.90 | 3.59 / 0.08 / 1.89 |
| jac | 0.1 | 1.04 / 0.99 / 0.02 | 1.00 / 0.99 / 0.20 |
| jac | 0.2 | 1.24 / 0.94 / 0.03 | 1.01 / 0.99 / 0.21 |
| jac | 0.3 | 3.96 / 0.00 / 0.29 | 1.05 / 0.93 / 0.41 |
| jac | 0.5 | 5.96 / 0.00 / 1.54 | 2.06 / 0.44 / 1.53 |
| official | 0.1 | 1.91 / 0.16 / 0.04 | 1.00 / 0.99 / 0.23 |
| official | 0.2 | 5.19 / 0.00 / 0.91 | 1.11 / 0.80 / 1.02 |
| official | 0.3 | 6.01 / 0.00 / 1.12 | 1.39 / 0.53 / 1.15 |
| official | 0.5 | 6.63 / 0.00 / 1.21 | 2.03 / 0.32 / 1.82 |

- The predictor pulls every model's OOD latents back toward the training latents (kNN falls toward 1 and in-support rises with each predicted step), as the windowed transformer predictor did on Dubins (`regrounding_gru.py`: transformer contracts, GRU diverges).
- It pulls toward the right state only if the encoder kept the state. jac at $\alpha = 0.3$: the encoder latent is off support but near the right state (0.29 rad); by step 5 it is 93% in support at 0.41 rad, which is why the plan veto works there. sigreg at $\alpha = 0.1$: off support and wrong (2.56 rad); by step 5 it is 88% in support and still wrong (2.16 rad), so re-grounding turns a detectable OOD latent into an in-support alias. Same for official at $\alpha \ge 0.2$ and jac at 0.5.
- Re-grounding is therefore not a goal in itself: it amplifies whatever state the encoder kept. The quantity to improve is the encoder's state retention under shift. An OOD check belongs on the step-0 encoder latent, where every failing case here is 100% off support (it would also flag jac at 0.3, where it still works).

## Sim-to-real shifts

`reacher_shift.py`: shifts a deployment would plausibly meet, severities fixed before any result. Camera: fixed camera moved 3 cm in x and y (cam shift), tilted 5° (cam tilt), moved 10 cm closer (cam zoom). Light: overhead light moved to (0.4, 0.4, 1) aimed at the origin (light dir). Sensor: Gaussian noise $\sigma = 10/255$, Gaussian blur $\sigma = 1.5$ px, JPEG quality 15, gamma 1.5 (exposure), white balance R ×1.2 / B ×0.8, a fixed 48 px grey square per episode (occlusion). Dynamics (planning only; frames unchanged): actuator gear ×0.7 (gear0.7), joint damping ×10 (damping10), 1-env-step action latency (delay1), Gaussian action noise $\sigma = 0.2$ (act noise). Start and goal frames are rendered under the shift. Epoch-10 world models, real+imag margin functions.

Table 10. Zero-shot margin function under sim-to-real appearance shifts: bal acc (AUC), 5000 held-out states.

| Shift (mean pixel change) | sigreg MLP | sigreg linear | jac MLP | jac linear | official MLP | official linear |
|---|---|---|---|---|---|---|
| cam shift (15.7) | 0.500 (0.838) | 0.500 (0.892) | 0.506 (0.935) | 0.860 (0.973) | 0.500 (0.777) | 0.500 (0.843) |
| cam tilt (19.4) | 0.500 (0.831) | 0.500 (0.872) | 0.503 (0.828) | 0.693 (0.854) | 0.500 (0.712) | 0.500 (0.741) |
| cam zoom (6.2) | 0.500 (0.995) | 0.904 (0.997) | 0.913 (0.999) | 0.966 (0.999) | 0.649 (0.999) | 0.977 (0.997) |
| light dir (5.2) | 0.500 (0.906) | 0.500 (0.894) | 0.997 (1.000) | 0.992 (0.999) | 0.500 (0.995) | 0.969 (0.997) |
| noise (8.0) | 0.824 (0.997) | 0.987 (0.998) | 0.997 (1.000) | 0.989 (0.999) | 0.998 (0.999) | 0.991 (0.999) |
| blur (1.7) | 0.812 (0.999) | 0.983 (0.999) | 0.996 (0.999) | 0.991 (0.999) | 0.992 (1.000) | 0.994 (0.999) |
| jpeg (4.5) | 0.500 (0.802) | 0.504 (0.952) | 0.996 (1.000) | 0.991 (0.999) | 0.500 (0.997) | 0.921 (0.997) |
| exposure (33.7) | 0.500 (0.583) | 0.500 (0.658) | 0.735 (0.998) | 0.980 (0.998) | 0.500 (0.675) | 0.500 (0.743) |
| white balance (12.7) | 0.500 (0.498) | 0.500 (0.348) | 0.997 (1.000) | 0.991 (0.999) | 0.500 (0.619) | 0.500 (0.459) |
| occlusion (1.9) | 0.500 (0.691) | 0.576 (0.683) | 0.592 (0.956) | 0.707 (0.955) | 0.500 (0.900) | 0.582 (0.869) |

Table 11. CEM under sim-to-real shifts, 50 box-crossing pairs: collision (success). In distribution (Table 4): raw 0.540-0.580, CCEM 0.020-0.060.

| Shift | sigreg raw | sigreg CCEM MLP | sigreg CCEM lin | jac raw | jac CCEM MLP | jac CCEM lin | official raw | official CCEM MLP | official CCEM lin |
|---|---|---|---|---|---|---|---|---|---|
| cam shift | 0.160 (0.080) | 0.160 (0.080) | 0.160 (0.080) | 0.640 (0.260) | 0.320 (0.200) | 0.220 (0.200) | 0.240 (0.120) | 0.240 (0.120) | 0.240 (0.120) |
| cam tilt | 0.260 (0.160) | 0.260 (0.160) | 0.260 (0.160) | 0.560 (0.160) | 0.440 (0.160) | 0.340 (0.160) | 0.320 (0.140) | 0.320 (0.140) | 0.320 (0.140) |
| cam zoom | 0.280 (0.540) | 0.280 (0.600) | 0.220 (0.420) | 0.400 (0.880) | 0.120 (0.700) | 0.100 (0.640) | 0.360 (0.720) | 0.060 (0.640) | 0.040 (0.560) |
| light dir | 0.540 (0.160) | 0.560 (0.160) | 0.520 (0.160) | 0.560 (0.680) | 0.060 (0.620) | 0.060 (0.520) | 0.640 (0.480) | 0.100 (0.580) | 0.060 (0.520) |
| noise | 0.540 (0.580) | 0.120 (0.620) | 0.060 (0.500) | 0.580 (0.600) | 0.060 (0.660) | 0.060 (0.500) | 0.600 (0.720) | 0.040 (0.680) | 0.040 (0.520) |
| blur | 0.520 (0.640) | 0.040 (0.740) | 0.040 (0.540) | 0.640 (0.560) | 0.080 (0.660) | 0.060 (0.540) | 0.560 (0.780) | 0.040 (0.780) | 0.040 (0.600) |
| jpeg | 0.500 (0.140) | 0.260 (0.240) | 0.200 (0.060) | 0.560 (0.540) | 0.040 (0.560) | 0.060 (0.440) | 0.560 (0.600) | 0.100 (0.620) | 0.040 (0.540) |
| exposure | 0.280 (0.140) | 0.280 (0.140) | 0.280 (0.140) | 0.500 (0.680) | 0.080 (0.680) | 0.060 (0.640) | 0.400 (0.240) | 0.400 (0.240) | 0.400 (0.240) |
| white balance | 0.600 (0.100) | 0.600 (0.100) | 0.600 (0.100) | 0.540 (0.680) | 0.060 (0.560) | 0.060 (0.460) | 0.360 (0.300) | 0.360 (0.300) | 0.360 (0.300) |
| occlusion | 0.280 (0.240) | 0.280 (0.240) | 0.260 (0.220) | 0.340 (0.420) | 0.240 (0.500) | 0.260 (0.420) | 0.320 (0.320) | 0.320 (0.300) | 0.280 (0.260) |
| gear0.7 | 0.520 (0.380) | 0.040 (0.500) | 0.040 (0.300) | 0.580 (0.380) | 0.040 (0.400) | 0.040 (0.340) | 0.520 (0.440) | 0.060 (0.320) | 0.040 (0.320) |
| damping10 | 0.120 (0.040) | 0.020 (0.060) | 0.040 (0.060) | 0.140 (0.040) | 0.020 (0.060) | 0.040 (0.040) | 0.140 (0.040) | 0.020 (0.040) | 0.040 (0.040) |
| delay1 | 0.560 (0.540) | 0.080 (0.540) | 0.080 (0.440) | 0.600 (0.440) | 0.120 (0.520) | 0.100 (0.420) | 0.560 (0.520) | 0.060 (0.460) | 0.060 (0.460) |
| act noise | 0.620 (0.460) | 0.220 (0.360) | 0.120 (0.360) | 0.620 (0.420) | 0.220 (0.340) | 0.180 (0.260) | 0.540 (0.460) | 0.200 (0.360) | 0.220 (0.280) |

- jac + CCEM keeps collision at 0.040-0.120 under every photometric and sensor shift, camera zoom and three of the four dynamics shifts; sigreg is at raw-CEM level under exposure, white balance, light direction and the camera shifts, and the official model under exposure, white balance and the camera shifts. Margin function sign acc (Table 10) predicts this: wherever it is 0.500 for both heads, CCEM equals raw CEM.
- jac's remaining failures are camera shift / tilt (CCEM collision 0.220-0.440, success 0.160-0.200) and occlusion (0.240-0.260). The camera shifts break planning for every model (success 0.080-0.200 even raw): the goal image and the dynamics are seen through a geometry the model never saw, so this is a task failure as much as a safety one.
- Dynamics shifts leave the margin function intact (frames unchanged), so CCEM works for all three models (collision 0.020-0.120) except action noise (0.120-0.220): noise injected after planning pushes the arm into the box regardless of the plan. Joint damping ×10 is too strong (no model reaches the goal, success 0.040-0.060); its low collision is from the arm barely moving.
- The linear margin function is at least as good as the MLP for jac under every shift.

## Threshold-free CEM

Dubins found that threshold-free elite rules need $h$ to rank imagined OOD states, which it stopped doing within 5-10 steps under purple. On Reacher the relevant question is narrower: can $h$ rank the candidate plans of one replan, which share the start frame (`reacher_within_start.py`)? 200 safe held-out starts whose recorded path enters the box within 25 steps, 64 random CEM-style 5-block plans each (35.5% enter the box, simulator labels); plan score $\min_k h(\hat z_k)$ over the 5 predicted blocks, start frame rendered at floor $\alpha$.

Table 12. Plan ranking: within-start AUC (mean over starts) / pooled AUC / plan false-safe at threshold 0, for the MLP and linear real+imag margin functions.

| WM | $\alpha$ | MLP | linear |
|---|---|---|---|
| jac ep10 | 0.0 | 0.979 / 0.976 / 0.037 | 0.978 / 0.975 / 0.027 |
| jac ep10 | 0.2 | 0.979 / 0.976 / 0.048 | 0.977 / 0.976 / 0.031 |
| jac ep10 | 0.3 | 0.900 / 0.874 / 0.327 | 0.899 / 0.902 / 0.276 |
| jac ep10 | 0.4 | 0.834 / 0.804 / 0.670 | 0.730 / 0.697 / 0.606 |
| jac+adv_h ep5 | 0.0 | 0.981 / 0.979 / 0.042 | 0.977 / 0.974 / 0.029 |
| jac+adv_h ep5 | 0.2 | 0.980 / 0.979 / 0.044 | 0.977 / 0.973 / 0.029 |
| jac+adv_h ep5 | 0.3 | 0.966 / 0.942 / 0.349 | 0.967 / 0.959 / 0.187 |
| jac+adv_h ep5 | 0.4 | 0.852 / 0.766 / 0.900 | 0.883 / 0.828 / 0.747 |

- The plan ranking survives the shift where the threshold does not: at $\alpha = 0.3$ jac ranks plans at within-start AUC 0.90 while 28-33% of unsafe plans pass the 0 threshold; jac+adv_h ranks them at 0.97 (its better static ranking, Table 8, carries into plan ranking). At 0.4 ranking degrades (0.73-0.88) and 61-90% of unsafe plans pass.

CEM with the Dubins threshold-free elite rules (`ccem_fallback`; pre-selection 50 of 300): safe→task = the 50 plans with the highest $\min_k h$, elites = the 30 with the best goal cost among them; task→safe = the 50 best goal costs, elites = the 30 safest; safe→task norm = safe→task with $h$ z-scored per predicted step across the candidates.

Table 13. CEM under the floor shift, 50 box-crossing pairs: collision / safe success / median closest goal distance (rad; in distribution about 0.03-0.05).

| WM | $\alpha$ | Margin fn | CCEM (minh) | safe→task | task→safe | safe→task norm |
|---|---|---|---|---|---|---|
| jac ep10 | 0.3 | MLP | 0.140 / 0.440 / 0.048 | 0.080 / 0.120 / 0.470 | 0.120 / 0.300 / 0.085 | 0.060 / 0.060 / 0.610 |
| jac ep10 | 0.3 | linear | 0.100 / 0.460 / 0.054 | 0.080 / 0.140 / 0.193 | 0.080 / 0.320 / 0.094 | 0.060 / 0.120 / 0.255 |
| jac ep10 | 0.4 | MLP | 0.240 / 0.280 / 0.138 | 0.100 / 0.180 / 0.458 | 0.160 / 0.180 / 0.128 | 0.100 / 0.040 / 0.526 |
| jac ep10 | 0.4 | linear | 0.220 / 0.260 / 0.128 | 0.260 / 0.200 / 0.181 | 0.260 / 0.200 / 0.131 | 0.380 / 0.100 / 0.230 |
| jac+adv_h ep5 | 0.3 | MLP | 0.180 / 0.500 / 0.047 | 0.040 / 0.100 / 0.464 | 0.080 / 0.200 / 0.124 | 0.020 / 0.060 / 0.598 |
| jac+adv_h ep5 | 0.3 | linear | 0.160 / 0.420 / 0.052 | 0.020 / 0.180 / 0.204 | 0.060 / 0.260 / 0.101 | 0.020 / 0.160 / 0.265 |
| jac+adv_h ep5 | 0.4 | MLP | not run | 0.040 / 0.100 / 0.575 | 0.100 / 0.120 / 0.316 | 0.020 / 0.060 / 0.607 |
| jac+adv_h ep5 | 0.4 | linear | not run | 0.060 / 0.100 / 0.416 | 0.100 / 0.120 / 0.300 | 0.020 / 0.060 / 0.576 |

- Ranking-first rules lower collision further, most with jac+adv_h (0.020-0.060 at $\alpha$ 0.3 and 0.4, against 0.160-0.180 for CCEM at 0.3), but by stalling: safe success falls to 0.060-0.260 and the arm ends 0.19-0.61 rad from the goal instead of about 0.05. Picking the safest plans first picks plans that barely move. task→safe keeps more progress but is still below CCEM's safe success at 0.3.
- This repeats the Dubins verdict (ranking-first trades task for safety; threshold-first with ranking as fallback is the better default), but adv_h's ranking advantage is visible here for the first time in planning: with the same rules it collides less than jac.

## Test-time recalibration of the threshold

Under these shifts the ranking often survives while the threshold fails (Tables 3, 10, 12). `reacher_calib.py` refits only $h$'s offset, $h \to h - c$, from calibration frames rendered under the shift: held-out states disjoint from the test set, joint-box labels from qpos (a robot reads them from its joint encoders). Labeled: 100 safe + 100 unsafe frames, $c$ = the threshold with the best balanced accuracy. Safe-only (no failure data): 100 safe frames; median-matching ($c$ = shift of the median $h$ on safe frames) or the 5% quantile ($c$ = 5% quantile of $h$ on the shifted safe frames).

Table 14. Static bal acc on the 5000 test states: uncalibrated / labeled / safe-only 5% quantile (real+imag margin functions).

| WM | Margin fn | floor 0.3 | floor 0.4 | exposure | cam shift | cam tilt | occlusion |
|---|---|---|---|---|---|---|---|
| sigreg | MLP | 0.500 / 0.517 / 0.529 | 0.500 / 0.585 / 0.466 | 0.500 / 0.647 / 0.649 | 0.500 / 0.751 / 0.711 | 0.500 / 0.759 / 0.651 | 0.500 / 0.650 / 0.583 |
| jac | MLP | 0.500 / 0.844 / 0.806 | 0.500 / 0.681 / 0.585 | 0.735 / 0.959 / 0.988 | 0.506 / 0.862 / 0.718 | 0.503 / 0.788 / 0.538 | 0.592 / 0.890 / 0.856 |
| jac | linear | 0.578 / 0.904 / 0.890 | 0.500 / 0.519 / 0.470 | 0.980 / 0.985 / 0.979 | 0.860 / 0.960 / 0.751 | 0.693 / 0.788 / 0.610 | 0.707 / 0.887 / 0.806 |
| jac+adv_h ep5 | MLP | 0.500 / 0.968 / 0.973 | 0.500 / 0.778 / 0.631 | 0.997 / 0.998 / 0.989 | 0.602 / 0.932 / 0.813 | 0.500 / 0.678 / 0.549 | 0.592 / 0.925 / 0.831 |
| jac+adv_h ep5 | linear | 0.572 / 0.980 / 0.980 | 0.500 / 0.794 / 0.680 | 0.986 / 0.988 / 0.989 | 0.913 / 0.954 / 0.900 | 0.648 / 0.705 / 0.603 | 0.680 / 0.926 / 0.852 |

- Recalibration recovers the threshold wherever the ranking survived, and the safe-only 5% quantile gets most of the labeled gain without any failure data (jac+adv_h, floor 0.3: 0.973 / 0.980 vs labeled 0.968 / 0.980). adv_h's better ranking is what makes it recalibrate best. Median-matching does nothing (bal acc unchanged, not shown): $h$ on safe states hardly moves under these shifts; it is the unsafe states that rise into the safe range, so only a quantile-style threshold catches them.

Table 15. CCEM with the recalibrated threshold (offset fit on real frames, applied to every predicted state): collision / safe success / progress (median fraction of the initial goal distance closed).

| WM | Margin fn | Condition | Uncalibrated | Labeled | Safe-only 5% |
|---|---|---|---|---|---|
| jac | MLP | floor 0.3 | 0.140 / 0.440 / 0.898 | 0.060 / 0.080 / 0.200 | 0.080 / 0.220 / 0.610 |
| jac | linear | floor 0.3 | 0.100 / 0.460 / 0.874 | 0.100 / 0.280 / 0.730 | 0.100 / 0.240 / 0.770 |
| jac | MLP | floor 0.4 | 0.240 / 0.280 / 0.646 | 0.060 / 0.120 / 0.192 | 0.100 / 0.200 / 0.320 |
| jac | MLP | occlusion | 0.240 / 0.440 / 0.897 | 0.080 / 0.080 / 0.100 | 0.120 / 0.220 / 0.690 |
| jac | MLP | cam shift | 0.320 / 0.160 / 0.736 | 0.080 / 0.060 / 0.000 | 0.180 / 0.100 / 0.310 |
| jac | MLP | exposure | 0.080 / 0.640 / 0.930 | 0.040 / 0.580 / 0.920 | 0.020 / 0.340 / 0.840 |
| jac+adv_h ep5 | MLP | floor 0.3 | 0.180 / 0.500 / 0.902 | 0.020 / 0.180 / 0.653 | 0.020 / 0.140 / 0.543 |
| jac+adv_h ep5 | linear | floor 0.3 | 0.160 / 0.420 / 0.867 | 0.020 / 0.200 / 0.704 | 0.020 / 0.160 / 0.714 |
| jac+adv_h ep5 | MLP | occlusion | 0.220 / 0.400 / 0.860 | 0.080 / 0.140 / 0.280 | 0.120 / 0.240 / 0.540 |

- Recalibration cuts collision (to 0.020-0.120) but costs safe success (for example 0.500 to 0.140-0.180), largely through lost progress: the planner gets safer by not reaching the goal.
- The returned plans are almost never infeasible under the calibrated threshold (0-1% of replans for jac+adv_h at floor 0.3, Table 16 data), so this is not CCEM's fallback: the feasible set itself shrinks to plans that do not approach the goal. The offset is fit on real frames, but predicted states re-ground toward in-distribution, where $h$'s safe-ward shift is about half as large (Table 6), so the same offset over-tightens them. Next to try: calibrate on predicted latents (offset per prediction step), with safe labels from the robot's own executed plans.

### Recalibrating on predicted latents

`reacher_calib_pred.py`: one offset per position CCEM reads, $c_0$ for the current real frame (from the real-frame calibration) and $c_1 \dots c_5$ for the predicted blocks. Calibration data: 300 held-out starts (episodes not used by the CEM pairs, half near the box) rendered under the shift, 4 random CEM-style plans each, predicted latents labeled by replaying the plan in the simulator (what a robot gets by executing plans and reading its joint encoders). Labeled: per step, the best balanced-accuracy threshold; safe-only: per step, the 5% quantile of $h$ on predicted states whose executed state was safe. The predicted-step offsets are about half the real-frame ones (safe-only, jac MLP floor 0.3: $c_{1..5}$ = 0.44-0.93 vs $c_0$ = 1.46; jac+adv_h: 0.51-0.94 vs 1.92), confirming that the real-frame offset over-tightens predicted states. Held-out per-step bal acc on predicted states rises most for jac+adv_h at floor 0.3 (0.53-0.77 to 0.82-0.92 labeled, 0.79-0.87 safe-only); on camera shift the safe-only offsets come out negative and lower it.

Table 17. CCEM collision / safe success / progress: uncalibrated, real-frame offset (labeled, safe-only 5%), per-step predicted-latent offsets (labeled, safe-only 5%).

| WM | Margin fn | Condition | Uncalibrated | Real, labeled | Real, safe-only | Predicted, labeled | Predicted, safe-only |
|---|---|---|---|---|---|---|---|
| jac | MLP | floor 0.3 | 0.140 / 0.440 / 0.898 | 0.060 / 0.080 / 0.201 | 0.080 / 0.220 / 0.613 | 0.060 / 0.180 / 0.604 | 0.080 / 0.240 / 0.743 |
| jac | linear | floor 0.3 | 0.100 / 0.460 / 0.874 | 0.100 / 0.280 / 0.732 | 0.100 / 0.240 / 0.774 | 0.100 / 0.320 / 0.790 | 0.100 / 0.340 / 0.792 |
| jac+adv_h ep5 | MLP | floor 0.3 | 0.180 / 0.500 / 0.902 | 0.020 / 0.180 / 0.653 | 0.020 / 0.140 / 0.543 | 0.080 / 0.220 / 0.723 | 0.180 / 0.220 / 0.798 |
| jac+adv_h ep5 | linear | floor 0.3 | 0.160 / 0.420 / 0.867 | 0.020 / 0.200 / 0.704 | 0.020 / 0.160 / 0.714 | 0.060 / 0.240 / 0.768 | 0.080 / 0.280 / 0.841 |
| jac | MLP | floor 0.4 | 0.240 / 0.280 / 0.646 | 0.060 / 0.120 / 0.192 | 0.100 / 0.200 / 0.324 | 0.100 / 0.120 / 0.178 | 0.140 / 0.240 / 0.407 |
| jac | linear | floor 0.4 | 0.220 / 0.260 / 0.662 | 0.240 / 0.080 / 0.320 | 0.200 / 0.260 / 0.562 | 0.120 / 0.100 / 0.187 | 0.260 / 0.240 / 0.709 |
| jac+adv_h ep5 | MLP | floor 0.4 | 0.360 / 0.140 / 0.511 | 0.020 / 0.080 / 0.049 | 0.040 / 0.080 / 0.119 | 0.080 / 0.080 / 0.092 | 0.260 / 0.120 / 0.432 |
| jac+adv_h ep5 | linear | floor 0.4 | 0.360 / 0.140 / 0.544 | 0.100 / 0.060 / 0.118 | 0.180 / 0.100 / 0.312 | 0.100 / 0.080 / 0.119 | 0.300 / 0.140 / 0.569 |
| jac | MLP | occlusion | 0.240 / 0.440 / 0.897 | 0.080 / 0.080 / 0.100 | 0.120 / 0.220 / 0.688 | 0.080 / 0.140 / 0.295 | 0.220 / 0.220 / 0.800 |
| jac | linear | occlusion | 0.260 / 0.400 / 0.859 | 0.140 / 0.200 / 0.721 | 0.180 / 0.360 / 0.856 | 0.180 / 0.360 / 0.831 | 0.220 / 0.400 / 0.865 |
| jac+adv_h ep5 | MLP | occlusion | 0.220 / 0.400 / 0.855 | 0.080 / 0.140 / 0.279 | 0.120 / 0.240 / 0.541 | 0.120 / 0.180 / 0.401 | 0.160 / 0.200 / 0.628 |
| jac+adv_h ep5 | linear | occlusion | 0.200 / 0.360 / 0.849 | 0.140 / 0.240 / 0.649 | 0.140 / 0.280 / 0.699 | 0.140 / 0.300 / 0.725 | 0.160 / 0.300 / 0.739 |
| jac | MLP | cam shift | 0.320 / 0.160 / 0.736 | 0.080 / 0.060 / 0.005 | 0.180 / 0.100 / 0.314 | 0.100 / 0.060 / 0.023 | 0.360 / 0.120 / 0.601 |
| jac | linear | cam shift | 0.220 / 0.200 / 0.659 | 0.180 / 0.140 / 0.546 | 0.340 / 0.200 / 0.773 | 0.100 / 0.160 / 0.365 | 0.420 / 0.140 / 0.738 |
| jac+adv_h ep5 | MLP | cam shift | 0.240 / 0.140 / 0.744 | 0.080 / 0.080 / 0.072 | 0.140 / 0.240 / 0.475 | 0.320 / 0.120 / 0.200 | 0.320 / 0.160 / 0.693 |
| jac+adv_h ep5 | linear | cam shift | 0.140 / 0.200 / 0.595 | 0.060 / 0.140 / 0.232 | 0.140 / 0.200 / 0.641 | 0.140 / 0.180 / 0.396 | 0.320 / 0.220 / 0.734 |

- Predicted-latent offsets are less conservative than real-frame ones (more progress, for example jac MLP floor 0.3: progress 0.60-0.74 vs 0.20-0.61), but they give back part of the collision reduction. Every calibrated run sits on the same safety-progress trade-off: safe success is at or below the uncalibrated run in 63 of 64 cells (the exception, jac+adv_h MLP under camera shift with the real-frame safe-only offset, 0.240 vs 0.140, is within about 1.5 standard errors at $n = 50$).
- Calibration therefore does not improve the joint outcome; it chooses a more conservative operating point. Only a better ranking can move the trade-off itself, and adv_h's better static ranking does not clearly do so in CEM (at matched collision, its safe success is within noise of jac's).
- Under calibration, residual collisions are more often noticed (the margin function reads the box once inside: 0.33-1.00 vs 0-0.08 uncalibrated, Table 16 vs `attr/` runs), so the threshold is closer, but still fooled on the plan that entered.

### The blur: veto rate vs distance to the box

Why does no threshold work? `reacher_veto_diag.py` (`CURVE=1`): 200 safe starts near the box × 64 random 5-block plans, each simulated step by step for its closest joint-space distance to the box. For each model and condition the CCEM plan score ($\min$ of $h$ over the current frame and 5 predicted blocks) is thresholded to catch 90% of unsafe plans; the table gives the share of safe plans vetoed, by their true closest distance (rad). In distribution the veto falls off within about 0.1 rad of the box: that is the margin function's resolution. A blurred latent pushes the fall-off outward; a flat row means no distance information.

Table 18. Safe-plan veto rate at 90% unsafe caught, by closest distance to the box (rad).

| WM | Margin fn | Condition | < 0.05 | 0.05-0.1 | 0.1-0.2 | 0.2-0.3 | 0.3-0.5 | 0.5-1 | > 1 |
|---|---|---|---|---|---|---|---|---|---|
| sigreg | MLP | in distribution | 0.36 | 0.25 | 0.12 | 0.03 | 0.02 | 0.01 | 0.15 |
| sigreg | linear | in distribution | 0.48 | 0.40 | 0.23 | 0.05 | 0.02 | 0.01 | 0.04 |
| sigreg | MLP | floor 0.2 | 0.87 | 0.89 | 0.89 | 0.90 | 0.87 | 0.93 | 0.99 |
| sigreg | linear | floor 0.2 | 0.93 | 0.95 | 0.95 | 0.97 | 0.97 | 0.98 | 0.99 |
| sigreg | MLP | floor 0.3 | 0.83 | 0.87 | 0.87 | 0.89 | 0.88 | 0.92 | 0.99 |
| sigreg | linear | floor 0.3 | 0.93 | 0.95 | 0.96 | 0.97 | 0.97 | 0.96 | 0.98 |
| sigreg | MLP | occlusion | 0.87 | 0.88 | 0.86 | 0.82 | 0.75 | 0.71 | 0.82 |
| sigreg | linear | occlusion | 0.85 | 0.87 | 0.85 | 0.82 | 0.80 | 0.78 | 0.83 |
| sigreg | MLP | cam shift | 0.84 | 0.83 | 0.82 | 0.81 | 0.80 | 0.71 | 0.69 |
| sigreg | linear | cam shift | 0.87 | 0.88 | 0.86 | 0.85 | 0.86 | 0.83 | 0.87 |
| jac | MLP | in distribution | 0.35 | 0.25 | 0.12 | 0.03 | 0.02 | 0.01 | 0.12 |
| jac | linear | in distribution | 0.53 | 0.46 | 0.29 | 0.07 | 0.02 | 0.01 | 0.04 |
| jac | MLP | floor 0.2 | 0.36 | 0.24 | 0.12 | 0.03 | 0.02 | 0.01 | 0.14 |
| jac | linear | floor 0.2 | 0.53 | 0.47 | 0.30 | 0.09 | 0.02 | 0.01 | 0.06 |
| jac | MLP | floor 0.3 | 0.83 | 0.82 | 0.68 | 0.42 | 0.12 | 0.03 | 0.37 |
| jac | linear | floor 0.3 | 0.72 | 0.71 | 0.56 | 0.35 | 0.11 | 0.08 | 0.29 |
| jac | MLP | occlusion | 0.88 | 0.87 | 0.85 | 0.80 | 0.56 | 0.16 | 0.48 |
| jac | linear | occlusion | 0.77 | 0.74 | 0.69 | 0.58 | 0.42 | 0.45 | 0.56 |
| jac | MLP | cam shift | 0.90 | 0.90 | 0.86 | 0.75 | 0.52 | 0.21 | 0.22 |
| jac | linear | cam shift | 0.83 | 0.81 | 0.76 | 0.66 | 0.49 | 0.21 | 0.07 |
| jac+adv_h ep5 | MLP | in distribution | 0.34 | 0.26 | 0.12 | 0.03 | 0.02 | 0.01 | 0.14 |
| jac+adv_h ep5 | linear | in distribution | 0.55 | 0.47 | 0.29 | 0.08 | 0.02 | 0.01 | 0.04 |
| jac+adv_h ep5 | MLP | floor 0.2 | 0.37 | 0.27 | 0.13 | 0.03 | 0.02 | 0.01 | 0.12 |
| jac+adv_h ep5 | linear | floor 0.2 | 0.56 | 0.47 | 0.29 | 0.06 | 0.02 | 0.01 | 0.04 |
| jac+adv_h ep5 | MLP | floor 0.3 | 0.71 | 0.62 | 0.46 | 0.25 | 0.09 | 0.03 | 0.30 |
| jac+adv_h ep5 | linear | floor 0.3 | 0.51 | 0.36 | 0.24 | 0.09 | 0.04 | 0.02 | 0.16 |
| jac+adv_h ep5 | MLP | occlusion | 0.87 | 0.83 | 0.87 | 0.82 | 0.55 | 0.17 | 0.48 |
| jac+adv_h ep5 | linear | occlusion | 0.74 | 0.71 | 0.68 | 0.49 | 0.28 | 0.19 | 0.48 |
| jac+adv_h ep5 | MLP | cam shift | 0.80 | 0.79 | 0.67 | 0.51 | 0.32 | 0.15 | 0.09 |
| jac+adv_h ep5 | linear | cam shift | 0.80 | 0.79 | 0.72 | 0.59 | 0.38 | 0.20 | 0.16 |

- The latent's state resolution under the shift, not the threshold, sets the trade-off. In distribution the 50% veto point is about 0.05 rad from the box; jac at floor 0.3 moves it to about 0.25 rad (matching its nearest-neighbour state error, 0.29-0.41 rad, Table 6c), and the safe detours that make progress pass within 0.15 rad (41% of them). sigreg's rows are flat at about 0.9 (no distance information); jac at floor 0.2 is unchanged from in distribution.
- jac+adv_h narrows the blur: at floor 0.3 its 50% point is about 0.15 rad (jac 0.25), and under camera shift it vetoes 10-20 points fewer safe plans at every distance; under occlusion it equals jac. The 50-pair CEM comparisons were too noisy to show this. So the adversarial reg acts on the right quantity.
- Plans more than 1 rad from the box are vetoed 12-15% even in distribution (rarely visited regions); this does not affect the comparison.

### Oracle check: a graded target does not remove the blur

If the blur were in the readout, a target with resolution near the boundary should help. Oracle (not a usable method, since it needs a geometric definition of the constraint): the same real + imagined latents, but $h$ regresses the signed joint-space distance to the box (MSE, clipped to [-0.5, 1] rad; `reacher_margin_imag.py TARGET=dist`, heads in `reacher_heads_dist*`). Plan-veto diagnostic as Table 18's set-up, threshold 0, MLP.

Table 20. CCEM plan veto, unsafe caught / near-safe vetoed / far-safe vetoed: classification (margin_gp) vs distance-regression margin function.

| WM | Target | In distribution | Floor 0.3 | Occlusion |
|---|---|---|---|---|
| jac | classification | 0.92 / 0.31 / 0.05 | 0.62 / 0.07 / 0.02 | 0.57 / 0.07 / 0.02 |
| jac | distance (oracle) | 0.83 / 0.11 / 0.02 | 0.59 / 0.06 / 0.01 | 0.55 / 0.06 / 0.02 |
| jac+adv_h ep5 | classification | 0.91 / 0.30 / 0.05 | 0.59 / 0.06 / 0.02 | 0.47 / 0.07 / 0.02 |
| jac+adv_h ep5 | distance (oracle) | 0.85 / 0.16 / 0.02 | 0.61 / 0.05 / 0.01 | 0.48 / 0.07 / 0.01 |

- In distribution the distance target sharpens the boundary (near-safe vetoed 0.11-0.16 vs 0.30-0.31, at a somewhat lower catch rate). Under the shifts the two targets are indistinguishable, with or without plan-level calibration: the information that separates near-safe from unsafe is not in the shifted latent, so no readout target can recover it. The lever is the encoder's state resolution under shift (Tables 6c, 18), not the margin function.

## Separating "cannot plan" from "unsafe"

`reacher_outcomes.py` on the saved trajectories: per episode, reached the goal or not × entered the box or not; progress = fraction of the initial goal distance closed by the closest approach; no-progress share = safe-but-failed episodes with progress < 0.25. For runs with $h$ logging (`reacher_cem.py`), each collision is attributed: fooled = the plan returned at the replan before entry had $\min_k h(\hat z_k) > 0$ (the model judged it safe); forced = $\le 0$ (no feasible plan); noticed = $h$ on the real frame at the entry step is $< 0$.

Table 16. Outcomes and collision attribution (CCEM): reached+safe / safe, no goal / entered box / progress / no-progress share of safe failures | fooled / forced / noticed (of collisions).

| WM | Margin fn | Condition | Outcomes | Attribution |
|---|---|---|---|---|
| jac | MLP | in distribution | 0.580 / 0.360 / 0.060 / 0.927 / 0.111 | 0.33 / 0.67 / 1.00 |
| jac | MLP | floor 0.3 | 0.440 / 0.420 / 0.140 / 0.898 / 0.190 | 1.00 / 0.00 / 0.00 |
| jac | MLP | floor 0.4 | 0.280 / 0.480 / 0.240 / 0.646 / 0.250 | 1.00 / 0.00 / 0.00 |
| jac | MLP | occlusion | 0.440 / 0.320 / 0.240 / 0.897 / 0.062 | 1.00 / 0.00 / 0.08 |
| jac | MLP | cam shift | 0.160 / 0.520 / 0.320 / 0.736 / 0.385 | 1.00 / 0.00 / 0.00 |
| jac | linear | cam shift | 0.200 / 0.580 / 0.220 / 0.659 / 0.448 | 0.73 / 0.27 / 0.73 |
| sigreg | MLP | floor 0.1 | 0.300 / 0.220 / 0.480 / 0.732 / 0.455 | 1.00 / 0.00 / 0.00 |
| jac+adv_h ep5 | MLP | floor 0.3 | 0.500 / 0.320 / 0.180 / 0.902 / 0.062 | 1.00 / 0.00 / 0.00 |
| jac+adv_h ep5 | MLP | floor 0.3, labeled calib | 0.180 / 0.800 / 0.020 / 0.653 / 0.300 | 1.00 / 0.00 / 1.00 |
| jac | MLP | damping ×10 (dynamics) | 0.040 / 0.920 / 0.040 / 0.204 / 0.848 | n/a (no logging) |

- In distribution the few collisions (3/50) are mostly forced: the model saw no feasible plan, and $h$ noticed the box. Under every appearance shift the collisions are the model's: the chosen plan looked safe (fooled 0.73-1.00) and $h$ did not read the box even with the arm inside (noticed 0-0.08). So OOD collisions are a safety-model failure, not a planner choosing a known risk.
- "Cannot plan" is separable: in distribution the safe failures are near-misses (progress 0.93, 11% no-progress); under camera shift and for sigreg at floor 0.1 about 40-45% of safe failures make no progress, and damping ×10 is almost purely a planning failure (85% no-progress, progress 0.20).

## adv_h at matched epochs (mid-run checkpoints)

The adv_h runs are paused or still training, so their checkpoints are compared with jac at the same epoch (same seed, same cosine schedule position; neither has finished annealing). jac+adv_h = `rc_jac_safeadv_linf` (MLP adversary head, the Dubins recipe: L∞ 8/255, 2 PGD steps, AUC gate 0.95), epoch 5; jac+adv_h-lin = `rc_jac_safeadv_linf_linhead` (linear adversary head), epoch 4. Same pipeline as above: real+imag margin functions (`reacher_heads_imag_ep<EP>/`), floor sweep, CEM on the 50 box-crossing pairs with the floor shift.

Table 8. Floor sweep, bal acc (AUC) of the real+imag margin functions.

| WM | Margin fn | $\alpha$ = 0.2 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|
| jac ep5 | MLP | 0.998 (0.999) | 0.500 (0.955) | 0.500 (0.746) | 0.500 (0.368) |
| jac+adv_h ep5 | MLP | 0.997 (1.000) | 0.500 (0.989) | 0.500 (0.853) | 0.500 (0.708) |
| jac ep5 | linear | 0.992 (0.999) | 0.635 (0.994) | 0.500 (0.661) | 0.500 (0.376) |
| jac+adv_h ep5 | linear | 0.988 (0.998) | 0.572 (0.997) | 0.500 (0.867) | 0.500 (0.781) |
| jac ep4 | MLP | 0.997 (0.999) | 0.500 (0.974) | 0.500 (0.756) | 0.500 (0.653) |
| jac+adv_h-lin ep4 | MLP | 0.981 (0.999) | 0.500 (0.598) | 0.500 (0.281) | 0.500 (0.307) |
| jac ep4 | linear | 0.992 (0.999) | 0.835 (0.996) | 0.500 (0.832) | 0.500 (0.496) |
| jac+adv_h-lin ep4 | linear | 0.993 (0.999) | 0.500 (0.474) | 0.500 (0.495) | 0.500 (0.463) |

Table 9. CEM collision (success), in distribution and under the floor shift.

| WM | Planner | $\alpha$ = 0 | 0.1 | 0.2 | 0.3 |
|---|---|---|---|---|---|
| jac ep5 | raw | 0.580 (0.620) | 0.580 (0.480) | 0.580 (0.520) | 0.540 (0.660) |
| jac ep5 | CCEM, MLP | 0.060 (0.560) | 0.060 (0.600) | 0.040 (0.600) | 0.200 (0.540) |
| jac ep5 | CCEM, linear | 0.060 (0.480) | 0.060 (0.460) | 0.080 (0.440) | 0.140 (0.460) |
| jac+adv_h ep5 | raw | 0.620 (0.580) | 0.580 (0.600) | 0.580 (0.660) | 0.340 (0.620) |
| jac+adv_h ep5 | CCEM, MLP | 0.060 (0.680) | 0.060 (0.580) | 0.060 (0.600) | 0.180 (0.580) |
| jac+adv_h ep5 | CCEM, linear | 0.080 (0.440) | 0.080 (0.500) | 0.060 (0.560) | 0.160 (0.440) |
| jac ep4 | raw | 0.580 (0.500) | 0.580 (0.480) | 0.600 (0.540) | 0.580 (0.640) |
| jac ep4 | CCEM, MLP | 0.040 (0.580) | 0.040 (0.620) | 0.060 (0.580) | 0.160 (0.660) |
| jac ep4 | CCEM, linear | 0.060 (0.520) | 0.040 (0.420) | 0.040 (0.460) | 0.100 (0.480) |
| jac+adv_h-lin ep4 | raw | 0.600 (0.660) | 0.560 (0.620) | 0.520 (0.640) | 0.280 (0.420) |
| jac+adv_h-lin ep4 | CCEM, MLP | 0.040 (0.560) | 0.060 (0.620) | 0.080 (0.540) | 0.280 (0.420) |
| jac+adv_h-lin ep4 | CCEM, linear | 0.040 (0.480) | 0.040 (0.420) | 0.040 (0.400) | 0.280 (0.420) |

- jac+adv_h (MLP adversary) reproduces the Dubins pattern: the margin function's ranking survives larger shifts (AUC at $\alpha = 0.5$: 0.708 / 0.781 vs jac 0.368 / 0.376), but the threshold breaks at the same $\alpha = 0.3$, and CCEM is unchanged (collision at 0.3: 0.180 / 0.160 vs 0.200 / 0.140). In distribution it costs nothing (margin accuracy and CEM equal; CCEM-MLP success 0.680 vs 0.560, within noise).
- jac+adv_h with a linear adversary head is worse than jac: its ranking collapses at $\alpha = 0.3$ (AUC 0.598 / 0.474 vs 0.974 / 0.996), and CCEM at 0.3 is no better than raw (0.280 vs jac 0.160 / 0.100).
- Mid-run checkpoints, one seed, $n = 50$ per CEM cell.

### Why the adversarial reg costs Reacher little: not because Reacher's $h$ is already robust

On Dubins adv_h raised prediction loss 1.7-6x and was mostly worse in CCEM on the jac base; on Reacher, jac+adv_h's prediction loss is within 0-5% of jac's from epoch 3 on (epoch 1-7: 0.0867 / 0.0426 / 0.0316 / 0.0289 / 0.0263 / 0.0246 / 0.0238 vs jac 0.0873 / 0.0355 / 0.0297 / 0.0276 / 0.0254 / 0.0243 / 0.0237; +20% only at epoch 2, when the penalty switches on). Hypothesis tested: on Reacher the safety signal is arm geometry, which a small pixel perturbation cannot move, so the reg has little to remove. `adv_sensitivity.py`: PGD (10 sign steps, random start) in an L∞ ball on [0, 1] pixels, unsafe states pushed toward safe and safe toward unsafe; held-out frames (last 10% of each dataset, 200 per class), real+imag MLP margin functions.

Table 19. Adversarial movability of $h$: median $|\Delta h|$ / clean safe-unsafe gap, and flip rates unsafe→safe / safe→unsafe.

| Benchmark | WM | 8/255 | 16/255 |
|---|---|---|---|
| Dubins | jac | 0.048, 0.09 / 0.02 | 0.106, 0.23 / 0.04 |
| Dubins | jac+adv_h | 0.018, 0.01 / 0.00 | 0.041, 0.02 / 0.01 |
| Reacher | jac | 0.070, 0.24 / 0.03 | 0.351, 1.00 / 0.05 |
| Reacher | jac+adv_h ep5 | 0.021, 0.02 / 0.00 | 0.085, 0.77 / 0.00 |

- Rejected: Reacher's $h$ is more movable than Dubins' (8/255 flips 24% of unsafe states to safe for jac, against 9%; 16/255 flips all of them), so the reg has more to remove on Reacher yet costs less. The cost must depend on whether making $h$ insensitive discards information the predictor uses (plausible: Reacher's arm is a high-contrast shape the reg can keep while dropping intensity sensitivity), not on how sensitive $h$ was; untested.
- adv_h cuts $h$'s movability about 3x within its own ball on both benchmarks. Even after the reg, 16/255 flips 77% of Reacher's unsafe states, so a 16/255 adversary would push much harder than the current one (expect a real prediction-loss cost).

## Which route do the regs take on Reacher?

Port of the Dubins / Safety Gym route decomposition (`safeadv_results/diagnostics/three_routes.py`) as `reacher_routes.py`: for a shift $\Delta z$ and the safety direction $\lambda$ (safe-minus-unsafe latent centroid difference), $|\Delta z \cdot \lambda_n| / \text{gap} = \cos(\Delta z, \lambda_n) \times (\lVert \Delta z \rVert / \text{spread}) / (\text{gap} / \text{spread})$; cos = orthogonality route (lower is better), $\lVert \Delta z \rVert$ / spread = invariance route (lower is better), gap / spread = widening route (higher is better), spread = RMS radius of the clean latent cloud. Added: motion / spread = median latent distance between a state and the state 5 env steps later (state resolution). 1000 held-out states (500 unsafe); PGD = 8/255 L∞, 2 sign steps maximizing $(\Delta z \cdot \lambda_n)^2$ (as in the Dubins script); real shifts from paired renders.

Table 21. Per shift: cos / $\lVert \Delta z \rVert$ / spread / along / gap.

| WM | gap / spread | motion / spread | pgd | floor 0.2 | floor 0.3 | light | arm color | exposure | white balance | occlusion | cam shift |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sigreg | 0.559 | 0.600 | 0.349 / 1.577 / 0.986 | 0.250 / 1.683 / 0.752 | 0.219 / 1.610 / 0.632 | 0.256 / 1.405 / 0.644 | 0.210 / 0.875 / 0.330 | 0.311 / 1.555 / 0.867 | 0.225 / 1.544 / 0.622 | 0.216 / 1.470 / 0.570 | 0.289 / 1.608 / 0.832 |
| jac ep5 | 0.574 | 0.567 | 0.625 / 0.112 / 0.122 | 0.107 / 0.109 / 0.020 | 0.289 / 0.653 / 0.329 | 0.135 / 0.148 / 0.035 | 0.165 / 0.404 / 0.116 | 0.224 / 0.371 / 0.145 | 0.180 / 0.079 / 0.025 | 0.299 / 0.584 / 0.304 | 0.171 / 0.770 / 0.229 |
| jac | 0.580 | 0.591 | 0.609 / 0.110 / 0.115 | 0.099 / 0.113 / 0.019 | 0.298 / 0.771 / 0.397 | 0.106 / 0.138 / 0.025 | 0.162 / 0.378 / 0.106 | 0.191 / 0.347 / 0.114 | 0.143 / 0.076 / 0.019 | 0.297 / 0.656 / 0.336 | 0.178 / 0.817 / 0.251 |
| jac+adv_h ep5 | 0.614 | 0.572 | 0.559 / 0.061 / 0.055 | 0.159 / 0.091 / 0.024 | 0.307 / 0.783 / 0.391 | 0.206 / 0.077 / 0.026 | 0.220 / 0.494 / 0.177 | 0.163 / 0.083 / 0.022 | 0.095 / 0.054 / 0.008 | 0.298 / 0.716 / 0.347 | 0.195 / 0.678 / 0.215 |
| official | 0.517 | 0.633 | 0.333 / 1.365 / 0.879 | 0.289 / 1.084 / 0.605 | 0.274 / 1.320 / 0.700 | 0.245 / 1.432 / 0.679 | 0.176 / 0.836 / 0.284 | 0.209 / 1.476 / 0.597 | 0.260 / 1.418 / 0.713 | 0.227 / 1.022 / 0.447 | 0.225 / 1.514 / 0.658 |

- jac does not compress the latent on Reacher: gap / spread and motion / spread are the same as sigreg's (0.58 vs 0.56; 0.59 vs 0.60), unlike Dubins where jac shrank the class gap to ×0.44. So preserving state distances against a jac-induced compression (relational distillation on pairs of different states) has nothing to fix here.
- jac's route on Reacher is plain invariance, and unlike Dubins it transfers to the real shifts: under light, floor 0.2 and white balance the latent moves 10-20x less than sigreg's (0.08-0.14 vs 1.41-1.68 of the spread). adv_h adds invariance where jac was already decent (exposure 0.083 vs 0.347; light 0.077 vs 0.106) and widens the gap slightly (0.614).
- Where jac breaks (floor 0.3, occlusion, camera shift) its invariance runs out: the latent moves 0.66-0.82 of the spread, more than 5 env steps of real motion (0.59), and no model turns the shift away from the safety axis (cos about 0.3 everywhere). adv_h does not change this linear picture at floor 0.3 (along / gap 0.391 vs 0.397); its narrower blur (Table 18) must sit in the nonlinear structure the MLP $h$ reads, which a single centroid direction does not capture.
- So the routes left for the large shifts are extending invariance to larger shifts (the size problem; a larger adversary radius, with the Dubins caveat that 16/255 hurt colour there) or turning the shift orthogonal to safety, which on Dubins only the jac+pull family did (colour ortho ×4.1-×5.0; forcing it with a loss hurt). The running jac+pull(2.4)+adv_h model tests the second on Reacher.

## Test-time BatchNorm adaptation

Label-free test-time adaptation from the Dubins work (imag-margin `bn_adapt_save.py`; Reacher port `reacher_bn_adapt.py`). LeWM's only BatchNorm layers are in the encoder's projector and the predictor's output projection. Their running statistics are reset and re-estimated (equal-weight average) under the shift, the projector from 2000 unlabeled shifted frames, then the predictor projection from 5-block random CEM-style imagined rollouts from those adapted latents. Only safe frames are used (a robot cannot collect failures). Nothing else changes, and the margin functions stay the ones trained on the unadapted model. The adapted checkpoint replaces the world model everywhere (current frame, goal frame, predicted rollout). Safe-q10 = plan with $h - c$, $c$ = 10% quantile of $h$ on the adapted safe frames (the Dubins safe-only threshold).

Table 22. Blur curve (as Table 18: safe-plan veto at 90% unsafe caught, by closest distance to the box, rad), MLP margin function.

| WM | Condition | Model | < 0.05 | 0.05-0.1 | 0.1-0.2 | 0.2-0.3 | 0.3-0.5 | 0.5-1 | > 1 |
|---|---|---|---|---|---|---|---|---|---|
| any | in distribution | reference (jac) | 0.35 | 0.25 | 0.12 | 0.03 | 0.02 | 0.01 | 0.12 |
| sigreg | floor 0.1 | BN-adapted | 0.34 | 0.26 | 0.13 | 0.04 | 0.03 | 0.01 | 0.28 |
| sigreg | floor 0.3 | unadapted | 0.83 | 0.87 | 0.87 | 0.89 | 0.88 | 0.92 | 0.99 |
| sigreg | floor 0.3 | BN-adapted | 0.40 | 0.30 | 0.18 | 0.11 | 0.05 | 0.02 | 0.35 |
| sigreg | floor 0.4 | BN-adapted | 0.66 | 0.59 | 0.45 | 0.23 | 0.11 | 0.04 | 0.40 |
| sigreg | exposure | BN-adapted | 0.78 | 0.77 | 0.67 | 0.51 | 0.27 | 0.13 | 0.18 |
| sigreg | white balance | BN-adapted | 0.51 | 0.44 | 0.29 | 0.14 | 0.04 | 0.02 | 0.18 |
| sigreg | occlusion | unadapted | 0.87 | 0.88 | 0.86 | 0.82 | 0.75 | 0.71 | 0.82 |
| sigreg | occlusion | BN-adapted | 0.83 | 0.83 | 0.84 | 0.78 | 0.65 | 0.55 | 0.73 |
| sigreg | cam shift | unadapted | 0.84 | 0.83 | 0.82 | 0.81 | 0.80 | 0.71 | 0.69 |
| sigreg | cam shift | BN-adapted | 0.91 | 0.91 | 0.89 | 0.86 | 0.77 | 0.45 | 0.21 |
| jac | floor 0.1 | BN-adapted | 0.36 | 0.26 | 0.13 | 0.03 | 0.02 | 0.01 | 0.12 |
| jac | floor 0.3 | unadapted | 0.83 | 0.82 | 0.68 | 0.42 | 0.12 | 0.03 | 0.37 |
| jac | floor 0.3 | BN-adapted | 0.74 | 0.67 | 0.47 | 0.21 | 0.05 | 0.02 | 0.28 |
| jac | floor 0.4 | BN-adapted | 0.94 | 0.95 | 0.95 | 0.96 | 0.96 | 0.79 | 0.83 |
| jac | white balance | BN-adapted | 0.36 | 0.25 | 0.12 | 0.03 | 0.02 | 0.01 | 0.12 |
| jac | occlusion | unadapted | 0.88 | 0.87 | 0.85 | 0.80 | 0.56 | 0.16 | 0.48 |
| jac | occlusion | BN-adapted | 0.87 | 0.87 | 0.84 | 0.79 | 0.53 | 0.14 | 0.47 |
| jac | cam shift | unadapted | 0.90 | 0.90 | 0.86 | 0.75 | 0.52 | 0.21 | 0.22 |
| jac | cam shift | BN-adapted | 0.88 | 0.86 | 0.80 | 0.69 | 0.46 | 0.16 | 0.13 |

Table 23. CEM with the BN-adapted world model, 50 box-crossing pairs: collision / safe success / progress. Unadapted = the same margin function and CCEM on the original model (Tables 5, 11).

| WM | Condition | CCEM MLP, unadapted | CCEM MLP, BN-adapted | CCEM linear, unadapted | CCEM linear, BN-adapted | CCEM MLP, BN + safe-q10 | raw CEM, BN-adapted |
|---|---|---|---|---|---|---|---|
| sigreg | floor 0.1 | 0.480 / 0.300 / 0.732 | 0.060 / 0.680 / 0.926 | 0.480 / 0.300 / 0.729 | 0.040 / 0.540 / 0.894 | 0.020 / 0.080 / 0.118 | 0.560 / 0.340 / 0.918 |
| sigreg | floor 0.3 | 0.380 / 0.180 / 0.409 | 0.140 / 0.600 / 0.931 | 0.380 / 0.180 / 0.409 | 0.120 / 0.480 / 0.905 | 0.020 / 0.180 / 0.368 | 0.380 / 0.380 / 0.940 |
| sigreg | floor 0.4 | 0.420 / 0.180 / 0.368 | 0.280 / 0.440 / 0.930 | 0.420 / 0.180 / 0.368 | 0.120 / 0.280 / 0.867 | 0.060 / 0.140 / 0.276 | 0.360 / 0.340 / 0.900 |
| sigreg | light | 0.420 / 0.120 / 0.359 | 0.440 / 0.200 / 0.828 | 0.420 / 0.120 / 0.359 | 0.340 / 0.260 / 0.794 | 0.140 / 0.200 / 0.462 | 0.460 / 0.180 / 0.849 |
| sigreg | exposure | 0.280 / 0.120 / 0.296 | 0.280 / 0.260 / 0.808 | 0.280 / 0.120 / 0.296 | 0.360 / 0.260 / 0.772 | 0.200 / 0.140 / 0.328 | 0.280 / 0.260 / 0.807 |
| sigreg | white balance | 0.600 / 0.100 / 0.324 | 0.240 / 0.380 / 0.873 | 0.600 / 0.100 / 0.324 | 0.160 / 0.460 / 0.865 | 0.040 / 0.220 / 0.638 | 0.380 / 0.320 / 0.879 |
| sigreg | occlusion | 0.280 / 0.180 / 0.553 | 0.220 / 0.260 / 0.675 | 0.260 / 0.180 / 0.460 | 0.180 / 0.260 / 0.565 | 0.180 / 0.140 / 0.443 | 0.340 / 0.220 / 0.676 |
| sigreg | cam shift | 0.160 / 0.080 / 0.268 | 0.500 / 0.140 / 0.788 | 0.160 / 0.080 / 0.268 | 0.280 / 0.160 / 0.530 | 0.200 / 0.080 / 0.057 | 0.680 / 0.180 / 0.748 |
| jac | floor 0.1 | 0.060 / 0.660 / 0.919 | 0.060 / 0.560 / 0.920 | 0.060 / 0.540 / 0.899 | 0.080 / 0.400 / 0.857 | 0.020 / 0.100 / 0.019 | 0.600 / 0.360 / 0.901 |
| jac | floor 0.3 | 0.140 / 0.440 / 0.898 | 0.160 / 0.420 / 0.898 | 0.100 / 0.460 / 0.874 | 0.100 / 0.440 / 0.886 | 0.040 / 0.100 / 0.281 | 0.520 / 0.380 / 0.918 |
| jac | floor 0.4 | 0.240 / 0.280 / 0.646 | 0.240 / 0.300 / 0.725 | 0.220 / 0.260 / 0.662 | 0.220 / 0.280 / 0.713 | 0.180 / 0.140 / 0.341 | 0.260 / 0.260 / 0.703 |
| jac | light | 0.060 / 0.580 / 0.919 | 0.080 / 0.520 / 0.912 | 0.060 / 0.520 / 0.885 | 0.080 / 0.400 / 0.880 | 0.020 / 0.100 / 0.015 | 0.600 / 0.300 / 0.883 |
| jac | exposure | 0.080 / 0.640 / 0.927 | 0.080 / 0.600 / 0.938 | 0.060 / 0.620 / 0.905 | 0.060 / 0.400 / 0.876 | 0.020 / 0.100 / 0.017 | 0.560 / 0.320 / 0.919 |
| jac | white balance | 0.060 / 0.560 / 0.908 | 0.060 / 0.540 / 0.902 | 0.060 / 0.460 / 0.883 | 0.100 / 0.440 / 0.884 | 0.020 / 0.080 / 0.015 | 0.640 / 0.280 / 0.896 |
| jac | occlusion | 0.240 / 0.440 / 0.897 | 0.300 / 0.400 / 0.888 | 0.260 / 0.400 / 0.859 | 0.240 / 0.400 / 0.893 | 0.060 / 0.100 / 0.271 | 0.440 / 0.400 / 0.903 |
| jac | cam shift | 0.320 / 0.160 / 0.736 | 0.240 / 0.240 / 0.763 | 0.220 / 0.200 / 0.659 | 0.100 / 0.160 / 0.486 | 0.040 / 0.060 / 0.040 | 0.620 / 0.160 / 0.728 |

- For sigreg, BN adaptation recovers most of what the shift destroyed whenever the shift is largely photometric: floor 0.1-0.4 and white balance. At floor 0.3 the blur curve goes from flat (0.83-0.89 at every distance) to close to in distribution (0.40 / 0.30 / 0.18 / 0.11 / 0.05), and CCEM-MLP goes from 0.380 collision / 0.180 safe success to 0.140 / 0.600, which is jac's level. Raw CEM on the adapted sigreg also reaches the goal again (progress 0.41 to 0.94): the predictor's inputs are corrected too, unlike with an offset on $h$.
- Correction to the blur reading (Table 6c): sigreg's "wrong state" under small floor shifts was mostly a recoverable per-dimension offset in the projector, not lost information, since re-estimating two BatchNorm layers on unlabeled frames brings the state resolution back. The resolution loss that adaptation cannot undo is what remains after it: floor 0.4 (partial), lighting and exposure (little CEM gain), occlusion and camera shift (none, as rotation on Dubins).
- For jac, adaptation adds nothing: jac's representation of these shifts is already close to unshifted, and where jac fails (floor 0.4, occlusion, camera shift) the failure is not an offset (CCEM changes within noise; at floor 0.4 the curve stays flat).
- The safe-q10 threshold stalls (safe success 0.06-0.22, progress 0.02-0.64): after adaptation $h$ is already about right, and the extra margin vetoes the detours, as the real-frame recalibration did (Table 15). Threshold 0 is the right setting on Reacher.
- Opposite outcome to Dubins, where the same adaptation doubled CCEM collisions under purple: there the unadapted $h$ called nearly every plan unsafe and CCEM's conservative fallback was itself the safety; on Reacher the unadapted $h$ calls everything safe (CCEM = raw CEM), so correcting it can only help.

## jac+adv_h at epoch 10

`rc_jac_safeadv_linf` finished (epoch 10, same seed and schedule as jac); full suite through `queue_ep10_eval_v2.sh`. Training prediction loss is the same as jac's (epoch 10: 0.0220 vs 0.0218); the adversary was on from epoch 2 (in-training head AUC 0.9993-0.9997).

Table 24. Floor sweep, bal acc (AUC), real+imag margin functions: jac vs jac+adv_h.

| WM | Margin fn | 0.2 | 0.3 | 0.4 | 0.5 | 1 |
|---|---|---|---|---|---|---|
| jac | MLP | 0.997 (0.999) | 0.500 (0.873) | 0.500 (0.667) | 0.500 (0.553) | 0.500 (0.566) |
| jac+adv_h | MLP | 0.997 (0.999) | 0.500 (0.986) | 0.500 (0.747) | 0.500 (0.565) | 0.500 (0.436) |
| jac | linear | 0.990 (0.999) | 0.578 (0.952) | 0.500 (0.402) | 0.500 (0.465) | 0.500 (0.699) |
| jac+adv_h | linear | 0.990 (0.999) | 0.517 (0.993) | 0.500 (0.896) | 0.500 (0.734) | 0.500 (0.425) |

Table 25. Blur curve (as Table 18), MLP: safe-plan veto at 90% unsafe caught, by closest distance to the box (rad).

| WM | Condition | < 0.05 | 0.05-0.1 | 0.1-0.2 | 0.2-0.3 | 0.3-0.5 | 0.5-1 | > 1 |
|---|---|---|---|---|---|---|---|---|
| jac | floor 0.3 | 0.83 | 0.82 | 0.68 | 0.42 | 0.12 | 0.03 | 0.37 |
| jac+adv_h | floor 0.3 | 0.54 | 0.44 | 0.28 | 0.13 | 0.07 | 0.02 | 0.31 |
| jac | camera shift | 0.90 | 0.90 | 0.86 | 0.75 | 0.52 | 0.21 | 0.22 |
| jac+adv_h | camera shift | 0.77 | 0.73 | 0.61 | 0.42 | 0.25 | 0.12 | 0.15 |
| jac | occlusion | 0.88 | 0.87 | 0.85 | 0.80 | 0.56 | 0.16 | 0.48 |
| jac+adv_h | occlusion | 0.80 | 0.76 | 0.78 | 0.73 | 0.47 | 0.10 | 0.43 |

Table 26. CEM, 50 box-crossing pairs: collision / safe success / progress, jac vs jac+adv_h (epoch 10, same margin-function recipe).

| Condition | jac CCEM-MLP | jac+adv_h CCEM-MLP | jac CCEM-linear | jac+adv_h CCEM-linear | jac raw | jac+adv_h raw |
|---|---|---|---|---|---|---|
| in distribution | 0.06 / 0.58 / 0.93 | 0.04 / 0.52 / 0.91 | 0.06 / 0.44 / 0.90 | 0.08 / 0.54 / 0.89 | 0.58 / 0.36 / 0.94 | 0.58 / 0.36 / 0.93 |
| floor 0.3 | 0.14 / 0.44 / 0.90 | 0.14 / 0.36 / 0.83 | 0.10 / 0.46 / 0.87 | 0.18 / 0.28 / 0.81 | 0.34 / 0.50 / 0.89 | 0.26 / 0.38 / 0.88 |
| floor 0.4 | 0.24 / 0.28 / 0.65 | 0.20 / 0.10 / 0.28 | 0.22 / 0.26 / 0.66 | 0.16 / 0.12 / 0.27 | 0.24 / 0.24 / 0.66 | 0.20 / 0.10 / 0.28 |
| camera shift | 0.32 / 0.16 / 0.74 | 0.16 / 0.22 / 0.77 | 0.22 / 0.20 / 0.66 | 0.14 / 0.22 / 0.52 | 0.64 / 0.16 / 0.73 | 0.60 / 0.16 / 0.68 |
| camera tilt | 0.44 / 0.10 / 0.44 | 0.26 / 0.12 / 0.50 | 0.34 / 0.12 / 0.45 | 0.28 / 0.14 / 0.48 | 0.56 / 0.10 / 0.56 | 0.52 / 0.12 / 0.62 |
| occlusion | 0.24 / 0.44 / 0.90 | 0.24 / 0.42 / 0.88 | 0.26 / 0.40 / 0.86 | 0.20 / 0.38 / 0.82 | 0.34 / 0.36 / 0.88 | 0.30 / 0.40 / 0.88 |
| photometric (light dir, noise, jpeg, exposure, white balance) | 0.04-0.08 / 0.56-0.66 | 0.04-0.06 / 0.52-0.66 | 0.06 / 0.44-0.62 | 0.06 / 0.48-0.56 | 0.50-0.58 | 0.54-0.62 |

- The adversarial reg sharpens the margin function's boundary under shift: at floor 0.3 the blur curve's 50% point moves from about 0.25 rad (jac) to about 0.05 rad, close to in distribution, and under camera shift by about 0.15 rad; the ranking survives further (floor 0.4 AUC 0.747 / 0.896 vs 0.667 / 0.402). This is stronger than at epoch 5 (Table 18).
- In planning this pays off under camera geometry (CCEM-MLP collision 0.32 to 0.16 under camera shift, 0.44 to 0.26 under tilt, at equal or better safe success); elsewhere CCEM is within noise of jac.
- But jac+adv_h plans worse under large floor shifts, regardless of safety: at floor 0.4 even raw CEM's progress falls to 0.28 (jac 0.66), so safe success drops to 0.10. The adversary makes $h$ invariant, not the goal cost; under a large floor shift the goal latent and the planner's distance to it are off. One seed, $n = 50$.

### BatchNorm adaptation on jac+adv_h (epoch 10)

Same recipe (2000 safe frames, threshold 0) on `rc_jac_safeadv_linf`, for the conditions where it fails or plans badly.

Table 27. jac+adv_h, unadapted → BN-adapted. Collision / safe success / progress; blur curve = safe-plan veto at 90% unsafe caught at < 0.05 / 0.1-0.2 / 0.2-0.3 / 0.3-0.5 rad (MLP).

| Condition | CCEM-MLP | CCEM-linear | raw CEM | blur curve, unadapted → adapted |
|---|---|---|---|---|
| floor 0.3 | 0.14 / 0.36 / 0.83 → 0.10 / 0.52 / 0.91 | 0.18 / 0.28 / 0.81 → 0.08 / 0.48 / 0.88 | 0.26 / 0.38 / 0.88 → 0.36 / 0.48 / 0.92 | 0.54 / 0.28 / 0.13 / 0.07 → 0.81 / 0.74 / 0.59 / 0.23 |
| floor 0.4 | 0.20 / 0.10 / 0.28 → 0.36 / 0.14 / 0.61 | 0.16 / 0.12 / 0.27 → 0.32 / 0.14 / 0.42 | 0.20 / 0.10 / 0.28 → 0.36 / 0.14 / 0.64 | n/a → 0.87 / 0.83 / 0.75 / 0.68 |
| camera shift | 0.16 / 0.22 / 0.77 → 0.16 / 0.16 / 0.67 | 0.14 / 0.22 / 0.52 → 0.18 / 0.18 / 0.27 | 0.60 / 0.16 / 0.68 → 0.68 / 0.16 / 0.63 | 0.77 / 0.61 / 0.42 / 0.25 → 0.75 / 0.62 / 0.44 / 0.27 |
| camera tilt | 0.26 / 0.12 / 0.50 → 0.36 / 0.16 / 0.41 | 0.28 / 0.14 / 0.48 → 0.26 / 0.12 / 0.37 | 0.52 / 0.12 / 0.62 → 0.58 / 0.10 / 0.49 | n/a → 0.91 / 0.94 / 0.93 / 0.83 |
| occlusion | 0.24 / 0.42 / 0.88 → 0.18 / 0.36 / 0.86 | 0.20 / 0.38 / 0.82 → 0.14 / 0.34 / 0.85 | 0.30 / 0.40 / 0.88 → 0.30 / 0.34 / 0.85 | 0.80 / 0.78 / 0.73 / 0.47 → 0.82 / 0.80 / 0.77 / 0.55 |

- Adaptation repairs jac+adv_h's goal reaching under floor shifts (raw CEM progress at floor 0.4: 0.28 → 0.64; at 0.3 safe success 0.36-0.28 → 0.52-0.48 with lower collision), so the floor-0.4 planning failure (Table 26) was an offset.
- But it blurs adv_h's safety boundary: at floor 0.3 the curve moves from near in distribution back to jac-like (0.54 → 0.81 at < 0.05 rad), and at floor 0.4 collision rises (0.20 → 0.36). The margin function was trained on the unadapted latent, and adv_h's invariance already kept that latent near its training position; re-centering the BatchNorm moves it away from what $h$ reads.
- So for adv_h, BN adaptation helps the goal cost and hurts the safety readout. Camera geometry: no gain or worse, as everywhere.

### Which BatchNorm set, and a split planner

Per-layer test (`reacher_bn_adapt.py BNSETS=proj|pred`) and a split CCEM (`H_WM`: goal cost and rollout from the adapted model, the feasibility check from $h$ on the unadapted model's rollout of the same candidates; with both models equal it reproduces the single-model run exactly). jac+adv_h epoch 10, collision / safe success / progress; blur = safe-plan veto at < 0.05 rad (90% unsafe caught, MLP).

Table 28. jac+adv_h under floor shifts.

| Condition | Margin fn | None | Both sets | Projector only | Predictor only | Split (both-sets planner) | Split (projector planner) |
|---|---|---|---|---|---|---|---|
| floor 0.3 | MLP | 0.14 / 0.36 / 0.83 | 0.10 / 0.52 / 0.91 | 0.14 / 0.66 / 0.92 | 0.16 / 0.28 / 0.79 | 0.14 / 0.60 / 0.94 | 0.20 / 0.56 / 0.93 |
| floor 0.3 | linear | 0.18 / 0.28 / 0.81 | 0.08 / 0.48 / 0.88 | 0.10 / 0.52 / 0.89 | 0.14 / 0.30 / 0.78 | 0.14 / 0.60 / 0.93 | 0.18 / 0.52 / 0.92 |
| floor 0.4 | MLP | 0.20 / 0.10 / 0.28 | 0.36 / 0.14 / 0.61 | 0.38 / 0.24 / 0.73 | 0.34 / 0.14 / 0.37 | 0.34 / 0.18 / 0.66 | 0.36 / 0.26 / 0.76 |
| floor 0.4 | linear | 0.16 / 0.12 / 0.27 | 0.32 / 0.14 / 0.42 | 0.38 / 0.28 / 0.70 | 0.26 / 0.10 / 0.22 | 0.20 / 0.16 / 0.57 | 0.22 / 0.24 / 0.65 |
| floor 0.3 | blur at < 0.05 rad | 0.54 | 0.81 | 0.84 | 0.57 | (unadapted $h$) | (unadapted $h$) |

- The projector BatchNorm carries both effects: adapting it alone repairs goal reaching (the best safe success at floor 0.3, 0.66) and blurs $h$ like adapting both; the predictor projection alone keeps $h$ sharp but does not repair planning. So the planning fix and the safety loss sit in the same encoder latent.
- The split planner keeps the adapted model's progress (floor 0.3: 0.92-0.94; floor 0.4: 0.57-0.76 vs 0.27-0.28 unadapted) with the unadapted, sharper $h$, but collision does not fall below the unadapted run (floor 0.3: 0.14-0.20 vs 0.14-0.18). A planner that reaches the goal takes more near-box routes, which exercises $h$'s remaining false-safes; part of the unadapted run's low collision was not getting anywhere. Best overall at floor 0.3: split with the linear margin function (0.14 / 0.60 / 0.93). One seed, $n = 50$.

### Paired tests vs jac

All CEM runs use the same 50 box-crossing pairs, so outcomes are paired per episode. Two-sided exact McNemar on collision and on safe success, against jac (epoch 10, same margin-function recipe and planner).

Table 29. jac → other: rate (episodes better for jac / better for the other model, p).

| Comparison | Condition | Margin fn | Collision | Safe success |
|---|---|---|---|---|
| jac+adv_h | floor 0.3 | MLP | 0.14 → 0.14 (3 / 3, 1.00) | 0.44 → 0.36 (12 / 8, 0.50) |
| jac+adv_h | floor 0.3 | linear | 0.10 → 0.18 (1 / 5, 0.22) | 0.46 → 0.28 (13 / 4, 0.049) |
| jac+adv_h | floor 0.4 | MLP | 0.24 → 0.20 (5 / 3, 0.73) | 0.28 → 0.10 (9 / 0, 0.004) |
| jac+adv_h | floor 0.4 | linear | 0.22 → 0.16 (6 / 3, 0.51) | 0.26 → 0.12 (8 / 1, 0.039) |
| jac+adv_h | camera shift | MLP | 0.32 → 0.16 (11 / 3, 0.057) | 0.16 → 0.22 (3 / 6, 0.51) |
| jac+adv_h | camera tilt | MLP | 0.44 → 0.26 (14 / 5, 0.064) | 0.10 → 0.12 (2 / 3, 1.00) |
| jac+adv_h | occlusion | MLP | 0.24 → 0.24 (5 / 5, 1.00) | 0.44 → 0.42 (8 / 7, 1.00) |
| jac+adv_h | in distribution | MLP | 0.06 → 0.04 (1 / 0, 1.00) | 0.58 → 0.52 (6 / 3, 0.51) |
| jac+adv_h + BN split | floor 0.3 | MLP | 0.14 → 0.14 (3 / 3, 1.00) | 0.44 → 0.60 (4 / 12, 0.077) |
| jac+adv_h + BN split | floor 0.4 | MLP | 0.24 → 0.34 (5 / 10, 0.30) | 0.28 → 0.18 (8 / 3, 0.23) |
| jac+adv_h + BN split | camera shift | MLP | 0.32 → 0.14 (12 / 3, 0.035) | 0.16 → 0.20 (5 / 7, 0.77) |
| jac+adv_h + BN split | occlusion | MLP | 0.24 → 0.22 (4 / 3, 1.00) | 0.44 → 0.42 (8 / 7, 1.00) |

- No general OOD improvement over jac in planning. The only significant gain is lower collision under camera shift for adv_h with the BN split planner (p = 0.035), with adv_h alone borderline under camera shift and tilt (p ≈ 0.06); about 20 tests were run here, so one p = 0.035 needs replication. adv_h significantly lowers safe success under large floor shifts (floor 0.4: p = 0.004 / 0.039; floor 0.3 linear: p = 0.049), which BN adaptation repairs only to jac's level.
- The claim that holds is diagnostic: adv_h sharpens the margin function's boundary under shift (Tables 24-25, 12,800 plans per condition), one training seed. Planning claims need more seeds and more CEM pairs.

### 150 paired episodes

The 50-pair tests left camera shift / tilt and the floor results borderline, so the key conditions were rerun on a new, larger draw of 150 box-crossing pairs (the same 150 for every model), CCEM-minh, epoch-10 models, real+imag margin functions. Each cell: collision / safe success / progress, then paired exact McNemar against jac (episodes better for jac / better for the other model, p). Split = jac+adv_h with the BN-adapted (both sets) planner and $h$ on the unadapted rollout (Table 28).

Table 30.

| Condition | Margin fn | jac | jac+adv_h | jac+adv_h + BN split |
|---|---|---|---|---|
| cam shift | MLP | 0.427 / 0.207 / 0.85 | 0.207 / 0.207 / 0.78 | coll 47/14 p=0.000; safe-succ 21/21 p=1.000 | 0.193 / 0.193 / 0.76 | coll 49/14 p=0.000; safe-succ 22/20 p=0.878 |
| cam shift | linear | 0.253 / 0.133 / 0.73 | 0.147 / 0.173 / 0.69 | coll 29/13 p=0.020; safe-succ 15/21 p=0.405 | 0.133 / 0.233 / 0.72 | coll 28/10 p=0.005; safe-succ 12/27 p=0.024 |
| cam tilt | MLP | 0.373 / 0.087 / 0.54 | 0.313 / 0.120 / 0.57 | coll 36/27 p=0.314; safe-succ 11/16 p=0.442 | 0.333 / 0.120 / 0.60 | coll 33/27 p=0.519; safe-succ 9/14 p=0.405 |
| cam tilt | linear | 0.293 / 0.067 / 0.48 | 0.220 / 0.127 / 0.48 | coll 29/18 p=0.144; safe-succ 6/15 p=0.078 | 0.220 / 0.120 / 0.44 | coll 26/15 p=0.117; safe-succ 6/14 p=0.115 |
| floor 0.3 | MLP | 0.233 / 0.447 / 0.91 | 0.140 / 0.327 / 0.84 | coll 22/8 p=0.016; safe-succ 35/17 p=0.018 | 0.267 / 0.500 / 0.94 | coll 13/18 p=0.473; safe-succ 29/37 p=0.389 |
| floor 0.3 | linear | 0.133 / 0.460 / 0.90 | 0.113 / 0.260 / 0.83 | coll 10/7 p=0.629; safe-succ 48/18 p=0.000 | 0.220 / 0.520 / 0.93 | coll 6/19 p=0.015; safe-succ 26/35 p=0.306 |
| floor 0.4 | MLP | 0.340 / 0.100 / 0.57 | 0.227 / 0.060 / 0.27 | coll 27/10 p=0.008; safe-succ 12/6 p=0.238 | 0.473 / 0.160 / 0.71 | coll 13/33 p=0.005; safe-succ 12/21 p=0.163 |
| floor 0.4 | linear | 0.320 / 0.100 / 0.56 | 0.213 / 0.067 / 0.27 | coll 25/9 p=0.009; safe-succ 11/6 p=0.332 | 0.353 / 0.113 / 0.69 | coll 17/22 p=0.522; safe-succ 13/15 p=0.851 |

- Under camera shift, jac+adv_h beats jac: collision halves (MLP 0.427 → 0.207, 47 / 14 episodes, p < 0.001; linear 0.253 → 0.147, p = 0.020) at the same safe success; with the split planner and the linear margin function safe success also rises (0.133 → 0.233, p = 0.024). One training seed; two margin functions and two planners agree.
- Under floor shifts adv_h is a more conservative operating point, not a better one: fewer collisions (floor 0.3 MLP p = 0.016; floor 0.4 p ≈ 0.008) because it reaches the goal less (floor 0.3 safe success p = 0.018 / < 0.001; floor 0.4 progress 0.27 vs 0.56). The split restores progress but gives the collisions back (floor 0.3 linear p = 0.015, floor 0.4 MLP p = 0.005 worse than jac).
- Camera tilt trends the same way as shift (collision 0.373 → 0.313, 0.293 → 0.220) without reaching significance.
- jac's own rates differ from the 50-pair table (camera shift MLP collision 0.427 vs 0.320): a different, larger draw of pairs.

### When does adv_h help? (both benchmarks)

`adv_align.py`: jac vs jac+adv_h on 400 held-out states per shift (Reacher epoch 10; Dubins `jacobian_w1_dubins` vs `lewm_dubins_jac_safeadv_linf50`, MLP real+imag margin functions). Alignment = fraction of jac's latent displacement under the shift that lies in the top-10 directions of jac's own 8/255 PGD displacements (random directions: 0.052; the PGD displacements themselves: 0.53 Reacher, 0.72 Dubins). Change in $h$ = median $|\Delta h|$ / safe-unsafe gap; latent shift = mean $\lVert \Delta z \rVert$ / spread. Pixel columns: mean $|\Delta x|$ / median per-image max / fraction of pixels beyond 8/255 / high-frequency energy share (the PGD perturbation itself: 0.02-0.03 / 0.031 / 0 / 0.88-0.96).

Table 31.

| Benchmark | Shift | Alignment (jac) | Change in $h$: jac → adv_h (ratio) | Latent shift: jac → adv_h (ratio) | Pixel mean / max / frac > 8/255 / hi-freq |
|---|---|---|---|---|---|
| Reacher | floor 0.1 | 0.239 | 0.002 -> 0.001 (0.42) | 0.035 -> 0.022 (0.62) | 0.028 / 0.067 / 0.916 / 0.018 |
| Reacher | floor 0.2 | 0.208 | 0.010 -> 0.004 (0.36) | 0.112 -> 0.096 (0.86) | 0.057 / 0.129 / 0.917 / 0.016 |
| Reacher | floor 0.3 | 0.248 | 0.325 -> 0.423 (1.30) | 0.764 -> 0.828 (1.08) | 0.084 / 0.196 / 0.917 / 0.016 |
| Reacher | floor 0.4 | 0.172 | 0.860 -> 0.721 (0.84) | 1.126 -> 1.042 (0.93) | 0.113 / 0.259 / 0.935 / 0.016 |
| Reacher | light | 0.273 | 0.015 -> 0.003 (0.18) | 0.138 -> 0.073 (0.53) | 0.064 / 0.192 / 0.995 / 0.011 |
| Reacher | exposure | 0.241 | 0.078 -> 0.003 (0.04) | 0.346 -> 0.079 (0.23) | 0.132 / 0.149 / 1.000 / 0.003 |
| Reacher | white balance | 0.228 | 0.004 -> 0.002 (0.48) | 0.075 -> 0.055 (0.73) | 0.050 / 0.200 / 1.000 / 0.008 |
| Reacher | noise | 0.175 | 0.001 -> 0.000 (0.64) | 0.015 -> 0.014 (0.90) | 0.031 / 0.180 / 0.782 / 0.985 |
| Reacher | blur | 0.173 | 0.003 -> 0.004 (1.36) | 0.065 -> 0.080 (1.23) | 0.007 / 0.376 / 0.086 / 0.966 |
| Reacher | cam zoom | 0.431 | 0.011 -> 0.010 (0.88) | 0.278 -> 0.278 (1.00) | 0.024 / 0.843 / 0.172 / 0.574 |
| Reacher | cam shift | 0.211 | 0.200 -> 0.099 (0.49) | 0.802 -> 0.718 (0.89) | 0.061 / 0.839 / 0.344 / 0.229 |
| Reacher | cam tilt | 0.205 | 0.340 -> 0.337 (0.99) | 0.991 -> 1.001 (1.01) | 0.076 / 0.847 / 0.479 / 0.166 |
| Reacher | occlusion | 0.263 | 0.086 -> 0.121 (1.41) | 0.646 -> 0.758 (1.17) | 0.007 / 0.365 / 0.046 / 0.106 |
| Dubins | purple | 0.259 | 0.343 -> 0.446 (1.30) | 1.116 -> 1.202 (1.08) | 0.059 / 0.545 / 0.185 / 0.065 |
| Dubins | diamond | 0.376 | 0.040 -> 0.007 (0.18) | 0.190 -> 0.168 (0.88) | 0.043 / 1.000 / 0.082 / 0.393 |
| Dubins | rot90 | 0.222 | 0.145 -> 0.149 (1.03) | 1.380 -> 1.395 (1.01) | 0.200 / 1.000 / 0.320 / 0.093 |

- Alignment with the adversary's directions does not predict the benefit: every real shift sits at 0.17-0.43, and shifts with the same alignment go opposite ways (camera shift and tilt both 0.21: helped / not; exposure 0.24 helped 25x, floor 0.3 0.25 worse). Real shifts also look nothing like the adversary (small, high-frequency): photometric shifts are dense and low-frequency, camera shifts sparse large edits.
- What predicts it is how far the shift already moves jac's $h$. Where it moves $h$ by up to about 0.2 of the gap, adv_h shrinks that 2-25x (exposure, light, white balance, floor ≤ 0.2, camera shift, Dubins diamond). Where it moves $h$ by about 0.3 of the gap or more, adv_h does nothing or makes it worse (floor 0.3 / 0.4, camera tilt, occlusion, Dubins purple and rotation): adv_h's robustness is local, consistent with jac breaking by shift size, not direction.
- A planning gain needs both headroom and reach. If jac is already good (exposure, light, white balance, small floor shifts, Dubins diamond), adv_h still improves $h$, but CCEM is already at its in-distribution level, so planning cannot improve. If the shift is too big for jac (floor ≥ 0.3, tilt, occlusion, Dubins purple, rotation), adv_h cannot improve $h$. Only a shift in between pays off: Reacher camera shift (jac's $h$ moves 0.20 of the gap, enough to cost collisions; adv_h halves it), which is exactly where adv_h beat jac in CCEM (Table 30). Dubins has no shift in that band, which is why adv_h never helped planning there. Under camera shift adv_h redirects the displacement rather than shrinking it (latent shift ×0.89, change in $h$ ×0.49).
- One training seed per model; the boundary (about 0.2-0.3 of the gap) is read off 16 shifts, not fitted.

## Varying adversary radius (jac+adv_h, radius log-uniform in [4/255, 32/255] per sequence)

`rc_jac_safeadv_linf_epsmix` (snapshot `code_point_92e7589_epsmix`, kwarg `eps_pix_max`): identical to jac+adv_h except that each training sequence's L∞ radius is drawn fresh every step. Prediction loss equals jac's (0.0220 vs 0.0218 at epoch 10). Reach (`adv_align.py MODEL2=...`, change in $h$ / gap, jac → model): floor 0.3 0.325 → 0.007 (fixed-radius adv_h 0.423), floor 0.4 0.860 → 0.075 (0.721), occlusion 0.086 → 0.022 (0.121), camera shift 0.200 → 0.107 (0.099), camera tilt 0.340 → 0.505 (0.337).

Table 32. 150 paired episodes, collision / safe success (progress); exact McNemar vs jac.

| Condition | Margin fn | jac | jac+adv_h | jac + whole-latent adv | jac+adv_h, varying radius | varying vs jac |
|---|---|---|---|---|---|---|
| floor 0.3 | MLP | 0.233 / 0.447 | 0.140 / 0.327 | 0.113 / 0.460 | 0.040 / 0.673 (0.94) | collision 29 / 0, p < 0.001; safe success 11 / 45, p < 0.001 |
| floor 0.3 | linear | 0.133 / 0.460 | 0.113 / 0.260 | 0.033 / 0.287 | 0.033 / 0.573 (0.92) | collision 15 / 0, p < 0.001; safe success p = 0.024 |
| floor 0.4 | MLP | 0.340 / 0.100 | 0.227 / 0.060 | 0.327 / 0.160 | 0.087 / 0.653 (0.94) | collision 43 / 5, p < 0.001; safe success 2 / 85, p < 0.001 |
| floor 0.4 | linear | 0.320 / 0.100 | 0.213 / 0.067 | 0.300 / 0.153 | 0.060 / 0.587 (0.92) | collision 42 / 3, p < 0.001; safe success 1 / 74, p < 0.001 |
| camera shift | MLP | 0.427 / 0.207 | 0.207 / 0.207 | 0.160 / 0.313 | 0.287 / 0.193 (0.66) | collision 37 / 16, p = 0.005 |
| camera shift | linear | 0.253 / 0.133 | 0.147 / 0.173 | 0.087 / 0.180 | 0.160 / 0.167 (0.63) | collision p = 0.038 |
| camera tilt | MLP | 0.373 / 0.087 | 0.313 / 0.120 | 0.240 / 0.140 | 0.480 / 0.047 (0.51) | collision 27 / 43, p = 0.072 (worse) |
| camera tilt | linear | 0.293 / 0.067 | 0.220 / 0.127 | 0.227 / 0.127 | 0.420 / 0.067 (0.40) | collision 20 / 39, p = 0.018 (worse) |

- The reach analysis predicted planning: the shifts the varying radius brought within reach (floor 0.3 / 0.4, occlusion) improve, at both collision and safe success, not as a trade. At floor 0.4 safe success rises from 0.10 to 0.59-0.65, close to the in-distribution level (0.58), and progress returns to 0.92-0.94: making the latent nearly invariant to the floor also fixes the goal cost, so fixed-radius adv_h's floor trade-off disappears. 50 pairs, occlusion: collision 0.24 → 0.10, safe success 0.44 → 0.62.
- The camera-shift gain survives but is smaller than fixed adv_h's or the whole-latent model's; camera tilt, the one shift the reach analysis showed getting worse, is worse than jac in planning too.
- No in-distribution cost (50 pairs, CCEM-MLP 0.04 / 0.64 vs jac 0.06 / 0.58), and at jac's level or better under every photometric, sensor and dynamics shift. One training seed.

## Varying-radius whole latent (jac + whole-latent adversary, radius log-uniform in [4/255, 32/255])

`rc_jac_safeadv_latent_epsmix` (snapshot `code_point_92e7589_epsmix`): the whole-latent target of `rc_jac_safeadv_latent` with the varying radius of `rc_jac_safeadv_linf_epsmix`. Reach (`adv_align.py MODEL2=...`, 400 states, change in $h$ / gap, jac → fixed whole latent / varying whole latent): floor 0.3 0.325 → 0.058 / 0.008, floor 0.4 0.860 → 0.417 / 0.262, occlusion 0.086 → 0.093 / 0.019, camera shift 0.200 → 0.104 / 0.123, camera tilt 0.340 → 0.407 / 0.309, camera zoom 0.011 → 0.019 / 0.017, exposure 0.078 → 0.005 / 0.003 (logs `adv_align_rc_jac_safeadv_latent{,_epsmix}.log`).

Table 35. 150 paired episodes (same pairs as Table 32), collision / safe success; exact McNemar vs jac (episodes better for jac / better for the model).

| Condition | Margin fn | jac | Varying whole latent | Collision | Safe success |
|---|---|---|---|---|---|
| floor 0.3 | MLP | 0.233 / 0.447 | 0.040 / 0.653 | 29 / 0, p < 0.001 | 10 / 41, p < 0.001 |
| floor 0.3 | linear | 0.133 / 0.460 | 0.033 / 0.427 | 15 / 0, p < 0.001 | 27 / 22, p = 0.568 |
| floor 0.4 | MLP | 0.340 / 0.100 | 0.200 / 0.333 | 35 / 14, p = 0.004 | 7 / 42, p < 0.001 |
| floor 0.4 | linear | 0.320 / 0.100 | 0.073 / 0.160 | 41 / 4, p < 0.001 | 10 / 19, p = 0.136 |
| camera shift | MLP | 0.427 / 0.207 | 0.233 / 0.140 | 43 / 14, p < 0.001 | 24 / 14, p = 0.143 |
| camera shift | linear | 0.253 / 0.133 | 0.107 / 0.127 | 31 / 9, p = 0.001 | 13 / 12, p = 1.000 |
| camera tilt | MLP | 0.373 / 0.087 | 0.280 / 0.133 | 36 / 22, p = 0.087 | 10 / 17, p = 0.248 |
| camera tilt | linear | 0.293 / 0.067 | 0.140 / 0.140 | 34 / 11, p = 0.001 | 7 / 18, p = 0.043 |

- The combination only half works. At floor 0.3 it matches varying-radius adv_h (MLP 0.040 / 0.653 vs 0.040 / 0.673); at floor 0.4 it gets about half of that model's gain (MLP 0.200 / 0.333 vs 0.087 / 0.653), as its reach there is smaller (change in $h$ cut to 0.30 of jac's vs 0.09); under camera shift it sits between the fixed whole latent and varying adv_h (collision 0.233 vs 0.160 / 0.287). No variant is best on every shift.
- Camera tilt breaks the reach account. Neither whole-latent model brings tilt within reach (change in $h$ 0.407 / 0.309 of the gap vs jac's 0.340), yet both collide less than jac under tilt: fixed whole latent MLP 0.373 → 0.240 (39 / 19, p = 0.012), varying whole latent linear 0.293 → 0.140 (p = 0.001), with safe success up (linear 0.067 → 0.140, p = 0.043). Varying-radius adv_h, whose reach under tilt is also out, is worse than jac there. So the tilt gain comes from somewhere the static real-frame measure does not see (candidates: the goal latent or the predicted latents CCEM reads, both of which the whole-latent target also constrains). Not yet tested.
- One training seed; numbers for the varying whole latent are on the 150 pairs of Table 32 (pair seed and CEM seed identical).

## Can the too-big shifts be detected?

Beyond some size every model's margin function calls every state safe (floor 0.5 and up for all six models, Table 3, Table 5), and CCEM degenerates to raw CEM. `reacher_detect.py` asks whether those failures announce themselves, with two label-free detectors that need only unlabeled in-distribution data. 1000 held-out states (half unsafe), each with the frame one action block (5 env steps, the recorded actions) later, both rendered under the shift; real+imag MLP margin function.

- Support: mean distance from $z$ to its $k = 10$ nearest neighbours among 20000 training latents (dataset frames, episodes < 9000).
- Surprise: the world model's own one-block prediction error, $\lVert \hat z_{t+5}(z_t, a_{t:t+5}) - z_{t+5} \rVert$, 1-frame history.

Each detector's threshold is the 99th percentile on clean frames of a calibration half of the states; rates are on the other half (500 states), so the clean false-alarm rate is about 0.01 per detector. Caught = the fraction of the margin function's false-safe failures (unsafe frames with $h \ge 0$) that the detector flags; "-" = no false-safe failure.

Table 33. Support detector: margin function bal acc / fraction of shifted frames flagged / fraction of false-safe failures caught.

| Condition | sigreg | official | jac | jac+adv_h | adv_h varying radius | whole latent |
|---|---|---|---|---|---|---|
| clean | 0.992 / 0.010 / - | 0.992 / 0.018 / - | 0.990 / 0.012 / - | 0.994 / 0.006 / - | 0.994 / 0.014 / - | 0.996 / 0.016 / - |
| floor 0.1 | 0.500 / 1.000 / 1.000 | 0.704 / 0.860 / 1.000 | 0.990 / 0.014 / - | 0.994 / 0.006 / - | 0.994 / 0.012 / - | 0.996 / 0.016 / - |
| floor 0.2 | 0.500 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 | 0.994 / 0.084 / - | 0.996 / 0.046 / - | 0.994 / 0.018 / - | 0.996 / 0.012 / - |
| floor 0.3 | 0.500 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 | 0.994 / 0.138 / - | 0.846 / 0.768 / 0.961 |
| floor 0.4 | 0.500 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 | 0.632 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 |
| floor 0.5 | 0.500 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 |
| floor 0.75 | 0.500 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 |
| floor 1 | 0.500 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 | 0.465 / 1.000 / 1.000 |
| light | 0.500 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 | 0.992 / 0.288 / - | 0.990 / 0.020 / - | 0.994 / 0.030 / - | 0.998 / 0.014 / - |
| light dir | 0.500 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 | 0.990 / 0.010 / - | 0.996 / 0.008 / - | 0.994 / 0.014 / - | 0.996 / 0.010 / - |
| exposure | 0.500 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 | 0.745 / 1.000 / 1.000 | 0.992 / 0.026 / - | 0.994 / 0.094 / - | 0.994 / 0.012 / 0.000 |
| white balance | 0.500 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 | 0.990 / 0.036 / - | 0.992 / 0.010 / - | 0.992 / 0.016 / - | 0.994 / 0.016 / - |
| noise | 0.767 / 0.856 / 0.983 | 0.998 / 0.058 / - | 0.990 / 0.014 / - | 0.994 / 0.006 / - | 0.994 / 0.012 / - | 0.996 / 0.016 / - |
| blur | 0.836 / 0.950 / 1.000 | 0.994 / 0.126 / 0.000 | 0.990 / 0.016 / - | 0.992 / 0.024 / - | 0.994 / 0.014 / - | 0.996 / 0.014 / - |
| jpeg | 0.500 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 | 0.990 / 0.018 / - | 0.992 / 0.016 / - | 0.994 / 0.018 / - | 0.996 / 0.014 / - |
| occlusion | 0.500 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 | 0.561 / 0.908 / 0.995 | 0.545 / 0.914 / 1.000 | 0.747 / 0.606 / 0.992 | 0.686 / 0.688 / 0.949 |
| camera zoom | 0.500 / 1.000 / 1.000 | 0.656 / 0.978 / 1.000 | 0.899 / 0.106 / 0.040 | 0.897 / 0.144 / 0.176 | 0.905 / 0.092 / 0.021 | 0.901 / 0.018 / 0.000 |
| camera shift | 0.500 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 | 0.506 / 1.000 / 1.000 | 0.587 / 1.000 / 1.000 | 0.555 / 0.998 / 1.000 | 0.684 / 0.918 / 0.897 |
| camera tilt | 0.500 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 | 0.514 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 | 0.500 / 0.992 / 0.984 | 0.551 / 0.976 / 0.946 |

Table 34. Surprise detector, same layout.

| Condition | sigreg | official | jac | jac+adv_h | adv_h varying radius | whole latent |
|---|---|---|---|---|---|---|
| clean | 0.992 / 0.012 / - | 0.992 / 0.012 / - | 0.990 / 0.012 / - | 0.994 / 0.012 / - | 0.994 / 0.014 / - | 0.996 / 0.020 / - |
| floor 0.1 | 0.500 / 0.374 / 0.372 | 0.704 / 0.014 / 0.000 | 0.990 / 0.018 / - | 0.994 / 0.012 / - | 0.994 / 0.014 / - | 0.996 / 0.020 / - |
| floor 0.2 | 0.500 / 1.000 / 1.000 | 0.500 / 0.224 / 0.300 | 0.994 / 0.026 / - | 0.996 / 0.010 / - | 0.994 / 0.014 / - | 0.996 / 0.016 / - |
| floor 0.3 | 0.500 / 1.000 / 1.000 | 0.500 / 0.134 / 0.166 | 0.500 / 0.262 / 0.296 | 0.500 / 0.032 / 0.012 | 0.994 / 0.010 / - | 0.846 / 0.230 / 0.329 |
| floor 0.4 | 0.500 / 0.914 / 0.889 | 0.500 / 0.074 / 0.079 | 0.500 / 0.282 / 0.304 | 0.500 / 0.000 / 0.000 | 0.632 / 0.094 / 0.134 | 0.500 / 0.026 / 0.040 |
| floor 0.5 | 0.500 / 0.750 / 0.723 | 0.500 / 0.056 / 0.059 | 0.500 / 0.182 / 0.209 | 0.500 / 0.000 / 0.000 | 0.500 / 0.000 / 0.000 | 0.500 / 0.012 / 0.024 |
| floor 0.75 | 0.500 / 0.854 / 0.838 | 0.500 / 0.114 / 0.119 | 0.500 / 0.310 / 0.320 | 0.500 / 0.000 / 0.000 | 0.500 / 1.000 / 1.000 | 0.500 / 1.000 / 1.000 |
| floor 1 | 0.500 / 0.638 / 0.605 | 0.500 / 0.184 / 0.186 | 0.500 / 0.962 / 0.980 | 0.500 / 0.140 / 0.119 | 0.500 / 1.000 / 1.000 | 0.465 / 1.000 / 1.000 |
| light | 0.500 / 0.952 / 0.933 | 0.500 / 0.498 / 0.498 | 0.992 / 0.020 / - | 0.990 / 0.012 / - | 0.994 / 0.012 / - | 0.998 / 0.018 / - |
| light dir | 0.500 / 0.990 / 1.000 | 0.500 / 0.042 / 0.012 | 0.990 / 0.016 / - | 0.996 / 0.016 / - | 0.994 / 0.016 / - | 0.996 / 0.018 / - |
| exposure | 0.500 / 1.000 / 1.000 | 0.500 / 0.748 / 0.715 | 0.745 / 0.018 / 0.000 | 0.992 / 0.014 / - | 0.994 / 0.014 / - | 0.994 / 0.016 / 0.000 |
| white balance | 0.500 / 0.912 / 0.893 | 0.500 / 0.368 / 0.387 | 0.990 / 0.020 / - | 0.992 / 0.012 / - | 0.992 / 0.014 / - | 0.994 / 0.018 / - |
| noise | 0.767 / 0.020 / 0.009 | 0.998 / 0.012 / - | 0.990 / 0.012 / - | 0.994 / 0.012 / - | 0.994 / 0.014 / - | 0.996 / 0.022 / - |
| blur | 0.836 / 0.010 / 0.000 | 0.994 / 0.010 / 0.000 | 0.990 / 0.014 / - | 0.992 / 0.012 / - | 0.994 / 0.012 / - | 0.996 / 0.022 / - |
| jpeg | 0.500 / 0.948 / 0.964 | 0.500 / 0.134 / 0.146 | 0.990 / 0.014 / - | 0.992 / 0.012 / - | 0.994 / 0.012 / - | 0.996 / 0.018 / - |
| occlusion | 0.500 / 0.882 / 0.870 | 0.500 / 0.428 / 0.462 | 0.561 / 0.162 / 0.189 | 0.545 / 0.110 / 0.087 | 0.747 / 0.162 / 0.323 | 0.686 / 0.120 / 0.166 |
| camera zoom | 0.500 / 0.148 / 0.111 | 0.656 / 0.032 / 0.006 | 0.899 / 0.044 / 0.040 | 0.897 / 0.040 / 0.020 | 0.905 / 0.050 / 0.043 | 0.901 / 0.068 / 0.020 |
| camera shift | 0.500 / 1.000 / 1.000 | 0.500 / 0.700 / 0.704 | 0.506 / 0.294 / 0.378 | 0.587 / 0.174 / 0.240 | 0.555 / 0.226 / 0.316 | 0.684 / 0.304 / 0.458 |
| camera tilt | 0.500 / 1.000 / 1.000 | 0.500 / 0.616 / 0.613 | 0.514 / 0.432 / 0.498 | 0.500 / 0.262 / 0.356 | 0.500 / 0.322 / 0.399 | 0.551 / 0.258 / 0.336 |

- The support detector catches the too-big shifts completely. At floor 0.5-1 every model is flagged on 100% of frames and 100% of its false-safe failures are caught, and the same holds for every other shift that drives a model to all-safe (bal acc 0.500-0.590): camera shift and tilt, occlusion, and the photometric shifts that break sigreg and the official model. Over all 66 (model, shift) cells with bal acc at most 0.85, the support detector catches at least 0.897 of the false-safe failures (mostly 1.000).
- It stays quiet where the margin function still works: jac under floor 0.1, light direction, white balance, noise, blur and JPEG is flagged at 1-4% (clean: 1-2%). The robust models push the alarm out together with the failure (flags at floor 0.3: jac 1.000, varying-radius adv_h 0.138, where its bal acc is still 0.994), so the alarm is mildly early rather than late. The one sizeable false alarm is jac under light intensity 0.3 (0.288 flagged, bal acc 0.992).
- One failure is not detected: camera zoom. Every jac-family model degrades moderately (bal acc 0.897-0.905, about 50 false-safe states of 253 unsafe) and the support detector catches only 0.000-0.176 of them (median kNN distance 1.2x clean): the zoomed frame lands among training latents of other poses, the in-support aliasing that was the dominant Dubins failure. On Reacher it is the exception: every other failure leaves the support.
- The world model's own prediction error is a poor detector. It catches 0.000-0.500 of the failures of the jac-family models under every shift up to floor 0.5, and for jac+adv_h it falls to 0.000 at floor 0.4-0.75: the predictor stays self-consistent on the shifted latents (it maps a shifted frame to the shifted next frame), so a shift can be invisible to the dynamics while the encoder has left the support. It fires only on the extreme shifts (floor 0.75 / 1 for varying-radius adv_h and whole latent, median error 4-25x clean).
- Consequence for the pipeline: a support check on the current frame, computed once per replan, can turn every too-big shift into an abstention (stop or hand over) instead of a silent all-safe planner; the residual risk is the moderate, in-support aliasing case (camera zoom), which only robustness training can address. Both detectors are per-frame and label-free; the threshold needs only clean in-distribution frames. One training seed, one margin-function seed.

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

Code on branch `worktree-reacher-imag-margin`: `le-wm/scripts/reacher_margin.py`, `reacher_margin_imag.py`, `reacher_ood.py` (`FLOOR_SWEEP`), `reacher_cem.py` (`CCEM=minh`, `FLOOR`, `LIGHT`, `ARM`), `reacher_floor_diag.py`, `reacher_depth_diag.py`, `reacher_nn_diag.py`, `reacher_jac_aniso.py`, `reacher_pull_ratio.py`, `reacher_shift.py`, `reacher_within_start.py`, `reacher_calib.py`, `reacher_outcomes.py`, `reacher_calib_pred.py`, `reacher_veto_diag.py`, `adv_sensitivity.py`, `reacher_routes.py`, `reacher_bn_adapt.py`; `le-wm/safe_solver.py` (CCEM, from `imag-margin`). Margin functions: `/data/seongbin/lewm/reacher_heads/<wm>/` (real-only), `/data/seongbin/lewm/reacher_heads_imag/<wm>/` (real+imag). CEM trajectories: `/data/seongbin/lewm/reacher_cem/`. Logs: `/data/seongbin/lewm/reacher_eval_logs/`.
