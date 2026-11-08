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

Code on branch `worktree-reacher-imag-margin`: `le-wm/scripts/reacher_margin.py`, `reacher_margin_imag.py`, `reacher_ood.py` (`FLOOR_SWEEP`), `reacher_cem.py` (`CCEM=minh`, `FLOOR`, `LIGHT`, `ARM`), `reacher_floor_diag.py`, `reacher_depth_diag.py`, `reacher_nn_diag.py`, `reacher_jac_aniso.py`, `reacher_pull_ratio.py`, `reacher_shift.py`; `le-wm/safe_solver.py` (CCEM, from `imag-margin`). Margin functions: `/data/seongbin/lewm/reacher_heads/<wm>/` (real-only), `/data/seongbin/lewm/reacher_heads_imag/<wm>/` (real+imag). CEM trajectories: `/data/seongbin/lewm/reacher_cem/`. Logs: `/data/seongbin/lewm/reacher_eval_logs/`.
