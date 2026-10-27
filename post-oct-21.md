# Dubins, Safety Gym and Cartpole

# Dubins

The same 3×3 grid on Dubins for reference: base {sigreg, + jacobian, + jacobian + pull} × safety reg {none, L∞ safety
reg, λ batch-detached}. Zero-shot margin_gp: the head is trained on red-obstacle frames and scored on the same states
re-rendered with purple obstacles (color), a 90° rotation (rotate) or diamond obstacles (shape); labels are exact under
every shift. WM seed 3072, ± = std over 3 eval seeds. Numbers are from `safety_adv_reg_report.md` (logs under
`/data/seongbin/lewm/safeadv_results/`). Blank rows: those WMs have not been trained on Dubins.

AUC

| Model | base | color | rotate | shape |
|---|---|---|---|---|
| sigreg | 0.979 ± 0.002 | 0.144 ± 0.032 | 0.956 ± 0.023 | 0.913 ± 0.024 |
| jacobian | 0.992 ± 0.001 | 0.673 ± 0.076 | 0.989 ± 0.002 | 0.992 ± 0.003 |
| jac+pull | 0.983 ± 0.001 | 0.753 ± 0.049 | 0.978 ± 0.003 | 0.977 ± 0.007 |
| L∞ safety reg | 0.998 ± 0.001 | 0.910 ± 0.098 | 0.996 ± 0.002 | 0.997 ± 0.001 |
| jacobian + L∞ | | | | |
| jac+pull + L∞ | 0.996 ± 0.001 | 0.965 ± 0.013 | 0.994 ± 0.002 | 0.994 ± 0.002 |
| λ batch-detached | 0.997 ± 0.001 | 0.948 ± 0.013 | 0.996 ± 0.001 | 0.996 ± 0.001 |
| jacobian + λ | | | | |
| jac+pull + λ | | | | |

Sign accuracy (h >= 0 = safe; the test set is 0.830 safe, so 0.830 = everything called safe, 0.170 = everything unsafe)

| Model | base | color | rotate | shape |
|---|---|---|---|---|
| sigreg | 0.943 ± 0.005 | 0.170 ± 0.006 | 0.830 ± 0.006 | 0.310 ± 0.025 |
| jacobian | 0.968 ± 0.002 | 0.171 ± 0.007 | 0.962 ± 0.004 | 0.968 ± 0.007 |
| jac+pull | 0.959 ± 0.003 | 0.830 ± 0.007 | 0.939 ± 0.003 | 0.957 ± 0.008 |
| L∞ safety reg | 0.981 ± 0.004 | 0.174 ± 0.006 | 0.975 ± 0.004 | 0.980 ± 0.003 |
| jacobian + L∞ | | | | |
| jac+pull + L∞ | 0.973 ± 0.003 | 0.830 ± 0.006 | 0.970 ± 0.009 | 0.978 ± 0.005 |
| λ batch-detached | 0.975 ± 0.005 | 0.170 ± 0.006 | 0.973 ± 0.005 | 0.973 ± 0.005 |
| jacobian + λ | | | | |
| jac+pull + λ | | | | |

Reading:

- Color is the only hard shift on Dubins. Shape and rotation stay at 0.91-0.997 AUC for every model.
- No threshold survives color: every model calls purple frames all unsafe (0.170) or all safe (0.830).
- Over 3 WM seeds the single-seed color AUCs above don't hold: jac+pull 0.498 ± 0.234, L∞ 0.643 ± 0.247,
  λ batch-detached 0.721 ± 0.434. Only jac+pull + L∞ is stable (0.957 ± 0.021).

# Safety Gym

Goal: a second benchmark where safety is coupled to dynamics (unlike a "teleportable" agent whose safe set is just the
obstacle geometry), to re-test the appearance-shift findings from Dubins. Code and logs: branch `point-dynamics-probe`
(`safety_gym_port/`, `le-wm/scripts/sg_point_*.py`, `run_logs/sg_point/`); full tables in `sg_point_ovh_results.md`.

## Point instead of Car: safety that depends on velocity

The earlier safety-gym port used SafetyCarGoal1. The Car brakes from top speed in 0.055 (a quarter of the hazard radius
0.2) and reaches top speed in 5 steps, so its safe set is essentially the hazard geometry. The Point robot (thrust along
heading, world-frame slide joints) needs 60 steps to reach top speed, brakes in 0.238 and coasts 0.78, so turning does not
redirect its momentum.

Probe (`safety_gym_port/point_doomed.py`): inject Point states near hazards, call a currently-safe state "doomed" if none of
13 escape policies avoids every hazard for 80 steps (an upper bound on true unsafety). Fraction doomed:

| distance past hazard edge | moving away | approach 0-0.01/step | 0.01-0.02 | 0.02-0.03 (top speed) |
|---|---|---|---|---|
| 0-0.05 | 0.04 | 0.20 | 0.88 | 1.00 |
| 0.05-0.10 | 0.01 | 0.01 | 0.28 | 0.87 |
| 0.10-0.15 | 0.00 | 0.00 | 0.01 | 0.57 |
| 0.15-0.20 | 0.00 | 0.00 | 0.00 | 0.28 |
| 0.20-0.30 | 0.00 | 0.00 | 0.00 | 0.04 |

At top approach speed the doomed band reaches about 0.24 past the hazard edge (the braking distance); moving away, almost
nothing is doomed. Heading barely matters, so the coupling is translational momentum.

## Problem 1: every Car dataset leaked the label

Safety-gym's `render()` draws a translucent red sphere at the agent whenever the previous step had cost > 0, which is exactly
our failure label. A red-pixel count alone predicts unsafe at AUC 0.998-1.000 on the Car training set and on every Car eval
set (color, camera rotation, shape, multicolor). All renders since then are "clean": the sphere and the lidar ring are off
(`--clean`; red-pixel AUC 0.08-0.55 on every clean set, i.e. no marker).

How much the Car models leaned on it (probe trained with the sphere, tested on the same frames without it): the gp margin
probe barely moved (zero-shot drop at most 0.017), but the jacobian/jac+pull latents encode the sphere along the safe/unsafe
direction (cosine ~0.8) and an unregularized probe dropped 1.000 -> 0.84. The earlier Car color/rotation safety-reg gains
should be treated as suspect.

## Problem 2: the old protocol measured layout memorization

Hazard layouts are random every episode. The old eval (one 5000-pose collection, shuffled 80/20 frame split) lets the probe's
train and test frames share layouts. Probes on clean fixedfar@128 Point models scored 0.99 in-distribution, but 0.47-0.52
on base-view frames from fresh layouts, i.e. chance. The camera-rotation axis failed for the same reason: a 90 degree turn
of this camera looks like the normal view of a different layout.

The benchmark itself was learnable. A supervised CNN on raw fixedfar@128 pixels reaches AUC 0.998-0.999 on unseen layouts
(and 0.958-0.999 on every OOD view). The bottleneck was the world model's pooled latent: a probe on the same encoder's patch
tokens reached 0.79 where the pooled latent gave 0.35.

Why Dubins never showed this: it has one fixed obstacle layout, so the agent's pose alone determines safety. Here the hazards
change every episode, never move within one and do not affect the Point's motion, so the world-model objective has little
reason to encode them.

## Fix: overhead camera at 224 px plus a layout-disjoint eval

- Camera: safety-gym's fixedfar camera moved straight above the arena (0, 0, 4), framing [-1.8, 1.8]. The agent goes from
  about 6 px to about 143 px, and the encoder's 224 px input no longer upsamples a 128 px render. A 90 degree camera turn is
  now an exact image rotation.
- Training data: `sg_point_ovh224.h5`, 170,400 clean frames, about 850 random layouts, same trajectories and labels as the
  fixedfar set.
- Eval: margin_gp probe trained on 50k frames of the training set, scored on 5000 poses of a separate collection (every
  layout unseen). Views: base, purple hazards (color), 90 degree camera rotation (rotate), box-shaped hazards (shape), Car
  robot instead of Point (agent).

Pilot, same probe on unseen layouts (AUC, base view): sigreg 0.362 (fixedfar) -> 0.581 (overhead); jac+pull 0.563 -> 0.836.

## Results: 9 world models on overhead@224, layout-disjoint

le-wm, 50 epochs, seed 3072. Safety-reg recipes copied from the earlier safety-gym reference runs. ± = std over 3 eval seeds
(probe subsample and init); one training seed per model. "train fit" = probe AUC on its own training frames.

AUC

| Model | train fit | base | color | rotate | shape | agent |
|---|---|---|---|---|---|---|
| sigreg | 0.989 ± 0.001 | 0.581 ± 0.023 | 0.464 ± 0.013 | 0.447 ± 0.020 | 0.514 ± 0.043 | 0.554 ± 0.020 |
| jacobian | 1.000 ± 0.000 | 0.954 ± 0.006 | 0.725 ± 0.014 | 0.932 ± 0.009 | 0.952 ± 0.007 | 0.824 ± 0.010 |
| jac+pull | 0.999 ± 0.000 | 0.836 ± 0.015 | 0.660 ± 0.035 | 0.740 ± 0.022 | 0.865 ± 0.010 | 0.673 ± 0.045 |
| L∞ safety reg | 0.996 ± 0.000 | 0.948 ± 0.003 | 0.746 ± 0.008 | 0.909 ± 0.009 | 0.619 ± 0.015 | 0.873 ± 0.008 |
| jacobian + L∞ | 0.998 ± 0.000 | 0.985 ± 0.001 | 0.937 ± 0.006 | 0.972 ± 0.003 | 0.984 ± 0.001 | 0.917 ± 0.000 |
| jac+pull + L∞ | 0.999 ± 0.000 | 0.939 ± 0.007 | 0.753 ± 0.032 | 0.821 ± 0.012 | 0.951 ± 0.004 | 0.852 ± 0.007 |
| λ batch-detached | 0.989 ± 0.001 | 0.885 ± 0.010 | 0.501 ± 0.009 | 0.899 ± 0.003 | 0.895 ± 0.009 | 0.757 ± 0.016 |
| jacobian + λ | 0.994 ± 0.001 | 0.969 ± 0.003 | 0.898 ± 0.024 | 0.961 ± 0.003 | 0.972 ± 0.001 | 0.892 ± 0.007 |
| jac+pull + λ | 0.995 ± 0.001 | 0.971 ± 0.003 | 0.894 ± 0.013 | 0.955 ± 0.004 | 0.974 ± 0.002 | 0.867 ± 0.005 |

Sign accuracy (h >= 0 = safe; the test set is 0.830 safe, so 0.830 = everything called safe, 0.170 = everything unsafe)

| Model | base | color | rotate | shape | agent |
|---|---|---|---|---|---|
| sigreg | 0.776 ± 0.009 | 0.820 ± 0.003 | 0.794 ± 0.019 | 0.808 ± 0.008 | 0.777 ± 0.007 |
| jacobian | 0.896 ± 0.001 | 0.468 ± 0.037 | 0.866 ± 0.001 | 0.890 ± 0.005 | 0.814 ± 0.009 |
| jac+pull | 0.847 ± 0.011 | 0.757 ± 0.012 | 0.839 ± 0.010 | 0.861 ± 0.013 | 0.813 ± 0.002 |
| L∞ safety reg | 0.931 ± 0.004 | 0.170 ± 0.000 | 0.918 ± 0.004 | 0.835 ± 0.003 | 0.884 ± 0.004 |
| jacobian + L∞ | 0.956 ± 0.001 | 0.879 ± 0.005 | 0.907 ± 0.003 | 0.949 ± 0.002 | 0.866 ± 0.002 |
| jac+pull + L∞ | 0.852 ± 0.007 | 0.809 ± 0.004 | 0.852 ± 0.008 | 0.879 ± 0.013 | 0.812 ± 0.011 |
| λ batch-detached | 0.879 ± 0.000 | 0.710 ± 0.010 | 0.888 ± 0.012 | 0.889 ± 0.005 | 0.841 ± 0.004 |
| jacobian + λ | 0.934 ± 0.009 | 0.618 ± 0.023 | 0.907 ± 0.013 | 0.937 ± 0.004 | 0.815 ± 0.006 |
| jac+pull + λ | 0.940 ± 0.005 | 0.752 ± 0.022 | 0.905 ± 0.007 | 0.937 ± 0.003 | 0.808 ± 0.004 |

Reading:

- Jacobian + L∞ safety reg is best on every axis (AUC 0.917-0.985) and the only model whose threshold holds on all five
  views (predicted-safe fraction 0.79-0.86 against a true 0.83).
- The safety regularizers help on this benchmark: sigreg 0.581 base -> 0.948 with L∞ and 0.885 with λ. Combined with the
  Jacobian term they form the top group (jacobian + λ and jac+pull + λ: about 0.97 base, 0.90 color).
- Color is the hardest shift and breaks thresholds even when the ranking holds: L∞ alone calls everything unsafe on purple
  hazards, and jacobian alone calls 70% unsafe.
- The action-separation pull hurts here (jac+pull 0.836 vs jacobian 0.954 base).
- sigreg stays near chance; the camera change does not rescue an unregularized latent.
- All probes fit their own training frames (0.989-1.000), so the gaps above are generalization to new layouts, not under-fitting.

## Caveats and next steps

- One training seed per model. Single-seed safety-reg results have been lucky before (Dubins). Seeds 1 and 2 of all 9
  recipes are training (`le-wm/scripts/run_point_ovh_seeds.sh`); the same eval runs automatically when they finish.
- Labels are position-only failure (agent inside a hazard). The velocity-dependent value (the doomed band above) is not yet
  used as a target; `point_doomed.py` can generate Monte Carlo value labels.
- Possible variant: a fixed-layout Point benchmark (Dubins-like), which would also make an exact HJ value oracle feasible.
- No Point Dreamer has been trained.

# Cartpole

*Updated 2026-10-27 22:40 UTC. LeWM, classic Gym CartPole-v1. Branch `worktree-cartpole-port` (code in `cartpole_port/`).*

## TL;DR

- **Why cartpole:** safety here is coupled to dynamics. A state can be inside the track and below 12° and still be lost, because the cart or pole is moving too fast to recover. The sim is the analytic cartpole ODE, so the safe set can be computed exactly (a 4D value table), and every frame is a pure function of the state, so OOD frames are exactly paired with in-distribution ones.
- **Three kinds of state:** failed (ℓ < 0: pole past 12° or cart past the ±2.4 m wall), doomed (ℓ ≥ 0 but V < 0: not failed yet, but no action sequence avoids failure), and safe (V ≥ 0). Unsafe = failed + doomed. The margin head learns "currently safe" from the failure labels; a separate reachability critic, trained by a safety Bellman backup with no doomed labels, has to find the doomed states.
- **In distribution, every model is about equal.** The margin detects failure at AUC 0.980–0.993. The critic finds doomed states at 0.939–0.962, as well as a probe trained directly on the oracle V (0.942–0.953) and close to the same critic on the true state (0.985 ± 0.004). The margin alone sees doom only partly (0.665–0.731), so the separate critic is needed.
- **Held-out purple pole:** λ batch-detached is the best column on every base, for both heads. jac+pull + λ batch-detached keeps the critic at 0.947 ± 0.003 (tan: 0.958) and the margin at 0.894 ± 0.011; sigreg alone falls to 0.785 / 0.656. The L∞ safety reg barely moves the margin; on the jacobian base it lifts the critic from 0.797 to 0.893.
- **Viewpoint (zoom out, horizontal pan, vertical pan) breaks every model.** Averaged over the three views, critic doomed AUC is 0.580–0.716 and margin failed AUC 0.572–0.745, and the V probe is just as low (0.554–0.694), so the latent itself moves. The ranking flips: the two λ batch-detached models on the jacobian bases, best on color, are the two worst here.
- **Thresholds don't survive either shift.** Critic sign accuracy is 0.790–0.895 in distribution, 0.444–0.736 on purple, and mostly 0.33–0.39 on viewpoint, about what calling everything unsafe scores (0.330 of test frames are unsafe).
- **Caveat:** one world-model training seed (3072) per model; ± is over 3 head seeds. On Dubins, single-seed rankings did not survive reseeding, so the color ranking here needs reseeds before it is a headline.

## Setup

**Environment and rendering.** Classic Gym CartPole-v1 dynamics (Euler step, τ = 0.02 s, discrete ±10 N push). Failure: |x| > 2.4 or |θ| > 12°. Frames are drawn directly from the state (`cartpole_port/render.py`, 128×128, 4× supersampled) rather than with gym's renderer: gym anti-aliases, so swm's exact-match color swap would leave the original color on edge pixels. The pole is drawn 2.5 m long (physics keeps 1.0 m) and the view is anisotropic, otherwise a 12° lean is about 4 px at 128 px wide. Walls are drawn at ±2.4 m.

**Value oracle** (`cartpole_port/value_dp.py`). Discrete-time safety value iteration on the env's own Euler step, V(s) = min(ℓ(s), max_a V(f(s, a))), with ℓ(s) = min(1 − |x|/2.4, 1 − |θ|/θ_max), on a 61⁴ grid with 5-step lookahead per backup (torch, GPU). Speeds outside |ẋ| ≤ 9 m/s or |θ̇| ≤ 4 rad/s count as doomed: at 9 m/s the cart can't stop within the 4.8 m track. Clamping speeds instead made V optimistic (21% of "safe" states hit the wall). Validation from 20k sampled states: no state with V < −0.02 is saved by any tested policy, and 91% of states with V > 0.02 survive 500 steps under the greedy-V policy. The remaining gap is grid optimism near V = 0 and shrinks with resolution (85% / 88% / 91% at 41 / 51 / 61 points per axis).

**Dataset** (`/data/seongbin/lewm/datasets/cartpole_expert.h5`, `cartpole_port/collect_dataset.py`). 170,040 frames in 5,175 episodes, in the dubins_expert.h5 schema plus `margin`, `value` and `unsafe`. Half the episodes start near upright at a random cart pose, half uniformly over the not-yet-failed state box (which includes doomed starts). The policy is V-greedy with each action replaced by a random one at a per-episode rate ε ~ U(0, 0.6). Episodes stop 8 frames after the first failure. All failing episodes are kept and safe ones subsampled to 17% failed frames (dubins: 16.7%). The result is 17.0% failed, 15.7% doomed, 67.3% safe. Example rollouts: `figs/cartpole_rollouts/cartpole_{greedy,noisy_eps0.4,doomed_start}.gif`.

**World models.** The same 3×3 grid as Point: base {sigreg, + jacobian, + jacobian + pull} × safety reg {none, L∞ safety reg, λ batch-detached}, 50 epochs, seed 3072, same recipes and code snapshots as the Dubins runs (L∞: `code_safeadv_cfa5e52`, AUC gate 0.95, sigreg base ungated; λ: `code_safeadv_bf16b3e`). The L∞ safety reg's in-training head reached AUC 0.991 on cartpole, so the gate opened (on Safety Gym Car it never did).

**Eval** (`cartpole_port/eval_cartpole.py`). WM frozen; heads are trained on 70% of episodes and scored on the other 30% (51,282 frames).
- **Margin ℓ(z_t):** the Dubins margin_gp head and loss, 50/50 safe/failed batches, trained on `failures`.
- **Critic Q(h_t, a):** offline fitted Q-iteration on dataset transitions with the discounted safety Bellman target (1 − γ) ℓ(z_t) + γ min(ℓ(z_t), max_a′ Q′(h_{t+1}, a′)), γ = 0.98, where ℓ is the frozen learned margin. Its input is the last three latents h_t = [z_{t−2}, z_{t−1}, z_t], the predictor's context: one frame carries no velocity, and doom depends on velocity. No value or doomed labels are used.
- **Probe:** supervised regression from h_t to the oracle V. It asks whether the information is in the latent at all.
- **Metrics** (score ≥ 0 means safe): failed AUC (safe = ℓ ≥ 0), unsafe AUC (safe = V ≥ 0), doomed AUC (among ℓ ≥ 0 frames only, safe = V ≥ 0), sign accuracy against V ≥ 0.
- **State ceiling:** the same critic on the true 4D state history with the true ℓ: failed AUC 0.960 ± 0.002, unsafe AUC 0.993 ± 0.002, doomed AUC 0.985 ± 0.004, sign accuracy 0.970 ± 0.005.

**OOD conditions:** the same test states re-rendered, labels unchanged (`figs/cartpole_rollouts/cartpole_ood_samples.png`).
- **Purple pole:** held-out pole color (tan in training).
- **zoom_out:** 0.8× zoom about the view center.
- **pan_x:** 0.5 m horizontal pan at the zoom_out scale.
- **pan_y:** 0.3 m vertical pan at native scale.

Zoom-in and a native-scale horizontal pan were dropped: the native view only just fits a pole leaning outward at the wall, so they crop the pole in 24–40% of failed test frames, which removes information instead of shifting the view. In the views kept, no not-failed frame is cropped.

All values below are mean ± std over 3 head seeds (one WM per row).

## In distribution (tan pole)

| model | margin failed AUC | margin doomed AUC | critic doomed AUC | critic unsafe AUC | probe doomed AUC | critic sign acc |
|---|---|---|---|---|---|---|
| sigreg | 0.991 ± 0.001 | 0.722 ± 0.002 | 0.953 ± 0.002 | 0.976 ± 0.001 | 0.953 ± 0.001 | 0.895 ± 0.007 |
| jacobian | 0.991 ± 0.001 | 0.731 ± 0.003 | 0.962 ± 0.002 | 0.975 ± 0.002 | 0.949 ± 0.001 | 0.875 ± 0.004 |
| jac+pull | 0.992 ± 0.002 | 0.719 ± 0.004 | 0.962 ± 0.002 | 0.979 ± 0.001 | 0.950 ± 0.002 | 0.865 ± 0.019 |
| sigreg + L∞ safety reg | 0.981 ± 0.002 | 0.689 ± 0.005 | 0.940 ± 0.001 | 0.968 ± 0.001 | 0.946 ± 0.001 | 0.894 ± 0.023 |
| jacobian + L∞ safety reg | 0.993 ± 0.002 | 0.705 ± 0.008 | 0.952 ± 0.004 | 0.969 ± 0.002 | 0.946 ± 0.001 | 0.863 ± 0.013 |
| jac+pull + L∞ safety reg | 0.991 ± 0.000 | 0.699 ± 0.002 | 0.948 ± 0.002 | 0.964 ± 0.003 | 0.947 ± 0.002 | 0.866 ± 0.010 |
| sigreg + λ batch-detached | 0.984 ± 0.001 | 0.707 ± 0.006 | 0.939 ± 0.003 | 0.966 ± 0.002 | 0.943 ± 0.002 | 0.884 ± 0.005 |
| jacobian + λ batch-detached | 0.980 ± 0.005 | 0.665 ± 0.012 | 0.947 ± 0.003 | 0.964 ± 0.003 | 0.942 ± 0.003 | 0.815 ± 0.030 |
| jac+pull + λ batch-detached | 0.986 ± 0.001 | 0.708 ± 0.005 | 0.958 ± 0.002 | 0.971 ± 0.002 | 0.944 ± 0.001 | 0.790 ± 0.036 |

- Every latent encodes velocity well enough from three frames: the critic finds doomed states at 0.939–0.962 without ever seeing a doomed label, matching the supervised probe and 0.02–0.05 below the true-state ceiling.
- The margin is a good failure detector but a poor doom detector (0.665–0.731): a single frame can't show that the cart is about to hit the wall.
- The regularizers cost little in distribution (critic doomed within 0.023 of the best model).

## Color: held-out purple pole

| model | margin failed AUC | margin doomed AUC | critic doomed AUC | critic unsafe AUC | probe doomed AUC | critic sign acc |
|---|---|---|---|---|---|---|
| sigreg | 0.656 ± 0.003 | 0.520 ± 0.005 | 0.785 ± 0.014 | 0.797 ± 0.011 | 0.772 ± 0.022 | 0.444 ± 0.046 |
| jacobian | 0.737 ± 0.006 | 0.569 ± 0.006 | 0.797 ± 0.008 | 0.819 ± 0.016 | 0.701 ± 0.020 | 0.464 ± 0.009 |
| jac+pull | 0.789 ± 0.010 | 0.597 ± 0.004 | 0.850 ± 0.011 | 0.877 ± 0.010 | 0.750 ± 0.008 | 0.536 ± 0.028 |
| sigreg + L∞ safety reg | 0.665 ± 0.002 | 0.560 ± 0.001 | 0.788 ± 0.023 | 0.816 ± 0.024 | 0.772 ± 0.015 | 0.563 ± 0.052 |
| jacobian + L∞ safety reg | 0.753 ± 0.014 | 0.582 ± 0.004 | 0.893 ± 0.007 | 0.902 ± 0.001 | 0.863 ± 0.006 | 0.632 ± 0.051 |
| jac+pull + L∞ safety reg | 0.750 ± 0.023 | 0.596 ± 0.008 | 0.863 ± 0.004 | 0.867 ± 0.007 | 0.814 ± 0.010 | 0.626 ± 0.026 |
| sigreg + λ batch-detached | 0.835 ± 0.007 | 0.609 ± 0.002 | 0.843 ± 0.020 | 0.862 ± 0.017 | 0.834 ± 0.009 | 0.736 ± 0.016 |
| jacobian + λ batch-detached | 0.891 ± 0.009 | 0.596 ± 0.008 | 0.913 ± 0.004 | 0.931 ± 0.003 | 0.878 ± 0.007 | 0.631 ± 0.047 |
| jac+pull + λ batch-detached | 0.894 ± 0.011 | 0.647 ± 0.008 | 0.947 ± 0.003 | 0.961 ± 0.002 | 0.924 ± 0.003 | 0.669 ± 0.046 |

- **λ batch-detached is the best column on every base, for both heads:** margin failed 0.835 / 0.891 / 0.894 against 0.656–0.789 without it, critic doomed 0.843 / 0.913 / 0.947 against 0.785–0.850. jac+pull + λ batch-detached loses only 0.011 critic AUC against tan.
- **The base matters too:** without a safety reg, sigreg < jacobian < jac+pull on both heads (critic 0.785 / 0.797 / 0.850).
- **L∞ safety reg:** the margin barely moves (sigreg 0.656 → 0.665, jacobian 0.737 → 0.753, jac+pull 0.789 → 0.750), but the critic gains on the jacobian base (0.797 → 0.893). This differs from Dubins, where jac+pull + L∞ safety reg was the clear color winner over 3 WM seeds.
- **Probe vs. critic:** for the robust models the probe stays high (jac+pull + λ batch-detached 0.924), so the robustness is in the latent, not an artifact of the critic's training.
- **Thresholds:** critic sign accuracy drops from 0.790–0.895 to 0.444–0.736; the ranking survives on the best models but the zero point drifts.

## Viewpoint

Critic doomed AUC:

| model | zoom_out | pan_x | pan_y | mean of views |
|---|---|---|---|---|
| sigreg | 0.681 ± 0.048 | 0.689 ± 0.040 | 0.770 ± 0.014 | 0.714 |
| jacobian | 0.699 ± 0.015 | 0.735 ± 0.020 | 0.618 ± 0.024 | 0.684 |
| jac+pull | 0.617 ± 0.006 | 0.666 ± 0.013 | 0.670 ± 0.033 | 0.651 |
| sigreg + L∞ safety reg | 0.740 ± 0.005 | 0.591 ± 0.014 | 0.732 ± 0.022 | 0.688 |
| jacobian + L∞ safety reg | 0.692 ± 0.060 | 0.687 ± 0.079 | 0.770 ± 0.035 | 0.716 |
| jac+pull + L∞ safety reg | 0.669 ± 0.078 | 0.692 ± 0.057 | 0.748 ± 0.031 | 0.703 |
| sigreg + λ batch-detached | 0.646 ± 0.019 | 0.512 ± 0.017 | 0.705 ± 0.019 | 0.621 |
| jacobian + λ batch-detached | 0.563 ± 0.057 | 0.587 ± 0.052 | 0.589 ± 0.016 | 0.580 |
| jac+pull + λ batch-detached | 0.642 ± 0.007 | 0.584 ± 0.019 | 0.626 ± 0.009 | 0.617 |

Margin failed AUC:

| model | zoom_out | pan_x | pan_y | mean of views |
|---|---|---|---|---|
| sigreg | 0.619 ± 0.010 | 0.620 ± 0.010 | 0.746 ± 0.014 | 0.662 |
| jacobian | 0.726 ± 0.005 | 0.753 ± 0.011 | 0.758 ± 0.016 | 0.745 |
| jac+pull | 0.708 ± 0.014 | 0.700 ± 0.014 | 0.796 ± 0.011 | 0.735 |
| sigreg + L∞ safety reg | 0.800 ± 0.006 | 0.601 ± 0.020 | 0.769 ± 0.007 | 0.724 |
| jacobian + L∞ safety reg | 0.592 ± 0.013 | 0.636 ± 0.002 | 0.777 ± 0.012 | 0.668 |
| jac+pull + L∞ safety reg | 0.642 ± 0.015 | 0.723 ± 0.016 | 0.801 ± 0.003 | 0.722 |
| sigreg + λ batch-detached | 0.707 ± 0.011 | 0.668 ± 0.004 | 0.699 ± 0.008 | 0.692 |
| jacobian + λ batch-detached | 0.530 ± 0.018 | 0.539 ± 0.011 | 0.690 ± 0.009 | 0.586 |
| jac+pull + λ batch-detached | 0.596 ± 0.007 | 0.569 ± 0.006 | 0.551 ± 0.010 | 0.572 |

Probe doomed AUC:

| model | zoom_out | pan_x | pan_y | mean of views |
|---|---|---|---|---|
| sigreg | 0.625 ± 0.020 | 0.616 ± 0.017 | 0.662 ± 0.023 | 0.635 |
| jacobian | 0.663 ± 0.017 | 0.730 ± 0.017 | 0.603 ± 0.015 | 0.666 |
| jac+pull | 0.672 ± 0.002 | 0.702 ± 0.008 | 0.703 ± 0.028 | 0.692 |
| sigreg + L∞ safety reg | 0.672 ± 0.005 | 0.559 ± 0.009 | 0.790 ± 0.005 | 0.673 |
| jacobian + L∞ safety reg | 0.665 ± 0.027 | 0.625 ± 0.042 | 0.793 ± 0.019 | 0.694 |
| jac+pull + L∞ safety reg | 0.629 ± 0.016 | 0.638 ± 0.011 | 0.693 ± 0.035 | 0.653 |
| sigreg + λ batch-detached | 0.605 ± 0.001 | 0.501 ± 0.004 | 0.682 ± 0.007 | 0.596 |
| jacobian + λ batch-detached | 0.556 ± 0.011 | 0.551 ± 0.038 | 0.556 ± 0.005 | 0.554 |
| jac+pull + λ batch-detached | 0.666 ± 0.006 | 0.632 ± 0.022 | 0.617 ± 0.016 | 0.639 |

Critic sign accuracy:

| model | zoom_out | pan_x | pan_y | mean of views |
|---|---|---|---|---|
| sigreg | 0.332 ± 0.002 | 0.330 ± 0.001 | 0.355 ± 0.005 | 0.339 |
| jacobian | 0.373 ± 0.033 | 0.370 ± 0.032 | 0.342 ± 0.004 | 0.361 |
| jac+pull | 0.388 ± 0.013 | 0.386 ± 0.013 | 0.337 ± 0.005 | 0.370 |
| sigreg + L∞ safety reg | 0.634 ± 0.059 | 0.428 ± 0.038 | 0.637 ± 0.067 | 0.566 |
| jacobian + L∞ safety reg | 0.351 ± 0.018 | 0.347 ± 0.013 | 0.449 ± 0.030 | 0.382 |
| jac+pull + L∞ safety reg | 0.344 ± 0.015 | 0.342 ± 0.011 | 0.345 ± 0.010 | 0.344 |
| sigreg + λ batch-detached | 0.616 ± 0.053 | 0.518 ± 0.034 | 0.633 ± 0.010 | 0.589 |
| jacobian + λ batch-detached | 0.352 ± 0.026 | 0.381 ± 0.048 | 0.416 ± 0.056 | 0.383 |
| jac+pull + λ batch-detached | 0.355 ± 0.026 | 0.345 ± 0.020 | 0.330 ± 0.001 | 0.344 |

- **No model is robust to viewpoint.** The best mean critic doomed AUC is 0.716 (tan: 0.94–0.96), and the probe is equally low, so the latent itself relocates rather than the heads failing to read it.
- **The color winner is the worst here.** jacobian + λ batch-detached and jac+pull + λ batch-detached are the two lowest rows on both heads (critic 0.580 / 0.617, margin 0.586 / 0.572). A possible reason, not tested: λ batch-detached makes the latent invariant along the safety direction to small pixel perturbations, and in cartpole that direction mostly encodes where the cart and pole are in the frame, which is exactly what a pan or zoom changes.
- **The jacobian base helps the margin a little** (0.745 / 0.735 vs sigreg 0.662), a weak echo of Safety Gym Car rotation, where only the Jacobian family held. It doesn't carry over to the critic.
- **Thresholds collapse to "all unsafe"** for most models (sign accuracy about 0.33–0.39 against 0.330 unsafe test frames). The two sigreg-base safety-reg models are the exception (0.566 / 0.589).
- **Many differences are within noise.** Several L∞ critic cells have std 0.06–0.08, and every row is one WM seed.

## Interpretation

- **Cartpole separates "currently safe" from "will stay safe" cleanly.** The margin and the critic answer different questions, and only the critic (or the probe) finds doom. Every latent carries the velocity information needed, given a 3-frame history.
- **Color and viewpoint robustness are different properties.** The regularizer that buys color robustness here (λ batch-detached) makes viewpoint robustness worse. This matches the Safety Gym Car lesson that color training gives no cross-type transfer, and here the transfer is negative.
- **A viewpoint fix probably needs geometric training** (pan/zoom augmentation, or an adversary over small camera shifts) rather than appearance invariance.

## Caveats

- One WM training seed per model (3072); ± is over head seeds only. On Dubins, single-seed L∞ and λ batch-detached color numbers swung by up to ±0.4 across training seeds.
- The oracle is slightly optimistic near V = 0 (9% of V > 0.02 states fail under the greedy policy), so a few "safe" test frames are probably doomed. Doomed labels are clean.
- The critic is offline fitted Q-iteration on dataset transitions, not imagination rollouts through the WM predictor as in the Dubins critic; it tests the encoder's latent, not the predictor's dynamics.
- The renderer exaggerates the pole (2.5 m drawn, anisotropic view); the OOD views are defined relative to that.

## Files

- Eval outputs (every metric, per-seed runs, summaries): `/data/seongbin/lewm/cartpole_eval/<model>.json` (tan + purple), `<model>_zoom_out+pan_x+pan_y.json` (viewpoint), `state_ceiling.json`. Logs: `run_logs/cartpole_eval/`.
- Checkpoints: `/data/seongbin/lewm/checkpoints/cp_*/weights_epoch_50.pt`. Value table: `/data/seongbin/lewm/datasets/cartpole_value.npz`.
- Figures: `figs/cartpole_rollouts/` (three rollout GIFs, `cartpole_ood_samples.png`).
- Code (`cartpole_port/`): `value_dp.py`, `render.py`, `collect_dataset.py`, `train_cp_reg.sh`, `eval_cartpole.py`, `report_tables.py` (regenerates every table above), `ood_samples.py`, `rollout_gifs.py`.
