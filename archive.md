# Archive

Experiments and findings that were dropped or are not in the main reports, kept for reference.

## Dubins goal marker (2026-11-04 to 2026-11-05, dropped)

### Where the marker is

- Training data (`dubins_expert.h5`): the green goal marker is static within an episode and moves only vertically between episodes: always on the right edge ($x \approx 1.3$, pixel column 118.6-118.7 of 128), with $y$ jittered per episode over about $[-0.9, 0.6]$ (pixel rows 39-87). The world model never saw the marker anywhere else.
- The marker is dark green, RGB about $(0, 128, 0)$.
- `DubinsSwmEnv` draws it at `goal_position`, which defaults to $(1.3, 0)$ (`x_max - 0.2`); `_set_goal_state` moves it to the goal state.
- The GP margin functions (`train_gp_margin_dubins.py`, `dub_fixed_margin.py`) are trained on uniform states re-rendered by the swm env, so only ever with the marker at the default $(1.3, 0)$.

### What the current sg25clean CEM shows

- Observations: the env draws the marker at the episode's goal state (`_set_goal_state`). sg25clean goals are states 25 steps ahead in an annulus around the obstacles, so during planning the marker is usually mid-arena, a placement absent from the training data.
- Goal image, red runs: the stored dataset frame, so the marker is on the $x \approx 1.3$ line. Observations and goal image therefore disagree about the marker in red runs.
- Goal image, OOD runs (purple, diamond): re-rendered by `eval.py` `_rerender_goal`, which is meant to put the marker where the dataset frame shows it, but its detector looks for bright green (G > 150, R < 100, B < 100) and never finds the dark-green marker, so it falls back to the goal state. The code comment ("goal marker where the dataset frame shows it") is wrong in practice. In OOD runs observations and goal image agree (both at the goal state), unlike red runs.
- So red and OOD CEM cells use different marker handling.

### Marker sensitivity of $h$ (start-frame A/B)

Found while explaining why `diag_imag_steps.py` disagreed with Table 1 of post-oct-21.md for sigreg and adv_h. Table 1 (`train_gp_margin_imag_dubins.py`) starts imagined rollouts from the stored dataset frame (marker jittered on the 1.3 line); `diag_imag_steps.py`, `diag_imag_color_dubins.py` and `diag_purple_drift.py` re-render the start state (marker at the default). The two frames differ only in the marker (about 0.3% of pixels). Same starts, plans, labels and margin function (CEM's 5000-step $h$), `imag_startframe_ab.py`:

| Model | Start frame | Imagined AUC | False-safe |
| --- | --- | ---: | ---: |
| sigreg | re-rendered | 0.814 | 0.423 |
| sigreg | dataset frame | 0.664 | 0.967 |
| adv_h | re-rendered | 0.661 | 0.796 |
| adv_h | dataset frame | 0.584 | 0.983 |
| jac | re-rendered | 0.969 | 0.115 |
| jac | dataset frame | 0.968 | 0.115 |

- The dataset-frame rows reproduce Table 1 (sigreg 0.642 / 0.968, adv_h 0.637 / 0.971).
- $h$ on sigreg and adv_h reads the marker height as if it carried safety information; jac ignores it. The earlier real-frame check (post-oct-21.md, Diagnostics) had found the marker to be a nuisance for sigreg only (AUC 0.985 -> 0.901 when the marker moves).

### Default-marker CEM experiment

Idea: draw the marker at the default $(1.3, 0)$ in every observation and goal image, decoupled from the goal used for success, so planning frames stay in the training distribution and observations and goal image always agree. (The alternative considered, goal at $(1.3, 0)$ with starts 25 steps before it, would have been a new protocol.)

- Implementation: branch `marker-default`, commit 9b3c78c, not merged. `DUBINS_MARKER=default` makes `dubins_swm_env.py` draw the marker at a fixed $(1.3, 0)$ (rotated with the world under `DUBINS_ROT_DEG`), while `goal_position` still drives success and termination; `eval.py` re-renders every goal frame under the same flag. Queue script `le-wm/scripts/marker_default_queue.sh`.
- Check: with a goal at $(0.9, 0.9)$, success uses $(0.9, 0.9)$ and the marker is drawn at $(1.32, 0.0)$ in the observation and the re-rendered goal frame; all 200 goal frames were re-rendered in the CEM run.
- Not validated for the 90° rotation condition.

Red, sg25clean, 200 episodes, success / collision (%). Baselines are the current main-table cells (raw CEM; GP safety-first, $w = 1000$, $C = 10^4$, open loop).

| Model | Planner | Current protocol | Default marker |
| --- | --- | ---: | ---: |
| jac | raw | 81.5 / 22.5 | 79.5 / 27.0 |
| jac | safety-first | 79.0 / 8.5 | 79.0 / 8.5 |
| sigreg | raw | 62.5 / 50.0 | 44.5 / 56.5 |
| sigreg | safety-first | 57.5 / 49.5 | 43.0 / 57.0 |
| adv_h | raw | 57.5 / 57.0 | 43.0 / 61.5 |

- jac is unchanged (safety-first identical; raw within noise and equal to another raw jac run, 79.5 / 26.0).
- sigreg and adv_h get 14.5-18 points worse in success, about 4-5 binomial SDs at 200 episodes. The opposite of what was expected.
- Untested explanation: in the current protocol every observation shows the marker at the goal location; marker-sensitive encoders (sigreg base) may use it as a goal cue, so the default marker removes information they rely on. If so, the current protocol has a mild leak (observations show the goal location) that helps the sigreg base, and the default marker would widen the jac-vs-sigreg gap.
- Stopped before adv_h safety-first and adv_λ (raw and safety-first); OOD conditions not run. Logs: `run_logs/softcem_*_gp_mkdef_*_red.log`.
- Dropped (2026-11-05): not expected to help the main results.

### Ideas not pursued

- Train $h$ on uniform states with the marker drawn at random positions, so $h$ cannot read the marker.
- Measure sigreg's goal fidelity with the marker at the goal location vs the default, to test the goal-cue explanation.
- Fix `_rerender_goal`'s detector (dark green) if the OOD goal images should really put the marker where the dataset frame shows it.

## Free-running rollout loss, SLS-squared style (2026-11-03 to 2026-11-05, paused)

Status: paused, not abandoned. At 50 epochs the rollout loss keeps the imagined-state ranking on red longer, but safety-first CEM is no better on red and mixed under purple (jac worse, jac+pull+adv_h encoder-only better, 50 episodes), and on sigreg it degraded the encoder. It failed the decision rule set before the 50-epoch runs (keep it only if red with replanning beats the original by about 10 points and purple is not worse). jac+pull was stopped at epoch 32 of 50.

What is saved:

- World models (weights of every epoch, `weights_epoch_<k>.pt`): `/data/seongbin/lewm/checkpoints/lewm_dubins_{sigreg,jac,jacpull_safeadv_linf_enc}_roll5b50/` (epochs 1-50) and `lewm_dubins_jacpull_roll5b50/` (epochs 1-32). Only weights are saved, no optimizer state, so finishing jac+pull would mean retraining or resuming with a fresh optimizer at the matching point of the cosine schedule.
- 15-epoch versions: `lewm_dubins_jac_roll5b/`, `lewm_dubins_jacpull_safeadv_linf_enc_roll5b/`; failed first version (rollout from 3 frames): `lewm_dubins_{sigreg,jac,jacpull,jacpull_safeadv_linf_enc}_roll5/`.
- Margin functions of the evaluated checkpoints: `/data/seongbin/dreamer/lewm/lewm_dubins_*_roll5b50_e50/margin_gp_fail_red_s0.pt` (and `_roll5b_e<k>/` for the screens). The `_e<k>` directories are symlink views of one epoch of the training directory.
- Code: branch `rollout-loss` (`le-wm/train.py`, `rollout_steps = K`; scripts `train_rollout5_dubins.sh`, `roll5b_check.sh`, `roll5b_quick_cem.sh`, `roll5b50_eval_queue.sh`, `diag_purple_drift.py`). CEM logs: `run_logs/softcem_*roll5b*`.

Tables in this section are numbered R1-R7; "Table N" alone refers to post-oct-21.md, numbered as after this section was moved out of it.

Question: does training the predictor on its own multi-step rollouts make CEM plan better, in-distribution and under OOD shifts? Code on branch `rollout-loss` (`train.py`, `rollout_steps = K`); scripts `train_rollout5_dubins.sh`, `roll5b_check.sh` (per-step test of Table 6), `roll5b_quick_cem.sh`, `roll5b50_eval_queue.sh`.

### How le-wm and SLS-squared train the predictor

- le-wm (all Dubins WMs so far): windows of 4 frames, one forward pass of the causal transformer on the 3 real context latents, MSE of the predictions at all 3 positions to the next real latents. So every context length (1, 2, 3 frames) is trained, but only one step ahead from real latents; the predictor never sees its own outputs.
- SLS-squared (`SLS-squared/<env>/train/mlpdyn_train.py`): an MLP predictor on a single Markov state $[z_t, z_t - z_{t-1}]$, unrolled 5 steps on its own predictions, MSE averaged over the 5 steps. This replaces the one-step loss (step 0 is the one-step prediction from a real state). Plus SIGReg and a trajectory-straightening loss; model error is calibrated afterwards with conformal prediction.
- Ours ("rollout loss", $K = 5$): replaces the le-wm loss with the mean MSE over 5 free-running steps; the regularizers keep their original 4-frame window. Everything else (seed, 50 epochs, regularizer weights) as the original model.

### First attempt: rollout from 3 real frames (failed)

The first version started each rollout from 3 real latents and scored only the last position. CEM never trains on this case: its rollouts start from 1 frame and grow to 3, so predictions from 1-2 frames of context lost their training signal. After 50 epochs the rollouts collapse after step 5 (imagined latents converge to one region; median $h$ on safe and unsafe states both 0.70), and planning breaks.

Table R1. Per-step AUC of $h$ on imagined latents from a red start (as Table 6), jac, original vs first rollout-loss version (both 50 epochs). The original row is from `diag_purple_drift.py` (400 held-out states), the rollout-loss row from `diag_imag_color_dubins.py` (600); same plan distribution.

| jac | Step 1 | Step 3 | Step 5 | Step 10 | Step 15 | Step 25 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 1.00 | 1.00 | 0.99 | 0.99 | 0.97 | 0.88 |
| rollout loss from 3 frames | 0.98 | 0.91 | 0.88 | 0.43 | 0.42 | 0.51 |

Table R2. CEM, replanning every 5 steps, goal cost at every imagined step, success / collision (%). Original: 200 episodes; first rollout-loss version: 100 episodes.

| Model | Planner | Red | Purple | Diamond | Rotated 90° |
| --- | --- | ---: | ---: | ---: | ---: |
| jac, original | raw | 77.5 / 48.5 | 20.5 / 75.5 | 79.5 / 29.5 | 47.5 / 54.0 |
| jac, rollout loss from 3 frames | raw | 19.0 / 61.0 | 15.0 / 73.0 | 23.0 / 53.0 | 33.0 / 59.0 |
| jac, original | safety-first | 84.0 / 8.0 | 34.5 / 43.0 | 80.0 / 6.5 | 50.0 / 40.0 |
| jac, rollout loss from 3 frames | safety-first | 17.0 / 63.0 | 15.0 / 73.0 | 23.0 / 53.0 | 33.0 / 58.0 |
| sigreg, original | raw | 63.0 / 55.5 | 14.5 / 72.0 | 59.5 / 46.5 | 56.0 / 58.5 |
| sigreg, rollout loss from 3 frames | raw | 26.0 / 70.0 | 18.0 / 59.0 | 37.0 / 51.0 | 33.0 / 58.0 |

### Fixed version: rollout from 1 real frame

The fixed version starts every training rollout from one real frame and lets the context grow to 3, exactly as CEM does, so every context length is trained. To get an answer fast, jac and jac+pull+adv_h encoder-only were trained for 15 epochs (learning-rate schedule fitted to 15) and compared with the original models' checkpoints at a similar number of epochs.

Table R3. Per-step AUC of $h$ on imagined latents (as Table 6), red and purple starts, at matched training length. The original jac only kept every 10th checkpoint.

| Model | Epoch | Start | Step 1 | Step 3 | Step 5 | Step 10 | Step 15 | Step 25 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| jac, original | 10 | red | 0.99 | 0.99 | 0.98 | 0.82 | 0.66 | 0.55 |
| jac, rollout loss | 10 | red | 0.99 | 0.99 | 0.98 | 0.93 | 0.83 | 0.75 |
| jac, rollout loss | 15 | red | 0.99 | 0.99 | 0.99 | 0.95 | 0.86 | 0.72 |
| jac+pull+adv_h, encoder-only, original | 15 | red | 0.99 | 0.99 | 0.99 | 0.85 | 0.68 | 0.55 |
| jac+pull+adv_h, encoder-only, rollout loss | 15 | red | 0.99 | 0.99 | 0.99 | 0.95 | 0.82 | 0.73 |
| jac+pull+adv_h, encoder-only, original | 15 | purple | 0.81 | 0.57 | 0.54 | 0.52 | 0.52 | 0.47 |
| jac+pull+adv_h, encoder-only, rollout loss | 15 | purple | 0.52 | 0.64 | 0.62 | 0.52 | 0.49 | 0.48 |
| jac+pull+adv_h, encoder-only, original | 50 | purple | 0.80 | 0.88 | 0.81 | 0.54 | 0.54 | 0.51 |

- On red the rollout loss keeps the ranking much longer at equal training: +10-18 AUC points at steps 10-25 on both models.
- On purple it makes no net difference at epoch 15; both are at chance from step 10. (jac is near chance on purple for every version, since it is not color-robust.)

Table R4. Quick CEM screen, 50 episodes, success / collision (%), goal cost at every imagined step. jac: original epoch 10 vs rollout loss epoch 15.

| Setting | Original, epoch 10 | Rollout loss, epoch 15 |
| --- | ---: | ---: |
| Red, open loop, raw | 58.0 / 50.0 | 56.0 / 48.0 |
| Red, open loop, safety-first | 42.0 / 50.0 | 46.0 / 54.0 |
| Purple, open loop, raw | 22.0 / 68.0 | 6.0 / 78.0 |
| Purple, open loop, safety-first | 14.0 / 62.0 | 32.0 / 50.0 |
| Red, replan every 5, raw | 64.0 / 52.0 | 56.0 / 52.0 |
| Red, replan every 5, safety-first | 52.0 / 48.0 | 50.0 / 46.0 |
| Purple, replan every 5, raw | 16.0 / 56.0 | 14.0 / 76.0 |
| Purple, replan every 5, safety-first | 24.0 / 56.0 | 20.0 / 46.0 |

Table R5. Same screen for jac+pull+adv_h encoder-only, original vs rollout loss, both at epoch 15.

| Setting | Original, epoch 15 | Rollout loss, epoch 15 |
| --- | ---: | ---: |
| Red, open loop, raw | 46.0 / 48.0 | 46.0 / 44.0 |
| Red, open loop, safety-first | 40.0 / 54.0 | 44.0 / 56.0 |
| Purple, open loop, raw | 28.0 / 58.0 | 4.0 / 82.0 |
| Purple, open loop, safety-first | 32.0 / 42.0 | 26.0 / 60.0 |
| Red, replan every 5, raw | 48.0 / 54.0 | 62.0 / 46.0 |
| Red, replan every 5, safety-first | 52.0 / 44.0 | 60.0 / 32.0 |
| Purple, replan every 5, raw | 22.0 / 68.0 | 14.0 / 78.0 |
| Purple, replan every 5, safety-first | 10.0 / 64.0 | 24.0 / 66.0 |

- At 50 episodes a gap needs to be about 15 points to be clear. Fully trained models plan much better than these 10-15-epoch checkpoints (jac at 50 epochs: red open loop safety-first 76.0 / 22.0, 200 episodes), so these tables compare training recipes at equal length, not final models.
- Red: a tie open loop; with replanning, modestly better for jac+pull+adv_h encoder-only (8-14 more points of success, 8-12 fewer of collision) and a tie for jac.
- Purple, raw CEM: worse with the rollout loss in all 4 comparisons (success 22 -> 6 and 28 -> 4 open loop). Purple, safety-first: mixed.

### 50-epoch results (before pausing)

Same per-step test and CEM screen as the 15-epoch versions, on the fully trained models (50 epochs, cosine schedule fitted to 50). CEM: 50 episodes per rollout-loss cell; originals are the 200-episode runs of the main report.

Table R6. Per-step AUC of $h$ on imagined latents (as Table 6), 50 epochs. Original rows from Table 6 of post-oct-21.md and Table R1; blank = not measured. Held-out AUC of the GP margin function on real red latents: sigreg 0.902 with the rollout loss (the original sigreg margin function reaches 0.987 on the Table 14 states), jac 0.992, jac+pull+adv_h encoder-only 0.999.

| Model | Start | Step 1 | Step 3 | Step 5 | Step 10 | Step 15 | Step 25 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sigreg, rollout loss | red | 0.87 | 0.84 | 0.81 | 0.65 | 0.56 | 0.54 |
| sigreg, rollout loss | purple | 0.72 | 0.74 | 0.74 | 0.56 | 0.53 | 0.51 |
| jac, original | red | 1.00 | 1.00 | 0.99 | 0.99 | 0.97 | 0.88 |
| jac, rollout loss | red | 0.98 | 0.99 | 0.99 | 0.98 | 0.97 | 0.91 |
| jac, rollout loss | purple | 0.51 | 0.49 | 0.49 | 0.49 | 0.51 | 0.52 |
| jac+pull+adv_h, encoder-only, original | red | 1.00 | 1.00 | 1.00 | 0.98 | 0.93 | 0.77 |
| jac+pull+adv_h, encoder-only, rollout loss | red | 1.00 | 1.00 | 1.00 | 0.98 | 0.96 | 0.88 |
| jac+pull+adv_h, encoder-only, original | purple | 0.80 | 0.88 | 0.81 | 0.54 | 0.54 | 0.51 |
| jac+pull+adv_h, encoder-only, rollout loss | purple | 0.50 | 0.59 | 0.59 | 0.48 | 0.43 | 0.51 |

Table R7. CEM at 50 epochs, goal cost at every imagined step, success / collision (%). Raw = no safety cost; safety-first = GP margin function, $w = 1000$, $C = 10^4$. Blank = not run.

| Model | Setting | Scenario | Rollout loss, raw | Original, raw | Rollout loss, safety-first | Original, safety-first |
| --- | --- | --- | --- | --- | --- | --- |
| sigreg | replan every 5 | red | 12.0 / 66.0 | 63.0 / 55.5 | 16.0 / 64.0 | 63.0 / 55.5 |
| sigreg | replan every 5 | purple | 8.0 / 70.0 | 14.5 / 72.0 | 16.0 / 64.0 | 14.0 / 72.5 |
| sigreg | open loop | red | 20.0 / 68.0 | 61.0 / 52.5 | 22.0 / 58.0 | 60.0 / 52.0 |
| sigreg | open loop | purple | 16.0 / 66.0 | 35.5 / 67.0 | 20.0 / 54.0 | 34.5 / 68.5 |
| jac | replan every 5 | red | 76.0 / 42.0 | 77.5 / 48.5 | 76.0 / 10.0 | 84.0 / 8.0 |
| jac | replan every 5 | purple | 16.0 / 62.0 | 20.5 / 75.5 | 20.0 / 60.0 | 34.5 / 43.0 |
| jac | open loop | red | 74.0 / 48.0 | 68.0 / 52.0 | 66.0 / 26.0 | 76.0 / 22.0 |
| jac | open loop | purple | 48.0 / 56.0 | 16.0 / 80.5 | 52.0 / 24.0 | 43.0 / 26.5 |
| jac+pull+adv_h, encoder-only | replan every 5 | red | 76.0 / 38.0 | | 70.0 / 20.0 | 76.0 / 20.5 |
| jac+pull+adv_h, encoder-only | replan every 5 | purple | 36.0 / 66.0 | | 38.0 / 38.0 | 30.5 / 47.5 |
| jac+pull+adv_h, encoder-only | open loop | red | 66.0 / 34.0 | | 66.0 / 28.0 | |
| jac+pull+adv_h, encoder-only | open loop | purple | 36.0 / 68.0 | | 32.0 / 38.0 | |

- Red: the gain in imagined-state ranking at long horizons (step 25: jac 0.88 -> 0.91, jac+pull+adv_h encoder-only 0.77 -> 0.88) does not reach safety-first CEM: jac 84.0 / 8.0 -> 76.0 / 10.0 with replanning and 76.0 / 22.0 -> 66.0 / 26.0 open loop; jac+pull+adv_h encoder-only ties (76.0 / 20.5 vs 70.0 / 20.0).
- Purple, safety-first with replanning: mixed. jac collides more (43.0 -> 60.0); jac+pull+adv_h encoder-only collides less (47.5 -> 38.0, 50 episodes, about 1.2 binomial SDs), although its early-step ranking on purple, the only window where the original had any signal (0.80-0.88 at steps 1-5), drops to 0.50-0.59. Worth a 200-episode confirmation if the rollout loss is picked up again.
- The one large gain is raw CEM, open loop, purple for jac (16.0 / 80.5 -> 48.0 / 56.0, 50 episodes), consistent with better long-horizon rollouts; it disappears once the safety cost is on.
- sigreg: the rollout loss with only SIGReg on the latent made the encoder less informative (margin function 0.902 held-out AUC on real red latents; imagined red AUC 0.87 at step 1) and planning collapsed (red success 60-63 -> 12-22). The jac-base encoders were unaffected (0.992, 0.999).
- If picked up again: longer rollouts ($K = 10$ or 25) would matter mainly for open loop; the purple failure is the size of $h$'s gap, not prediction (Table 10), so a predictor-side loss is not expected to fix it.
