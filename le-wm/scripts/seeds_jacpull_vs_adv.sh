#!/usr/bin/env bash
# Is jac+pull+adv_h's planning gap vs jac+pull noise? Raw CEM (sg25clean, red) + rollout diagnostic on WM seeds 1 and 2
# of both models (seed 3072 already done).
set -uo pipefail
cd "$(dirname "$0")"
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
ANY=/data/seongbin/dreamer/lewm/sigreg_only_dubins/margin_gp_fail_red_s0.pt   # unused at w=0 (raw CEM)
{ ./soft_cem.sh 0 lewm_dubins_jacpull50_s1 $ANY jacpull_s1 0 red;
  ./soft_cem.sh 0 lewm_dubins_jacpull50_s2 $ANY jacpull_s2 0 red; } &
{ ./soft_cem.sh 1 lewm_dubins_jacpull_safeadv_linf50_s1 $ANY jacpull_linf_s1 0 red;
  ./soft_cem.sh 1 lewm_dubins_jacpull_safeadv_linf50_s2 $ANY jacpull_linf_s2 0 red; } &
CUDA_VISIBLE_DEVICES=0 /home/seongbin/latent/le-wm/.venv/bin/python diag_rollout_safety.py \
  jacpull_s1=lewm_dubins_jacpull50_s1 jacpull_s2=lewm_dubins_jacpull50_s2 \
  jacpull_linf_s1=lewm_dubins_jacpull_safeadv_linf50_s1 jacpull_linf_s2=lewm_dubins_jacpull_safeadv_linf50_s2 2>&1 | grep -E "^DIAG|Error|Traceback"
wait; echo SEEDS_DONE
