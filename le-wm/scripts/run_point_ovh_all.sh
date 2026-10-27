#!/usr/bin/env bash
# All 9 clean-Point recipes on the OVERHEAD@224 data (suffix _ovh, seed 3072). pt_baseline_ovh / pt_jacpull_ovh exist from
# the pilot and are skipped (cached). Supervisor relaunches OOM deaths, then runs the layout-disjoint eval (3 eval seeds).
CODE=/data/seongbin/lewm/code_point_92e7589; D=/data/seongbin/lewm/datasets; R=/home/seongbin/latent/run_logs
export DATA=sg_point_ovh224_safety
( cd $CODE && SFX=_ovh NOWAIT=1 ./run_point.sh pt_jacobian pt_safeadv_linf_g0 pt_jac_safeadv_linf pt_jacpull_safeadv_linf \
    pt_lam_bdet pt_jac_lam_bdet pt_jacpull_lam_bdet > $R/sg_point_queue_ovh.log 2>&1 ) &
cd "$(dirname "${BASH_SOURCE[0]}")"
SFXS=_ovh TRAIN_H5=$D/sg_point_ovh224.h5 SG_DATA=$D/sg_point_ovh224_multi_eval.npz ./run_point_eval_when_ready.sh
