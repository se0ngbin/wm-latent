#!/usr/bin/env bash
# Training seeds 1 and 2 of all 9 recipes on the overhead@224 data (suffixes _ovh_s1/_ovh_s2), one memory-aware queue,
# then the supervisor (OOM retries) and the layout-disjoint eval (3 eval seeds each).
CODE=/data/seongbin/lewm/code_point_92e7589; D=/data/seongbin/lewm/datasets; R=/home/seongbin/latent/run_logs
export DATA=sg_point_ovh224_safety
( cd $CODE && { SEED=1 SFX=_ovh_s1 NOWAIT=1 ./run_point.sh; SEED=2 SFX=_ovh_s2 NOWAIT=1 ./run_point.sh; } > $R/sg_point_queue_ovh_seeds.log 2>&1 ) &
cd "$(dirname "${BASH_SOURCE[0]}")"
SFXS="_ovh_s1 _ovh_s2" TRAIN_H5=$D/sg_point_ovh224.h5 SG_DATA=$D/sg_point_ovh224_multi_eval.npz ./run_point_eval_when_ready.sh
