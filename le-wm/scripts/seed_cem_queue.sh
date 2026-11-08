#!/usr/bin/env bash
# Paper grid (paper-results.tex), LeWM protocol: 50 episodes per WM seed, averaged over 3 training seeds afterwards.
# Per spec <wm_dir>:<tag>, in order: wait for <wm_dir>/weights_epoch_50.pt; train the GP and linear margin functions if
# missing and score them on real frames once (seed_head_eval.sh -> run_logs/seed_head_eval_q.log); then CEM on red,
# purple and rotated 90° (SCENS) for each planner in PLANNERS: raw (w = 0, C = 0) and sf (safety-first, GP margin
# function, w = 1000, C = 1e4). Default open loop (RECS=25) with the final-step goal cost (GOALCOST=final);
# replanning is RECS="5" GOALCOST=mean. Usage: GPU=<id> [PLANNERS="raw sf"] seed_cem_queue.sh <spec> [...]
set -uo pipefail
cd "$(dirname "$0")"
export GOALCOST=${GOALCOST:-final} NEVAL=${NEVAL:-50}
M=/data/seongbin/dreamer/lewm; C=/data/seongbin/lewm/checkpoints; G=${GPU:-1}; LOG=/home/seongbin/latent/run_logs
for spec in "$@"; do
  IFS=: read -r ck t <<< "$spec"
  until [ -f $C/$ck/weights_epoch_50.pt ]; do sleep 300; done
  grep -q "HEADOOD $ck " $LOG/seed_head_eval_*.log $LOG/lin_h_metrics.log 2>/dev/null || GPU=$G ./seed_head_eval.sh $ck >> $LOG/seed_head_eval_q.log 2>&1
  [ -f $M/$ck/margin_gp_fail_red_s0.pt ] || { echo "$(date +%H:%M) no margin function for $ck, skipping"; continue; }
  for rec in ${RECS:-25}; do
    for p in ${PLANNERS:-raw sf}; do
      if [ "$p" = raw ]; then CONST=0 REC=$rec ./soft_cem.sh $G $ck $M/$ck/margin_gp_fail_red_s0.pt ${t}_gr 0 ${SCENS:-red purple fullrot90}
      else CONST=10000 REC=$rec ./soft_cem.sh $G $ck $M/$ck/margin_gp_fail_red_s0.pt ${t}_gp_gr 1000 ${SCENS:-red purple fullrot90}; fi
    done
  done
done
echo SEED_CEM_DONE
