#!/usr/bin/env bash
# Wait for all 9 clean-Point WMs + the multi-variant eval set, then run sg_point_eval.py for eval seeds 0-2
# on whichever GPU has the most free memory. Prints the SGPT lines at the end.
cd "$(dirname "${BASH_SOURCE[0]}")"
CK=/data/seongbin/lewm/checkpoints; L=/home/seongbin/latent/run_logs/sg_point
NAMES="pt_baseline pt_jacobian pt_jacpull pt_safeadv_linf_g0 pt_jac_safeadv_linf pt_jacpull_safeadv_linf pt_lam_bdet pt_jac_lam_bdet pt_jacpull_lam_bdet"
ready(){ [ -f /data/seongbin/lewm/datasets/sg_point_multi_eval.npz ] || return 1
  for n in $NAMES; do [ -f "$CK/$n/weights_epoch_50.pt" ] || return 1; done; }
until ready; do
  # give up on runs whose training process died (FAILED in the queue log) instead of waiting forever
  # (pt_jacpull_safeadv_linf runs under the first launcher, whose log also holds the 8 startup-OOM FAILED lines)
  if grep -qh "FAILED" /home/seongbin/latent/run_logs/sg_point_queue.log 2>/dev/null || grep -q "FAILED pt_jacpull_safeadv_linf " /home/seongbin/latent/run_logs/sg_point_launcher.log; then
    echo "a run FAILED:"; grep -h FAILED /home/seongbin/latent/run_logs/sg_point_queue.log; grep "FAILED pt_jacpull_safeadv_linf " /home/seongbin/latent/run_logs/sg_point_launcher.log; exit 1; fi
  sleep 300
done
gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
for s in 0 1 2; do CUDA_VISIBLE_DEVICES=$gpu SG_SEED=$s /home/seongbin/latent/le-wm/.venv/bin/python sg_point_eval.py > $L/eval_seed$s.log 2>&1; done
grep -h SGPT $L/eval_seed*.log
