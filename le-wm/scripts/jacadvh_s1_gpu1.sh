#!/usr/bin/env bash
# jac+adv_h Dubins WM seed 1 on GPU ${G:-1}: wait until it has >= ${NEED:-30000} MiB free, train (from scratch), then the
# follow-up (margin functions, sweep, raw + CCEM K = 10 / 15) on the same GPU. Seed 2 is not touched.
set -uo pipefail
cd "$(dirname "$0")"
G=${G:-1}
echo "$(date +%H:%M) waiting for >= ${NEED:-30000} MiB free on GPU $G"
while :; do
  f=$(timeout 60 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$G" | tr -d ' ')
  [ -n "$f" ] && [ "$f" -ge "${NEED:-30000}" ] && break
  sleep 120
done
echo "$(date +%H:%M) GPU $G has $f MiB free; training seed 1"
SEEDS=1 G1=$G ./train_jacadvh_seeds_dubins.sh
G=$G ./jacadvh_s1_followup.sh 0
