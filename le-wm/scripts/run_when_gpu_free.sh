#!/usr/bin/env bash
# Wait until some GPU has >= MIN_FREE_MB free (default 12000), then run "<script> <gpu>:<rest>" with that GPU id.
# Usage: run_when_gpu_free.sh <script> <spec-without-gpu e.g. jacobian_w1_dubins:jacobian>
set -uo pipefail
MIN=${MIN_FREE_MB:-12000}
while true; do
  g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' -v m="$MIN" '$2 > m {print $1; exit}')
  [ -n "$g" ] && break
  sleep 300
done
exec "$1" "$g:$2"
