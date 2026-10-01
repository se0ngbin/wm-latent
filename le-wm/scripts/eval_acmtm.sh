#!/usr/bin/env bash
# T1a+T1b evals for the AC-MTM WMs (see train_acmtm.sh): encoder mechanism probe + margin_gp OOD,
# dubins and safety-gym. Waits for each WM's epoch-50 checkpoint. Reference encoders are re-run at
# probe seed 0 as a reproduction check against run_logs/ood_gp_matrix.out / sg_robust_margin.log.
# Usage: eval_acmtm.sh [wm_suffix]   ("" = seed 3072, "_s1", "_s2")
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=/home/seongbin/latent/le-wm/.venv/bin/python
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm
SFX=${1:-}; GPU=${GPU:-0}; LOG=/home/seongbin/latent/run_logs/acmtm; mkdir -p "$LOG"
CK=$STABLEWM_HOME/checkpoints
wait_ck(){ until [ -f "$CK/$1/weights_epoch_50.pt" ]; do sleep 300; done; sleep 60; }
run(){ local log=$1; shift; echo "[$(date +%F' '%H:%M:%S)] $log"; CUDA_VISIBLE_DEVICES=$GPU "$@" > "$LOG/$log.log" 2>&1 \
  || echo "  FAILED $log"; grep -h "^MECH\|^KEY\|effrank=" "$LOG/$log.log"; }

DA=acmtm:dubins_acmtm$SFX,acmtm+jac:dubins_acmtm_jac$SFX
REF_D=baseline:sigreg_only_dubins,jacobian:jacobian_w1_dubins,jacobian+pull:lewm_dubins_jacpull50
SA=acmtm:sg_acmtm$SFX,acmtm+jac:sg_acmtm_jac$SFX
REF_S=baseline:sg_baseline,jacobian:sg_jacobian,jacobian+pull:sg_jacpull

wait_ck dubins_acmtm$SFX; wait_ck dubins_acmtm_jac$SFX
[ -z "$SFX" ] && run jac_singular_dubins $PY -u jac_singular.py acmtm_dubins
ENCODERS=$([ -z "$SFX" ] && echo "$REF_D,$DA" || echo "$DA") run dubins_mech$SFX $PY -u dubins_mechanism.py
for axis in color shape rotate; do for s in 0 1 2; do
  enc=$DA; [ -z "$SFX" ] && [ $s = 0 ] && enc="$REF_D,$DA"
  ENCODERS=$enc run ood_gp_dubins${SFX}_${axis}_s$s $PY -u ood_margin_gp_jepa.py $axis $s
done; done

wait_ck sg_acmtm$SFX; wait_ck sg_acmtm_jac$SFX
ENCODERS=$([ -z "$SFX" ] && echo "$REF_S,$SA" || echo "$SA") run sg_mech$SFX $PY -u sg_mechanism.py
for s in 0 1; do
  enc=$SA; [ -z "$SFX" ] && [ $s = 0 ] && enc="$REF_S,$SA"
  SG_ENCODERS=$enc SG_SEED=$s run sg_robmarg${SFX}_s$s $PY -u sg_robust_margin.py
done
echo "ACMTM_EVAL_DONE$SFX"
