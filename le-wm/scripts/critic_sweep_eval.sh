#!/usr/bin/env bash
# Eval every critic in the seed sweep and emit normalized RESULT lines:
#   RESULT <model> <column> <cond> <seed> <acc> <auc>
# column: indist (in-dist critic on red) | zeroshot (in-dist critic on a shift) |
#         retrained (per-shift critic on its own shift).
set -uo pipefail
cd /home/seongbin/latent/latent_cbf/src/latent_cbf
export PYTHONPATH=/home/seongbin/latent/le-wm STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm
PY=/home/seongbin/latent/latent_cbf/.venv/bin/python
VDIR=/data/seongbin/dreamer/lewm/_value_ood; MDIR=$VDIR/margins; SW=$VDIR/_sweep; LOG=/home/seongbin/latent/run_logs
RSSM=/data/seongbin/dreamer/dreamer/enc_lip_sweep/baseline/rssm_ckpt.pt
SEEDS=${1:-"0 1 2"}
declare -A WM=( [baseline]=sigreg_only_dubins [jacobian]=jacobian_w1_dubins [jacpull]=lewm_dubins_jacpull50 )
OUT="$LOG/critic_sweep_results.txt"; : > "$OUT"
pol() { ls -d "$1"/PyHJ/gp/epoch_id_* 2>/dev/null | sort -t_ -k3 -n | tail -1; }

eval_policy() {  # tag policy [is_dreamer]
  local tag=$1 p=$2 dr=${3:-0}
  local w=${WM[${tag%%_*}]:-}   # base model = first token (baseline_s0 -> baseline)
  if [ "$dr" = 1 ]; then
    $PY scripts/critic_ood_eval_dreamer.py --tag "$tag" --rssm_ckpt "$RSSM" --policy "$p" 2>/dev/null | grep "KEY critic"
  else
    $PY scripts/critic_ood_eval.py --tag "$tag" \
      --lewm_ckpt /data/seongbin/lewm/checkpoints/$w/weights_epoch_50.pt \
      --margin_ckpt /data/seongbin/dreamer/lewm/$w/margin_heads.pt --policy "$p" 2>/dev/null | grep "KEY critic"
  fi
}

emit() {  # model column cond seed line
  local acc auc
  acc=$(echo "$5" | grep -oE "acc=[0-9.]+" | cut -d= -f2)
  auc=$(echo "$5" | grep -oE "auc=[0-9.]+" | cut -d= -f2)
  echo "RESULT $1 $2 $3 $4 $acc $auc" | tee -a "$OUT"
}

for s in $SEEDS; do
  for tag in baseline jacobian jacpull dreamer; do
    dr=0; [ "$tag" = dreamer ] && dr=1
    # in-dist critic: red=in-dist, shifts=zero-shot
    ip=$(pol "$SW/${tag}_s${s}")/policy.pth
    if [ -f "$ip" ]; then
      while read -r ln; do
        c=$(echo "$ln" | grep -oE "cond=[a-z0-9]+" | cut -d= -f2)
        col=indist; [ "$c" != red ] && col=zeroshot
        emit "$tag" "$col" "$c" "$s" "$ln"
      done < <(eval_policy "${tag}_s${s}" "$ip" "$dr")
    else echo "no indist policy $tag s$s"; fi
    # retrained critics
    for cond in purple diamond rot90; do
      rp=$(pol "$SW/${tag}_${cond}_s${s}")/policy.pth
      [ -f "$rp" ] || { echo "no retr policy $tag/$cond s$s"; continue; }
      ln=$(eval_policy "${tag}_${cond}_s${s}" "$rp" "$dr" | grep "cond=${cond} ")
      [ -n "$ln" ] && emit "$tag" retrained "$cond" "$s" "$ln"
    done
  done
done
echo CRITIC_SWEEP_EVAL_DONE | tee -a "$OUT"
