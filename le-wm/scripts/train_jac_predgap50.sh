#!/usr/bin/env bash
# jac + predictor gap at the full 50 epochs (2026-11-08), the one term that beat its jac control in the 15-epoch screen
# (jac_predgap15: 12-15 fewer collision points than jac15 in every condition). Same recipe as jac_predgap15: jac weight 1
# (L=1), safety_gap pred_gap_weight 1, gamma 0.5, 5-step rollout; seed 3072. Code: /data/seongbin/lewm/code_gapreg_766810f
# (= le-wm/gapreg at 766810f: the rollout encodes only the main 4-frame window). Waits for a GPU with >= 30 GB free.
set -uo pipefail
C=/data/seongbin/lewm/code_gapreg_766810f; cd "$C"
PY=/home/seongbin/latent/le-wm/.venv/bin/python
export STABLEWM_HOME=/data/seongbin/lewm LOCAL_DATASET_DIR=/data/seongbin/lewm WANDB_MODE=disabled OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
n=lewm_dubins_jac_predgap50
while :; do
  read -r g free < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | tr -d ',')
  [ "$free" -ge 30000 ] && break; sleep 120
done
echo "$(date +%H:%M) launching $n on gpu $g"
mkdir -p ".run_$n"; cd ".run_$n"
CUDA_VISIBLE_DEVICES=$g $PY ../train.py trainer.max_epochs=50 seed=3072 wandb.enabled=false output_model_name=$n subdir=$n \
  data=dubins_safety '+loss.regularizers.jacobian.weight=1.0' '+loss.regularizers.jacobian.kwargs={target_L:1.0,n_probes:1}' \
  '+loss.regularizers.safety_gap.weight=1.0' '+loss.regularizers.safety_gap.kwargs={pred_gap_weight:1.0,gamma:0.5,rollout_steps:5}' \
  > /home/seongbin/latent/run_logs/train_$n.log 2>&1
echo "exit $?"
