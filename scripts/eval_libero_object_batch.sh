#!/bin/bash
set -euo pipefail

GPUS=(0 1 2 3)
MAX_PER_GPU=2
NUM_GPUS=${#GPUS[@]}
TOTAL_SLOTS=$((NUM_GPUS * MAX_PER_GPU))

LOG_DIR=../logs/discrete_diffusion_libero_spatial/$(date +'%m%d_%H%M')
mkdir -p "$LOG_DIR"

# 要跑的 STEPS（以列表形式定义，方便增删）
STEPS=(
  300000
  290000
  280000
  270000
  260000
  250000
  ...  # please add all your checkpoints
  60000
  70000
  50000
)

# initialization
declare -a JOB_PIDS
for ((i=0; i<TOTAL_SLOTS; i++)); do
  JOB_PIDS[i]=0
done

start_job() {
  local STEP=$1
  local SLOT=$2
  # calculate respective slot
  local GPU_INDEX=$(( SLOT / MAX_PER_GPU ))
  local GPU=${GPUS[$GPU_INDEX]}

  echo "[$(date +'%H:%M:%S')] START STEP=${STEP} on GPU=${GPU} (slot ${SLOT})"
  CUDA_VISIBLE_DEVICES=$GPU \
    python ../experiments/robot/libero/run_libero_eval.py \
      --pretrained_checkpoint "/path/to/xxx--${STEP}_chkpt" \
      --task_suite_name libero_object \
      --use_l1_regression False \
      --use_diffusion False \
      --use_discrete_diffusion True \
      --use_film False \
      --num_images_in_input 2 \
      --use_proprio True \
      --topk_filter_thres 0.0 \
    > "$LOG_DIR/eval_${STEP}.log" 2>&1 &

  JOB_PIDS[$SLOT]=$!
}

# tranverse all STEP，start if there are empty slots
for STEP in "${STEPS[@]}"; do
  while :; do
    for ((slot=0; slot<TOTAL_SLOTS; slot++)); do
      pid=${JOB_PIDS[slot]}
      if [[ $pid -eq 0 ]] || ! kill -0 "$pid" 2>/dev/null; then
        start_job "$STEP" "$slot"
        break 2
      fi
    done
    sleep 2
  done
done

# wait
for pid in "${JOB_PIDS[@]}"; do
  [[ $pid -ne 0 ]] && wait "$pid"
done

echo "All evaluations finished."
