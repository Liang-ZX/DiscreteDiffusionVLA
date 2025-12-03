#!/bin/bash
set -euo pipefail

GPUS=(3 4 5 6 7)
MAX_PER_GPU=1
NUM_GPUS=${#GPUS[@]}
TOTAL_SLOTS=$((NUM_GPUS * MAX_PER_GPU))

LOG_DIR=./logs/discrete_diffusion_libero_object/$(date +'%m%d_%H%M')
mkdir -p "$LOG_DIR"

# 要跑的 STEPS（以列表形式定义，方便增删）
STEPS=(
  10000
  30000
  50000
  70000
  80000
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
  ASCEND_RT_VISIBLE_DEVICES=$GPU \
    python ./experiments/robot/libero/run_libero_eval.py \
      --pretrained_checkpoint "/data/ruanyifan/DiscreteDiffusionVLA/checkpoints/ddopenvla-libero-object/openvla-7b-exp+libero_object_no_noops+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--parallel_dec--8_acts_chunk--bin_acts--discrete_diffusion--3rd_person_img--wrist_img--proprio_state--20251127_1210--${STEP}_chkpt" \
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
