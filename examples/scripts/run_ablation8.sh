#!/bin/bash
# 8-condition DSRL ablation on one LIBERO task (baseline / vlm / buf1 / buf2 /
# na / vlm_buf2 / na_buf2 / vlm_na_buf2), sequential on one GPU.
#   full : 25k steps, eval every 2.5k x 10 episodes, no 100%-early-stop
#   smoke: ~300 steps, 1-episode evals, tiny warmup - pipeline check only
# Usage:
#   SUITE=libero_90 TASK_ID=46 bash examples/scripts/run_ablation8.sh full [variant ...]
# (no variant args = all 8)
MODE=${1:-smoke}; shift || true
VARIANTS=("$@")
if [ ${#VARIANTS[@]} -eq 0 ]; then
  VARIANTS=(baseline vlm buf1 buf2 na vlm_buf2 na_buf2 vlm_na_buf2)
fi
device_id=0

SUITE=${SUITE:-libero_90}
TASK_ID=${TASK_ID:-46}
# Overridable at launch (full mode): step budget, eval/log cadence, checkpointing, output dir.
EXP_NAME=${EXP_NAME:-DSRL_pi0_Libero_ABL8}
CKPT_INTERVAL=${CKPT_INTERVAL:--1}

source /home/fuyx/anaconda3/etc/profile.d/conda.sh
conda activate dsrl_pi0

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$device_id

export OPENPI_DATA_HOME=/media/fuyx/系统/lanzc/openpi_data_home
BIGDISK=/media/fuyx/系统/lanzc/dsrl_runtime
mkdir -p "$BIGDISK/logs" "$BIGDISK/jax_cache"
if [ "$MODE" = "smoke" ]; then
  export EXP="$BIGDISK/logs/${EXP_NAME}_smoke"
else
  export EXP="$BIGDISK/logs/${EXP_NAME}"
fi
mkdir -p "$EXP"

export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd /home/fuyx/lanzc/dsrl_pi0
export PYTHONPATH=/home/fuyx/lanzc/dsrl_pi0:$PYTHONPATH

if [ "$MODE" = "smoke" ]; then
  COMMON=(--max_steps 300 --eval_interval 150 --log_interval 50 --eval_episodes 1
          --multi_grad_step 10 --start_online_updates 40 --batch_size 64
          --tb_project "${EXP_NAME}_smoke" --checkpoint_interval "$CKPT_INTERVAL")
  WARMUP_N=2
else
  # full mode: step budget + eval/log cadence overridable via env (defaults = the original 25k config)
  MAX_STEPS=${MAX_STEPS:-25000}
  EVAL_INTERVAL=${EVAL_INTERVAL:-2500}
  LOG_INTERVAL=${LOG_INTERVAL:-500}
  COMMON=(--max_steps "$MAX_STEPS" --eval_interval "$EVAL_INTERVAL" --log_interval "$LOG_INTERVAL" --eval_episodes 10
          --multi_grad_step 20 --start_online_updates 500 --batch_size 256
          --tb_project "${EXP_NAME}" --checkpoint_interval "$CKPT_INTERVAL")
  WARMUP_N=10
fi
COMMON+=(--algorithm pixel_sac --env libero --task_suite "$SUITE" --task_id "$TASK_ID"
         --discount 0.999 --seed 0 --resize_image 64 --action_magnitude 1.0
         --query_freq 20 --hidden_dims 128 --early_stop_success 0)

variant_flags() {
  case "$1" in
    baseline)     echo "" ;;
    vlm)          echo "--obs_mode vlm" ;;
    buf1)         echo "--warmup_trajs $WARMUP_N" ;;
    buf2)         echo "--warmup_trajs $WARMUP_N --dual_buffer 1" ;;
    na)           echo "--algorithm pixel_sac_na" ;;
    vlm_buf2)     echo "--obs_mode vlm --warmup_trajs $WARMUP_N --dual_buffer 1" ;;
    na_buf2)      echo "--algorithm pixel_sac_na --warmup_trajs $WARMUP_N --dual_buffer 1" ;;
    vlm_na_buf2)  echo "--obs_mode vlm --algorithm pixel_sac_na --warmup_trajs $WARMUP_N --dual_buffer 1" ;;
    *) echo "UNKNOWN" ;;
  esac
}

SUMMARY="$EXP/queue_summary_$MODE.txt"
echo "=== ablation queue $(date) mode=$MODE suite=$SUITE task=$TASK_ID variants=${VARIANTS[*]} ===" >> "$SUMMARY"
for v in "${VARIANTS[@]}"; do
  flags=$(variant_flags "$v")
  if [ "$flags" = "UNKNOWN" ]; then echo "unknown variant $v" | tee -a "$SUMMARY"; continue; fi
  LOG="$EXP/run_${MODE}_${v}.log"
  echo "[queue] START $v  $(date +%H:%M:%S)  flags: $flags" | tee -a "$SUMMARY"
  # shellcheck disable=SC2086
  python3 examples/launch_train_sim.py "${COMMON[@]}" $flags --prefix "exp8_${v}" > "$LOG" 2>&1
  rc=$?
  tailmsg=$(grep -E "Success rate:" "$LOG" | tail -3 | tr '\n' ' ')
  echo "[queue] END   $v  rc=$rc  $(date +%H:%M:%S)  last evals: $tailmsg" | tee -a "$SUMMARY"
done
echo "[queue] ALL DONE $(date)" | tee -a "$SUMMARY"
