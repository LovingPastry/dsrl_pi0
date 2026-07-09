#!/bin/bash
# ============================================================================
# Multi-task DSRL ablation on LIBERO-Long (libero_10, 10 tasks) — the 8
# conditions, one shared SAC agent per run, 500k steps each, ONE Gaussian warmup
# trajectory PER TASK, resume-capable, with a multi-GPU pool scheduler
# (one experiment per GPU; the 8 queue across the GPUs you list).
#
#   # multi-GPU server (auto-allocates one experiment per card, queues the rest):
#   GPUS=0,1,2,3 bash examples/scripts/run_ablation8_multitask.sh full
#   GPUS=2,5     bash examples/scripts/run_ablation8_multitask.sh full            # 8 queue on 2 cards
#
#   # local single-GPU, run only DSRL-NA + VLM, sequentially (31GB-RAM safe):
#   GPUS=0 bash examples/scripts/run_ablation8_multitask.sh full na vlm
#
#   # tiny end-to-end smoke (2 tasks, ~60 steps, saves+reloads a checkpoint):
#   bash examples/scripts/run_ablation8_multitask.sh smoke na
#
#   # validate the scheduling on any machine (no GPU / no training):
#   DRY_RUN=1 GPUS=0,1,2 bash examples/scripts/run_ablation8_multitask.sh full
#
# RESUME: every job passes --resume 1, so simply re-running the SAME variant
# continues from its last checkpoint (SAC params + replay buffers + step + RNG).
# Re-run after a crash/OOM and it picks up where it stopped.
#
# View curves:  tensorboard --logdir "$EXP_ROOT/logs/DSRL_pi0_LiberoLong_MT8_500k"
#   evaluation/success_rate            -> per-run task-MEAN (the 8-curve headline)
#   evaluation/success_rate/task_<id>  -> per-task drill-down
#
# ⚠ HOST RAM: each job loads its own frozen pi0 (~14 GB) plus up to 10 LIBERO
#   EGL envs. On this 31 GB box run ONE at a time (GPUS=0). Multi-GPU parallelism
#   is for a big-RAM server (needs ~N x (14 GB + envs)).
# ============================================================================
set -u
MODE=${1:-smoke}; shift || true
VARIANTS=("$@")
if [ ${#VARIANTS[@]} -eq 0 ]; then
  VARIANTS=(baseline vlm buf2 vlm_buf2 na vlm_na na_buf2 vlm_na_buf2)
fi

GPUS=${GPUS:-0}
read -r -a GPU_ARR <<< "${GPUS//,/ }"
DRY_RUN=${DRY_RUN:-0}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME=${ENV_NAME:-dsrl_pi0}

EXP_NAME=${EXP_NAME:-DSRL_pi0_LiberoLong_MT8_500k}   # separate TB project from the previous single-task round
MT_TASK_IDS=${MT_TASK_IDS:-0,1,2,3,4,5,6,7,8,9}      # LIBERO-Long = 10 tasks
WARMUP_N=${WARMUP_N:-10}                             # one Gaussian warmup traj per task

# --- environment (skipped in DRY_RUN so the scheduler can be validated with no GPU) ---
if [ "$DRY_RUN" != "1" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
  export DISPLAY=${DISPLAY:-:0} MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
  export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-/media/fuyx/系统/lanzc/openpi_data_home}
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  cd "$REPO_DIR"
  export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
fi

EXP_ROOT=${EXP_ROOT:-$HOME/dsrl_runtime}
mkdir -p "$EXP_ROOT/logs" "$EXP_ROOT/jax_cache"
if [ "$MODE" = "smoke" ]; then export EXP="$EXP_ROOT/logs/${EXP_NAME}_smoke"; else export EXP="$EXP_ROOT/logs/${EXP_NAME}"; fi
mkdir -p "$EXP"

# --- step / eval budgets ---
if [ "$MODE" = "smoke" ]; then
  MT_TASK_IDS=${SMOKE_TASK_IDS:-0,1}
  WARMUP_N=2
  COMMON=(--max_steps ${MAX_STEPS:-60} --eval_interval ${EVAL_INTERVAL:-40} --log_interval 10 --eval_episodes 1
          --multi_grad_step 1 --start_online_updates 5 --batch_size 64 --checkpoint_interval ${CKPT_INTERVAL:-30})
else
  MAX_STEPS=${MAX_STEPS:-500000}; EVAL_INTERVAL=${EVAL_INTERVAL:-25000}; LOG_INTERVAL=${LOG_INTERVAL:-1000}
  COMMON=(--max_steps "$MAX_STEPS" --eval_interval "$EVAL_INTERVAL" --log_interval "$LOG_INTERVAL" --eval_episodes ${EVAL_EP:-5}
          --multi_grad_step 20 --start_online_updates 500 --batch_size 256 --checkpoint_interval ${CKPT_INTERVAL:-50000})
fi
COMMON+=(--algorithm pixel_sac --env libero --task_suite libero_10
         --multitask 1 --task_ids "$MT_TASK_IDS" --tasks_per_eval ${TASKS_PER_EVAL:--1}
         --action_magnitude 1.0 --query_freq 20 --discount 0.999
         --tb_project "$EXP_NAME" --seed 0 --resize_image 64 --hidden_dims 128
         --early_stop_success 0 --resume 1 --max_live_envs ${MAX_LIVE_ENVS:-2})

variant_flags() {
  case "$1" in
    baseline)     echo "--warmup_trajs $WARMUP_N" ;;
    vlm)          echo "--obs_mode vlm --warmup_trajs $WARMUP_N" ;;
    buf2)         echo "--warmup_trajs $WARMUP_N --dual_buffer 1" ;;
    na)           echo "--algorithm pixel_sac_na --warmup_trajs $WARMUP_N" ;;
    vlm_buf2)     echo "--obs_mode vlm --warmup_trajs $WARMUP_N --dual_buffer 1" ;;
    vlm_na)       echo "--obs_mode vlm --algorithm pixel_sac_na --warmup_trajs $WARMUP_N" ;;
    na_buf2)      echo "--algorithm pixel_sac_na --warmup_trajs $WARMUP_N --dual_buffer 1" ;;
    vlm_na_buf2)  echo "--obs_mode vlm --algorithm pixel_sac_na --warmup_trajs $WARMUP_N --dual_buffer 1" ;;
    *) echo "UNKNOWN" ;;
  esac
}

SUMMARY="$EXP/queue_multitask_$MODE.txt"
echo "=== multitask ablation $(date) mode=$MODE gpus=[${GPU_ARR[*]}] tasks=$MT_TASK_IDS variants=${VARIANTS[*]} ===" | tee -a "$SUMMARY"

run_job() {  # $1=gpu  $2=variant  — logs its own START/END; runs training (or a mock in DRY_RUN)
  local gpu=$1 v=$2 flags log rc ev
  flags=$(variant_flags "$v")
  if [ "$flags" = "UNKNOWN" ]; then echo "[queue] SKIP unknown variant '$v'" | tee -a "$SUMMARY"; return; fi
  log="$EXP/run_${MODE}_${v}.log"
  echo "[queue] START $v  gpu=$gpu  $(date +%H:%M:%S)  flags: $flags" | tee -a "$SUMMARY"
  if [ "$DRY_RUN" = "1" ]; then
    { echo "[DRY] CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu python3 examples/launch_train_sim.py ${COMMON[*]} $flags --prefix exp8mt_$v"
      sleep 1; } > "$log" 2>&1
    rc=$?
  else
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu \
      python3 examples/launch_train_sim.py "${COMMON[@]}" $flags --prefix "exp8mt_${v}" > "$log" 2>&1
    rc=$?
  fi
  ev=$(grep -E "mean success over|Success rate:" "$log" 2>/dev/null | tail -2 | tr '\n' ' ')
  echo "[queue] END   $v  gpu=$gpu  rc=$rc  $(date +%H:%M:%S)  last: $ev" | tee -a "$SUMMARY"
}

# --- GPU-pool scheduler: assign each pending variant to a free GPU; poll; reap; repeat ---
declare -A busy   # gpu_id -> background pid of its running job
pending=("${VARIANTS[@]}")
while [ ${#pending[@]} -gt 0 ] || [ ${#busy[@]} -gt 0 ]; do
  for gpu in "${GPU_ARR[@]}"; do
    [ ${#pending[@]} -eq 0 ] && break
    if [ -z "${busy[$gpu]:-}" ]; then
      v=${pending[0]}; pending=("${pending[@]:1}")
      run_job "$gpu" "$v" &
      busy[$gpu]=$!
    fi
  done
  sleep 5
  for gpu in "${!busy[@]}"; do
    pid=${busy[$gpu]}
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
      unset "busy[$gpu]"
    fi
  done
done
echo "[queue] ALL DONE $(date)" | tee -a "$SUMMARY"
