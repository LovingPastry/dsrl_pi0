#!/bin/bash
# ============================================================================
# Parallel multi-GPU scheduler for the 8-condition DSRL ablation.
# One experiment per GPU; the 8 experiments queue across the GPUs you list —
# if you give fewer GPUs than experiments, they wait in a pool and each GPU
# picks up the next pending experiment as soon as it frees.
#
# Works for BOTH envs:
#   LIBERO (default):  bash examples/scripts/run_ablation8_parallel.sh full
#   ALOHA:             ENV_KIND=aloha bash examples/scripts/run_ablation8_parallel.sh full
#   (or use the thin wrapper run_ablation8_aloha_parallel.sh)
#
# Pick the usable GPUs with GPUS (comma-separated). Examples:
#   GPUS=0,1,2,3 bash examples/scripts/run_ablation8_parallel.sh full   # 8 jobs across 4 cards
#   GPUS=0,1,2   bash examples/scripts/run_ablation8_parallel.sh full   # 8 jobs queue on 3 cards
#   GPUS=2,5     bash examples/scripts/run_ablation8_parallel.sh full baseline vlm na   # subset on cards 2,5
#   DRY_RUN=1 GPUS=0,1,2 bash examples/scripts/run_ablation8_parallel.sh full  # validate scheduling, no training
#
# View curves:  tensorboard --logdir "$EXP_ROOT/logs/DSRL_pi0_Libero_ABL8"   (or DSRL_pi0_Aloha_ABL8)
#
# ⚠ HOST RAM: every job loads its own frozen pi0 (~14 GB host RAM). N parallel
#   jobs need ~N×14 GB of system RAM (plus ~10 GB VRAM per GPU). If the server
#   is RAM-limited, list FEWER GPUs than you have — the pool still runs all 8,
#   just with less concurrency. (We OOM-killed a run this way on a 31 GB box.)
# ============================================================================
set -u
MODE=${1:-smoke}; shift || true
VARIANTS=("$@")
if [ ${#VARIANTS[@]} -eq 0 ]; then
  VARIANTS=(baseline vlm buf2 vlm_buf2 na vlm_na na_buf2 vlm_na_buf2)
fi

GPUS=${GPUS:-0}
read -r -a GPU_ARR <<< "${GPUS//,/ }"
ENV_KIND=${ENV_KIND:-libero}
DRY_RUN=${DRY_RUN:-0}
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME=${ENV_NAME:-dsrl_pi0}
CKPT_INTERVAL=${CKPT_INTERVAL:--1}

# --- env-specific flags/defaults ---
if [ "$ENV_KIND" = "aloha" ]; then
  EXP_NAME=${EXP_NAME:-DSRL_pi0_Aloha_ABL8}
  ENV_FLAGS=(--env aloha_cube --action_magnitude 2.0 --query_freq 50 --target_entropy 0.0)
  START_ONLINE=1000
else
  EXP_NAME=${EXP_NAME:-DSRL_pi0_Libero_ABL8}
  SUITE=${SUITE:-libero_90}; TASK_ID=${TASK_ID:-46}
  ENV_FLAGS=(--env libero --task_suite "$SUITE" --task_id "$TASK_ID" --action_magnitude 1.0 --query_freq 20)
  START_ONLINE=500
fi

# --- environment (skipped in DRY_RUN so the scheduler can be validated with no GPU) ---
if [ "$DRY_RUN" != "1" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
  export DISPLAY=${DISPLAY:-:0} MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
  export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-$HOME/.cache/openpi}
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  cd "$REPO_DIR"
  export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
fi

EXP_ROOT=${EXP_ROOT:-$HOME/dsrl_runtime}
mkdir -p "$EXP_ROOT/logs" "$EXP_ROOT/jax_cache"
if [ "$MODE" = "smoke" ]; then export EXP="$EXP_ROOT/logs/${EXP_NAME}_smoke"; else export EXP="$EXP_ROOT/logs/${EXP_NAME}"; fi
mkdir -p "$EXP"

# --- step / eval budgets ---
if [ "$MODE" = "smoke" ]; then
  COMMON=(--max_steps 400 --eval_interval 200 --log_interval 50 --eval_episodes 1
          --multi_grad_step 10 --start_online_updates 40 --batch_size 64 --checkpoint_interval "$CKPT_INTERVAL")
  WARMUP_N=2
else
  MAX_STEPS=${MAX_STEPS:-250000}; EVAL_INTERVAL=${EVAL_INTERVAL:-12500}; LOG_INTERVAL=${LOG_INTERVAL:-500}
  COMMON=(--max_steps "$MAX_STEPS" --eval_interval "$EVAL_INTERVAL" --log_interval "$LOG_INTERVAL" --eval_episodes 10
          --multi_grad_step 20 --start_online_updates "$START_ONLINE" --batch_size 256 --checkpoint_interval "$CKPT_INTERVAL")
  WARMUP_N=10
fi
COMMON+=(--algorithm pixel_sac "${ENV_FLAGS[@]}" --tb_project "$EXP_NAME"
         --discount 0.999 --seed 0 --resize_image 64 --hidden_dims 128 --early_stop_success 0)

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

SUMMARY="$EXP/queue_parallel_$MODE.txt"
echo "=== parallel ablation $(date) env=$ENV_KIND mode=$MODE gpus=[${GPU_ARR[*]}] variants=${VARIANTS[*]} ===" | tee -a "$SUMMARY"

run_job() {  # $1=gpu  $2=variant  — logs its own START/END; runs the training (or a mock in DRY_RUN)
  local gpu=$1 v=$2 flags log rc ev
  flags=$(variant_flags "$v")
  if [ "$flags" = "UNKNOWN" ]; then echo "[queue] SKIP unknown variant '$v'" | tee -a "$SUMMARY"; return; fi
  log="$EXP/run_${MODE}_${v}.log"
  echo "[queue] START $v  gpu=$gpu  $(date +%H:%M:%S)  flags: $flags" | tee -a "$SUMMARY"
  if [ "$DRY_RUN" = "1" ]; then
    { echo "[DRY] CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu python3 examples/launch_train_sim.py ${COMMON[*]} $flags --prefix exp8_$v"
      sleep $(( (RANDOM % 3) + 1 )); } > "$log" 2>&1
    rc=$?
  else
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$gpu MUJOCO_EGL_DEVICE_ID=$gpu \
      python3 examples/launch_train_sim.py "${COMMON[@]}" $flags --prefix "exp8_${v}" > "$log" 2>&1
    rc=$?
  fi
  ev=$(grep -E "Success rate:" "$log" 2>/dev/null | tail -2 | tr '\n' ' ')
  echo "[queue] END   $v  gpu=$gpu  rc=$rc  $(date +%H:%M:%S)  last: $ev" | tee -a "$SUMMARY"
}

# --- GPU-pool scheduler: assign each pending variant to a free GPU; poll; reap; repeat ---
declare -A busy   # gpu_id -> background pid of its running job
pending=("${VARIANTS[@]}")
while [ ${#pending[@]} -gt 0 ] || [ ${#busy[@]} -gt 0 ]; do
  # dispatch pending variants onto any currently-free GPUs
  for gpu in "${GPU_ARR[@]}"; do
    [ ${#pending[@]} -eq 0 ] && break
    if [ -z "${busy[$gpu]:-}" ]; then
      v=${pending[0]}; pending=("${pending[@]:1}")
      run_job "$gpu" "$v" &
      busy[$gpu]=$!
    fi
  done
  sleep 5
  # reap finished jobs, freeing their GPUs for the next pending variant
  for gpu in "${!busy[@]}"; do
    pid=${busy[$gpu]}
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" 2>/dev/null || true
      unset "busy[$gpu]"
    fi
  done
done
echo "[queue] ALL DONE $(date)" | tee -a "$SUMMARY"
