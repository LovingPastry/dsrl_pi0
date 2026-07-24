#!/bin/bash
# ============================================================================
# 8-condition DSRL ablation on the ALOHA cube-transfer sim (gym-aloha).
# Mirrors run_ablation8.sh (LIBERO) but for --env aloha_cube.
#
# 2³ factorial ablation across three modules:
#   Module 1 (obs):      CNN (off) vs VLM (on)        → --obs_mode vlm
#   Module 2 (buffer):   single (off) vs dual (on)    → --dual_buffer 1
#   Module 3 (alg):      pixel_sac (off) vs NA (on)   → --algorithm pixel_sac_na
#   All 8 conditions share --warmup_trajs N (Gaussian exploration before updates).
#
#   baseline      CNN  + single + SAC     (0,0,0)
#   vlm           VLM  + single + SAC     (1,0,0)
#   buf2          CNN  + dual   + SAC     (0,1,0)
#   vlm_buf2      VLM  + dual   + SAC     (1,1,0)
#   na            CNN  + single + NA      (0,0,1)
#   vlm_na        VLM  + single + NA      (1,0,1)
#   na_buf2       CNN  + dual   + NA      (0,1,1)
#   vlm_na_buf2   VLM  + dual   + NA      (1,1,1)
#
# Usage (after setup_env.sh):
#   bash examples/scripts/run_ablation8_aloha.sh full            # all 8, 250k steps
#   bash examples/scripts/run_ablation8_aloha.sh smoke           # quick pipeline check
#   bash examples/scripts/run_ablation8_aloha.sh full baseline vlm na   # subset
# Overridable env:
#   MAX_STEPS=250000 EVAL_INTERVAL=12500 CKPT_INTERVAL=50000 DEVICE=0 \
#   OPENPI_DATA_HOME=~/.cache/openpi EXP_ROOT=~/dsrl_runtime \
#   bash examples/scripts/run_ablation8_aloha.sh full
#   View curves:  tensorboard --logdir "$EXP_ROOT/logs/DSRL_pi0_Aloha_ABL8"
# ============================================================================
set -e
MODE=${1:-smoke}; shift || true
VARIANTS=("$@")
if [ ${#VARIANTS[@]} -eq 0 ]; then
  VARIANTS=(baseline vlm buf2 vlm_buf2 na vlm_na na_buf2 vlm_na_buf2)
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
device_id=${DEVICE:-0}
ENV_NAME=${ENV_NAME:-dsrl_pi0}
EXP_NAME=${EXP_NAME:-DSRL_pi0_Aloha_ABL8}
CKPT_INTERVAL=${CKPT_INTERVAL:--1}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# --- rendering (EGL headless) ---
export DISPLAY=${DISPLAY:-:0}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$device_id

# --- data / logs / cache (portable defaults; override via env) ---
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-$HOME/.cache/openpi}
EXP_ROOT=${EXP_ROOT:-$HOME/dsrl_runtime}
mkdir -p "$EXP_ROOT/logs" "$EXP_ROOT/jax_cache"
if [ "$MODE" = "smoke" ]; then
  export EXP="$EXP_ROOT/logs/${EXP_NAME}_smoke"
else
  export EXP="$EXP_ROOT/logs/${EXP_NAME}"
fi
mkdir -p "$EXP"

# --- gpu / jax ---
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# ALOHA-specific hyperparams (from examples/scripts/run_aloha.sh):
#   query_freq 50, action_magnitude 2.0, target_entropy 0.0, env aloha_cube.
if [ "$MODE" = "smoke" ]; then
  COMMON=(--max_steps 400 --eval_interval 200 --log_interval 50 --eval_episodes 1
          --multi_grad_step 10 --start_online_updates 40 --batch_size 64
          --tb_project "${EXP_NAME}_smoke" --checkpoint_interval "$CKPT_INTERVAL")
  WARMUP_N=2
else
  MAX_STEPS=${MAX_STEPS:-250000}
  EVAL_INTERVAL=${EVAL_INTERVAL:-12500}
  LOG_INTERVAL=${LOG_INTERVAL:-500}
  COMMON=(--max_steps "$MAX_STEPS" --eval_interval "$EVAL_INTERVAL" --log_interval "$LOG_INTERVAL" --eval_episodes 10
          --multi_grad_step 20 --start_online_updates 1000 --batch_size 256
          --tb_project "${EXP_NAME}" --checkpoint_interval "$CKPT_INTERVAL")
  WARMUP_N=10
fi
COMMON+=(--algorithm pixel_sac --env aloha_cube
         --discount 0.999 --seed 0 --resize_image 64 --action_magnitude 2.0
         --query_freq 50 --hidden_dims 128 --target_entropy 0.0 --early_stop_success 0)

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

SUMMARY="$EXP/queue_summary_$MODE.txt"
echo "=== aloha ablation queue $(date) mode=$MODE variants=${VARIANTS[*]} ===" >> "$SUMMARY"
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
