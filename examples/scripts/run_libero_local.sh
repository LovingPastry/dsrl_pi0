#!/bin/bash
# Local launcher for DSRL-pi0 LIBERO, adapted for this machine:
#  - reuse the user's existing openpi data home (has pi0_libero ckpt)
#  - put logs/compilation cache on the big disk (root fs is small)
#  - log to TensorBoard locally under $EXP (videos/images saved next to events)
# Usage: bash examples/scripts/run_libero_local.sh [smoke|full]
#   View logs:  tensorboard --logdir "$EXP"
set -e
MODE=${1:-smoke}
device_id=0

# LIBERO task selection (override at launch, e.g. `SUITE=libero_10 TASK_ID=3 bash ... full`)
SUITE=${SUITE:-libero_10}
TASK_ID=${TASK_ID:-3}

source /home/fuyx/anaconda3/etc/profile.d/conda.sh
conda activate dsrl_pi0

# --- rendering (EGL headless) ---
export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$device_id

# --- data / logs / cache redirected to big disk ---
export OPENPI_DATA_HOME=/media/fuyx/系统/lanzc/openpi_data_home
BIGDISK=/media/fuyx/系统/lanzc/dsrl_runtime
mkdir -p "$BIGDISK/logs" "$BIGDISK/jax_cache"
export EXP="$BIGDISK/logs/DSRL_pi0_Libero"

# --- gpu / jax ---
export CUDA_VISIBLE_DEVICES=$device_id
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd /home/fuyx/lanzc/dsrl_pi0
export PYTHONPATH=/home/fuyx/lanzc/dsrl_pi0:$PYTHONPATH

if [ "$MODE" = "smoke" ]; then
  # minimal: prove the full pipeline (pi0 load -> env -> SAC loop -> baseline eval) works fast.
  # low start_online_updates so the initial (baseline pi0) eval fires after a few episodes.
  python3 examples/launch_train_sim.py \
    --algorithm pixel_sac --env libero --prefix dsrl_pi0_libero \
    --task_suite "$SUITE" --task_id "$TASK_ID" \
    --tb_project DSRL_pi0_Libero --batch_size 256 --discount 0.999 --seed 0 \
    --max_steps 800 --eval_interval 400 --log_interval 100 --eval_episodes 5 \
    --multi_grad_step 10 --start_online_updates 40 --resize_image 64 \
    --action_magnitude 1.0 --query_freq 20 --hidden_dims 128
else
  # full training (paper settings)
  python3 examples/launch_train_sim.py \
    --algorithm pixel_sac --env libero --prefix dsrl_pi0_libero \
    --task_suite "$SUITE" --task_id "$TASK_ID" \
    --tb_project DSRL_pi0_Libero --batch_size 256 --discount 0.999 --seed 0 \
    --max_steps 500000 --eval_interval 10000 --log_interval 500 --eval_episodes 10 \
    --multi_grad_step 20 --start_online_updates 500 --resize_image 64 \
    --action_magnitude 1.0 --query_freq 20 --hidden_dims 128
fi
