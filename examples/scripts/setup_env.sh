#!/bin/bash
# ============================================================================
# One-click environment setup for DSRL-pi0 (LIBERO + ALOHA sim).
#
# Installs the conda env, python deps, the openpi / openpi-client / LIBERO
# editable submodule packages, the mujoco version the ALOHA sim needs, and
# pre-downloads the frozen pi0 checkpoint(s) from the public openpi S3 bucket.
#
# Usage (on a fresh server, after `git clone` of this repo):
#   bash examples/scripts/setup_env.sh
# Overridable:
#   ENV_NAME=dsrl_pi0 PY_VER=3.11 CKPTS="pi0_aloha_sim pi0_libero" \
#   OPENPI_DATA_HOME=/big/disk/openpi_data_home bash examples/scripts/setup_env.sh
#
# NOTE on GPU: this repo is pinned to jax 0.5.0. The cuDNN/CUDA runtime must
# match your GPU driver. This project was validated with CUDA 12.8 + cuDNN
# 9.10.2 on driver 575. If JAX fails to initialise cuDNN, install a matching
# CUDA/cuDNN set (see docs/) rather than changing the jax version.
# ============================================================================
set -euo pipefail

ENV_NAME=${ENV_NAME:-dsrl_pi0}
PY_VER=${PY_VER:-3.11}
CKPTS=${CKPTS:-pi0_aloha_sim}          # space-separated; add pi0_libero if you also want LIBERO
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
echo "[setup] repo: $REPO_DIR   env: $ENV_NAME   python: $PY_VER   ckpts: $CKPTS"

# --- 0. submodules (openpi + LIBERO) ----------------------------------------
if [ ! -f openpi/pyproject.toml ] || [ ! -e LIBERO/setup.py ]; then
  echo "[setup] initialising git submodules (openpi, LIBERO)..."
  if ! git submodule update --init --recursive; then
    echo "[setup] SSH fetch failed; switching submodule URLs to HTTPS and retrying"
    git config submodule.LIBERO.url https://github.com/nakamotoo/LIBERO.git
    git config submodule.openpi.url https://github.com/nakamotoo/openpi.git
    git submodule update --init --recursive
  fi
fi

# --- 1. conda env -----------------------------------------------------------
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[setup] creating conda env $ENV_NAME (python $PY_VER)"
  conda create -y -n "$ENV_NAME" python="$PY_VER"
fi
conda activate "$ENV_NAME"

# --- 2. python dependencies -------------------------------------------------
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# GPU JAX (cuda12 extra pulls the matching pjrt/plugin wheels for jax 0.5.0)
python -m pip install -U "jax[cuda12]==0.5.0"
# ALOHA sim (gym-aloha) renders through mujoco 2.3.7 under EGL; newer mujoco breaks it
python -m pip install mujoco==2.3.7
# editable submodule packages
python -m pip install -e openpi
python -m pip install -e openpi/packages/openpi-client
python -m pip install -e LIBERO

# --- 3. pre-download frozen pi0 checkpoint(s) (public anonymous S3) ----------
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-$HOME/.cache/openpi}
mkdir -p "$OPENPI_DATA_HOME"
echo "[setup] downloading checkpoints into OPENPI_DATA_HOME=$OPENPI_DATA_HOME (each ~12 GB)"
CKPTS="$CKPTS" python - <<'PY'
import os
from openpi.shared import download
for name in os.environ["CKPTS"].split():
    p = download.maybe_download(f"s3://openpi-assets/checkpoints/{name}")
    print("  ok:", name, "->", p)
PY

echo "[setup] DONE."
echo "[setup] Activate:  conda activate $ENV_NAME"
echo "[setup] Run ALOHA 8-experiment ablation:  bash examples/scripts/run_ablation8_aloha.sh full"
echo "[setup] (set OPENPI_DATA_HOME to the same path when running, so it finds the checkpoints)"
