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
# NOTE on GPU: this repo is pinned to jax 0.5.0 for pre-Blackwell GPUs.
# RTX 50-series (Blackwell, sm_120) cards require jax 0.6.2+ and a distrax
# compat patch (patch_distrax_jax060.sh). The script auto-detects the GPU and
# picks the right jax version.
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

# Detect RTX 50-series (Blackwell, compute capability >= 12.0) and pick the right
# jax version. 50-series cards need jax >= 0.6.2 for sm_120 codegen and fp8 fixes;
# older cards stick to the pinned jax 0.5.0.
IS_BLACKWELL=0
if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
  while IFS= read -r cap; do
    major=$(echo "$cap" | cut -d. -f1)
    if [ "$major" -ge 12 ]; then IS_BLACKWELL=1; break; fi
  done < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)
fi

if [ "$IS_BLACKWELL" -eq 1 ]; then
  echo "[setup] Blackwell (RTX 50-series) detected — installing jax 0.6.2 + distrax compat patch"
  python -m pip install -U "jax[cuda12]==0.6.2"
  bash "$REPO_DIR/examples/scripts/patch_distrax_jax060.sh"
else
  echo "[setup] pre-Blackwell GPU (or none) — installing jax 0.5.0"
  python -m pip install -U "jax[cuda12]==0.5.0"
fi
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
