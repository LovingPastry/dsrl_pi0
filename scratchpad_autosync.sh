#!/bin/bash
source /home/fuyx/anaconda3/etc/profile.d/conda.sh && conda activate dsrl_pi0
WB="/tmp/tmpj4pgc9_4/dsrl_pi0_libero_2026_06_30_12_02_16_0000--s-0/wandb/offline-run-20260630_120217-dsrl_pi0_libero_2026_06_30_12_02_16_0000--s-0"
SLOG=/home/fuyx/lanzc/dsrl_pi0/scratchpad_autosync.log
: > "$SLOG"
for k in $(seq 1 30); do   # up to ~15h
  # re-read the current key from ~/.bashrc each cycle (self-heals once user fixes it)
  K=$(grep -E '^[[:space:]]*export[[:space:]]+WANDB_API_KEY=' ~/.bashrc 2>/dev/null | tail -1 | cut -d= -f2- | tr -d '"'\'' ')
  ts=$(date '+%H:%M:%S')
  if [ -z "$K" ]; then
    echo "[$ts] attempt $k: no WANDB_API_KEY in ~/.bashrc, skipping" >> "$SLOG"
  else
    out=$(WANDB_API_KEY="$K" timeout 180 wandb sync --no-mark-synced "$WB" 2>&1)
    if echo "$out" | grep -qiE "invalid api key|401"; then
      echo "[$ts] attempt $k: 401 INVALID KEY (waiting for you to fix ~/.bashrc)" >> "$SLOG"
    elif echo "$out" | grep -qi "done"; then
      url=$(echo "$out" | grep -oE "https://wandb.ai/[^ ]+" | tail -1)
      echo "[$ts] attempt $k: SYNCED OK -> $url" >> "$SLOG"
    else
      echo "[$ts] attempt $k: $(echo "$out" | tail -1)" >> "$SLOG"
    fi
  fi
  pgrep -f launch_train_sim >/dev/null 2>&1 || { echo "[$ts] training finished -> final sync done, exiting loop" >> "$SLOG"; break; }
  sleep 1800
done
