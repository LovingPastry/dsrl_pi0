LOG=/home/fuyx/lanzc/dsrl_pi0/scratchpad_smoke.log
for k in $(seq 1 220); do
  if grep -qE "Success rate:" "$LOG" 2>/dev/null; then
    echo "=== EVAL DONE (baseline pi0 success rate) ==="
    grep -nE "Loaded pi0 policy|performing evaluation|Rollout [0-9]+ :|Success rate:|Average return:|Reward >=" "$LOG" | tail -20
    exit 0
  fi
  if grep -qiE "Traceback|No visible GPU|Killed|CUDA_ERROR|RuntimeError|XlaRuntimeError|MemoryError|Errno|cannot|Exception" "$LOG" 2>/dev/null; then
    echo "=== ERROR DETECTED ==="; grep -niE "Traceback|No visible GPU|Killed|Error|Exception|cannot" "$LOG" | tail -8
    echo "--- last 15 lines ---"; tail -15 "$LOG"
    exit 1
  fi
  if ! pgrep -f "launch_train_sim" >/dev/null 2>&1; then
    echo "=== PROCESS EXITED without Success rate ==="; tail -25 "$LOG"; exit 2
  fi
  sleep 10
done
echo "=== WATCH TIMEOUT (still running after ~37min) ==="; tail -15 "$LOG"; exit 3
