#!/bin/bash
# ============================================================================
# ALOHA parallel multi-GPU ablation — thin wrapper over run_ablation8_parallel.sh
# with ENV_KIND=aloha. Same GPU-pool scheduling (one experiment per GPU, 8 queue
# across the GPUs you list).
#
# Usage:
#   GPUS=0,1,2,3 bash examples/scripts/run_ablation8_aloha_parallel.sh full
#   GPUS=0,1     bash examples/scripts/run_ablation8_aloha_parallel.sh full   # 8 queue on 2 cards
#   DRY_RUN=1 GPUS=0,1,2 bash examples/scripts/run_ablation8_aloha_parallel.sh full
#
# See run_ablation8_parallel.sh for the full option list and the host-RAM caveat
# (each parallel job loads its own ~14 GB pi0).
# ============================================================================
exec env ENV_KIND=aloha bash "$(dirname "${BASH_SOURCE[0]}")/run_ablation8_parallel.sh" "$@"
