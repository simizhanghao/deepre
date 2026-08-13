#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_file "$BASE_MODEL/config.json"
require_file "$LLAMAFACTORY_ROOT/data/eca_qwen3_30b_coldstart_train.jsonl"

log="$PROJECT_ROOT/logs/sft_$(date +%Y%m%d_%H%M%S).log"
cd "$LLAMAFACTORY_ROOT"
export FORCE_TORCHRUN=1
export NNODES=1
export NODE_RANK=0
export NPROC_PER_NODE="$N_GPUS"
export MASTER_PORT=${MASTER_PORT:-29531}
export TOKENIZERS_PARALLELISM=false
run_llamafactory train "$PROJECT_ROOT/config/sft_lora.yaml" 2>&1 | tee "$log"
echo "SFT_LOG=$log"

