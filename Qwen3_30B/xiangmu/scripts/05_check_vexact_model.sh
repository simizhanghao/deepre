#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

model=${1:-$BASE_MODEL}
tag=${2:-base}
require_file "$model/config.json"
out="$PROJECT_ROOT/results/vexact_compat/$tag"
mkdir -p "$out/capture"
log="$PROJECT_ROOT/logs/vexact_compat_${tag}_$(date +%Y%m%d_%H%M%S).log"

export VERL_USE_EXTERNAL_MODULES=vexact.integrations.verl.register
export MODELING_BACKEND=veomni
export VEOMNI_USE_LIGER_KERNEL=0
export INFER_FA_IMPL=triton-invariant
export TOKENIZERS_PARALLELISM=false

cd "$VEXACT_ROOT"
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=${COMPAT_GPU:-0} \
  "$VEXACT_ROOT/.venv/bin/python" tests/scripts/hf_inference.py \
    --model_path "$model" \
    --simulate_requests 4 \
    --max_length 256 \
    --max_new_tokens 1 \
    --max_num_batched_tokens 1024 \
    --max_cache_blocks 64 \
    --request_interval 0.01 \
    --temperature 0.9 \
    --top_p 0.95 \
    --top_k -1 \
    --do_sample \
    --seed 42 \
    --enable_batch_invariant \
    --use_fp32_logits \
    --enforce_eager \
    --attn_impl triton-invariant \
    --output_dir "$out/capture" \
    2>&1 | tee "$log"

env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=${COMPAT_GPU:-0} \
  "$VEXACT_ROOT/.venv/bin/python" tests/scripts/verify_logits_vs_native_hf.py \
    --model_path "$model" \
    --data_dir "$out/capture" \
    --model_backend veomni \
    --attn_impl triton-invariant \
    --enable_batch_invariant \
    --use_remove_padding \
    --skip_backward \
    --rtol 1e-5 \
    --atol 1e-6 \
    --log_file "$out/verify.log" \
    2>&1 | tee -a "$log"

sha256sum "$model/config.json" "$model/tokenizer.json" >"$out/model_identity.sha256"
echo "VEXACT_MODEL_COMPAT_PASS tag=$tag model=$model"

