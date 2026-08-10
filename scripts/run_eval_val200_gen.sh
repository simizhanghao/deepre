#!/usr/bin/env bash
# Phase 3C-GEN: frozen val-200 Agentic eval for SFT-v1 / 3B@100 / 3C@400.
# Prerequisites: HF merges under outputs/rl/02_* and 03_* (see export_verl_fsdp_to_hf.sh)
set -euo pipefail

REPO=${REPO:-/data1/hcc/deepresearch}
cd "$REPO"
EVAL=${EVAL:-data/eval/hotpotqa_200.jsonl}
GPU_SFT=${GPU_SFT:-4}
GPU_3B=${GPU_3B:-5}
GPU_3C=${GPU_3C:-6}
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_ROOT=${OUT_ROOT:-results/10_eval_grpo_evidence_val200_${STAMP}}
mkdir -p "$OUT_ROOT" logs

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

run_one() {
  local gpu="$1" model="$2" tag="$3"
  echo "=== GEN $tag gpu=$gpu model=$model ==="
  CUDA_VISIBLE_DEVICES=$gpu python scripts/run_agent_rollout_smoke.py \
    --model-path "$model" \
    --eval-file "$EVAL" \
    --max-samples 200 \
    --top-k 5 \
    --max-search-turns 2 \
    --temperature 0.0 \
    --max-new-tokens 512 \
    --output-dir "$OUT_ROOT" \
    --run-tag "$tag" \
    2>&1 | tee "logs/10_eval_val200_${tag}_${STAMP}.log"
}

SFT=${SFT:-outputs/00_sft_v1_merged}
B3=${B3:-outputs/rl/02_hf_answer_only_step100}
C3=${C3:-outputs/rl/03_hf_evidence_step400}

for d in "$SFT" "$B3" "$C3"; do
  [[ -f "$d/config.json" ]] || { echo "missing HF model: $d (merge first)"; exit 1; }
done

# Sequential on one GPU by default if ONLY_GPU set; else three GPUs in sequence
# (agent loop is single-process; parallel optional via background jobs)
if [[ "${PARALLEL:-0}" == "1" ]]; then
  run_one "$GPU_SFT" "$SFT" "gen_sftv1_val200" &
  run_one "$GPU_3B" "$B3" "gen_3b100_val200" &
  run_one "$GPU_3C" "$C3" "gen_3c400_val200" &
  wait
else
  GPU=${ONLY_GPU:-$GPU_SFT}
  run_one "$GPU" "$SFT" "gen_sftv1_val200"
  run_one "${ONLY_GPU:-$GPU_3B}" "$B3" "gen_3b100_val200"
  run_one "${ONLY_GPU:-$GPU_3C}" "$C3" "gen_3c400_val200"
fi

echo "GEN_DONE root=$OUT_ROOT"
ls -lt "$OUT_ROOT" | head
