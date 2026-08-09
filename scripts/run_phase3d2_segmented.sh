#!/usr/bin/env bash
# Phase 3D2 segmented training orchestrator (capability refresh between windows).
#
# Protocol:
#   1) tool-free n=4 p_int refresh on current HF policy
#   2) GRPO for REFRESH_EVERY steps with frozen table
#   3) export FSDP→HF, repeat until TOTAL_STEPS
#
# THIS SCRIPT IS READY BUT NOT AUTO-STARTED. Review knobs first.
#
# Usage:
#   bash scripts/run_phase3d2_segmented.sh          # full plan
#   DRY_RUN=1 bash scripts/run_phase3d2_segmented.sh  # print plan only
set -euo pipefail

REPO=${REPO:-/data1/hcc/deepresearch}
CONTAINER=${CONTAINER:-eca-verl}
TOTAL_STEPS=${TOTAL_STEPS:-400}
REFRESH_EVERY=${REFRESH_EVERY:-50}
SAVE_FREQ=${SAVE_FREQ:-50}
ECA_SEARCH_COST_WEIGHT=${ECA_SEARCH_COST_WEIGHT:-0.30}
ECA_EVIDENCE_WEIGHT=${ECA_EVIDENCE_WEIGHT:-0.5}
N_PINT=${N_PINT:-4}
PINT_TEMP=${PINT_TEMP:-0.9}
PINT_GPU=${PINT_GPU:-4}
BATCH=${BATCH:-16}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.60}
MICRO_BATCH=${MICRO_BATCH:-2}
DRY_RUN=${DRY_RUN:-0}

INIT_MODEL=${INIT_MODEL:-$REPO/outputs/sft_qwen25_3b_coldstart_v1_merged}
OUT_DIR_HOST=${OUT_DIR_HOST:-$REPO/outputs/rl/grpo_sftv1_cap_3d2}
OUT_DIR_C=/workspace/deepresearch/outputs/rl/grpo_sftv1_cap_3d2
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_sftv1_cap_3d2}
CAP_DIR=$REPO/outputs/rl/capability
HF_DIR=$REPO/outputs/rl/hf_merged
TRAIN_PQ=$REPO/data/rl/grpo_smoke_128/train.parquet

cd "$REPO"
mkdir -p "$CAP_DIR" "$OUT_DIR_HOST" "$HF_DIR" logs

echo "======== 3D2 segmented plan ========"
echo "TOTAL_STEPS=$TOTAL_STEPS REFRESH_EVERY=$REFRESH_EVERY λ_s=$ECA_SEARCH_COST_WEIGHT"
echo "INIT_MODEL=$INIT_MODEL"
echo "segments=$(( (TOTAL_STEPS + REFRESH_EVERY - 1) / REFRESH_EVERY ))"
echo "DRY_RUN=$DRY_RUN"
echo "===================================="

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] exit without building p_int or launching GRPO"
  exit 0
fi

# docker group helper
run_docker() {
  if docker info >/dev/null 2>&1; then
    docker "$@"
  else
    sg docker -c "docker $(printf '%q ' "$@")"
  fi
}

current_hf=$INIT_MODEL
done_steps=0
seg=0
resume=disable

while [[ "$done_steps" -lt "$TOTAL_STEPS" ]]; do
  seg=$((seg + 1))
  remain=$((TOTAL_STEPS - done_steps))
  delta=$REFRESH_EVERY
  [[ "$delta" -gt "$remain" ]] && delta=$remain
  target_step=$((done_steps + delta))

  echo "-------- segment $seg: advance to global_step=$target_step (+$delta) | refresh from $current_hf --------"

  # 1) capability refresh on free host GPU (not the 4 training GPUs if busy)
  pint_out="$CAP_DIR/p_int_seg${seg}_step${done_steps}.json"
  CUDA_VISIBLE_DEVICES=$PINT_GPU HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python "$REPO/scripts/build_capability_pint_table.py" \
      --model-path "$current_hf" \
      --train-parquet "$TRAIN_PQ" \
      --n "$N_PINT" \
      --temperature "$PINT_TEMP" \
      --out "$pint_out" \
      --symlink-latest "$CAP_DIR/p_int_latest.json"

  python "$REPO/scripts/audit_pint_table.py" \
    --table "$pint_out" \
    --train-parquet "$TRAIN_PQ" \
    --require-full-coverage \
    --out "$OUT_DIR_HOST/pint_audit_seg${seg}_step${done_steps}.json"

  # segment ledger (hash / coverage / histogram)
  python3 - "$pint_out" "$OUT_DIR_HOST" "$seg" "$done_steps" "$target_step" <<'PY'
import json, hashlib, sys
from pathlib import Path
pint, out_dir, seg, done, target = sys.argv[1:6]
p = Path(pint).resolve()
h = hashlib.sha256(p.read_bytes()).hexdigest()
raw = json.loads(p.read_text())
ledger = Path(out_dir) / "segment_ledger.jsonl"
row = {
    "segment": int(seg),
    "global_step_start": int(done),
    "global_step_target": int(target),
    "p_int_table_path": str(p),
    "p_int_table_sha256": h,
    "p_int_mean": raw.get("mean_p_int"),
    "histogram": raw.get("histogram"),
    "coverage": raw.get("coverage", 1.0),
    "missing_count": raw.get("missing_count", 0),
}
with ledger.open("a") as f:
    f.write(json.dumps(row) + "\n")
print("[ledger]", row)
PY

  # 2) GRPO window
  run_docker start "$CONTAINER" >/dev/null
  run_docker exec "$CONTAINER" bash -lc \
    'ray stop --force >/dev/null 2>&1 || true; pkill -9 -f launch_grpo_main >/dev/null 2>&1 || true; pkill -9 -f run_grpo_capability >/dev/null 2>&1 || true; exit 0' \
    || true
  sleep 2

  STAMP=$(date +%Y%m%d_%H%M%S)
  LOG="$REPO/logs/grpo_${EXPERIMENT_NAME}_seg${seg}_to${target_step}_${STAMP}.log"
  : >"$LOG"
  ln -sfn "$LOG" "$REPO/logs/grpo_3d2_latest.log"

  # Train always inits config from SFT-v1 path; weights come from resume after seg1.
  # HF export (current_hf) is ONLY for the p_int refresh above.
  if [[ "$seg" -eq 1 ]]; then
    resume=disable
  else
    resume=auto
  fi
  model_c=/workspace/deepresearch/outputs/sft_qwen25_3b_coldstart_v1_merged

  # veRL trainer.total_training_steps is an absolute stop step (not +delta).
  run_docker exec -d "$CONTAINER" bash -lc "\
    export PYTHONPATH=/workspace/deepresearch:/workspace/verl; \
    export TENSORBOARD_DIR=/workspace/deepresearch/outputs/rl/tensorboard/${EXPERIMENT_NAME}; \
    export ECA_EVIDENCE_WEIGHT=${ECA_EVIDENCE_WEIGHT}; \
    export ECA_SEARCH_COST_WEIGHT=${ECA_SEARCH_COST_WEIGHT}; \
    export ECA_PINT_TABLE=/workspace/deepresearch/outputs/rl/capability/p_int_latest.json; \
    export ECA_PINT_DEFAULT=0.0; \
    export ECA_PINT_STRICT=1; \
    cd /workspace/deepresearch; \
    STEPS=${target_step} SAVE_FREQ=${SAVE_FREQ} OUT_DIR=${OUT_DIR_C} \
    EXPERIMENT_NAME=${EXPERIMENT_NAME} RESUME_MODE=${resume} \
    MODEL_PATH=${model_c} \
    BATCH=${BATCH} GPU_MEM_UTIL=${GPU_MEM_UTIL} MICRO_BATCH=${MICRO_BATCH} \
    bash /workspace/deepresearch/scripts/run_grpo_capability.sh \
      >/workspace/deepresearch/logs/$(basename "$LOG") 2>&1"

  echo "[3D2] training segment $seg; log=$LOG"
  # wait until process ends
  while run_docker exec "$CONTAINER" bash -lc 'pgrep -f launch_grpo_main >/dev/null || pgrep -f run_grpo_capability >/dev/null'; do
    sleep 30
  done

  done_steps=$target_step
  ckpt_host="$OUT_DIR_HOST/global_step_${done_steps}"
  if [[ ! -d "$ckpt_host/actor" ]]; then
    # find latest global_step_*
    ckpt_host=$(ls -d "$OUT_DIR_HOST"/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1 || true)
  fi
  if [[ -z "${ckpt_host}" || ! -d "${ckpt_host}/actor" ]]; then
    echo "ERROR: no ckpt after segment $seg under $OUT_DIR_HOST"
    exit 1
  fi

  # 3) export HF for next p_int refresh
  hf_out="$HF_DIR/${EXPERIMENT_NAME}_step${done_steps}"
  bash "$REPO/scripts/export_verl_fsdp_to_hf.sh" "$ckpt_host" "$hf_out"
  current_hf=$hf_out
done

echo "======== 3D2 segmented DONE steps=$done_steps ========"
echo "final ckpt dir: $OUT_DIR_HOST"
echo "final HF: $current_hf"
