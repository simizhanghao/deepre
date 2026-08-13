#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

echo "[paths]"
for path in "$PROJECT_ROOT" "$LLAMAFACTORY_ROOT" "$VEXACT_ROOT"; do
  require_dir "$path"
  echo "OK $path"
done
for path in "$RL_TRAIN" "$RL_VAL" "$BM25_INDEX" "$FROZEN_DEV"; do
  require_file "$path"
  echo "OK $path"
done

echo "[data hashes]"
sha256sum \
  "$SFT_DATA_DIR/eca_coldstart_v1_train.jsonl" \
  "$SFT_DATA_DIR/eca_coldstart_v1_dev.jsonl" \
  "$RL_TRAIN" "$RL_VAL" "$BM25_INDEX" "$FROZEN_DEV" \
  | tee "$PROJECT_ROOT/results/input_sha256.txt"

echo "[gpu]"
command -v nvidia-smi >/dev/null
gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[[ "$gpu_count" -ge "$N_GPUS" ]] || {
  echo "ERROR need at least $N_GPUS visible GPUs, found $gpu_count" >&2
  exit 1
}
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

echo "[disk]"
df -h "$PROJECT_ROOT"
free_gb=$(df -Pk "$PROJECT_ROOT" | awk 'NR==2 {printf "%d", $4/1024/1024}')
[[ "$free_gb" -ge 450 ]] || {
  echo "ERROR recommend >=450 GiB free for base + merged SFT + one resumable RL state; found ${free_gb} GiB" >&2
  exit 1
}

echo "[framework support]"
grep -q 'Qwen3-30B-A3B-Instruct-2507' "$LLAMAFACTORY_ROOT/src/llamafactory/extras/constants.py"
grep -q 'name="qwen3_nothink"' "$LLAMAFACTORY_ROOT/src/llamafactory/data/template.py"
require_file "$VEXACT_ROOT/vexact/models/qwen3_moe/modeling_qwen3_moe.py"
require_file "$VEXACT_ROOT/.venv/bin/python"
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 "$VEXACT_ROOT/.venv/bin/python" - <<'PY'
import torch, transformers, verl, veomni, vexact
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("verl", verl.__file__)
print("veomni", veomni.__file__)
print("vexact", vexact.__file__)
assert torch.cuda.is_available()
print("P0_PREFLIGHT_PASS")
PY
