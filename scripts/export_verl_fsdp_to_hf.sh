#!/usr/bin/env bash
# Merge veRL FSDP actor shards → HuggingFace dir (inside eca-verl).
# Usage:
#   bash scripts/export_verl_fsdp_to_hf.sh \
#     outputs/rl/grpo_sftv1_smoke/global_step_100 \
#     outputs/rl/hf_merged/grpo_sftv1_smoke_step100
set -euo pipefail

REPO=${REPO:-/data1/hcc/deepresearch}
CONTAINER=${CONTAINER:-eca-verl}
SRC_REL=${1:?usage: $0 <global_step_dir> <target_hf_dir>}
DST_REL=${2:?usage: $0 <global_step_dir> <target_hf_dir>}

# Accept host-relative or absolute; map into container /workspace/deepresearch
to_container() {
  local p="$1"
  if [[ "$p" == /workspace/* ]]; then echo "$p"; return; fi
  if [[ "$p" == /* ]]; then
    echo "/workspace/deepresearch/${p#${REPO}/}"
  else
    echo "/workspace/deepresearch/$p"
  fi
}

SRC_C=$(to_container "$SRC_REL")
DST_C=$(to_container "$DST_REL")
ACTOR="${SRC_C%/}/actor"

echo "[export] container=$CONTAINER"
echo "[export] actor=$ACTOR"
echo "[export] target=$DST_C"

docker start "$CONTAINER" >/dev/null
docker exec "$CONTAINER" bash -lc "
  set -euo pipefail
  test -d '$ACTOR' || { echo missing actor dir: $ACTOR; exit 1; }
  test -f '$ACTOR/model_world_size_4_rank_0.pt' || { echo missing FSDP shards; ls -la '$ACTOR'; exit 1; }
  mkdir -p '$(dirname "$DST_C")'
  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir '$ACTOR' \
    --target_dir '$DST_C' \
    --use_cpu_initialization
  ls -la '$DST_C' | head
  test -f '$DST_C/config.json'
  echo EXPORT_OK '$DST_C'
"
echo "[export] done → $DST_REL"
