#!/usr/bin/env bash
set -euo pipefail
repo=/data1/hcc/deepresearch
python=/home/hanchengcheng/miniconda3/envs/deepresearch/bin/python
exec "$python" "$repo/scripts/start_candidate_retrieval_server.py" \
  --index "$repo/data/cur/dssr_val2_fresh128/contexts_index.jsonl" \
  --port 8004
