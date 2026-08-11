#!/usr/bin/env bash
# A4: frozen Evidence@400, historical 32 questions x 4 real AgentLoop rollouts.
set -euo pipefail

exec bash /data1/hcc/deepresearch/scripts/run_vexact_a3_gate_b.sh \
  --stage a4 \
  "$@"
