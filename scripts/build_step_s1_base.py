#!/usr/bin/env python3
"""Freeze the Phase-25 S1 Train640 deterministic base-trace dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "data/cur/cur1_fresh896/train.parquet"
IDS = REPO / "data/cur/cur1_fresh896/train_ids.txt"
OUT = REPO / "data/step_adaptive/s1_train640"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    frozen_ids = [line for line in IDS.read_text().splitlines() if line]
    if len(frozen_ids) != 640 or len(set(frozen_ids)) != 640:
        raise RuntimeError("S1 requires exactly 640 unique CUR-1 Train IDs")
    source_rows = pq.read_table(SOURCE).to_pylist()
    by_id = {str(row["extra_info"]["sample_id"]): row for row in source_rows}
    rows = []
    for index, sample_id in enumerate(frozen_ids):
        row = by_id[sample_id]
        row["agent_name"] = "eca_step_adaptive_agent"
        extra = dict(row["extra_info"])
        for key in ("cur_forced_arm", "cur_pair_id", "cur_phase", "cur_split"):
            extra.pop(key, None)
        extra.update(
            {
                "step_phase": "s1_base",
                "step_question_index": index,
                "step_action_plan": [],
                "step_branch_id": "",
                "step_target_index": -1,
                "step_branch_arm": "base",
                "step_padding": False,
            }
        )
        row["extra_info"] = extra
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(OUT / "base640.parquet"))
    manifest = {
        "gate": "STEP_S1_BASE_FREEZE_PASS",
        "source": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "questions": 640,
        "question_ids_sha256": hashlib.sha256("\n".join(frozen_ids).encode()).hexdigest(),
        "selection_rule": "earliest+latest eligible; singleton once",
        "eligible": "query != NONE; nonduplicate; SEARCH and CONTINUE both state-valid",
        "completion_policy": "greedy deterministic fixed_completion_action",
        "max_sampled_checkpoints_per_question": 2,
        "original_test_read": False,
        "val3_read": False,
        "artifact_sha256": {"base640.parquet": sha256(OUT / "base640.parquet")},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
