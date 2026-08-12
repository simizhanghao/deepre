#!/usr/bin/env python3
"""Build fixed CUR-1 Train-only Phase25 S0 smoke rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "data/cur/cur1_fresh896/train.parquet"
IDS = REPO / "data/cur/cur1_fresh896/train_ids.txt"
OUT = REPO / "data/step_adaptive/s0_train32"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    frozen_ids = [x for x in IDS.read_text().splitlines() if x][:32]
    by_id = {}
    for row in pq.read_table(SOURCE).to_pylist():
        sample_id = str(row["extra_info"]["sample_id"])
        by_id.setdefault(sample_id, row)
    rows = []
    for index, sample_id in enumerate(frozen_ids):
        row = by_id[sample_id]
        row["agent_name"] = "eca_step_adaptive_agent"
        extra = dict(row["extra_info"])
        for key in ("cur_forced_arm", "cur_pair_id", "cur_phase", "cur_split"):
            extra.pop(key, None)
        extra.update(
            {
                "step_phase": "s0",
                "step_smoke_index": index,
                # Force CONTINUE on replay subset; ANSWER remains a separate
                # external action rather than query-NONE aliasing.
                "step_action_plan": ["continue", "answer"] if index < 8 else [],
            }
        )
        row["extra_info"] = extra
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(OUT / "train32.parquet"))
    Dataset.from_list(rows[:8]).to_parquet(str(OUT / "replay8.parquet"))
    Dataset.from_list(rows[:2]).to_parquet(str(OUT / "smoke2.parquet"))
    manifest = {
        "gate": "STEP_S0_DATA_FREEZE_PASS",
        "source": str(SOURCE),
        "source_sha256": sha256(SOURCE),
        "questions": 32,
        "replay_subset": frozen_ids[:8],
        "question_ids": frozen_ids,
        "actions": ["continue", "search", "answer"],
        "budgets": {
            "response_tokens": 2048,
            "observation_tokens": 384,
            "answer_reserve": 256,
            "step_tokens": 128,
            "checkpoints": 4,
            "searches": 3,
        },
        "artifact_sha256": {
            "train32.parquet": sha256(OUT / "train32.parquet"),
            "replay8.parquet": sha256(OUT / "replay8.parquet"),
            "smoke2.parquet": sha256(OUT / "smoke2.parquet"),
        },
        "val3_read": False,
        "test_read": False,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in {"question_ids", "replay_subset"}}, indent=2))


if __name__ == "__main__":
    main()
