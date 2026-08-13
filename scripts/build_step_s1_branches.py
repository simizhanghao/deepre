#!/usr/bin/env python3
"""Select frozen S1 checkpoints and build paired SEARCH_NOW/CONTINUE_NOW rows."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "data/step_adaptive/s1_train640/base640.parquet"
BASE_RECORDS = REPO / "results/25_step_adaptive/s1/base/step_records.jsonl"
OUT = REPO / "data/step_adaptive/s1_train640"
BATCH = 64


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    base = [json.loads(line) for line in BASE_RECORDS.read_text().splitlines() if line.strip()]
    if len(base) != 640:
        raise RuntimeError(f"expected 640 base records, got {len(base)}")
    if len({row["sample_id"] for row in base}) != 640:
        raise RuntimeError("base sample IDs are not unique")
    source = {
        str(row["extra_info"]["sample_id"]): row
        for row in pq.read_table(SOURCE).to_pylist()
    }
    selections = []
    branch_rows = []
    for base_row in sorted(base, key=lambda row: row["sample_id"]):
        if not base_row["finish"] or not base_row["metrics"]["parser_valid"]:
            raise RuntimeError(f"invalid base trajectory: {base_row['sample_id']}")
        eligible = [
            step
            for step in base_row["step_records"]
            if not step["query_is_none"]
            and not step["duplicate_query"]
            and step["step_index"] + 1 < 4
            and step["num_previous_searches"] < 3
        ]
        chosen = eligible[:1]
        if len(eligible) >= 2:
            chosen.append(eligible[-1])
        for step in chosen:
            target = int(step["step_index"])
            prior_actions = [
                previous["action"] for previous in base_row["step_records"] if previous["step_index"] < target
            ]
            branch_id = f"{base_row['sample_id']}:cp{target}"
            selections.append(
                {
                    "branch_id": branch_id,
                    "sample_id": base_row["sample_id"],
                    "target_index": target,
                    "candidate_query": step["candidate_query"],
                    "state_prefix_sha256": step["state_prefix_sha256"],
                    "checkpoint_token_ids": step["checkpoint_token_ids"],
                    "prior_actions": prior_actions,
                }
            )
            for arm in ("search", "continue"):
                row = copy.deepcopy(source[base_row["sample_id"]])
                extra = dict(row["extra_info"])
                extra.update(
                    {
                        "step_phase": "s1_branch",
                        "step_action_plan": prior_actions + [arm],
                        "step_branch_id": branch_id,
                        "step_target_index": target,
                        "step_branch_arm": arm,
                        "step_padding": False,
                    }
                )
                row["extra_info"] = extra
                branch_rows.append(row)
    scientific_rows = len(branch_rows)
    pad_pairs = 0
    while len(branch_rows) % BATCH:
        if len(branch_rows) + 2 > ((scientific_rows + BATCH - 1) // BATCH) * BATCH:
            raise RuntimeError("pair padding cannot reach batch multiple")
        source_pair = branch_rows[:2]
        for row in source_pair:
            padded = copy.deepcopy(row)
            extra = dict(padded["extra_info"])
            extra["step_padding"] = True
            extra["step_branch_id"] = f"padding:{pad_pairs}:{extra['step_branch_arm']}"
            padded["extra_info"] = extra
            branch_rows.append(padded)
        pad_pairs += 1
    Dataset.from_list(branch_rows).to_parquet(str(OUT / "branches.parquet"))
    with (OUT / "checkpoint_selections.jsonl").open("w") as handle:
        for selection in selections:
            handle.write(json.dumps(selection) + "\n")
    manifest = {
        "gate": "STEP_S1_BRANCH_FREEZE_PASS",
        "base_records_sha256": sha256(BASE_RECORDS),
        "questions": 640,
        "selected_states": len(selections),
        "scientific_rows": scientific_rows,
        "padded_rows": len(branch_rows) - scientific_rows,
        "runner_rows": len(branch_rows),
        "batch_size": BATCH,
        "selection_rule": "earliest+latest eligible; singleton once",
        "arms": ["search", "continue"],
        "same_future_completion_policy": True,
        "original_test_read": False,
        "val3_read": False,
        "artifact_sha256": {
            "branches.parquet": sha256(OUT / "branches.parquet"),
            "checkpoint_selections.jsonl": sha256(OUT / "checkpoint_selections.jsonl"),
        },
    }
    (OUT / "branch_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
