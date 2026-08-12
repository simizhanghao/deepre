#!/usr/bin/env python3
"""Independent hard audit of the immutable CUR-1 split/data contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    import pyarrow.parquet as pq

    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    errors = []
    expected = {"train": 640, "validation": 128, "test": 128}
    all_ids: set[str] = set()
    for split, n_questions in expected.items():
        rows = pq.read_table(args.data_dir / f"{split}.parquet").to_pylist()
        by_id: dict[str, list[dict]] = {}
        for row in rows:
            sid = str(row["extra_info"]["sample_id"])
            by_id.setdefault(sid, []).append(row)
        if len(rows) != n_questions * 2 or len(by_id) != n_questions:
            errors.append(f"{split}: rows/questions={len(rows)}/{len(by_id)}")
        if all_ids.intersection(by_id):
            errors.append(f"{split}: overlaps an earlier split")
        all_ids.update(by_id)
        for sid, pair in by_id.items():
            arms = Counter(x["extra_info"]["cur_forced_arm"] for x in pair)
            prompts = {json.dumps(x["prompt"], sort_keys=True) for x in pair}
            if arms != {"search": 1, "internal": 1} or len(prompts) != 1:
                errors.append(f"{split}/{sid}: invalid paired arms or prompt identity")
            prompt_blob = next(iter(prompts))
            if "supporting_facts" in prompt_blob or "contexts" in prompt_blob:
                errors.append(f"{split}/{sid}: reward/tool data leaked into prompt")
        id_file = (args.data_dir / f"{split}_ids.txt").read_text().splitlines()
        digest = hashlib.sha256("\n".join(id_file).encode()).hexdigest()
        if id_file != manifest["question_ids"][split] or digest != manifest["split_id_sha256"][split]:
            errors.append(f"{split}: ID file/hash mismatch")

    context_ids = {
        str(json.loads(line)["sample_id"])
        for line in (args.data_dir / "contexts_index.jsonl").read_text().splitlines() if line.strip()
    }
    prompt_rows = [
        json.loads(line) for line in (args.data_dir / "prompt_manifest.jsonl").read_text().splitlines() if line.strip()
    ]
    if context_ids != all_ids or {str(x["sample_id"]) for x in prompt_rows} != all_ids:
        errors.append("contexts/prompt manifest IDs do not equal selected split IDs")
    if len(prompt_rows) != 896 or any(not x["canonical_prompt_sha256"] for x in prompt_rows):
        errors.append("prompt manifest count/hash failure")

    for name, expected_hash in manifest["frozen_artifact_sha256"].items():
        if sha256_file(args.data_dir / name) != expected_hash:
            errors.append(f"frozen artifact hash mismatch: {name}")
    for name, expected_hash in manifest["model_contract_sha256"].items():
        if sha256_file(args.model / name) != expected_hash:
            errors.append(f"model contract hash mismatch: {name}")

    summary = {
        "gate": "CUR1_FREEZE_AUDIT_PASS" if not errors else "CUR1_FREEZE_AUDIT_FAIL",
        "seed": manifest["seed"],
        "historical_id_count": manifest["historical_id_count"],
        "eligible_pool_count": manifest["eligible_pool_count"],
        "split_sizes": expected,
        "selected_questions": len(all_ids),
        "planned_trajectories": manifest["rollout_plan"]["total_new_trajectories"],
        "test_sealed": manifest["test_sealed"],
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
