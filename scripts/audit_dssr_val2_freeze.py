#!/usr/bin/env python3
"""Independent audit of the immutable DSSR Val2 split and prompt contract."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    ap.add_argument("--cur1-manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    import pyarrow.parquet as pq

    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    cur1 = json.loads(args.cur1_manifest.read_text())
    errors: list[str] = []
    rows = pq.read_table(args.data_dir / "search.parquet").to_pylist()
    ids = [str(row["extra_info"]["sample_id"]) for row in rows]
    cur1_ids = {str(x) for values in cur1["question_ids"].values() for x in values}
    if len(rows) != 128 or len(set(ids)) != 128:
        errors.append(f"search rows/unique IDs={len(rows)}/{len(set(ids))}")
    if set(ids) & cur1_ids:
        errors.append("Val2 overlaps CUR-1 Train/Validation/original-Test")
    if ids != manifest["question_ids"] or ids != (args.data_dir / "val2_ids.txt").read_text().splitlines():
        errors.append("parquet/manifest/ID-file order mismatch")
    if hashlib.sha256("\n".join(ids).encode()).hexdigest() != manifest["split_id_sha256"]:
        errors.append("split ID hash mismatch")
    for row in rows:
        extra = row["extra_info"]
        prompt_blob = json.dumps(row["prompt"], sort_keys=True)
        if extra.get("cur_forced_arm") != "search" or extra.get("cur_split") != "val2":
            errors.append(f"{extra.get('sample_id')}: arm/split contract failure")
        if "contexts" in prompt_blob or "supporting_facts" in prompt_blob:
            errors.append(f"{extra.get('sample_id')}: tool/reward data leaked into prompt")

    contexts = [
        json.loads(line)
        for line in (args.data_dir / "contexts_index.jsonl").read_text().splitlines()
        if line.strip()
    ]
    prompts = [
        json.loads(line)
        for line in (args.data_dir / "prompt_manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if {str(x["sample_id"]) for x in contexts} != set(ids):
        errors.append("context IDs do not equal selected IDs")
    if [str(x["sample_id"]) for x in prompts] != ids or len(prompts) != 128:
        errors.append("prompt-manifest IDs/order/count mismatch")
    if any(not x.get("canonical_prompt_sha256") or not x.get("canonical_prompt_ids") for x in prompts):
        errors.append("prompt manifest contains empty token/hash data")

    if manifest.get("seed") != 2026081202 or not manifest.get("original_test_sealed"):
        errors.append("seed or Test-seal contract mismatch")
    if manifest.get("cur1_manifest_sha256") != sha256_file(args.cur1_manifest):
        errors.append("CUR-1 manifest hash mismatch")
    for name, expected in manifest["frozen_artifact_sha256"].items():
        if sha256_file(args.data_dir / name) != expected:
            errors.append(f"frozen artifact hash mismatch: {name}")
    for name, expected in manifest["model_contract_sha256"].items():
        if sha256_file(args.model / name) != expected:
            errors.append(f"model contract hash mismatch: {name}")

    summary = {
        "gate": "DSSR_VAL2_FREEZE_AUDIT_PASS" if not errors else "DSSR_VAL2_FREEZE_AUDIT_FAIL",
        "seed": manifest.get("seed"),
        "questions": len(set(ids)),
        "historical_id_count": manifest.get("historical_id_count"),
        "eligible_pool_count": manifest.get("eligible_pool_count"),
        "planned_probe_trajectories": manifest["acquisition_plan"]["probe"]["trajectories"],
        "planned_search_trajectories": manifest["acquisition_plan"]["search"]["trajectories"],
        "original_test_sealed": manifest.get("original_test_sealed"),
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
