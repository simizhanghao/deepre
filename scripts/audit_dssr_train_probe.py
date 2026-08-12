#!/usr/bin/env python3
"""Audit Train640 Probe capture and pair it with the existing Search N=1 outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-dir", type=Path, required=True)
    ap.add_argument("--train-ids", type=Path, required=True)
    ap.add_argument("--search-outcomes", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    errors: list[str] = []
    frozen_ids = [line for line in args.train_ids.read_text().splitlines() if line]
    probes = [
        json.loads(line)
        for line in (args.probe_dir / "probes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    probe_ids = [str(row["sample_id"]) for row in probes]
    hidden = np.load(args.probe_dir / "hidden_states.npz")
    hidden_ids = [str(x) for x in hidden["sample_ids"]]
    if len(frozen_ids) != 640 or len(set(frozen_ids)) != 640:
        errors.append("frozen Train IDs are not exactly 640 unique IDs")
    if probe_ids != frozen_ids or hidden_ids != frozen_ids:
        errors.append("Probe/hidden order differs from frozen Train IDs")
    for layer in (18, 27, 36):
        values = hidden[f"layer{layer}"]
        if values.shape != (640, 2048) or not np.isfinite(values).all():
            errors.append(f"layer{layer}: invalid shape or non-finite values")

    required_scalars = (
        "prefix20_mean_entropy",
        "prefix20_mean_margin",
        "prefix20_mean_logprob",
        "answer_mean_logprob",
        "answer_p10_logprob",
        "answer_min_logprob",
        "cosine_18_27",
        "cosine_27_36",
        "cosine_18_36",
        "relative_update_18_27",
        "relative_update_27_36",
    )
    for row in probes:
        if row.get("forced_prefix_ids") != [4159, 2978, 29]:
            errors.append(f"{row['sample_id']}: forced-prefix mismatch")
        if not row.get("probe_valid") or not row.get("closed_answer"):
            errors.append(f"{row['sample_id']}: invalid/unclosed Probe")
        if not 3 <= int(row.get("response_tokens", 0)) <= 96:
            errors.append(f"{row['sample_id']}: response-token cap failure")
        if not 1 <= int(row.get("prefix20_tokens", 0)) <= 20:
            errors.append(f"{row['sample_id']}: Prefix-20 count failure")
        if any(row.get(key) is None or not np.isfinite(float(row[key])) for key in required_scalars):
            errors.append(f"{row['sample_id']}: missing/non-finite feature")

    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for line in args.search_outcomes.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        grouped[str(row["sample_id"])][str(row["cur_forced_arm"])].append(row)
    if set(grouped) != set(frozen_ids):
        errors.append("existing paired-outcome IDs differ from frozen Train IDs")
    search_by_id: dict[str, dict] = {}
    search_policy_failures = 0
    search_action_invalid = 0
    for sample_id in frozen_ids:
        rows = grouped.get(sample_id, {}).get("search", [])
        if len(rows) != 1:
            errors.append(f"{sample_id}: Search outcome count={len(rows)} instead of 1")
        else:
            search_by_id[sample_id] = rows[0]
            search_action_invalid += int(not int(rows[0].get("cur_forced_action_valid", 0)))
            search_policy_failures += int(rows[0].get("cur_policy_failure", 0))

    regrets = []
    direction = Counter()
    if not errors:
        for probe in probes:
            search_f1 = float(search_by_id[probe["sample_id"]]["answer_f1"])
            probe_f1 = float(probe["answer_f1"])
            delta = search_f1 - probe_f1
            regrets.append(max(0.0, delta))
            direction["search_positive" if delta > 0 else "probe_positive" if delta < 0 else "tie"] += 1
    summary = {
        "gate": "DSSR_TRAIN_PROBE_AUDIT_PASS" if not errors else "DSSR_TRAIN_PROBE_AUDIT_FAIL",
        "questions": len(probes),
        "probe_valid_rate": float(np.mean([bool(row.get("probe_valid")) for row in probes])),
        "closed_answer_rate": float(np.mean([bool(row.get("closed_answer")) for row in probes])),
        "mean_probe_f1": float(np.mean([float(row["answer_f1"]) for row in probes])),
        "mean_search_f1": float(np.mean([float(search_by_id[x]["answer_f1"]) for x in frozen_ids if x in search_by_id])) if search_by_id else None,
        "mean_skip_regret": float(np.mean(regrets)) if regrets else None,
        "positive_skip_regret_rate": float(np.mean(np.asarray(regrets) > 0)) if regrets else None,
        "direction_counts": dict(direction),
        "search_action_invalid_retained": search_action_invalid,
        "search_policy_failures_retained": search_policy_failures,
        "artifact_sha256": {
            "train_ids": sha256_file(args.train_ids),
            "probes": sha256_file(args.probe_dir / "probes.jsonl"),
            "hidden_states": sha256_file(args.probe_dir / "hidden_states.npz"),
            "search_outcomes": sha256_file(args.search_outcomes),
        },
        "errors": errors[:20],
        "error_count": len(errors),
        "test_read": False,
        "val2_outcomes_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
