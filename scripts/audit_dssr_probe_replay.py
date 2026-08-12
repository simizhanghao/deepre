#!/usr/bin/env python3
"""Audit two independent DSSR probe captures for deterministic replay parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=Path, required=True)
    ap.add_argument("--second", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    first = load_jsonl(args.first / "probes.jsonl")
    second = load_jsonl(args.second / "probes.jsonl")
    errors: list[str] = []
    if [x["sample_id"] for x in first] != [x["sample_id"] for x in second]:
        errors.append("sample IDs/order differ")
    if [x["response_token_ids"] for x in first] != [x["response_token_ids"] for x in second]:
        errors.append("greedy response token IDs differ")
    scalar_exclusions = {"response_text"}
    for i, (left, right) in enumerate(zip(first, second)):
        if {k: v for k, v in left.items() if k not in scalar_exclusions} != {
            k: v for k, v in right.items() if k not in scalar_exclusions
        }:
            errors.append(f"scalar/token feature mismatch at row {i}")
            break
    a = np.load(args.first / "hidden_states.npz")
    b = np.load(args.second / "hidden_states.npz")
    max_abs = 0.0
    for key in ("layer18", "layer27", "layer36"):
        if a[key].shape != b[key].shape:
            errors.append(f"{key}: shape differs")
            continue
        max_abs = max(max_abs, float(np.max(np.abs(a[key] - b[key]))))
        if not np.array_equal(a[key], b[key]):
            errors.append(f"{key}: hidden states are not bit-exact")
    summary = {
        "gate": "DSSR_SK0_REPLAY_PASS" if not errors else "DSSR_SK0_REPLAY_FAIL",
        "n": len(first),
        "response_tokens_bit_exact": not any("response token" in x for x in errors),
        "hidden_max_abs_delta": max_abs,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
