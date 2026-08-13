#!/usr/bin/env python3
"""Apply the already-frozen full-Train B3 to exact batch32 Val3 root features."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_step_preference_gate import full_b3_scores, sha256_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--root-model-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    ids, scores = full_b3_scores(args.features, args.root_model_dir)
    payload = {
        "gate": "STEP_VAL3_ROOT_B3_FREEZE_PASS",
        "n": len(ids),
        "definition": "frozen CUR1 full-Train B3 on native-HF canonical-prompt L27 extracted in fixed ordered batch32",
        "scores": {sample_id: float(score) for sample_id, score in zip(ids, scores)},
        "features_sha256": sha256_file(args.features),
        "root_model_sha256": {f"seed{seed}": sha256_file(args.root_model_dir / f"seed{seed}.pt") for seed in (1, 2, 3)},
        "val3_outcomes_read": False,
        "test_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "scores"}, indent=2))


if __name__ == "__main__":
    main()
