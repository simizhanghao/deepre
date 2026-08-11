#!/usr/bin/env python3
"""Summarize frozen root-action log-prob margins from a VeXact capture."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()

    import torch

    capture = Path(args.capture_dir)
    logits = torch.load(capture / "all_logits.pt", map_location="cpu")
    metadata = json.loads((capture / "metadata.json").read_text())
    by_boundary: dict[str, list[float]] = defaultdict(list)
    rows = []
    for sample_id, tensors in logits.items():
        logp = torch.log_softmax(tensors[0].reshape(-1).float(), dim=-1)
        search = float(logp[27])
        internal = float(logp[4159])
        margin = search - internal
        boundary = metadata[sample_id]["boundary"]
        by_boundary[boundary].append(margin)
        rows.append(
            {
                "sample_id": sample_id,
                "boundary": boundary,
                "logp_search": search,
                "logp_internal": internal,
                "route_margin": margin,
            }
        )
    summary = {
        "step": args.step,
        "n": len(rows),
        "definition": "logP(<search>)-logP(<internal>)",
        "mean_route_margin": {
            label: sum(values) / len(values) for label, values in sorted(by_boundary.items())
        },
        "rows": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
