#!/usr/bin/env python3
"""Extract pre-action static features for frozen DSSR IDs without reading outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LAYERS = (18, 27, 36)
ROUTE_TOKEN_IDS = {"search": 27, "internal": 4159}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-manifest", type=Path, required=True)
    ap.add_argument("--ids", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    if "test" in str(args.prompt_manifest).lower() or "test" in str(args.ids).lower():
        raise RuntimeError("sealed Test paths are forbidden")

    import torch
    from transformers import AutoModelForCausalLM

    rows = [json.loads(line) for line in args.prompt_manifest.read_text().splitlines() if line]
    ids = [line for line in args.ids.read_text().splitlines() if line]
    if [str(row["sample_id"]) for row in rows] != ids or len(ids) != 128:
        raise RuntimeError("prompt manifest order differs from frozen 128 IDs")
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), dtype=torch.bfloat16, attn_implementation="eager", trust_remote_code=True
    ).cuda().eval()
    captured = {layer: [] for layer in LAYERS}
    margins = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            lengths = [len(row["canonical_prompt_ids"]) for row in batch]
            width = max(lengths)
            input_ids = torch.zeros((len(batch), width), dtype=torch.long, device="cuda")
            mask = torch.zeros_like(input_ids)
            for i, row in enumerate(batch):
                values = torch.tensor(row["canonical_prompt_ids"], device="cuda")
                input_ids[i, : len(values)] = values
                mask[i, : len(values)] = 1
            output = model(input_ids=input_ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
            row_index = torch.arange(len(batch), device="cuda")
            token_index = torch.tensor(lengths, device="cuda") - 1
            for layer in LAYERS:
                captured[layer].append(output.hidden_states[layer][row_index, token_index].float().cpu().numpy())
            logits = output.logits[row_index, token_index].float()
            margins.append((logits[:, ROUTE_TOKEN_IDS["search"]] - logits[:, ROUTE_TOKEN_IDS["internal"]]).cpu().numpy())
            print(f"static features={min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        sample_ids=np.asarray(ids),
        layer18=np.concatenate(captured[18]),
        layer27=np.concatenate(captured[27]),
        layer36=np.concatenate(captured[36]),
        root_margin=np.concatenate(margins),
    )
    summary = {
        "gate": "DSSR_STATIC_FEATURE_CAPTURE_PASS",
        "n": len(ids),
        "artifact_sha256": sha256_file(args.output),
        "outcomes_read": False,
        "test_read": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
