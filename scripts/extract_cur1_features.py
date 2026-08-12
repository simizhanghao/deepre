#!/usr/bin/env python3
"""Extract the frozen CUR-1 pre-action feature contract for open splits only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

LAYERS = (18, 27, 36)
ROUTE_TOKEN_IDS = {"search": 27, "internal": 4159}
EXPECTED = {"train": 640, "validation": 128}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_open_rows(manifest: Path, split: str) -> list[dict]:
    """Stop at the exact open-split count; never consume a test record."""
    rows: list[dict] = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == split:
                rows.append(row)
                if len(rows) == EXPECTED[split]:
                    break
    if len(rows) != EXPECTED[split]:
        raise RuntimeError(f"{split}: expected {EXPECTED[split]} rows, found {len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--split", choices=sorted(EXPECTED), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    rows = load_open_rows(args.manifest, args.split)
    frozen_ids = [line.strip() for line in args.ids.read_text().splitlines() if line.strip()]
    sample_ids = [str(row["sample_id"]) for row in rows]
    if sample_ids != frozen_ids:
        raise RuntimeError(f"{args.split}: prompt manifest order differs from frozen IDs")

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    ).cuda().eval()
    captured = {layer: [] for layer in LAYERS}
    margins: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            lengths = [len(row["canonical_prompt_ids"]) for row in batch]
            width = max(lengths)
            input_ids = torch.zeros((len(batch), width), dtype=torch.long, device="cuda")
            attention_mask = torch.zeros_like(input_ids)
            for index, row in enumerate(batch):
                token_ids = torch.tensor(row["canonical_prompt_ids"], dtype=torch.long, device="cuda")
                input_ids[index, : len(token_ids)] = token_ids
                attention_mask[index, : len(token_ids)] = 1
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            if len(outputs.hidden_states) != 37:
                raise RuntimeError(f"expected 37 hidden-state slots, got {len(outputs.hidden_states)}")
            row_index = torch.arange(len(batch), device="cuda")
            token_index = torch.tensor(lengths, device="cuda") - 1
            for layer in LAYERS:
                states = outputs.hidden_states[layer][row_index, token_index].float().cpu().numpy()
                captured[layer].append(states)
            root_logits = outputs.logits[row_index, token_index].float()
            margin = (
                root_logits[:, ROUTE_TOKEN_IDS["search"]]
                - root_logits[:, ROUTE_TOKEN_IDS["internal"]]
            )
            margins.append(margin.cpu().numpy())
            print(f"{args.split}: captured={min(start + len(batch), len(rows))}/{len(rows)}", flush=True)

    schema = {
        "hidden_definition": "transformers hidden_states[k] at final canonical-prompt token",
        "layers": list(LAYERS),
        "root_margin": "logP(first_search_token)-logP(first_internal_token), equal to logit difference",
        "route_token_ids": ROUTE_TOKEN_IDS,
        "dtype_saved": "float32",
        "test_read": False,
    }
    schema_sha = hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        sample_ids=np.asarray(sample_ids),
        layer18=np.concatenate(captured[18]),
        layer27=np.concatenate(captured[27]),
        layer36=np.concatenate(captured[36]),
        root_margin=np.concatenate(margins),
    )
    summary = {
        "gate": "CUR1_FEATURE_CAPTURE_PASS",
        "split": args.split,
        "n": len(rows),
        "model": str(args.model),
        "feature_schema": schema,
        "feature_schema_sha256": schema_sha,
        "artifact_sha256": sha256(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
