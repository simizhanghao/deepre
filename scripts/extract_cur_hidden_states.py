#!/usr/bin/env python3
"""Extract frozen pre-action last-prompt-token states for CUR layers 18/27/36."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM

    rows = [json.loads(x) for x in args.manifest.read_text().splitlines() if x.strip()]
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), dtype=torch.bfloat16, attn_implementation="eager", trust_remote_code=True
    ).cuda().eval()
    layers = (18, 27, 36)
    captured = {layer: [] for layer in layers}
    ids_out = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            lengths = [len(x["canonical_prompt_ids"]) for x in batch]
            width = max(lengths)
            input_ids = torch.zeros((len(batch), width), dtype=torch.long, device="cuda")
            attention = torch.zeros_like(input_ids)
            for i, row in enumerate(batch):
                ids = torch.tensor(row["canonical_prompt_ids"], dtype=torch.long, device="cuda")
                input_ids[i, : len(ids)] = ids
                attention[i, : len(ids)] = 1
            output = model(input_ids=input_ids, attention_mask=attention, output_hidden_states=True)
            if len(output.hidden_states) != 37:
                raise RuntimeError(f"expected 37 hidden-state slots, got {len(output.hidden_states)}")
            index = torch.tensor(lengths, device="cuda") - 1
            batch_index = torch.arange(len(batch), device="cuda")
            for layer in layers:
                state = output.hidden_states[layer][batch_index, index].float().cpu().numpy()
                captured[layer].append(state)
            ids_out.extend(str(x["sample_id"]) for x in batch)
            print(f"captured={min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        sample_ids=np.asarray(ids_out),
        layer18=np.concatenate(captured[18]),
        layer27=np.concatenate(captured[27]),
        layer36=np.concatenate(captured[36]),
    )
    summary = {
        "gate": "CUR_HIDDEN_CAPTURE_PASS",
        "n": len(rows),
        "layers": list(layers),
        "definition": "transformers hidden_states[k] at final canonical-prompt token, before action intervention",
        "dtype_saved": "float32",
        "model": str(args.model),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
