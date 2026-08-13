#!/usr/bin/env python3
"""Exact-replay S1 checkpoints and extract the frozen Step-Gate features.

Run with torchrun.  Every rank owns an independent Evidence@400 replica and a
disjoint question-state shard; rank 0 combines the deterministic shard files.
Val3/Test paths are forbidden here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.rl.eca_step_adaptive_agent_loop import _with_step_protocol


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def id_hash(values: list[int]) -> str:
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()


def lexical_similarity(query: str, previous: list[str]) -> float:
    left = set(query.lower().split())
    if not left or not previous:
        return 0.0
    scores = []
    for value in previous:
        right = set(value.lower().split())
        scores.append(len(left & right) / max(1, len(left | right)))
    return float(max(scores))


def build_examples(args, tokenizer) -> list[dict]:
    import pandas as pd
    from verl.utils.tokenizer.chat_template import apply_chat_template

    frame = pd.read_parquet(args.parquet)
    data = {str(row.extra_info["sample_id"]): row for _, row in frame.iterrows()}
    dumps = {
        str(row["sample_id"]): row
        for row in (json.loads(line) for line in args.base_dump.read_text().splitlines() if line)
    }
    pairs = {
        str(row["branch_id"]): row
        for row in (json.loads(line) for line in args.pairs.read_text().splitlines() if line)
    }
    selections = [json.loads(line) for line in args.selections.read_text().splitlines() if line]
    if len(selections) != 1022 or set(pairs) != {str(row["branch_id"]) for row in selections}:
        raise RuntimeError("expected the frozen 1022 S1 paired states")
    close_ids = list(tokenizer.encode("</think>", add_special_tokens=False))
    examples = []
    for order, selected in enumerate(selections):
        sample_id = str(selected["sample_id"])
        dump = dumps[sample_id]
        target_index = int(selected["target_index"])
        records = [r for r in dump["step_records"] if int(r["step_index"]) == target_index]
        if len(records) != 1:
            raise RuntimeError(f"{sample_id}/cp{target_index}: target record missing or duplicated")
        record = records[0]
        raw_prompt = [dict(value) for value in data[sample_id].prompt.tolist()]
        prompt_ids = apply_chat_template(
            tokenizer, _with_step_protocol(raw_prompt), tools=None,
            add_generation_prompt=True, tokenize=True,
        )
        prompt_ids = list(prompt_ids.tolist() if hasattr(prompt_ids, "tolist") else prompt_ids)
        if id_hash(prompt_ids) != dump["step_prompt_sha256"]:
            raise RuntimeError(f"{sample_id}: step prompt SHA mismatch")
        end = int(record["checkpoint_response_end"])
        start = int(record["checkpoint_response_start"])
        sequence = prompt_ids + list(dump["response_token_ids"][:end])
        if id_hash(sequence) != selected["state_prefix_sha256"]:
            raise RuntimeError(f"{sample_id}/cp{target_index}: state prefix SHA mismatch")
        if sequence[-len(close_ids):] != close_ids:
            raise RuntimeError(f"{sample_id}/cp{target_index}: checkpoint does not end in </think>")
        logps = list(record["reasoning_raw_logprobs"]) + list(record["query_raw_logprobs"])
        if not logps:
            raise RuntimeError(f"{sample_id}/cp{target_index}: no raw chosen-token logprobs")
        pair = pairs[str(selected["branch_id"])]
        query = str(record["candidate_query"])
        examples.append({
            "order": order,
            "branch_id": str(selected["branch_id"]),
            "sample_id": sample_id,
            "target_index": target_index,
            "sequence": sequence,
            "prompt_length": len(prompt_ids),
            "checkpoint_start": start,
            "checkpoint_end": end,
            "query_position": len(sequence) - len(close_ids) - 1,
            "mean_logp": float(np.mean(logps)),
            "p10_logp": float(np.quantile(logps, 0.10)),
            "step_index": float(record["step_index"]),
            "previous_searches": float(record["num_previous_searches"]),
            "query_length": float(len(tokenizer.encode(query, add_special_tokens=False))),
            "duplicate_similarity": lexical_similarity(query, list(record["previous_queries"])),
            "label": 1.0 if pair["preference"] == "search" else 0.0,
            "delta_f1": float(pair["delta_f1"]),
            "search_f1": float(pair["search"]["f1"]),
            "continue_f1": float(pair["continue"]["f1"]),
            "search_calls": float(pair["search"]["search_calls"]),
            "continue_calls": float(pair["continue"]["search_calls"]),
            "search_tokens": float(pair["search"]["total_token_proxy"]),
            "continue_tokens": float(pair["continue"]["total_token_proxy"]),
        })
    return examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--base-dump", type=Path, required=True)
    ap.add_argument("--selections", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--entropy-chunk", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if any(word in str(value).lower() for value in vars(args).values() for word in ("val3", "test")):
        raise RuntimeError("Val3/Test paths are forbidden during Train feature extraction")

    import torch
    import torch.distributed as dist
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    examples = build_examples(args, tokenizer)
    if args.limit:
        examples = examples[: args.limit]
    owned = examples[rank::world]
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), dtype=torch.bfloat16, attn_implementation="eager", trust_remote_code=True
    ).cuda().eval()
    base = model.model
    pad_id = int(tokenizer.pad_token_id)
    rows = []
    with torch.inference_mode():
        for offset in range(0, len(owned), args.batch_size):
            batch = owned[offset: offset + args.batch_size]
            lengths = [len(row["sequence"]) for row in batch]
            width = max(lengths)
            input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long, device="cuda")
            mask = torch.zeros_like(input_ids)
            for i, row in enumerate(batch):
                values = torch.tensor(row["sequence"], dtype=torch.long, device="cuda")
                input_ids[i, : len(values)] = values
                mask[i, : len(values)] = 1
            captured = []
            handle = base.layers[26].register_forward_hook(
                lambda _module, _inputs, output: captured.append(output[0] if isinstance(output, tuple) else output)
            )
            output = base(input_ids=input_ids, attention_mask=mask, use_cache=False, return_dict=True)
            handle.remove()
            if len(captured) != 1:
                raise RuntimeError("L27 hook did not fire exactly once")
            h27_all = captured[0]
            final_all = output.last_hidden_state
            for i, row in enumerate(batch):
                query_h27 = h27_all[i, int(row["query_position"])].float().cpu().numpy()
                # Predict every normalized checkpoint token from its preceding state.
                first = int(row["prompt_length"] + row["checkpoint_start"] - 1)
                last = int(row["prompt_length"] + row["checkpoint_end"] - 1)
                states = final_all[i, first:last]
                entropy_parts = []
                for begin in range(0, len(states), args.entropy_chunk):
                    logits = model.lm_head(states[begin: begin + args.entropy_chunk]).float()
                    logz = torch.logsumexp(logits, dim=-1)
                    probs = torch.softmax(logits, dim=-1)
                    entropy_parts.append(logz - (probs * logits).sum(dim=-1))
                mean_entropy = float(torch.cat(entropy_parts).mean().cpu())
                scalar = np.asarray([
                    row["mean_logp"], row["p10_logp"], mean_entropy,
                    row["step_index"], row["previous_searches"], row["query_length"],
                    row["duplicate_similarity"],
                ], dtype=np.float32)
                rows.append((row, query_h27, scalar))
            done = min(offset + len(batch), len(owned))
            print(f"rank={rank} extracted={done}/{len(owned)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shard = args.output.with_name(f"{args.output.stem}.rank{rank}.npz")
    np.savez(
        shard,
        order=np.asarray([r[0]["order"] for r in rows]),
        branch_ids=np.asarray([r[0]["branch_id"] for r in rows]),
        sample_ids=np.asarray([r[0]["sample_id"] for r in rows]),
        target_index=np.asarray([r[0]["target_index"] for r in rows]),
        layer27=np.stack([r[1] for r in rows]),
        scalars=np.stack([r[2] for r in rows]),
        label=np.asarray([r[0]["label"] for r in rows]),
        delta_f1=np.asarray([r[0]["delta_f1"] for r in rows]),
        search_f1=np.asarray([r[0]["search_f1"] for r in rows]),
        continue_f1=np.asarray([r[0]["continue_f1"] for r in rows]),
        search_calls=np.asarray([r[0]["search_calls"] for r in rows]),
        continue_calls=np.asarray([r[0]["continue_calls"] for r in rows]),
        search_tokens=np.asarray([r[0]["search_tokens"] for r in rows]),
        continue_tokens=np.asarray([r[0]["continue_tokens"] for r in rows]),
    )
    if world > 1:
        dist.barrier()
    if rank == 0:
        parts = [np.load(args.output.with_name(f"{args.output.stem}.rank{i}.npz")) for i in range(world)]
        order = np.concatenate([part["order"] for part in parts])
        permutation = np.argsort(order)
        keys = [key for key in parts[0].files if key != "order"]
        combined = {key: np.concatenate([part[key] for part in parts], axis=0)[permutation] for key in keys}
        np.savez(args.output, **combined)
        for part in parts:
            part.close()
        summary = {
            "gate": "STEP_GATE_FEATURE_CAPTURE_PASS",
            "n": len(permutation),
            "world_size": world,
            "batch_size_per_gpu": args.batch_size,
            "feature_schema": {
                "layer27": "native-HF hidden_states[27] at candidate-query final token",
                "scalars": ["step_mean_logp", "step_p10_logp", "checkpoint_mean_entropy", "step_index", "previous_searches", "query_token_length", "max_previous_query_jaccard"],
                "entropy": "mean next-token categorical entropy over normalized checkpoint tokens",
            },
            "artifact_sha256": sha256_file(args.output),
            "val3_read": False,
            "test_read": False,
        }
        args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
