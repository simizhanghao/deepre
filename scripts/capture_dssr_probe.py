#!/usr/bin/env python3
"""Capture the frozen deterministic DSSR internal probe and its features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

LAYERS = (18, 27, 36)
FORCED_PREFIX = "<internal>"
OPEN_TAG = "<answer>"
CLOSE_TAG = "</answer>"
MAX_RESPONSE_TOKENS = 96


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def rel_update(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a))
    return float(np.linalg.norm(b - a) / denom) if denom else 0.0


def answer_from_text(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--prompt-manifest", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

    from src.eval.metrics import exact_match, token_f1

    class CloseThenEos(LogitsProcessor):
        def __init__(self, close_ids: list[int], eos_id: int):
            self.close_ids = close_ids
            self.eos_id = eos_id

        def __call__(self, input_ids, scores):
            width = len(self.close_ids)
            matched = (input_ids[:, -width:] == input_ids.new_tensor(self.close_ids)).all(dim=1)
            if matched.any():
                scores[matched] = -float("inf")
                scores[matched, self.eos_id] = 0.0
            return scores

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    prompt_rows = [
        json.loads(line)
        for line in args.prompt_manifest.read_text().splitlines()
        if line.strip()
    ]
    all_data_rows = pq.read_table(args.parquet).to_pylist()
    if args.limit:
        prompt_rows = prompt_rows[: args.limit]
    prompt_ids = [str(row["sample_id"]) for row in prompt_rows]
    data_by_id = {}
    for row in all_data_rows:
        data_by_id.setdefault(str(row["extra_info"]["sample_id"]), row)
    missing = [sample_id for sample_id in prompt_ids if sample_id not in data_by_id]
    if missing:
        raise RuntimeError(f"parquet lacks {len(missing)} prompt IDs; first={missing[0]}")
    data_rows = [data_by_id[sample_id] for sample_id in prompt_ids]
    data_ids = [str(row["extra_info"]["sample_id"]) for row in data_rows]
    if prompt_ids != data_ids:
        raise RuntimeError("prompt manifest and parquet IDs/order differ")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), dtype=torch.bfloat16, attn_implementation="eager", trust_remote_code=True
    ).cuda().eval()
    prefix_ids = tokenizer.encode(FORCED_PREFIX, add_special_tokens=False)
    close_ids = tokenizer.encode(CLOSE_TAG, add_special_tokens=False)
    if prefix_ids != [4159, 2978, 29]:
        raise RuntimeError(f"forced-prefix token contract changed: {prefix_ids}")
    eos_id = int(tokenizer.eos_token_id)
    pad_id = int(tokenizer.pad_token_id)

    records: list[dict] = []
    hidden_by_layer = {layer: [] for layer in LAYERS}
    started = time.perf_counter()
    for batch_start in range(0, len(prompt_rows), args.batch_size):
        batch_prompts = prompt_rows[batch_start : batch_start + args.batch_size]
        batch_data = data_rows[batch_start : batch_start + args.batch_size]
        source = [list(row["canonical_prompt_ids"]) + prefix_ids for row in batch_prompts]
        source_lengths = [len(x) for x in source]
        width = max(source_lengths)
        input_ids = torch.full((len(source), width), pad_id, dtype=torch.long, device="cuda")
        attention_mask = torch.zeros_like(input_ids)
        for i, values in enumerate(source):
            input_ids[i, width - len(values) :] = torch.tensor(values, device="cuda")
            attention_mask[i, width - len(values) :] = 1

        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=MAX_RESPONSE_TOKENS - len(prefix_ids),
                eos_token_id=eos_id,
                pad_token_id=pad_id,
                logits_processor=[CloseThenEos(close_ids, eos_id)],
                use_cache=True,
            )
        responses: list[list[int]] = []
        for row in generated.tolist():
            continuation = row[width:]
            if eos_id in continuation:
                continuation = continuation[: continuation.index(eos_id)]
            while continuation and continuation[-1] == pad_id:
                continuation.pop()
            responses.append(prefix_ids + continuation)

        complete = [list(row["canonical_prompt_ids"]) + response for row, response in zip(batch_prompts, responses)]
        complete_lengths = [len(x) for x in complete]
        full_width = max(complete_lengths)
        full_ids = torch.full((len(complete), full_width), pad_id, dtype=torch.long, device="cuda")
        full_mask = torch.zeros_like(full_ids)
        offsets = []
        for i, values in enumerate(complete):
            offset = full_width - len(values)
            offsets.append(offset)
            full_ids[i, offset:] = torch.tensor(values, device="cuda")
            full_mask[i, offset:] = 1
        with torch.inference_mode():
            outputs = model(
                input_ids=full_ids,
                attention_mask=full_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        if len(outputs.hidden_states) != 37:
            raise RuntimeError(f"expected 37 hidden-state slots, got {len(outputs.hidden_states)}")

        for i, (prompt, data, response) in enumerate(zip(batch_prompts, batch_data, responses)):
            response_text = tokenizer.decode(response, skip_special_tokens=False)
            open_char = response_text.find(OPEN_TAG)
            content_start = open_char + len(OPEN_TAG) if open_char >= 0 else -1
            close_char = response_text.find(CLOSE_TAG, content_start) if content_start >= 0 else -1
            closed = open_char >= 0 and close_char >= 0
            pieces = [tokenizer.decode([token], skip_special_tokens=False) for token in response]
            spans = []
            cursor = 0
            for piece in pieces:
                spans.append((cursor, cursor + len(piece)))
                cursor += len(piece)
            if "".join(pieces) != response_text:
                raise RuntimeError("per-token decoded spans do not reconstruct response")
            content_indices = (
                [
                    j
                    for j, ((start, end), piece) in enumerate(zip(spans, pieces))
                    if start >= content_start and end <= close_char and piece.strip()
                ]
                if closed
                else []
            )
            valid = bool(closed and content_indices)
            prediction = answer_from_text(response_text)
            golds = list(data["reward_model"]["ground_truth"]["target"])

            token_logps: list[float] = []
            entropies: list[float] = []
            margins: list[float] = []
            for response_index in content_indices:
                absolute = offsets[i] + len(prompt["canonical_prompt_ids"]) + response_index
                logits = outputs.logits[i, absolute - 1].float()
                token_id = response[response_index]
                log_z = torch.logsumexp(logits, dim=-1)
                token_logps.append(float(logits[token_id] - log_z))
                probs = torch.softmax(logits, dim=-1)
                entropies.append(float(log_z - torch.sum(probs * logits)))
                top2 = torch.topk(logits, 2).values
                margins.append(float(top2[0] - top2[1]))

            endpoint = content_indices[-1] if valid else max(len(response) - 1, 0)
            absolute_endpoint = offsets[i] + len(prompt["canonical_prompt_ids"]) + endpoint
            states: dict[int, np.ndarray] = {}
            for layer in LAYERS:
                state = outputs.hidden_states[layer][i, absolute_endpoint].float().cpu().numpy()
                states[layer] = state
                hidden_by_layer[layer].append(state)
            prefix_n = min(20, len(token_logps))
            prefix_slice = slice(0, prefix_n)
            record = {
                "sample_id": prompt["sample_id"],
                "canonical_prompt_sha256": prompt["canonical_prompt_sha256"],
                "canonical_prompt_tokens": len(prompt["canonical_prompt_ids"]),
                "forced_prefix_ids": prefix_ids,
                "response_token_ids": response,
                "response_tokens": len(response),
                "response_text": response_text,
                "answer": prediction,
                "golds": golds,
                "answer_em": float(exact_match(prediction, golds)),
                "answer_f1": float(token_f1(prediction, golds)),
                "closed_answer": bool(closed),
                "probe_valid": valid,
                "answer_content_tokens": len(content_indices),
                "prefix20_tokens": prefix_n,
                "prefix20_mean_entropy": float(np.mean(entropies[prefix_slice])) if prefix_n else None,
                "prefix20_mean_margin": float(np.mean(margins[prefix_slice])) if prefix_n else None,
                "prefix20_mean_logprob": float(np.mean(token_logps[prefix_slice])) if prefix_n else None,
                "answer_mean_logprob": float(np.mean(token_logps)) if token_logps else None,
                "answer_p10_logprob": float(np.percentile(token_logps, 10)) if token_logps else None,
                "answer_min_logprob": float(np.min(token_logps)) if token_logps else None,
                "cosine_18_27": cosine(states[18], states[27]),
                "cosine_27_36": cosine(states[27], states[36]),
                "cosine_18_36": cosine(states[18], states[36]),
                "relative_update_18_27": rel_update(states[18], states[27]),
                "relative_update_27_36": rel_update(states[27], states[36]),
            }
            if any(isinstance(value, float) and not math.isfinite(value) for value in record.values()):
                raise RuntimeError(f"non-finite feature for {prompt['sample_id']}")
            records.append(record)
        print(f"probe captured={len(records)}/{len(prompt_rows)}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "probes.jsonl"
    detail_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    feature_path = args.output_dir / "hidden_states.npz"
    np.savez(
        feature_path,
        sample_ids=np.asarray(prompt_ids),
        layer18=np.asarray(hidden_by_layer[18], dtype=np.float32),
        layer27=np.asarray(hidden_by_layer[27], dtype=np.float32),
        layer36=np.asarray(hidden_by_layer[36], dtype=np.float32),
    )
    deterministic_payload = [
        {
            key: value
            for key, value in row.items()
            if key not in {"response_text"}
        }
        for row in records
    ]
    payload_hash = hashlib.sha256(
        json.dumps(deterministic_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        "gate": "DSSR_PROBE_CAPTURE_PASS",
        "n": len(records),
        "valid_probe_rate": float(np.mean([row["probe_valid"] for row in records])),
        "closed_answer_rate": float(np.mean([row["closed_answer"] for row in records])),
        "mean_probe_f1": float(np.mean([row["answer_f1"] for row in records])),
        "response_token_mean": float(np.mean([row["response_tokens"] for row in records])),
        "response_token_max": int(max(row["response_tokens"] for row in records)),
        "generation_and_feature_wall_seconds": time.perf_counter() - started,
        "deterministic_payload_sha256": payload_hash,
        "details_sha256": sha256_file(detail_path),
        "hidden_states_sha256": sha256_file(feature_path),
        "model": str(args.model),
        "seed": args.seed,
        "probe_contract": {
            "forced_prefix": FORCED_PREFIX,
            "forced_prefix_ids": prefix_ids,
            "temperature": 0,
            "max_total_response_tokens": MAX_RESPONSE_TOKENS,
            "tools": False,
            "confidence_scope": "non-empty answer-content tokens only",
            "post_hidden_scope": "last non-empty answer-content token; invalid probes route Search",
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
