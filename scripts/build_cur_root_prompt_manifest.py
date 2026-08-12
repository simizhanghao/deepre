#!/usr/bin/env python3
"""Reconstruct and hash-check CUR canonical prompts for frozen root probing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids).encode("utf-8")).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--outcomes", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    from datasets import Dataset
    from transformers import AutoTokenizer
    from verl.utils.tokenizer.chat_template import apply_chat_template

    hashes: dict[str, str] = {}
    for line in args.outcomes.read_text().splitlines():
        row = json.loads(line)
        hashes.setdefault(str(row["sample_id"]), str(row["canonical_prompt_sha256"]))
    tok = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)
    seen = set()
    out = []
    for row in Dataset.from_parquet(str(args.parquet)):
        sid = str(row["extra_info"]["sample_id"])
        if sid in seen:
            continue
        seen.add(sid)
        ids = list(
            apply_chat_template(
                tok, row["prompt"], tools=None, tokenize=True, add_generation_prompt=True
            )
        )
        got = digest(ids)
        if hashes.get(sid) != got:
            raise SystemExit(f"canonical prompt mismatch {sid}: capture={hashes.get(sid)} rebuilt={got}")
        out.append({
            "sample_id": sid,
            "boundary": "CUR",
            "canonical_prompt_sha256": got,
            "canonical_prompt_ids": ids,
        })
    if len(out) != 128:
        raise SystemExit(f"expected 128 prompts, got {len(out)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(x) + "\n" for x in out))
    print(json.dumps({"gate": "CUR_ROOT_PROMPT_MANIFEST_PASS", "n": len(out)}, indent=2))


if __name__ == "__main__":
    main()
