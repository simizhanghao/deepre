#!/usr/bin/env python3
"""Phase 3B0 hard gate: decode(response_mask==1) must not contain observations.

Supports:
  1) Offline synthetic audit (tokenize a fake trajectory)
  2) Audit dumped trajectory json from a rollout worker
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.rl.mask_audit import dump_mask_audit


def audit_token_lists(tokenizer, response_ids: List[int], response_mask: List[int]) -> Dict[str, Any]:
    return dump_mask_audit(tokenizer, response_ids, response_mask)


def _synthetic(tokenizer) -> Dict[str, Any]:
    """Build a tiny synthetic trajectory and verify masking logic helpers."""
    policy = (
        "<think>need docs</think>\n"
        "<search>\nChristopher Nolan birthplace\n</search>"
    )
    obs = (
        "<observation>\n"
        "[hotpotqa_x_ctx_0] Christopher Nolan: Christopher Nolan was born in London.\n"
        "</observation>\n"
        "Continue. Prefer <evidence> then <think> then <answer>."
    )
    answer = (
        "<evidence>\nLondon\n</evidence>\n"
        "<answer>\nLondon\n</answer>"
    )
    p_ids = tokenizer.encode(policy, add_special_tokens=False)
    o_ids = tokenizer.encode(obs, add_special_tokens=False)
    a_ids = tokenizer.encode(answer, add_special_tokens=False)
    response_ids = p_ids + o_ids + a_ids
    response_mask = [1] * len(p_ids) + [0] * len(o_ids) + [1] * len(a_ids)
    return dump_mask_audit(tokenizer, response_ids, response_mask)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-path",
        type=Path,
        default=REPO / "outputs/sft_qwen25_3b_coldstart_v1_merged",
    )
    ap.add_argument("--trajectory-json", type=Path, default=None, help="optional dump with response_ids/mask")
    ap.add_argument("--out", type=Path, default=REPO / "results/phase3b0_mask_audit/synthetic.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)

    if args.trajectory_json:
        data = json.loads(args.trajectory_json.read_text(encoding="utf-8"))
        report = audit_token_lists(tok, list(data["response_ids"]), list(data["response_mask"]))
    else:
        report = _synthetic(tok)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k not in ("full_text", "mask1_text", "mask0_text")}, indent=2))
    if not report.get("pass", False):
        raise SystemExit("FAIL: observation leaked into response_mask==1")
    print(f"[ok] mask audit PASS -> {args.out}")


if __name__ == "__main__":
    main()
