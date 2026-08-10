"""Phase 2E2: build coldstart_v1 from labeled pools + optional Kimi teacher cache.

Example:
  python scripts/build_sft_coldstart_v1.py \\
    --teacher-cache results/teacher_reasoning_n20_*/reasoning_cache.jsonl \\
    --n-teacher-reasoning 20
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.sft.coldstart_v1_builder import (  # noqa: E402
    DEFAULT_TARGETS_V1,
    assign_coldstart_v1,
    load_frozen_ids,
)
from src.sft.prototype_builder import index_by_sample_id, load_jsonl  # noqa: E402
from src.sft.teacher_reasoning import oracle_em_map_from_metrics  # noqa: E402

DEFAULT_FROZEN = REPO_ROOT / "data/eval/hotpotqa_200_ids.txt"
DEFAULT_TRAIN = REPO_ROOT / "data/sft/source/hotpotqa_distractor_train_pool_n8000.jsonl"
DEFAULT_DIRECT = (
    REPO_ROOT
    / "results/phase2e1_direct_label_n8000_20260807_202826_phase2e1/labels.jsonl"
)
DEFAULT_BASE_ORACLE = (
    REPO_ROOT
    / "results/phase2e1_base_oracle_n8000_20260807_205154/merged/metrics.json"
)
DEFAULT_RETRIEVAL = (
    REPO_ROOT
    / "results/retrieval_candidate_bm25_n8000_20260807_162150/retrieval_results.jsonl"
)
DEFAULT_MODEL = "/data1/hcc/.hf_home/Qwen2.5-3B-Instruct"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Phase 2E2 coldstart_v1 JSONL.")
    p.add_argument("--train-file", type=str, default=str(DEFAULT_TRAIN))
    p.add_argument("--direct-labels", type=str, default=str(DEFAULT_DIRECT))
    p.add_argument("--base-oracle-metrics", type=str, default=str(DEFAULT_BASE_ORACLE))
    p.add_argument("--retrieval-cache", type=str, default=str(DEFAULT_RETRIEVAL))
    p.add_argument("--teacher-cache", type=str, default=None)
    p.add_argument("--frozen-val-ids", type=str, default=str(DEFAULT_FROZEN))
    p.add_argument("--output-jsonl", type=str, default="data/sft/coldstart_v1.jsonl")
    p.add_argument("--audit-dir", type=str, default="results/01_build_sft_v1_mix")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-internal", type=int, default=DEFAULT_TARGETS_V1["internal"])
    p.add_argument(
        "--n-search-required", type=int, default=DEFAULT_TARGETS_V1["search_required"]
    )
    p.add_argument(
        "--n-evidence-bm25", type=int, default=DEFAULT_TARGETS_V1["evidence_bm25"]
    )
    p.add_argument(
        "--n-evidence-reasoning",
        type=int,
        default=DEFAULT_TARGETS_V1["evidence_reasoning"],
    )
    p.add_argument(
        "--n-search-format", type=int, default=DEFAULT_TARGETS_V1["search_format"]
    )
    p.add_argument("--n-teacher-reasoning", type=int, default=400)
    p.add_argument("--tokenizer-path", type=str, default=DEFAULT_MODEL)
    p.add_argument("--spot-check", type=int, default=12)
    p.add_argument(
        "--accepted-teacher-only",
        action="store_true",
        default=True,
        help="Only use teacher rows with validation.accepted=true",
    )
    return p.parse_args()


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else REPO_ROOT / p


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_teacher_accepted(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if not (r.get("teacher_validation") or {}).get("accepted"):
                continue
            if not r.get("think"):
                continue
            out[r["sample_id"]] = r
    return out


def main() -> None:
    args = parse_args()
    train = load_jsonl(str(resolve(args.train_file)))
    direct_rows = load_jsonl(str(resolve(args.direct_labels)))
    direct = index_by_sample_id(direct_rows)
    base_oracle = oracle_em_map_from_metrics(
        json.loads(resolve(args.base_oracle_metrics).read_text(encoding="utf-8"))
    )
    retrieval = index_by_sample_id(load_jsonl(str(resolve(args.retrieval_cache))))
    teacher: Dict[str, Dict[str, Any]] = {}
    if args.teacher_cache:
        teacher = load_teacher_accepted(resolve(args.teacher_cache))

    frozen = load_frozen_ids(str(resolve(args.frozen_val_ids)))
    targets = {
        "internal": args.n_internal,
        "search_required": args.n_search_required,
        "evidence_bm25": args.n_evidence_bm25,
        "evidence_reasoning": args.n_evidence_reasoning,
        "search_format": args.n_search_format,
    }
    accepted, rejected, stats = assign_coldstart_v1(
        train,
        frozen_ids=frozen,
        direct_labels=direct,
        base_oracle=base_oracle,
        retrieval=retrieval,
        teacher_accepted=teacher,
        seed=args.seed,
        targets=targets,
        n_teacher_reasoning=args.n_teacher_reasoning,
    )

    out_jsonl = resolve(args.output_jsonl)
    write_jsonl(out_jsonl, accepted)
    audit_dir = resolve(args.audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(audit_dir / "rejected.jsonl", rejected)
    (audit_dir / "build_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n"
    )

    # overlap check
    overlap = [r["sample_id"] for r in accepted if r["sample_id"] in frozen]
    report = {
        "total": len(accepted),
        "categories": dict(Counter(r["category"] for r in accepted)),
        "mix_tags": dict(Counter((r.get("metadata") or {}).get("mix_tag") for r in accepted)),
        "reasoning_sources": dict(
            Counter((r.get("provenance") or {}).get("reasoning_source") for r in accepted)
        ),
        "overlap_val200": len(overlap),
        "n_teacher_in_mix": stats.get("n_teacher_reasoning"),
        "n_teacher_cache_accepted": len(teacher),
        "output_jsonl": str(out_jsonl),
        "build_stats": stats,
        "gates": {
            "zero_val200_overlap": len(overlap) == 0,
            "has_rows": len(accepted) > 0,
        },
    }
    (audit_dir / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    md = [
        "# Phase 2E2 coldstart_v1 audit",
        "",
        f"- total: **{report['total']}**",
        f"- val200 overlap: **{report['overlap_val200']}**",
        f"- teacher rows in mix: **{report['n_teacher_in_mix']}** "
        f"(cache accepted={report['n_teacher_cache_accepted']})",
        "",
        "## Categories",
        "",
    ]
    for k, v in sorted(report["categories"].items()):
        md.append(f"- {k}: {v}")
    md.append("")
    md.append("## Mix tags")
    md.append("")
    for k, v in sorted((report["mix_tags"] or {}).items(), key=lambda x: (-x[1], str(x[0]))):
        md.append(f"- {k}: {v}")
    (audit_dir / "report.md").write_text("\n".join(md) + "\n")
    print(json.dumps(report, indent=2))
    print(f"[coldstart_v1] wrote {out_jsonl}")
    print(f"[coldstart_v1] audit -> {audit_dir}")


if __name__ == "__main__":
    main()
