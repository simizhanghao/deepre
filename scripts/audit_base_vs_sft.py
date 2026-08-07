"""Phase 2D3-D: Base vs SFT taxonomy + paired EM transitions (CPU only).

Usage (repo root):
  python scripts/audit_base_vs_sft.py \
    --base-direct results/baseline_direct_n200_*_phase1_final_n200/metrics.json \
    --base-oracle results/baseline_oracle_n200_*_phase1_final_n200/metrics.json \
    --base-bm25 results/baseline_candidate_bm25_n200_*_phase1_final_n200/metrics.json \
    --sft-direct results/baseline_direct_n200_*_phase2d3_sft_n200/metrics.json \
    --sft-oracle results/baseline_oracle_n200_*_phase2d3_sft_n200/metrics.json \
    --sft-bm25 results/baseline_candidate_bm25_n200_*_phase2d3_sft_n200/metrics.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

LABEL_MEANING = {
    "A": "Direct❌ Oracle✅ BM25✅ — retrieval helps",
    "B": "Direct❌ Oracle✅ BM25❌ — retrieval miss / noise",
    "C": "Direct❌ Oracle❌ — evidence/reasoning gap",
    "D": "Direct✅ Oracle✅ BM25✅ — search likely unnecessary",
    "E": "Direct✅ BM25❌ — retrieval hurts",
    "O": "other pattern",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Base vs SFT paired + taxonomy audit.")
    p.add_argument("--base-direct", type=str, required=True)
    p.add_argument("--base-oracle", type=str, required=True)
    p.add_argument("--base-bm25", type=str, required=True)
    p.add_argument("--sft-direct", type=str, required=True)
    p.add_argument("--sft-oracle", type=str, required=True)
    p.add_argument("--sft-bm25", type=str, required=True)
    p.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "results"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--run-tag", type=str, default="phase2d3d")
    return p.parse_args()


def resolve(path: str) -> Path:
    """Resolve a concrete path or a glob (newest mtime if multiple)."""
    raw = Path(path)
    candidates: List[Path] = []
    if raw.is_absolute():
        if raw.exists():
            return raw
        candidates = sorted(Path("/").glob(str(raw).lstrip("/")))
    else:
        direct = REPO_ROOT / raw
        if direct.exists():
            return direct
        candidates = sorted(REPO_ROOT.glob(path))
    if not candidates:
        raise SystemExit(f"path not found: {path}")
    if len(candidates) == 1:
        return candidates[0]
    return sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True)[0]


def load_metrics(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {r["sample_id"]: r for r in rows}


def is_correct(row: Dict[str, Any]) -> bool:
    return float(row.get("metrics", {}).get("exact_match", 0.0)) == 1.0


def classify(d_ok: bool, o_ok: bool, b_ok: bool) -> str:
    if (not d_ok) and o_ok and b_ok:
        return "A"
    if (not d_ok) and o_ok and (not b_ok):
        return "B"
    if (not d_ok) and (not o_ok):
        return "C"
    if d_ok and o_ok and b_ok:
        return "D"
    if d_ok and (not b_ok):
        return "E"
    return "O"


def taxonomy_table(
    direct: Dict[str, Dict[str, Any]],
    oracle: Dict[str, Dict[str, Any]],
    bm25: Dict[str, Dict[str, Any]],
    ids: Sequence[str],
) -> Dict[str, Any]:
    labels: List[str] = []
    for sid in ids:
        labels.append(
            classify(
                is_correct(direct[sid]),
                is_correct(oracle[sid]),
                is_correct(bm25[sid]),
            )
        )
    counts = Counter(labels)
    n = len(ids)
    rates = {k: round(counts.get(k, 0) / n, 4) for k in "ABCDEO"}
    return {
        "num_samples": n,
        "taxonomy_counts": {k: counts.get(k, 0) for k in "ABCDEO"},
        "taxonomy_rates": rates,
        "mean_em": {
            "direct": round(sum(1 for s in ids if is_correct(direct[s])) / n, 4),
            "oracle": round(sum(1 for s in ids if is_correct(oracle[s])) / n, 4),
            "candidate_bm25": round(sum(1 for s in ids if is_correct(bm25[s])) / n, 4),
        },
        "per_sample_labels": dict(zip(ids, labels)),
    }


def paired_for_method(
    base: Dict[str, Dict[str, Any]],
    sft: Dict[str, Dict[str, Any]],
    ids: Sequence[str],
) -> Dict[str, Any]:
    both_right = both_wrong = w2r = r2w = 0
    for sid in ids:
        b_ok = is_correct(base[sid])
        s_ok = is_correct(sft[sid])
        if b_ok and s_ok:
            both_right += 1
        elif (not b_ok) and (not s_ok):
            both_wrong += 1
        elif (not b_ok) and s_ok:
            w2r += 1
        else:
            r2w += 1
    n = len(ids)
    base_em = (both_right + r2w) / n
    sft_em = (both_right + w2r) / n
    return {
        "n": n,
        "both_right": both_right,
        "both_wrong": both_wrong,
        "wrong_to_right": w2r,
        "right_to_wrong": r2w,
        "net_em_gain_count": w2r - r2w,
        "base_em": round(base_em, 4),
        "sft_em": round(sft_em, 4),
        "delta_em": round(sft_em - base_em, 4),
    }


def mcnemar_exact_p(w2r: int, r2w: int) -> Optional[float]:
    """Two-sided exact McNemar via binomial on discordant pairs."""
    n = w2r + r2w
    if n == 0:
        return None
    # P(X<=k) + P(X>=n-k) under Bin(n, 0.5); use smaller tail * 2
    k = min(w2r, r2w)
    # sum_{i=0..k} C(n,i) / 2^n
    # careful with large n: use recursive probabilities
    # cdf
    from math import comb

    left = sum(comb(n, i) for i in range(0, k + 1)) / (2**n)
    p = min(1.0, 2 * left)
    return round(p, 6)


def bootstrap_delta_ci(
    base_ok: List[bool],
    sft_ok: List[bool],
    n_boot: int,
    seed: int,
) -> Dict[str, float]:
    rng = random.Random(seed)
    n = len(base_ok)
    deltas = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        b = sum(base_ok[i] for i in idx) / n
        s = sum(sft_ok[i] for i in idx) / n
        deltas.append(s - b)
    deltas.sort()
    lo = deltas[int(0.025 * n_boot)]
    hi = deltas[int(0.975 * n_boot)]
    return {
        "delta_mean": round(sum(deltas) / len(deltas), 4),
        "ci95_low": round(lo, 4),
        "ci95_high": round(hi, 4),
    }


def main() -> None:
    args = parse_args()
    paths = {
        "base_direct": resolve(args.base_direct),
        "base_oracle": resolve(args.base_oracle),
        "base_bm25": resolve(args.base_bm25),
        "sft_direct": resolve(args.sft_direct),
        "sft_oracle": resolve(args.sft_oracle),
        "sft_bm25": resolve(args.sft_bm25),
    }
    loaded = {k: load_metrics(v) for k, v in paths.items()}
    ids = [
        r["sample_id"]
        for r in json.loads(paths["base_direct"].read_text(encoding="utf-8"))
    ]
    for k, m in loaded.items():
        missing = [i for i in ids if i not in m]
        if missing:
            raise SystemExit(f"{k} missing {len(missing)} ids e.g. {missing[0]}")

    tax_base = taxonomy_table(
        loaded["base_direct"], loaded["base_oracle"], loaded["base_bm25"], ids
    )
    tax_sft = taxonomy_table(
        loaded["sft_direct"], loaded["sft_oracle"], loaded["sft_bm25"], ids
    )

    # C-set transitions
    base_labels = tax_base["per_sample_labels"]
    sft_labels = tax_sft["per_sample_labels"]
    c_ids = [i for i in ids if base_labels[i] == "C"]
    c_dest = Counter(sft_labels[i] for i in c_ids)
    label_transitions = Counter(
        f"{base_labels[i]}->{sft_labels[i]}" for i in ids
    )

    paired: Dict[str, Any] = {}
    for method, bk, sk in [
        ("direct", "base_direct", "sft_direct"),
        ("oracle", "base_oracle", "sft_oracle"),
        ("candidate_bm25", "base_bm25", "sft_bm25"),
    ]:
        stats = paired_for_method(loaded[bk], loaded[sk], ids)
        base_ok = [is_correct(loaded[bk][i]) for i in ids]
        sft_ok = [is_correct(loaded[sk][i]) for i in ids]
        stats["mcnemar_exact_p"] = mcnemar_exact_p(
            stats["wrong_to_right"], stats["right_to_wrong"]
        )
        stats["bootstrap_delta_em"] = bootstrap_delta_ci(
            base_ok, sft_ok, args.bootstrap, args.seed
        )
        paired[method] = stats

    # Drop bulky per-sample from summary dump of tax_*
    tax_base_pub = {k: v for k, v in tax_base.items() if k != "per_sample_labels"}
    tax_sft_pub = {k: v for k, v in tax_sft.items() if k != "per_sample_labels"}

    summary = {
        "num_samples": len(ids),
        "inputs": {k: str(v) for k, v in paths.items()},
        "taxonomy_base": tax_base_pub,
        "taxonomy_sft": tax_sft_pub,
        "taxonomy_delta_rates": {
            k: round(
                tax_sft_pub["taxonomy_rates"][k] - tax_base_pub["taxonomy_rates"][k], 4
            )
            for k in "ABCDEO"
        },
        "c_base_count": len(c_ids),
        "c_base_rate": tax_base_pub["taxonomy_rates"]["C"],
        "c_sft_rate": tax_sft_pub["taxonomy_rates"]["C"],
        "c_destinations_from_base_C": dict(c_dest),
        "label_transitions": dict(sorted(label_transitions.items())),
        "paired": paired,
    }

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"audit_base_vs_sft_n{len(ids)}_{stamp}_{args.run_tag}"
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "diagnosis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with (run_dir / "per_sample_taxonomy.jsonl").open("w", encoding="utf-8") as f:
        for sid in ids:
            row = {
                "sample_id": sid,
                "base_label": base_labels[sid],
                "sft_label": sft_labels[sid],
                "base_direct_em": 1.0 if is_correct(loaded["base_direct"][sid]) else 0.0,
                "sft_direct_em": 1.0 if is_correct(loaded["sft_direct"][sid]) else 0.0,
                "base_oracle_em": 1.0 if is_correct(loaded["base_oracle"][sid]) else 0.0,
                "sft_oracle_em": 1.0 if is_correct(loaded["sft_oracle"][sid]) else 0.0,
                "base_bm25_em": 1.0 if is_correct(loaded["base_bm25"][sid]) else 0.0,
                "sft_bm25_em": 1.0 if is_correct(loaded["sft_bm25"][sid]) else 0.0,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Markdown report
    md = [
        "# Phase 2D3-D — Base vs SFT Taxonomy & Paired Audit",
        "",
        f"n = **{len(ids)}**",
        "",
        "## Taxonomy rates",
        "",
        "| Label | Base | SFT | Δ | Meaning |",
        "|------:|-----:|----:|--:|---------|",
    ]
    for k in "ABCDEO":
        md.append(
            f"| {k} | {tax_base_pub['taxonomy_rates'][k]:.1%} | "
            f"{tax_sft_pub['taxonomy_rates'][k]:.1%} | "
            f"{summary['taxonomy_delta_rates'][k]:+.1%} | {LABEL_MEANING[k]} |"
        )
    md += [
        "",
        f"**C: {summary['c_base_rate']:.1%} → {summary['c_sft_rate']:.1%}** "
        f"(base-C n={summary['c_base_count']})",
        "",
        "Base-C destinations under SFT labels:",
        "",
    ]
    for k, v in sorted(c_dest.items()):
        md.append(f"- {k}: {v}")
    md += ["", "## Paired EM transitions", ""]
    for method, st in paired.items():
        md += [
            f"### {method}",
            "",
            f"- Base EM {st['base_em']} → SFT EM {st['sft_em']} (Δ {st['delta_em']:+})",
            f"- Wrong→Right: **{st['wrong_to_right']}**",
            f"- Right→Wrong: **{st['right_to_wrong']}**",
            f"- Net EM count: **{st['net_em_gain_count']:+d}**",
            f"- Both right / both wrong: {st['both_right']} / {st['both_wrong']}",
            f"- McNemar exact p: `{st['mcnemar_exact_p']}`",
            f"- Bootstrap ΔEM 95% CI: "
            f"`{st['bootstrap_delta_em']['ci95_low']}` … "
            f"`{st['bootstrap_delta_em']['ci95_high']}`",
            "",
        ]
    md += [
        "## Note",
        "",
        "Protocol / Evidence F1 are **not** in this CPU audit — run "
        "`scripts/run_protocol_eval.py` (2D3-C) for those cells.",
        "",
    ]
    (run_dir / "report.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[audit_base_vs_sft] wrote {run_dir}")


if __name__ == "__main__":
    main()
