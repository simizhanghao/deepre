# Phase 2C Cold-start v0 Audit

- total: **3000**
- source_split: `train`
- overlap with frozen validation-200: **0**
- deterministic validation pass: **100.00%**
- duplicate targets: **128**
- template_v0 reasoning marked: **1500** (100.0% of evidence_reasoning)

## Gates

- `train_only`: PASS
- `zero_val200_overlap`: PASS
- `validation_near_100`: PASS
- `approx_3k`: PASS
- `search_format_le_20pct`: PASS
- `internal_in_10_20`: PASS

## Category mixture

| Category | Count | Rate |
|---|---:|---:|
| evidence_reasoning | 1500 | 50.0% |
| evidence | 600 | 20.0% |
| internal | 450 | 15.0% |
| search_format | 450 | 15.0% |

## Build stats

- targets: `{'evidence_reasoning': 1500, 'evidence': 600, 'internal': 450, 'search_format': 450}`
- shortfall: `{'evidence_reasoning': 0, 'evidence': 0, 'internal': 0, 'search_format': 0}`
- direct_correct available: **747**
- search eligible: **5847**
- context views: `{'n/a': 900, 'noisy': 1082, 'clean': 1018}`

## Token stats (for SFT cutoff)

- **prompt_tokens**: p50=429.0 p95=1173.0 max=2395.0 mean=531.0
- **target_tokens**: p50=307.0 p95=1194.0 max=2224.0 mean=396.3
- **total_tokens**: p50=970.5 p95=1580.0 max=2877.0 mean=927.3

## Notes

- Train-only; never includes Phase-1 frozen validation-200.
- Reasoning uses `template_v0` (no Teacher LLM).
- Large regenerable train pool JSONL is gitignored; rebuild with `scripts/prepare_hotpotqa_train.py`.
- Direct labels / BM25 caches live under `results/` (gitignored).

Next: Phase 2D first Qwen2.5-3B Cold-start SFT.
