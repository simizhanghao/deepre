# Run Log

One line per smoke/full run. Format:

```
YYYY-MM-DD | stage | command | output_dir | pass/fail | notes
```

---

2026-08-07 | phase2d3-b | `run_baseline.py` Direct/Oracle/Candidate on merged coldstart_v0 | `results/baseline_*_phase2d3_sft_n200/` | pass | EM 0.170 / 0.650 / 0.470
2026-08-07 | phase2d3-d | `audit_base_vs_sft.py` taxonomy + paired | `results/audit_base_vs_sft_n200_20260807_181820_phase2d3d/` | pass | C 38%→33%; Oracle W→R/R→W 24/13
2026-08-07 | phase2d3-c | `run_protocol_eval.py` evidence_oracle/candidate + routing | `results/protocol_*_20260807_18242*_phase2d3c/` | pass | Evid F1 0.818/0.665; routing internal 29%/search 71%; protocol valid 1.0
2026-08-07 | phase2d3-docs | freeze `docs/PHASE2D3_DIAGNOSIS.md` | — | pass | v1 priorities: routing cal > BM25 hard-neg evid > subset teacher rationale

