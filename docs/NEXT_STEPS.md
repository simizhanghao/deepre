# Next Steps — Text ECA (v2)

> [ROADMAP.md](ROADMAP.md) · [RESULTS_BOARD.md](RESULTS_BOARD.md)

## Done

- [x] 3C CLOSED @400 · **3C-GEN PASS** (dev-200)
- [x] **3D0** → λ_s=0.40 offline ([PHASE3D0.md](PHASE3D0.md))
- [x] **3D1 λ=0.40 FAIL** — search→0 after step5, KL→0.58; stopped @250 ([PHASE3D1.md](PHASE3D1.md))

## NOW — recover Cost (pick one) ⬜

Uniform Cost @0.40 **triggered** the “global bias only” gate.

1. **3D1b (recommended first):** short GRPO with `λ_s∈{0.10,0.15,0.20}`, stop ≤150 if search∈[0.4,0.9]  
2. **3D2:** Capability-aware cost if lower λ still has no stratified routing  

Do **not** continue λ=0.40 to 400. Do **not** GEN-eval the collapsed 3D1 ckpt as a success.

## Later

3E Full-Corpus · Phase4 larger train + final held-out · 5M multimodal

## Naming

`hotpotqa_200` = **dev-200** (selection). Phase4 needs a disjoint final test.
