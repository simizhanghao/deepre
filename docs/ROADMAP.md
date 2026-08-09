# Roadmap — Evidence-Cost-Aware Deep Research Agent

> Frozen **2026-08-09**. Text ECA causal chain first; multimodal is a **later branch**.  
> Board: [RESULTS_BOARD.md](RESULTS_BOARD.md) · Next actions: [NEXT_STEPS.md](NEXT_STEPS.md)

## Principle

```text
Do NOT insert multimodal into Phase 3D.
Changing base model + modality + tools + reward + data + retrieval
at once destroys the 3B / 3C / 3D ablation.
```

Interview / report story stays:

```text
Sparse Answer Reward  →  no-search shortcut   (3B)
+ Evidence            →  always-search        (3C)
+ Cost                →  search when needed   (3D)
```

---

## A. Text ECA mainline (current — do in order)

```text
NOW
│
├── ✅ 3B CLOSED @100     Answer+Format → search=0
├── ✅ 3C CLOSED @400     +Evidence     → search≈1
│
├── ⬜ 3C-GEN             (gate — short, mandatory)
│     FSDP ckpt → HF / Agent eval loader
│     frozen HotpotQA val-200
│     same protocol: Candidate-BM25, max_search=2
│     compare: SFT-v1 | 3B@100 | 3C@400
│     metrics: Answer EM/F1, Evid F1, search_rate,
│              search_count (+ P0/P1/P2), finish, obs tokens
│     Purpose: mechanism vs smoke128 memorization
│     NOT: continue grinding 3C to 500/1000
│
├── ⬜ 3D0
│     audit search_count / duplicate on 3C rollouts
│     offline λ_search sweep (center ~0.1–0.3)
│     prefer penalizing wasteful search, not all search
│     Duplicate = secondary (late 3C search_count≈1, not 2)
│
├── ⬜ 3D1
│     fresh from SFT-v1 (NOT continue 3C@400)
│     R = Answer + 0.5 Evid + 0.1 Format − λ_s C_search
│         (− λ_d C_dup optional / later)
│     success = quality–cost tradeoff, not “search↓ alone”
│
├── ⬜ 3D2
│     cost–quality Pareto vs 3B/3C (train windows + GEN)
│
├── ⬜ 3E Full-Corpus     (after Candidate 3D mechanism OK)
│     Wikipedia BM25 + HTTP retriever
│     Candidate Tool → Full-Corpus Tool
│
└── ⬜ Phase 4
      larger train pool (if 3C-GEN weak / for claims)
      matched-step 3B/3C/3D
      frozen eval, ablations, Pareto, README/interview pack
```

### Ablation tree (locked)

```text
SFT-v1
 ├── 3B Answer+Format              CLOSED
 ├── 3C Answer+Evidence+Format     CLOSED
 └── 3D Answer+Evidence+Format−Cost(+Dup)   NEXT (fresh)
```

Optional later (still text, not multimodal):

```text
ECA-v3 — Knowledge-boundary / process-aware cost
         (penalize unnecessary search; do not punish necessary search)
```

### Retrieval maturity (text)

| Level | Scope | When |
|-------|--------|------|
| **L1** | Candidate-BM25 + `sample_id` | **now** (3B–3D) |
| **L2** | Full Wikipedia BM25 | **3E** (parallel prep OK) |
| **L3** | Open web (Serper/Bing/…) | **after** text ECA + preferably after M1; not in 3D critical path |

---

## B. Multimodal branch (deferred — Phase 5M)

Start **only after** text mainline through 3D (and preferably 3E/Phase4 claims) is solid.

```text
Phase 5M — Multimodal ECA Extension
│
├── M1  Tool-augmented (keep Qwen2.5-3B text actor)
│     + image_search / image_inspect(frozen VLM) / OCR / crop
│     discovery ≠ load (metadata first, inspect on demand)
│     schema: VisualDocument / VisualObservation / VisualEvidence
│     50–200 multimodal bench → inference first → decide RL
│
└── M2  End-to-end VLM actor (Qwen2.5-VL + GRPO)
      text/image/crop/OCR in the loop
      toward MMSearch-R1 / OpenSearch-VL class systems
```

Do **not** swap to Qwen-VL for the current 3D run.

Suggested framing later:

> Not “add an image tool”, but **modality-aware information acquisition**  
> under Answer / Evidence / Cost (text search vs image search vs vision ops).

Reading list (branch prep, not blocking 3D): MMSearch-R1, WebWatcher, Vision-DeepResearch, OpenSearch-VL, ProMMSearchAgent, DeepVoyager-VL, MTA-Agent / MM-DeepResearch.

---

## C. Explicit non-goals (now)

- Grind 3C to 500/1000  
- Continue GRPO from 3C@400 for the **ablation** 3D line  
- Claim smoke128 `answer≈0.61` as HotpotQA val EM  
- Open-web or multimodal as 3D blockers  
- Rebuild SFT/RL stack on VL before text Cost story is done
