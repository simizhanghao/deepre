# Phase 2A — Cold-start SFT Data Contract

> Status: **design only**. No training, no formal dataset generation, no GPU.  
> Upstream freeze: `docs/PHASE1_BASELINE_RESULTS.md` (n=200).  
> Interfaces: `docs/hotpotqa_data_contract.md`, `src/eval/trace_schema.py`, `docs/searchR1_to_eca_mapping.md`.

---

## 0. Why Phase 2 now (from Phase 1)

Frozen n=200 (Qwen2.5-3B-Instruct):

| Signal | Value | Implication |
|--------|------:|-------------|
| Direct EM | 0.180 | parametric knowledge insufficient |
| Candidate-BM25 EM | 0.435 | search helps |
| Oracle EM | 0.595 | even gold docs leave a large gap |
| C (Direct❌ Oracle❌) | **38%** | **top bottleneck = evidence use / multi-hop** |
| B | 15.5% | retrieval miss/noise still real |
| D+E | 17.5% | always-search is not optimal |

**Phase 2 goal is not “search more.”**  
It is to turn the 3B model into a cold-start policy that:

1. emits a stable agent protocol  
2. selects sentence-level evidence from documents  
3. does short, evidence-grounded multi-hop inference  
4. has a basic internal vs external routing prior  
5. can emit `<search>` queries in the right format  

Phase 3 (Search-R1 GRPO/RLOO) then optimizes *how often / what / when to stop*.  
Entering RL with a base model that cannot emit format, evidence, or routing yields near-all-zero rewards → `zero_std_group_rate` collapse (see `docs/searchR1_read.md`).

---

## 1. Phase 2 objective (what SFT teaches)

| Teach | Do **not** teach (this phase) |
|-------|-------------------------------|
| Output protocol (`internal` / `search` / `evidence` / `reasoning` / `answer`) | Memorizing HotpotQA answers as closed-book facts |
| Evidence selection with provenance | Open-domain full-corpus retrieval mastery |
| Short multi-hop: evidence → inference → answer | Long free-form CoT / “talkative” reasoning |
| Basic routing prior (when *not* to search) | Optimal search budget / stop policy (→ Phase 3) |
| Legal `<search>query</search>` action shape | Reward weighting / GRPO advantages |

Cold-start SFT = **behavior prior**, not final policy.

---

## 2. Unified SFT trajectory protocol

### 2.1 Surface tags (model-visible)

| Tag | Meaning |
|-----|---------|
| `<internal>...</internal>` | Explicit choice to rely on parametric knowledge (routing act, not hidden CoT) |
| `<search>...</search>` | Tool call: search query string |
| `<observation>...</observation>` | Environment-injected retrieval text (**not** model-generated; see §10) |
| `<evidence>...</evidence>` | Model-selected supporting snippets with provenance |
| `<reasoning>...</reasoning>` | Short, evidence-grounded multi-hop rationale |
| `<answer>...</answer>` | Final answer (exactly one; last) |

### 2.2 Mapping to TraceRecord (`src/eval/trace_schema.py`)

| Surface tag | `step_type` | `loss_mask` |
|-------------|-------------|-------------|
| `<internal>` | `internal` | True |
| `<search>` | `search` | True |
| `<observation>` | `observation` | **False** |
| `<evidence>` | `evidence` | True |
| `<reasoning>` | `think` | True |
| `<answer>` | `answer` | True |

**Why `<reasoning>` → `think`:** Phase 0 schema already has `think`, not a separate `reasoning` type. Phase 2 uses the surface name `<reasoning>` to stress *short, evidence-grounded* hops; converters must set `step_type="think"` and may set `metadata["think_kind"]="evidence_grounded"`.

Do **not** extend `STEP_TYPES` in Phase 2A.

### 2.3 Legal trajectory templates (Cold-start v1)

Exactly one `<answer>` at the end. `internal` and `search` are **mutually exclusive** on a single short trajectory (multi-turn search deferred to Phase 3 rollouts).

| Type | Legal step order (surface) |
|------|----------------------------|
| **I Internal** | `internal` → `answer` |
| **II Evidence** | *(docs in prompt / observation)* → `evidence` → `answer` |
| **III Evidence+Reasoning** | *(docs in prompt / observation)* → `evidence` → `reasoning` → `answer` |
| **IV Search-format** | `search` → `observation` → `evidence` → [`reasoning`] → `answer` |

Illegal / rejected at build time:

- `internal` + `search` in the same sample  
- `evidence` / `reasoning` without any document context in the prompt (for Types II–IV)  
- missing `<answer>` / answer not last  
- answer text ≠ gold (after build-time check; see §8)  
- evidence spans inventing facts not in provided documents  
- Base-model wrong outputs used as targets  

Optional: Type II may omit `<reasoning>`; Type III must include it.

---

## 3. Type I — Internal routing samples

### 3.1 Source priority (Phase 1 taxonomy)

| Priority | Source | Rationale |
|----------|--------|-----------|
| 1 | **E** Direct✅ BM25❌ | Search can hurt; strongest “don’t always search” signal |
| 2 | **D** Direct✅ Oracle✅ BM25✅ | Search unnecessary given cost |
| 3 | Other Direct✅ (if needed for volume) | Weaker; use sparingly |

### 3.2 Target shape

```text
<internal>
Use internal knowledge.
</internal>
<answer>
{gold_answer}
</answer>
```

### 3.3 Semantics

- `<internal>` = **routing decision**: this turn uses parametric knowledge.  
- It is **not** a hidden scratchpad and must not dump long chain-of-thought.  
- Body may be a short fixed phrase (above) or a one-line reason; keep it short and templated in v1 for format stability.  
- Does **not** mean “always trust yourself”; mixture with Types II–IV prevents collapse to all-internal.

### 3.4 Prompt

Question only (same spirit as Direct baseline). No retrieved documents in the user/context for Type I.

---

## 4. Type II — Evidence extraction samples

### 4.1 Source

Prefer taxonomy **A**, **B**, and a slice of **C** where gold supporting facts resolve cleanly.  
Input context = **Oracle supporting documents** (unique titles from `supporting_facts`; full docs per `docs/hotpotqa_data_contract.md` §7).

### 4.2 Gold evidence construction (no Teacher guessing)

HotpotQA gold identity = `(title, sentence_id)`.

For each supporting fact:

1. Resolve `title` → `contexts[].document_id`  
2. Take `contexts[].sentences[sentence_id]` as evidence text  
3. Emit with provenance:

```text
<evidence>
[document_id={document_id} | title={title} | sentence_id={sentence_id}]
{sentence_text}
...
</evidence>
<answer>
{gold_answer}
</answer>
```

### 4.3 Hard rules

| Rule | Requirement |
|------|-------------|
| Provenance | Every evidence line must carry resolvable `document_id` + `title` + `sentence_id` |
| No invention | Teacher / builder must not invent evidence text |
| Sentence-level | Identity remains `(title, sentence_id)`; text is a copy from `sentences` |
| Order | Prefer order of first appearance in `supporting_facts` |
| Oracle docs | Prompt includes full supporting documents (not gold sentences alone as the only context) so the model must **select** |

### 4.4 Why this hits Phase 1 pain

Candidate Top-K dumps many tokens; models fail to pick the few supporting sentences. Type II teaches selection under oracle documents first (clean supervision).

---

## 5. Type III — Evidence + multi-hop reasoning (C-focused)

### 5.1 Source

Primary: taxonomy **C** (Direct❌ Oracle❌), where gold docs + gold answer still exist.  
Also: hard **A** cases that need bridging.

### 5.2 Target shape

```text
<evidence>
... gold supporting sentences with provenance ...
</evidence>
<reasoning>
{short bridge: only connect evidence sentences → answer}
</reasoning>
<answer>
{gold_answer}
</answer>
```

### 5.3 Reasoning constraints

| Constraint | Detail |
|------------|--------|
| Grounded | Every factual claim must be supported by an emitted evidence line |
| Short | Prefer 1–3 sentences; no essay CoT |
| No new facts | Forbidden: entities/dates/relations not in evidence |
| Answer-consistent | Must entail the gold answer; builder rejects mismatches |
| Style | Structured bridge (“Doc A states … Doc B states … Therefore …”), not free speculation |

### 5.4 Who writes reasoning

**Teacher** (or carefully constrained template) may draft `<reasoning>`;  
**Gold** always owns evidence identity + answer.  
Teacher must not rewrite answer or evidence (§8).

---

## 6. Type IV — Search-format samples

### 6.1 Purpose

Teach the **action shape** `<search>query</search>` and the post-observation habit of emitting `evidence` → (`reasoning`) → `answer`.

### 6.2 Scope honesty (critical)

| Scope | Allowed as Phase 2 supervision? |
|-------|----------------------------------|
| Candidate-BM25 (in-sample `contexts`) | **Yes** — prototype / format / noisy-context practice only |
| Claimed as full-corpus open-domain search | **No** |
| Full-corpus BM25 | Later parallel track; not a Phase 2A blocker |

Metadata must record `retriever.scope=candidate` when observations come from Candidate-BM25.

### 6.3 Target sketch

```text
<search>
{query}
</search>
<observation>
{serialized Top-K docs — environment; loss_mask=False}
</observation>
<evidence>
... preferably gold facts if present in observation; else skip sample or use oracle-observation variant ...
</evidence>
[<reasoning>...</reasoning>]
<answer>
{gold_answer}
</answer>
```

### 6.4 Query supervision

- Teacher or heuristic may propose `query` (often ≈ question, or entity-focused rewrite).  
- Query quality is secondary in Phase 2; **format + post-obs evidence discipline** is primary.  
- Do not use Base wrong answers as targets even if the search format looked fine.

### 6.5 Mixture role

Type IV is **minority** (~15–20% provisional). Dominating with Candidate search would teach the wrong skill relative to the C-class bottleneck.

---

## 7. Sample source strategy (taxonomy → types)

| Label | Meaning (Phase 1) | Primary SFT use |
|-------|-------------------|-----------------|
| **A** | Direct❌ Oracle✅ BM25✅ | Type II / IV positives (search+evidence helps) |
| **B** | Direct❌ Oracle✅ BM25❌ | Type II on **Oracle** docs; optional hard-neg Type IV (noisy Top-K) with gold evidence if still in bag — never train to copy Base wrong answer |
| **C** | Direct❌ Oracle❌ | Type III (evidence + short reasoning); Type II subset |
| **D** | Direct✅ all✅ | Type I internal |
| **E** | Direct✅ BM25❌ | Type I internal (+ optional “retrieval harmful” note in metadata only) |

### Hard prohibition

**Never** use Base / Direct / BM25 **incorrect** model outputs as SFT `target`.  
Wrong generations may appear only in offline analysis metadata, never as supervision.

---

## 8. Teacher vs Gold

### 8.1 Gold provides (deterministic)

- `question`  
- `gold_answers` / canonical `gold_answer` for target  
- `supporting_facts` identity `(title, sentence_id)`  
- Evidence **text** via `contexts[].sentences[sentence_id]`  
- Oracle document pack  

### 8.2 Teacher may provide

- Concise `<reasoning>` text (Type III)  
- `<search>` query string (Type IV)  
- Optional short `<internal>` body paraphrase (Type I; default template preferred)

### 8.3 Teacher must **not**

- Override gold answer  
- Invent or alter evidence spans / ids  
- Introduce unsupported facts in reasoning  

### 8.4 Deterministic validation (build gate)

Before a sample is written to JSONL:

1. Tags parse; legal template for `category`  
2. Exactly one answer; equals gold under the same normalization policy as eval (`normalize_answer` for check only; stored answer stays raw gold string)  
3. Every evidence ref resolves to eval `contexts` / provided docs  
4. Evidence text equals the referenced sentence (exact or whitespace-normalized equality — pick one in 2B and freeze)  
5. If reasoning present: reject if it contains obvious unsupported named entities not in evidence∪question (heuristic OK in 2B; LLM-as-judge optional later)  
6. `loss_mask` plan matches §10  

**Teacher model identity is not chosen in 2A** — only this interface + quality bar.

---

## 9. SFT sample JSONL schema

One JSON object per line. Fields:

| Field | Type | Required | Semantics |
|-------|------|----------|-----------|
| `sample_id` | str | ✅ | Same as HotpotQA eval `sample_id` (+ optional suffix for multi-views, see below) |
| `sft_id` | str | ✅ | Unique training row id: `{sample_id}__{category}__{view}` |
| `source_dataset` | str | ✅ | e.g. `hotpotqa` |
| `category` | str | ✅ | `internal` \| `evidence` \| `evidence_reasoning` \| `search_format` |
| `taxonomy_label` | str | ✅ | `A`–`E` / `O` / `unknown` from Phase 1 compare (or recomputed) |
| `messages` | list | ✅ | Chat messages for training (see below) |
| `target` | str | ✅ | Full model target string (assistant content) with tags |
| `gold_answer` | str | ✅ | Canonical gold string used in target |
| `gold_answers` | list[str] | ✅ | Full list from eval |
| `evidence_refs` | list[obj] | ✅ | `[{document_id,title,sentence_id,text}]` (empty for internal) |
| `documents` | list[obj] | ✅ | Docs shown in context (empty for internal) |
| `provenance` | obj | ✅ | `{supporting_facts, builder, teacher_id, retriever}` |
| `metadata` | obj | ✅ | `{level,type,seed,phase,"mix_tag",...}` |

**Multi-view:** same HotpotQA question may yield multiple `sft_id`s (e.g. evidence view + search_format view). Never collide `sft_id`.

### `messages` convention (chat)

```json
[
  {"role": "system", "content": "<agent protocol instructions>"},
  {"role": "user", "content": "<question and optional documents / prior observation>"}
]
```

Assistant target is stored in `target` (and/or final assistant message — 2B must pick one serialization and stick to it). Recommendation: **`messages` = system+user only; `target` = assistant string** to keep loss masking simple.

### 9.1 Example — Internal

```json
{
  "sample_id": "hotpotqa_distractor_validation_EXAMPLE_INT",
  "sft_id": "hotpotqa_distractor_validation_EXAMPLE_INT__internal__v0",
  "source_dataset": "hotpotqa",
  "category": "internal",
  "taxonomy_label": "E",
  "messages": [
    {"role": "system", "content": "You are an evidence-cost-aware research agent. Use tags: internal, search, evidence, reasoning, answer."},
    {"role": "user", "content": "Question: Who wrote the novel Pride and Prejudice?"}
  ],
  "target": "<internal>\nUse internal knowledge.\n</internal>\n<answer>\nJane Austen\n</answer>",
  "gold_answer": "Jane Austen",
  "gold_answers": ["Jane Austen"],
  "evidence_refs": [],
  "documents": [],
  "provenance": {
    "supporting_facts": [],
    "builder": "phase2_sft_builder",
    "teacher_id": null,
    "retriever": null
  },
  "metadata": {"phase": "2B", "mix_tag": "internal_E"}
}
```

### 9.2 Example — Evidence

```json
{
  "sample_id": "hotpotqa_distractor_validation_EXAMPLE_EV",
  "sft_id": "hotpotqa_distractor_validation_EXAMPLE_EV__evidence__oracle_v0",
  "source_dataset": "hotpotqa",
  "category": "evidence",
  "taxonomy_label": "A",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Question: ...\n\nDocuments:\n[DOC] id=... title=Film X\n...\n[DOC] id=... title=Person Y\n..."}
  ],
  "target": "<evidence>\n[document_id=..._ctx_0 | title=Film X | sentence_id=2]\nFilm X was directed by Person Y.\n\n[document_id=..._ctx_1 | title=Person Y | sentence_id=5]\nPerson Y was born in Canada.\n</evidence>\n<answer>\nCanada\n</answer>",
  "gold_answer": "Canada",
  "gold_answers": ["Canada"],
  "evidence_refs": [
    {"document_id": "..._ctx_0", "title": "Film X", "sentence_id": 2, "text": "Film X was directed by Person Y."},
    {"document_id": "..._ctx_1", "title": "Person Y", "sentence_id": 5, "text": "Person Y was born in Canada."}
  ],
  "documents": [{"document_id": "..._ctx_0", "title": "Film X", "text": "..."}, {"document_id": "..._ctx_1", "title": "Person Y", "text": "..."}],
  "provenance": {
    "supporting_facts": [{"title": "Film X", "sentence_id": 2}, {"title": "Person Y", "sentence_id": 5}],
    "builder": "phase2_sft_builder",
    "teacher_id": null,
    "retriever": {"name": "oracle", "scope": "oracle_supporting_docs"}
  },
  "metadata": {"phase": "2B", "mix_tag": "evidence_oracle"}
}
```

### 9.3 Example — Evidence + Reasoning

```json
{
  "sample_id": "hotpotqa_distractor_validation_EXAMPLE_ER",
  "sft_id": "hotpotqa_distractor_validation_EXAMPLE_ER__evidence_reasoning__oracle_v0",
  "source_dataset": "hotpotqa",
  "category": "evidence_reasoning",
  "taxonomy_label": "C",
  "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "Question + Oracle documents..."}],
  "target": "<evidence>\n[document_id=... | title=Film X | sentence_id=2]\nFilm X was directed by Person Y.\n\n[document_id=... | title=Person Y | sentence_id=5]\nPerson Y was born in Canada.\n</evidence>\n<reasoning>\nThe first evidence identifies Person Y as the director of Film X. The second states Person Y was born in Canada. Therefore the director was born in Canada.\n</reasoning>\n<answer>\nCanada\n</answer>",
  "gold_answer": "Canada",
  "gold_answers": ["Canada"],
  "evidence_refs": [],
  "documents": [],
  "provenance": {"supporting_facts": [], "builder": "phase2_sft_builder", "teacher_id": "TBD", "retriever": {"name": "oracle", "scope": "oracle_supporting_docs"}},
  "metadata": {"phase": "2B", "mix_tag": "evidence_reasoning_C"}
}
```

*(Full `evidence_refs` / `documents` omitted in the sketch above for brevity; real rows must populate them like §9.2.)*

### 9.4 Example — Search-format

```json
{
  "sample_id": "hotpotqa_distractor_validation_EXAMPLE_SF",
  "sft_id": "hotpotqa_distractor_validation_EXAMPLE_SF__search_format__cand_bm25_v0",
  "source_dataset": "hotpotqa",
  "category": "search_format",
  "taxonomy_label": "A",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Question: ..."}
  ],
  "target": "<search>\nWho directed Film X; where was Person Y born\n</search>\n<observation>\n...\n</observation>\n<evidence>\n...\n</evidence>\n<answer>\nCanada\n</answer>",
  "gold_answer": "Canada",
  "gold_answers": ["Canada"],
  "evidence_refs": [],
  "documents": [],
  "provenance": {
    "supporting_facts": [],
    "builder": "phase2_sft_builder",
    "teacher_id": "TBD",
    "retriever": {"name": "bm25s", "scope": "candidate", "top_k": 5}
  },
  "metadata": {"phase": "2B", "mix_tag": "search_format_candidate", "note": "candidate scope ≠ full corpus"}
}
```

In the serialized `target`, `<observation>...</observation>` is present for trajectory fidelity, but **must be excluded from token loss** (§10).

---

## 10. Loss mask contract

Aligned with `EXPECTED_LOSS_MASK` in `src/eval/trace_schema.py`:

| Content | In target string? | Enter SFT loss? |
|---------|-------------------|-----------------|
| `<internal>`, `<search>`, `<evidence>`, `<reasoning>`, `<answer>` (tags + bodies) | Yes | **Yes** |
| `<observation>` body (retrieved docs) | May appear in trajectory string | **No** |
| User-provided documents in `messages` | Prompt only | **No** (prompt tokens) |
| System protocol text | Prompt only | **No** |

Implementation note (Phase 2D collator, not 2A):

- Prefer packing: assistant target = only loss-bearing spans; inject observation as a non-label segment, **or**  
- Build full string then apply token spans with `loss_mask=False` on observation ranges (Search-R1 `info_mask` analogue).

**Evidence is never masked** even though its text was copied from documents — selection is a model action.

---

## 11. Prototype scale

| Stage | Scale | Purpose |
|-------|------:|---------|
| **Phase 2B** | **200–500** | Builder + automatic gates + light human audit |
| **Phase 2C** | **2k–5k** | Cold-start corpus after 2B passes |
| Forbidden now | tens of thousands | Premature scale hides schema bugs |

Prototype should already mix all four categories (skewed toward evidence/reasoning), not a single type.

---

## 12. Data mixture (provisional hyperparameter)

First-cut target for 2B/2C (subject to change after audit):

| Category | Share | Why |
|----------|------:|-----|
| `evidence` + `evidence_reasoning` | **~60%** | C is the largest failure class |
| `internal` | **~15–20%** | D+E routing prior |
| `search_format` | **~15–20%** | Format for Phase 3; Candidate only |
| hard negatives / format stress | **~0–5%** | Optional; keep tiny |

Marked **provisional** — treat as experiment knobs, not frozen science.  
Invariant: evidence/reasoning remains the plurality.

Within evidence/reasoning, prefer oversampling **C** relative to its raw HotpotQA frequency when building from the taxonomy join.

---

## 13. Validation & evaluation (beyond EM/F1)

### 13.1 Offline data validation (builder)

- Schema + tag parse  
- Gold answer consistency  
- Evidence provenance resolve + text match  
- Reasoning groundedness heuristics  
- Category template legality  

### 13.2 Post-SFT model evaluation (Phase 2E)

| Family | Metrics |
|--------|---------|
| Answer | EM, token F1 (existing `src/eval/metrics.py`) |
| Format | `format_valid_rate` (TraceRecord validator); tag coverage rates |
| Evidence | Precision / Recall / F1 vs gold `(title, sentence_id)` (or document_id+sentence_id) |
| Reasoning | Defined now, light impl later: gold-answer consistency; evidence-groundedness (heuristic or optional LLM judge — **not required for 2A/2B gate**) |
| Routing | `internal_rate`, `external_rate` (`search` rate); confusion vs taxonomy oracle proxy (e.g. E/D prefer internal; A/B prefer external) |

Evidence metrics are mandatory for claiming “we taught evidence use.” Answer-only gains are insufficient.

---

## 14. SFT acceptance gate (enter Phase 3 only if)

Qualitative hard gate (numeric thresholds **deferred** until 2B/2E measurements — do not invent tight cutoffs here):

1. **Format stable** — high `format_valid_rate`; tags parse reliably  
2. **Evidence usable** — evidence P/R/F1 clearly above Base (Base often ~0 structured evidence)  
3. **Oracle answer improves** vs Base on the same Oracle eval  
4. **C-class improves** — error rate or EM on C-tagged slice moves in the right direction  
5. **No routing collapse** — not ~100% search and not ~100% internal on a mixed eval  

Fail any → fix data/mixture/SFT, **do not** start GRPO.

---

## 15. Non-goals (Phase 2A)

This phase does **not**:

- Generate formal SFT JSONL  
- Choose a concrete teacher model  
- Run training / LoRA / full finetune  
- Implement GRPO / RLOO  
- Implement reward functions  
- Build full-corpus Wikipedia index  
- Modify Phase 1 frozen numbers  
- Expand HotpotQA eval to 500/1000  

Full-Corpus BM25 may proceed **in parallel** but must not block the SFT mainline.

---

## 16. Next phase

**Phase 2B — Prototype SFT data builder**

1. Implement builder + validator (CPU)  
2. Join Phase 1 taxonomy (`compare_baselines` per-sample labels) with `data/eval/hotpotqa_*.jsonl`  
3. Emit **200–500** rows under e.g. `data/sft/prototype_v0.jsonl` (path chosen in 2B)  
4. Automatic audit report + small human spot-check checklist  
5. Stop for review before 2C scale-up  

---

## Decision summary

```text
Goal:     cold-start agent protocol + evidence + short hops + routing prior
Tags:     internal | search | observation | evidence | reasoning | answer
Map:      reasoning → TraceRecord.think; observation loss_mask=False
Gold:     answer + supporting_facts → evidence text
Teacher:  reasoning + search query only; never overrides gold
Types:    I internal | II evidence | III evidence+reasoning | IV search-format
Mix:      ~60% evidence/reasoning | ~15–20% internal | ~15–20% search-format
Scale:    2B: 200–500 → gate → 2C: 2k–5k
Eval+:    evidence P/R/F1 + format + routing (not EM alone)
Gate:     format + evidence + Oracle↑ + C↑ + no collapse → then Phase 3
Stop:     after this contract; implement builder only in Phase 2B
```

*Document version: Phase 2A / 2026-08-07*
