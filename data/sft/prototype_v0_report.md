# Phase 2B SFT Prototype Audit

- name: `prototype_v0`
- builder: `phase2_sft_builder_v0`
- seed: `42`
- eval: `/data1/hcc/deepresearch/data/eval/hotpotqa_200.jsonl`
- taxonomy: `/data1/hcc/deepresearch/results/compare_baselines_n200_20260807_155225/per_sample.jsonl`
- retrieval: `/data1/hcc/deepresearch/results/retrieval_candidate_bm25_n200_20260807_154802/retrieval_results.jsonl`
- evidence text equality: `whitespace_normalized`
- reasoning: `template_v0` (no LLM teacher)
- search query: `question_copy`

## Counts

- accepted: **299**
- rejected: **0**

### By category

| Category | Count | Rate |
|----------|------:|-----:|
| internal | 35 | 11.7% |
| evidence | 128 | 42.8% |
| evidence_reasoning | 76 | 25.4% |
| search_format | 60 | 20.1% |

### By taxonomy label (of accepted rows)

| Label | Count |
|------:|------:|
| A | 104 |
| B | 44 |
| C | 116 |
| D | 20 |
| E | 15 |

## Mixture check (provisional)

- evidence + evidence_reasoning: **68.2%** (target ~60%)
- internal: **11.7%** (target ~15–20%)
- search_format: **20.1%** (target ~15–20%)

## Validation

- All accepted rows passed `validate_sft_row` (tags, answer=gold, evidence provenance, template legality).
- Base-model wrong outputs were never used as targets.
- Candidate-BM25 search_format only when all gold titles ∈ Top-K.

## Spot checks

### `hotpotqa_distractor_validation_5a7309a15542992359bc320c__internal__v0` (internal / D)

User (truncated):

```text
Question: Which writer of the song "Money, Money, Money" by pop group ABBA was born on 16 December 1946 ?
```

Target:

```text
<internal>
Use internal knowledge.
</internal>
<answer>
Benny Andersson
</answer>
```

### `hotpotqa_distractor_validation_5a7129685542994082a3e5fa__evidence__dual_v0` (evidence / C)

User (truncated):

```text
Question: Which "Blackzilians" fighter is currently competing in the Middleweight division of Ultimate Fighting Championship?

Documents:
[DOC] document_id=hotpotqa_distractor_validation_5a7129685542994082a3e5fa_ctx_2 title=Blackzilians
The Blackzilians is a defunct professional team of fighters in mixed martial arts, boxing, kickboxing, amateur wrestling and catch wrestling located in South Florida.  They are most well known for being the team that includes former UFC light heavyweight champions 'Sugar' Rashad Evans and Vitor Belfort and former UFC lightweight champion Eddie Alvarez.

[DOC] document_id=hotpotqa_distractor_validation_5a7129685542994082a3e5fa_ctx_0 title=Vitor Belfort
Vitor Vieira Belfort (] ; born April 1, 1977) is a Brazilian mixed martial artist currently competing in th…
```

Target:

```text
<evidence>
[document_id=hotpotqa_distractor_validation_5a7129685542994082a3e5fa_ctx_2 | title=Blackzilians | sentence_id=1]
They are most well known for being the team that includes former UFC light heavyweight champions 'Sugar' Rashad Evans and Vitor Belfort and former UFC lightweight champion Eddie Alvarez.

[document_id=hotpotqa_distractor_validation_5a7129685542994082a3e5fa_ctx_0 | title=Vitor Belfort | sentence_id=0]
Vitor Vieira Belfort (] ; born April 1, 1977) is a Brazilian mixed martial artist currently competing in the Middleweight division of the Ultimate Fighting Championship.
</evidence>
<answer>
Vitor Belfort
</answer>
```

### `hotpotqa_distractor_validation_5a7129685542994082a3e5fa__evidence_reasoning__v0` (evidence_reasoning / C)

User (truncated):

```text
Question: Which "Blackzilians" fighter is currently competing in the Middleweight division of Ultimate Fighting Championship?

Documents:
[DOC] document_id=hotpotqa_distractor_validation_5a7129685542994082a3e5fa_ctx_2 title=Blackzilians
The Blackzilians is a defunct professional team of fighters in mixed martial arts, boxing, kickboxing, amateur wrestling and catch wrestling located in South Florida.  They are most well known for being the team that includes former UFC light heavyweight champions 'Sugar' Rashad Evans and Vitor Belfort and former UFC lightweight champion Eddie Alvarez.

[DOC] document_id=hotpotqa_distractor_validation_5a7129685542994082a3e5fa_ctx_0 title=Vitor Belfort
Vitor Vieira Belfort (] ; born April 1, 1977) is a Brazilian mixed martial artist currently competing in th…
```

Target:

```text
<evidence>
[document_id=hotpotqa_distractor_validation_5a7129685542994082a3e5fa_ctx_2 | title=Blackzilians | sentence_id=1]
They are most well known for being the team that includes former UFC light heavyweight champions 'Sugar' Rashad Evans and Vitor Belfort and former UFC lightweight champion Eddie Alvarez.

[document_id=hotpotqa_distractor_validation_5a7129685542994082a3e5fa_ctx_0 | title=Vitor Belfort | sentence_id=0]
Vitor Vieira Belfort (] ; born April 1, 1977) is a Brazilian mixed martial artist currently competing in the Middleweight division of the Ultimate Fighting Championship.
</evidence>
<reasoning>
Evidence 1 (Blackzilians): They are most well known for being the team that includes former UFC light heavyweight champions 'Sugar' Rashad Evans and Vitor Belfort and former UFC lightweight champion Eddie Alvarez.
Evidence 2 (Vitor Belfort): Vitor Vieira Belfort (] ; born April 1, 1977) is a Brazilian mixed martial artist currently competing in the Middleweight division of the Ultimate Fighting Championship.
Combining these supporting facts yields the answer: Vitor Belfort.
</reasoning>
<answer>
Vitor Belfort
</answer>
```

### `hotpotqa_distractor_validation_5a7349125542994cef4bc505__search_format__v0` (search_format / A)

User (truncated):

```text
Question: Baadshah is an Indian action comedy film that was inspired by what Hong Kong action movie starring Jackie Chan and Richard Norton?
```

Target:

```text
<search>
Baadshah is an Indian action comedy film that was inspired by what Hong Kong action movie starring Jackie Chan and Richard Norton?
</search>
<observation>
[hotpotqa_distractor_validation_5a7349125542994cef4bc505_ctx_7] Baadshah (1999 film): Baadshah (translation: "King") is a 1999 Indian action comedy film directed by Abbas-Mustan.  The film stars Shah Rukh Khan opposite Twinkle Khanna in lead roles.  It was released on 27 August 1999.  Shahrukh Khan earned a Filmfare Award nomination for Best Performance in a Comic Role.  It is inspired from films such as "Nick of Time", "Rush Hour", "If Looks Could Kill", "Mr Nice Guy", "The Mask".
[hotpotqa_distractor_validation_5a7349125542994cef4bc505_ctx_9] Mr. Nice Guy (1997 film): Mr. Nice Guy (一個好人, LSHK "Jat1 go3 hou2 jan4") is a 1997 Hong Kong action film directed by Sammo Hung, who makes a cameo as an unfortunate cyclist.  The film stars Jackie Chan and Richard Norton.  The film was released in the Hong Kong on January 31, 1997.
[hotpotqa_distractor_validation_5a7349125542994cef4bc505_ctx_6] City Hunter (film): City Hunter (; "Sing si lip yan") is a 1993 Hong Kong action comedy film written and directed by Wong Jing, starring J…
```

### `hotpotqa_distractor_validation_5a7323ef5542994cef4bc477__internal__v0` (internal / E)

User (truncated):

```text
Question: Which battle took place first out of the Battle of the Ch'ongch'on River and the Battle of Tarawa?
```

Target:

```text
<internal>
Use internal knowledge.
</internal>
<answer>
The Battle of Tarawa
</answer>
```

### `hotpotqa_distractor_validation_5a7c76275542990527d554b4__internal__v0` (internal / D)

User (truncated):

```text
Question: Which film was released first out of The Hunchback of Notre Dame and Miracle of the White Stallions?
```

Target:

```text
<internal>
Use internal knowledge.
</internal>
<answer>
Miracle of the White Stallions
</answer>
```

### `hotpotqa_distractor_validation_5a7f275d55429934daa2fd01__internal__v0` (internal / D)

User (truncated):

```text
Question: Are both Jonathan Marray and Wayne Black British?
```

Target:

```text
<internal>
Use internal knowledge.
</internal>
<answer>
no
</answer>
```

### `hotpotqa_distractor_validation_5a80721b554299485f5985ef__internal__v0` (internal / D)

User (truncated):

```text
Question: The Livesey Hal War Memorial commemorates the fallen of which war, that had over 60 million casualties?
```

Target:

```text
<internal>
Use internal knowledge.
</internal>
<answer>
World War II
</answer>
```

## Human audit checklist

- [ ] Internal rows: no documents; short routing tag; gold answer
- [ ] Evidence rows: provenance ids match HotpotQA sentences
- [ ] Reasoning rows: short, no new facts, answer-consistent
- [ ] Search-format: observation is candidate scope; not claimed full-corpus
- [ ] No Base wrong answers as targets

## Next

After human spot-check passes → Phase 2C scale to 2k–5k.
