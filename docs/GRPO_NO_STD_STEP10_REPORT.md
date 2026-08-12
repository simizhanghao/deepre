# GRPO-no-std @10 — Final Causal Ablation Report

## Verdict

**`GRPO_NO_STD_DIRECTION_FAIL`**. The clean four-GPU run completed all ten
optimizer steps from Evidence@400, saved resumable states at steps 5 and 10,
and passed exact-alignment, trajectory and optimizer-health gates. It failed
the pre-registered conditional-routing gate because both Boundary classes
moved strongly toward `<internal>`.

This closes the planned trajectory-level optimizer/normalization sweep. The
next method branch is Root-Pivot decision-aligned credit; do not automatically
try full Dr.GRPO, RF++, GSPO, SAPO, reward retuning or a new loss aggregation.

## Frozen-20 direction gate

The route margin is `log P(<search>) - log P(<internal>)`.

| Frozen subset | Evidence@400 | Required at @10 | GRPO-no-std @10 | Result |
|---|---:|---:|---:|---|
| NoSearch | `0.864` | `<0.864` | `-1.409` | direction passed |
| NeedSearch | `1.472` | `>=1.272` | `-0.889` | preservation failed |

The NoSearch improvement is therefore not conditional learning: NeedSearch
lost `2.361` margin points relative to Evidence@400 and crossed to an internal
preference. This is the pre-registered **global-internal** failure signature.

## Exactness and health

- VeOmni ↔ VeXact full logits: PASS, maximum absolute delta `0.0`.
- Fused-LCE logprobs: PASS, maximum absolute delta `0.0`.
- finish rate `1.0`; response clip ratio `.015625`.
- final-answer missing rate and reserve violations: `0`.
- mixed-action group rate `.625`.
- gradient norm `.351`; PPO clip fraction `0`.
- KL loss `.01166`; importance-ratio P99 `1.0786`.

The failure cannot be attributed to TIM, missing action support, clipping,
importance drift, trajectory truncation or an unresolved optimizer config.

## Ten-step behavior

| Step | Need SR | NoSearch SR | Boundary delta | Internal rate |
|---:|---:|---:|---:|---:|
| 1 | `.917` | `.583` | `.333` | `.203` |
| 2 | `.893` | `.333` | `.560` | `.234` |
| 3 | `.800` | `.500` | `.300` | `.266` |
| 4 | `.714` | `.714` | `.000` | `.250` |
| 5 | `.472` | `.250` | `.222` | `.578` |
| 6 | `.321` | `.188` | `.134` | `.672` |
| 7 | `.375` | `.250` | `.125` | `.656` |
| 8 | `.375` | `.125` | `.250` | `.734` |
| 9 | `.250` | `.000` | `.250` | `.750` |
| 10 | `.227` | `.000` | `.227` | `.781` |

The early non-collapse signal was real but transient. From step 5 onward the
policy acquired a monotonic global internal tendency. At step 10 the training
batch had OSR `0`, but USR was `.773`; the model avoided unnecessary searches
by also suppressing necessary searches.

Answer reward at step 10 was `.125` and Evidence F1 `.1374`. These noisy batch
values are secondary because the hard root-policy gate already failed.

## Engineering result

The run used GPUs 0–3, FSDP size 4, micro-batch 2/GPU and a 4096-block VeXact
cache. VeXact rollout phases reached about `60.02 GB/GPU`, satisfying the
requested utilization target. Ten steps took `26m39s`; average recorded step
time was `159.6s`. Full resumable checkpoints exist at steps 5 and 10 (about
46 GB each), plus a lightweight step-10 HF artifact.

The detached controller stopped after @10. Its post-evaluation JSON reader used
an unavailable bare `python` command; this occurred only after the node summary
had been written. Since that summary is FAIL, stopping was the protocol-correct
outcome. The controller now uses the pinned VeXact environment Python.

## Scientific interpretation

With Exact rollout and the same Evidence@400 initialization:

- standard GRPO produced global-search behavior;
- RF++ baseline produced global-internal behavior;
- GRPO without per-group std normalization also produced global-internal
  behavior, although more gradually.

Removing local std normalization changes strength and early dynamics but is not
sufficient to learn opposite root decisions for NeedSearch and NoSearch. Along
with the measured `12.642x` search/internal policy-token length gap, this
supports the next hypothesis: trajectory-wide task credit interferes with the
pivotal root routing decision through shared Transformer parameters.

The next registered experiment is Root-Pivot v0: preserve task credit while
adding a Boundary-masked route loss directly on the two root token logits,
initially with equal NeedSearch/NoSearch weights and Undetermined masked. Its
coefficient must be fixed from an initial gradient-scale calibration rather
than a dev-set sweep.
