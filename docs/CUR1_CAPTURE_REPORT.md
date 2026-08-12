# CUR-1 Fresh Train/Validation Capture Report

Date: 2026-08-12

Policy: frozen Evidence@400

Backend: forward-only VeXact exact rollout

Test status: **SEALED — not opened**

## Verdict

Both fresh acquisition splits pass the paired capture contract. The completed
artifact contains 640 train questions at N=1 per arm and 128 validation
questions at N=4 per arm: 2304 trajectories in total. Search has a large
positive average F1 effect on both splits, while a non-trivial minority of
questions still favors internal. This establishes useful outcome supervision;
it does **not** pass Gates A/B/C, which require a locked router and the single
sealed-test invocation.

## Contract and integrity

| Check | Train | Validation |
|---|---:|---:|
| Questions | 640 | 128 |
| Rollouts per arm | 1 | 4 |
| Rows | 1280 | 1024 |
| Canonical prompt pairing | PASS | PASS |
| Action-valid rate | 99.9219% | 100% |
| Internal tool violations | 0 | 0 |
| Retained policy failures | 1 | 0 |
| Finish rate | 99.8438% | 99.6094% |
| Retrieval infrastructure failures | 0 | 0 |

The one train policy failure is retained as an outcome by design: its forced
search opening did not execute (`search_count=0`). Two additional train search
rollouts and four validation search rollouts did not finish. These are policy
outcomes, not dropped rows. All internal trajectories finished and made zero
tool calls.

The immutable split audit independently passes: 1240 historical question IDs
are excluded, train/validation/test are disjoint 640/128/128, all frozen hashes
match, and the complete design remains 896 fresh questions / 4352 planned
trajectories.

## Paired outcomes

| Metric | Train N=1 | Validation N=4 |
|---|---:|---:|
| F1, do(search) | 0.5495 | 0.4859 |
| F1, do(internal) | 0.2473 | 0.2371 |
| Mean F1 uplift | **+0.3022** | **+0.2488** |
| Question-bootstrap 95% CI of mean uplift | [0.2623, 0.3423] | [0.1768, 0.3253] |
| EM, do(search) | 0.4609 | 0.4102 |
| EM, do(internal) | 0.1844 | 0.1875 |
| Mean EM uplift | +0.2766 | +0.2227 |
| Search-positive questions | 271 | 61 |
| Internal-positive questions | 42 | 15 |
| Ties | 327 | 52 |
| Oracle F1 from observed arm means | 0.5944 | 0.5223 |

The positive mean effect is stable across the broad N=1 train split and the
replicated N=4 validation split. However, the deployment problem is not solved
by this mean: on validation, 15/128 questions favor internal and the observed
arm-mean oracle reaches 0.5223 versus 0.4859 for always-search. Thus the maximum
observed validation headroom over always-search is only 0.0364 F1, while search
is much more expensive.

Search trajectories average 636.5 response tokens and 380.7 observation
tokens on validation, versus 53.3 response tokens and zero observation tokens
for internal. The routing problem therefore remains decision-relevant even
though always-search is a strong accuracy baseline.

Train N=1 direction labels are deliberately noisy and are used as individual
arm outcomes for potential-outcome fitting, not as hard per-question truth.
Validation N=4 is reserved for architecture/calibration selection.

## Runtime diagnosis

The original acquisition profile used train batch 16 / 80 steps and validation
batch 16 / 16 steps.

| Runtime | Train | Validation |
|---|---:|---:|
| Rollout wall time | 64m39s | 20m05s |
| Median step | 36.4s | 55.0s |
| Mean step | 48.3s | 75.0s |
| Maximum step | 325.1s | 312.5s |
| Mean generation time | 44.8s | 69.0s |

Generation dominates runtime. A few multi-turn trajectories create five-minute
synchronous batch tails; actor update is disabled and old-log-prob plus weight
transfer are small by comparison.

The next-run scheduling profile is now train batch 64 / 20 steps, validation
batch 32 / 8 steps, `max_num_batched_tokens=16384`, VeXact
`max_cache_blocks=1024`, and actor/ref/rollout micro-batch 4. This preserves
the samples, interventions and sampling distribution. It should improve
throughput, but larger batches may amplify straggler latency, so it remains a
smoke-gated profile.

## Provenance note

The validation trainer completed all 16/16 steps and wrote all 1024 rows. The
capture script was edited while its Bash process was still alive to install
the next-run scheduling profile. After the trainer returned, Bash reread the
changed tail and executed one Hydra override as a shell command, producing a
post-run wrapper exit 127 before the automatic audit. No generation step was
affected. The standalone immutable audit and paired analysis were then run on
the complete artifact and both passed. Future scripts must not be edited in
place while running; copy/version the launcher before changing profiles.

## Next scientific step

Extract the frozen pre-action Layer-18 PCA64 semantic view and Layer
18/27/36 dynamics/margins for train and validation, then fit the pre-registered
B0–B6 matrix. Lock one candidate, PCA, calibration and budget policy on
validation. Only then may the single final evaluator open fresh test and emit
Gates A/B/C together.
