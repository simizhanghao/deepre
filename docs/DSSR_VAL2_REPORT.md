# DSSR Val2 Final Report

Date: 2026-08-12  
Decision: **`SELF_KNOWLEDGE_ROUTER_FAIL`**  
Original Test: **sealed and unread**

## Acquisition integrity

- Val2 freeze: 128 fresh HotpotQA distractor-train questions, seed
  `2026081202`, 3046 historical IDs excluded.
- Probe: 128/128 valid and closed; deterministic Probe F1 `0.3249`.
- Search: 512/512 trajectories (N=4/question), action-valid rate `1.0`, no
  policy failures, finish rate `0.9941`; Always-Search F1 `0.6116`.
- Search main process returned `0`; a DataLoader worker emitted a shutdown-only
  weakref warning after all four steps. Independent row/ID/action audit PASS.
- K0-K3 were frozen before Val2 outcomes were scored. K3 used strict OOF B3
  priors on Train to prevent stacking leakage.

## Val2 results

| Router | F1@25 / Recovery | F1@50 / Recovery | F1@75 / Recovery |
|---|---:|---:|---:|
| K0 Static B3 | .4374 / .2450 | .5343 / .3498 | .5970 / .4873 |
| K1 Prefix-20 | .4377 / .2470 | .5309 / .3316 | **.6119 / .6143** |
| K2 Confidence | .4351 / .2312 | .5174 / .2605 | .5864 / .3964 |
| K3 Dual-Signal | **.4646 / .4083** | .5294 / .3240 | .5800 / .3419 |

K3@50 TokenCost is `707.41`, ratio `0.6228` against Always-Search. Thus the
cost gate passes, but the quality/safety gates do not.

## Frozen four-gate decision

| Gate | Requirement | Result | Pass |
|---|---|---:|---:|
| Recovery@50 | `>= .65` | `.3240` | no |
| F1 preservation@50 | `>= AlwaysSearch-.02 = .5916` | `.5294` | no |
| TokenCost@50 | `<= .65 × AlwaysSearch` | `.6228` | yes |
| K3 superiority | beats max(K0,K1,K2) at 2/3 budgets | wins only 25% | no |

Final decision is therefore `SELF_KNOWLEDGE_ROUTER_FAIL`; no model-selection
exception is allowed. K1 at 75% is an informative diagnostic, not a passing
deployment candidate, because it does not meet the frozen 50% quality/cost
operating point.

## Interpretation

The short Probe adds real signal—K3 is strongest at the lowest 25% search
budget—but it does not identify enough of the Search-rescuable questions at
50%. The failure is no longer attributable to rollout mismatch, optimizer,
reward, static representation alone, or missing self-confidence features.
The remaining structural problem is decision timing: on multi-hop questions,
the external fact deficit often becomes observable only after reasoning has
started.

Per preregistration, question-level Router development is closed. No larger
MLP, new confidence feature, conformal calibration, Val2 tuning, or original
Test acquisition is permitted. The sole next research phase is reasoning-step-
level adaptive retrieval.

