"""Phase-25 step-level adaptive retrieval AgentLoop.

This is a new registry and deliberately does not alter EcaSearchAgentLoop.
Candidate-query generation and external Search/Continue/Answer decisions are
separate operations.
"""

from __future__ import annotations

import hashlib
import json
import fcntl
import os
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import ToolListWrap
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

from src.rl.eca_search_agent_loop import _rollout_backend, _truncate_at_complete_sequence
from src.rl.step_checkpoint import (
    CLOSE,
    StepAction,
    StepState,
    checkpoint_prompt,
    fixed_completion_action,
    parse_step_checkpoint,
    validate_action,
)

STEP_SYSTEM_PROMPT = (
    "You are an evidence-cost-aware research agent operating in bounded reasoning steps. "
    "Do not choose a root <search> or <internal> action. At each reasoning checkpoint, "
    "follow the exact step protocol. A Search query is only a proposal and never executes "
    "a tool by itself. When an observation is provided, use it as evidence. Only produce "
    "<answer>...</answer> after the external controller requests the final answer."
)

REASONING_PREFIX = "<think>Reasoning step: "
QUERY_PREFIX = "\nSearch query: "
LEGACY_STRUCTURE_MARKERS = (
    "<search>",
    "</search>",
    "<answer>",
    "</answer>",
    "<internal>",
    "</internal>",
    "<observation>",
    "</observation>",
    "<evidence>",
    "</evidence>",
)


def _normalise_generated_span(
    tokenizer: Any,
    token_ids: list[int],
    markers: tuple[str, ...],
    *,
    first_line: bool = False,
) -> tuple[str, list[int]]:
    """Turn a bounded raw proposal into controller-safe continuation text.

    VeXact currently generates beyond string stops.  We retain the raw proposal
    in the audit record, while the controller re-tokenises only the text before
    a structural marker.  These controller-normalised tokens are masked and are
    never treated as policy-generated tokens in Phase 25.
    """

    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    cut = len(text)
    for marker in markers:
        position = text.find(marker)
        if position >= 0:
            cut = min(cut, position)
    text = text[:cut].strip()
    if first_line:
        text = text.splitlines()[0].strip() if text else ""
    ids = list(tokenizer.encode(text, add_special_tokens=False)) if text else []
    return text, ids


def _step_dump(row: dict[str, Any]) -> None:
    path = (os.environ.get("ECA_STEP_CAPTURE_JSONL") or "").strip()
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _action_plan(extra_info: dict[str, Any]) -> list[StepAction]:
    raw = extra_info.get("step_action_plan") or []
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",") if part.strip()]
    if not isinstance(raw, list):
        raise ValueError("step_action_plan must be a list or comma-separated string")
    return [StepAction(str(value).lower()) for value in raw]


def _with_step_protocol(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace the incompatible root-routing system contract for this new agent."""

    updated = [dict(message) for message in messages]
    protocol = STEP_SYSTEM_PROMPT + "\n\nStep-adaptive protocol: " + checkpoint_prompt()
    if updated and updated[0].get("role") == "system":
        updated[0]["content"] = protocol
    else:
        updated.insert(0, {"role": "system", "content": protocol.strip()})
    return updated


@register("eca_step_adaptive_agent")
class EcaStepAdaptiveAgentLoop(AgentLoopBase):
    """Bounded checkpoint loop with externally applied three-way actions."""

    def __init__(self, *args, tools: Optional[ToolListWrap] = None, **kwargs):
        super().__init__(*args, **kwargs)
        tool_list = tools.tools if tools else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.search_tool = self.tools.get("search")
        if self.search_tool is None:
            raise ValueError("eca_step_adaptive_agent requires a search BaseTool")
        mt = self.rollout_config.multi_turn
        self.prompt_length = int(self.rollout_config.prompt_length)
        self.response_length = int(self.rollout_config.response_length)
        self.max_tool_response_length = min(int(mt.max_tool_response_length or 384), 384)
        self.tool_response_truncate_side = mt.tool_response_truncate_side or "middle"
        self.max_checkpoints = 4
        self.max_searches = 3
        self.step_token_cap = 128
        self.final_answer_reserve = 256

    async def _masked_turn(self, content: str) -> list[int]:
        ids = list(
            await self.apply_chat_template(
                [{"role": "user", "content": content}],
                tools=None,
                remove_system_prompt=True,
            )
        )
        return list(self.turn_separator or []) + ids

    async def _observation_turn(self, body: str) -> list[int]:
        body_ids = list(self.tokenizer.encode(body, add_special_tokens=False))
        keep = min(len(body_ids), max(0, self.max_tool_response_length - 64))
        while keep >= 0:
            rendered = self.tokenizer.decode(body_ids[:keep], skip_special_tokens=False)
            content = (
                f"<observation>\n{rendered}\n</observation>\n"
                "Use this evidence in the next bounded reasoning checkpoint. "
                + checkpoint_prompt()
            )
            ids = await self._masked_turn(content)
            if len(ids) <= self.max_tool_response_length:
                return ids
            keep -= max(1, len(ids) - self.max_tool_response_length)
        return []

    async def _generate(
        self,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        cap: int,
        closing_tags: tuple[str, ...],
        metrics: dict[str, Any],
    ) -> tuple[list[int], list[float], str | None, bool, TokenOutput]:
        sp = dict(sampling_params)
        sp.pop("stop", None)
        # Exact rollout currently cannot accept per-request max_new_tokens.
        if _rollout_backend() == "vexact":
            sp.pop("max_new_tokens", None)
            sp.pop("stop_token_ids", None)
        else:
            sp["max_new_tokens"] = cap
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sp,
            )
        raw = list(output.token_ids or [])
        sequences = {
            tag: list(self.tokenizer.encode(tag, add_special_tokens=False)) for tag in closing_tags
        }
        tokens, matched, truncated = _truncate_at_complete_sequence(
            raw, sequences, cap, tokenizer=self.tokenizer
        )
        logprobs = list(output.log_probs[: len(tokens)]) if output.log_probs else []
        return tokens, logprobs, matched, truncated, output

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        raw_messages = list(kwargs["raw_prompt"])
        extra_info = kwargs.get("extra_info", {}) or {}
        tools_kwargs = kwargs.get("tools_kwargs", {}) or extra_info.get("tools_kwargs") or {}
        sample_id = extra_info.get("sample_id") or (
            (tools_kwargs.get("search") or {}).get("create_kwargs", {}).get("sample_id")
        )
        if not sample_id:
            raise ValueError("eca_step_adaptive_agent requires sample_id")
        plan = _action_plan(extra_info)
        branch_id = str(extra_info.get("step_branch_id") or "")
        target_index_raw = extra_info.get("step_target_index")
        target_index = int(target_index_raw) if target_index_raw is not None else None
        branch_arm = str(extra_info.get("step_branch_arm") or "")
        step_padding = bool(extra_info.get("step_padding", False))
        messages = _with_step_protocol(raw_messages)
        canonical_prompt_ids = list(await self.apply_chat_template(raw_messages, tools=None))
        prompt_ids = list(await self.apply_chat_template(messages, tools=None))
        step_prompt_ids = list(prompt_ids)
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        records: list[dict[str, Any]] = []
        previous_queries: list[str] = []
        searches_used = 0
        finished = False
        request_id = uuid4().hex
        metrics: dict[str, Any] = {
            "sample_id": str(sample_id),
            "search_count": 0,
            "finish": 0,
            "parser_valid": 1,
            "checkpoint_count": 0,
            "duplicate_query_count": 0,
            "tool_violations": 0,
            "final_answer_reserve_violations": 0,
            "response_clipped": 0,
        }
        create_kwargs = {"sample_id": str(sample_id)}
        if isinstance(tools_kwargs.get("search"), dict):
            value = (tools_kwargs["search"].get("create_kwargs") or {}).get("sample_id")
            if value:
                create_kwargs["sample_id"] = str(value)
        instance_id, _ = await self.search_tool.create(create_kwargs=create_kwargs)
        min_global_steps: int | None = None
        max_global_steps: int | None = None

        def append_generated(tokens: list[int], logprobs: list[float]) -> tuple[int, int]:
            start = len(response_mask)
            prompt_ids.extend(tokens)
            response_mask.extend([1] * len(tokens))
            response_logprobs.extend(logprobs if logprobs else [0.0] * len(tokens))
            return start, len(response_mask)

        async def append_masked(content_ids: list[int]) -> None:
            prompt_ids.extend(content_ids)
            response_mask.extend([0] * len(content_ids))
            response_logprobs.extend([0.0] * len(content_ids))

        def update_versions(output: TokenOutput) -> None:
            nonlocal min_global_steps, max_global_steps
            extra = getattr(output, "extra_fields", None) or {}
            step = extra.get("min_global_steps", extra.get("global_steps"))
            if step is not None:
                step = int(step)
                min_global_steps = step if min_global_steps is None else min(min_global_steps, step)
                high = int(extra.get("max_global_steps", step))
                max_global_steps = high if max_global_steps is None else max(max_global_steps, high)

        try:
            for checkpoint_index in range(self.max_checkpoints):
                remaining = self.response_length - len(response_mask)
                if remaining <= self.final_answer_reserve:
                    metrics["final_answer_reserve_violations"] += 1
                    break
                forced_think_ids = list(
                    self.tokenizer.encode(REASONING_PREFIX, add_special_tokens=False)
                )
                checkpoint_start = len(response_mask)
                await append_masked(forced_think_ids)
                reasoning_raw, reasoning_raw_logps, _, reasoning_truncated, output = await self._generate(
                    request_id,
                    prompt_ids,
                    sampling_params,
                    min(96, remaining - self.final_answer_reserve - len(forced_think_ids)),
                    (QUERY_PREFIX.strip(), CLOSE),
                    metrics,
                )
                update_versions(output)
                reasoning, reasoning_ids = _normalise_generated_span(
                    self.tokenizer,
                    reasoning_raw,
                    (QUERY_PREFIX.strip(), CLOSE, *LEGACY_STRUCTURE_MARKERS),
                )
                reasoning_fallback = not bool(reasoning)
                if reasoning_fallback:
                    reasoning = "the remaining uncertainty must be resolved"
                    reasoning_ids = list(
                        self.tokenizer.encode(reasoning, add_special_tokens=False)
                    )
                await append_masked(reasoning_ids)

                query_prefix_ids = list(
                    self.tokenizer.encode(QUERY_PREFIX, add_special_tokens=False)
                )
                await append_masked(query_prefix_ids)
                query_raw, query_raw_logps, _, query_truncated, query_output = await self._generate(
                    request_id,
                    prompt_ids,
                    sampling_params,
                    min(32, remaining - self.final_answer_reserve - len(response_mask) + checkpoint_start),
                    (CLOSE,),
                    metrics,
                )
                update_versions(query_output)
                query, query_ids = _normalise_generated_span(
                    self.tokenizer,
                    query_raw,
                    (CLOSE, *LEGACY_STRUCTURE_MARKERS),
                    first_line=True,
                )
                query_fallback = not bool(query)
                if query_fallback:
                    query = "NONE"
                    query_ids = list(self.tokenizer.encode(query, add_special_tokens=False))
                await append_masked(query_ids)
                close_ids = list(self.tokenizer.encode(CLOSE, add_special_tokens=False))
                await append_masked(close_ids)
                end = len(response_mask)
                text = REASONING_PREFIX + reasoning + QUERY_PREFIX + query + CLOSE
                checkpoint = parse_step_checkpoint(text)
                if checkpoint is None:
                    metrics["parser_valid"] = 0
                    metrics["response_clipped"] += int(reasoning_truncated or query_truncated)
                    break
                state = StepState(checkpoint_index, searches_used, self.max_checkpoints, self.max_searches)
                forced_action = checkpoint_index < len(plan)
                action = plan[checkpoint_index] if forced_action else fixed_completion_action(checkpoint, state)
                if (
                    not forced_action
                    and action is StepAction.SEARCH
                    and checkpoint.proposed_query in previous_queries
                ):
                    action = StepAction.CONTINUE
                validate_action(action, checkpoint, state)
                duplicate = int(checkpoint.proposed_query in previous_queries and not checkpoint.query_is_none)
                metrics["duplicate_query_count"] += duplicate
                record = {
                    "step_index": checkpoint_index,
                    "reasoning_text": checkpoint.reasoning,
                    "candidate_query": checkpoint.proposed_query,
                    "query_is_none": checkpoint.query_is_none,
                    "num_previous_searches": searches_used,
                    "previous_queries": list(previous_queries),
                    "checkpoint_response_start": checkpoint_start,
                    "checkpoint_response_end": end,
                    "state_prefix_sha256": hashlib.sha256(
                        json.dumps(prompt_ids).encode()
                    ).hexdigest(),
                    "checkpoint_token_ids": forced_think_ids + reasoning_ids + query_prefix_ids + query_ids + close_ids,
                    "forced_prefix_token_count": len(forced_think_ids),
                    "checkpoint_logprobs": [],
                    "reasoning_raw_token_ids": reasoning_raw,
                    "reasoning_raw_logprobs": reasoning_raw_logps,
                    "query_raw_token_ids": query_raw,
                    "query_raw_logprobs": query_raw_logps,
                    "reasoning_fallback": reasoning_fallback,
                    "query_fallback": query_fallback,
                    "action": action.value,
                    "duplicate_query": duplicate,
                    "matched_close": CLOSE,
                    "was_truncated": bool(reasoning_truncated or query_truncated),
                }
                records.append(record)
                metrics["checkpoint_count"] += 1

                if action is StepAction.SEARCH:
                    previous_queries.append(checkpoint.proposed_query)
                    tool_response, _, tool_metrics = await self.search_tool.execute(
                        instance_id, {"query": checkpoint.proposed_query}
                    )
                    searches_used += 1
                    metrics["search_count"] = searches_used
                    observation_ids = await self._observation_turn(
                        (tool_response.text or "[no documents retrieved]").strip()
                    )
                    record["observation_tokens"] = len(observation_ids)
                    record["tool_metrics"] = tool_metrics
                    if not observation_ids:
                        metrics["final_answer_reserve_violations"] += 1
                        break
                    await append_masked(observation_ids)
                    continue

                if action is StepAction.CONTINUE:
                    control = await self._masked_turn(
                        "Retrieval skipped for this checkpoint. Continue with one new bounded "
                        "reasoning step; future Search remains available. " + checkpoint_prompt()
                    )
                    record["observation_tokens"] = 0
                    await append_masked(control)
                    continue

                answer_nudge = await self._masked_turn(
                    "Information is sufficient. Give only the final closed <answer>...</answer>."
                )
                await append_masked(answer_nudge)
                forced_answer_ids = list(self.tokenizer.encode("<answer>", add_special_tokens=False))
                await append_masked(forced_answer_ids)
                remaining = self.response_length - len(response_mask)
                answer_tokens, answer_logps, answer_close, answer_truncated, answer_output = await self._generate(
                    request_id,
                    prompt_ids,
                    sampling_params,
                    min(self.final_answer_reserve - len(forced_answer_ids), remaining),
                    ("</answer>",),
                    metrics,
                )
                update_versions(answer_output)
                append_generated(answer_tokens, answer_logps)
                record["answer_token_ids"] = forced_answer_ids + answer_tokens
                record["answer_forced_prefix_token_count"] = len(forced_answer_ids)
                record["answer_closed"] = answer_close == "</answer>"
                # Trimming tokens emitted *after* a complete close tag is normal
                # VeXact stop emulation, not a trajectory-budget violation.
                metrics["response_clipped"] += int(answer_truncated and answer_close is None)
                finished = answer_close == "</answer>"
                metrics["finish"] = int(finished)
                break
        finally:
            await self.search_tool.release(instance_id)

        response_ids = prompt_ids[-len(response_mask) :] if response_mask else []
        prompt_ids_out = prompt_ids[: len(prompt_ids) - len(response_mask)] if response_mask else prompt_ids
        min_global_steps = 0 if min_global_steps is None else min_global_steps
        max_global_steps = min_global_steps if max_global_steps is None else max_global_steps
        canonical_sha = hashlib.sha256(json.dumps(canonical_prompt_ids).encode()).hexdigest()
        step_prompt_sha = hashlib.sha256(json.dumps(step_prompt_ids).encode()).hexdigest()
        metrics.update(
            {
                "finish": int(finished),
                "search_count": searches_used,
                "response_tokens": len(response_mask),
                "assistant_tokens": int(sum(response_mask)),
                "step_prompt_sha256": step_prompt_sha,
            }
        )
        extra_fields = {
            "turn_scores": [],
            "tool_rewards": [],
            "min_global_steps": min_global_steps,
            "max_global_steps": max_global_steps,
            "sample_id": str(sample_id),
            "canonical_prompt_ids": canonical_prompt_ids,
            "canonical_prompt_sha256": canonical_sha,
            "step_prompt_ids": step_prompt_ids,
            "step_prompt_sha256": step_prompt_sha,
            "step_records": records,
            "search_count": searches_used,
            "finish": int(finished),
            "response_tokens": len(response_mask),
            "metrics": metrics,
            "step_branch_id": branch_id,
            "step_target_index": target_index,
            "step_branch_arm": branch_arm,
            "step_padding": step_padding,
        }
        _step_dump(
            {
                "sample_id": str(sample_id),
                "canonical_prompt_sha256": canonical_sha,
                "step_prompt_sha256": step_prompt_sha,
                "step_records": records,
                "search_count": searches_used,
                "finish": int(finished),
                "response_tokens": len(response_mask),
                "assistant_tokens": int(sum(response_mask)),
                "response_token_ids": response_ids[: self.response_length],
                "response_mask": response_mask[: self.response_length],
                "metrics": metrics,
                "step_branch_id": branch_id,
                "step_target_index": target_index,
                "step_branch_arm": branch_arm,
                "step_padding": step_padding,
            }
        )
        return AgentLoopOutput(
            prompt_ids=prompt_ids_out,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length],
            multi_modal_data={},
            num_turns=len(records) * 2 + 1,
            metrics=metrics,
            extra_fields=extra_fields,
        )
