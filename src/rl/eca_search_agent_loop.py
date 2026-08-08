"""ECA Search AgentLoop for veRL — preserves SFT-v1 XML dialect.

Flow:
  prompt (system+question, no gold/contexts)
    → generate until </search>|</answer>|</internal>
    → if <search>: CandidateBM25Tool(sample_id, query) → <observation> (mask=0)
    → continue (max_search_turns=2)
    → terminate on <answer> or budgets

Registered via configs/rl/eca_agent_loop.yaml as agent_name=eca_search_agent.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import ToolListWrap
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel("INFO")

_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_INTERNAL_RE = re.compile(r"<internal>(.*?)</internal>", re.DOTALL | re.IGNORECASE)

_STOP_STRINGS = ["</search>", "</answer>", "</internal>"]


@register("eca_search_agent")
class EcaSearchAgentLoop(AgentLoopBase):
    """Multi-turn search agent using Candidate-BM25 BaseTool + XML tags."""

    def __init__(self, *args, tools: Optional[ToolListWrap] = None, **kwargs):
        super().__init__(*args, **kwargs)
        tool_list = tools.tools if tools else []
        self.tools = {t.name: t for t in tool_list}
        self.search_tool = self.tools.get("search")
        if self.search_tool is None:
            raise ValueError(
                "EcaSearchAgentLoop requires a BaseTool named 'search' "
                "(see configs/rl/candidate_bm25_tool.yaml)"
            )

        mt = self.rollout_config.multi_turn
        self.max_assistant_turns = int(mt.max_assistant_turns or 6)
        self.max_user_turns = int(mt.max_user_turns or 4)
        self.max_tool_response_length = int(mt.max_tool_response_length or 2048)
        self.tool_response_truncate_side = mt.tool_response_truncate_side or "middle"

        # Locked for 3B0/3B1; raise later via config if needed.
        self.max_search_turns = 2
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        extra_info = kwargs.get("extra_info", {}) or {}
        tools_kwargs = kwargs.get("tools_kwargs", {}) or extra_info.get("tools_kwargs") or {}

        sample_id = (
            extra_info.get("sample_id")
            or (tools_kwargs.get("search") or {}).get("create_kwargs", {}).get("sample_id")
        )
        if not sample_id:
            raise ValueError("eca_search_agent requires extra_info.sample_id")

        create_kwargs = {"sample_id": str(sample_id)}
        # Prefer dataset tools_kwargs if present.
        if "search" in tools_kwargs and isinstance(tools_kwargs["search"], dict):
            ck = tools_kwargs["search"].get("create_kwargs") or {}
            if ck.get("sample_id"):
                create_kwargs["sample_id"] = str(ck["sample_id"])

        metrics: dict[str, Any] = {
            "sample_id": create_kwargs["sample_id"],
            "search_count": 0,
            "duplicate_query_count": 0,
            "observation_tokens": 0,
            "finish": 0,
        }
        request_id = uuid4().hex

        # Initial prompt — DO NOT inject OpenAI tool schemas (SFT-v1 never saw them).
        prompt_ids = await self.apply_chat_template(messages, tools=None)
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        search_queries: list[str] = []
        assistant_turns = 0
        user_turns = 0
        finished = False
        # Propagate weight-version tags from generate → TransferQueue metrics.
        # Missing these makes trainer_base._compute_metrics crash (None → int).
        min_global_steps: int | None = None
        max_global_steps: int | None = None

        # Bind trajectory tool instance once.
        instance_id, _ = await self.search_tool.create(create_kwargs=create_kwargs)

        try:
            sp = dict(sampling_params)
            # Prefer stop_token_ids. String `stop` crashes SGLang 0.5.5 under
            # veRL hybrid (skip_tokenizer_init → scheduler tokenizer is None).
            sp.pop("stop", None)
            stop_ids: list[int] = []
            for s in _STOP_STRINGS:
                ids = self.tokenizer.encode(s, add_special_tokens=False)
                if ids:
                    # Use last token as stop signal (Qwen often splits tags).
                    stop_ids.append(ids[-1])
            if stop_ids:
                existing = list(sp.get("stop_token_ids") or [])
                sp["stop_token_ids"] = sorted(set(existing + stop_ids))

            while (
                assistant_turns < self.max_assistant_turns
                and user_turns < self.max_user_turns
                and len(response_mask) < self.response_length
            ):
                with simple_timer("generate_sequences", metrics):
                    output: TokenOutput = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=prompt_ids,
                        sampling_params=sp,
                    )

                # Track trajectory weight versions across multi-turn generate calls.
                out_extra = getattr(output, "extra_fields", None) or {}
                step = out_extra.get("min_global_steps", out_extra.get("global_steps"))
                if step is not None:
                    step_i = int(step)
                    min_global_steps = step_i if min_global_steps is None else min(min_global_steps, step_i)
                    max_g = out_extra.get("max_global_steps", step_i)
                    max_global_steps = (
                        int(max_g)
                        if max_global_steps is None
                        else max(max_global_steps, int(max_g))
                    )

                gen_ids = list(output.token_ids or [])
                if not gen_ids:
                    break

                # Cap to remaining response budget.
                remain = self.response_length - len(response_mask)
                gen_ids = gen_ids[:remain]
                prompt_ids = prompt_ids + gen_ids
                response_mask.extend([1] * len(gen_ids))
                if output.log_probs:
                    response_logprobs.extend(list(output.log_probs[: len(gen_ids)]))
                assistant_turns += 1

                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                # Some engines omit stop string; tolerate either.
                has_answer = bool(_ANSWER_RE.search(text))
                search_hits = list(_SEARCH_RE.finditer(text))
                has_internal = bool(_INTERNAL_RE.search(text))

                if has_answer:
                    finished = True
                    metrics["finish"] = 1
                    break

                if search_hits:
                    if metrics["search_count"] >= self.max_search_turns:
                        # Budget exhausted — stop without executing more search.
                        break
                    query = search_hits[-1].group(1).strip()
                    if query in search_queries:
                        metrics["duplicate_query_count"] += 1
                    search_queries.append(query)

                    tool_resp, _, tool_metrics = await self.search_tool.execute(
                        instance_id, {"query": query}
                    )
                    metrics["search_count"] = int(metrics["search_count"]) + 1
                    obs_body = (tool_resp.text or "[no documents retrieved]").strip()
                    if len(obs_body) > self.max_tool_response_length:
                        side = self.tool_response_truncate_side
                        n = self.max_tool_response_length
                        if side == "left":
                            obs_body = "(truncated)..." + obs_body[-n:]
                        elif side == "right":
                            obs_body = obs_body[:n] + "...(truncated)"
                        else:
                            half = n // 2
                            obs_body = obs_body[:half] + "...(truncated)..." + obs_body[-half:]

                    cont = (
                        f"<observation>\n{obs_body}\n</observation>\n"
                        "Continue. Prefer <evidence> then <think> then <answer>. "
                        f"You may <search> again only if necessary "
                        f"(searches used: {metrics['search_count']}/{self.max_search_turns})."
                    )
                    obs_messages = [{"role": "user", "content": cont}]
                    # Tokenize observation user turn; mask=0 (env tokens).
                    obs_ids = await self.apply_chat_template(
                        obs_messages, tools=None, remove_system_prompt=True
                    )
                    if self.turn_separator:
                        obs_ids = list(self.turn_separator) + list(obs_ids)
                    if len(response_mask) + len(obs_ids) >= self.response_length:
                        # Keep truncated env tokens if any room left.
                        room = self.response_length - len(response_mask)
                        if room <= 0:
                            break
                        obs_ids = obs_ids[:room]
                        prompt_ids = prompt_ids + obs_ids
                        response_mask.extend([0] * len(obs_ids))
                        if response_logprobs:
                            response_logprobs.extend([0.0] * len(obs_ids))
                        metrics["observation_tokens"] += len(obs_ids)
                        break

                    prompt_ids = prompt_ids + obs_ids
                    response_mask.extend([0] * len(obs_ids))
                    if response_logprobs:
                        response_logprobs.extend([0.0] * len(obs_ids))
                    metrics["observation_tokens"] += len(obs_ids)
                    user_turns += 1
                    metrics["last_tool_metrics"] = tool_metrics
                    continue

                if has_internal:
                    # Ask for final answer without search (same as Phase 3A).
                    nudge = (
                        "You chose <internal>. Now give the final answer in "
                        "<answer>...</answer> (optionally with short <think>)."
                    )
                    nudge_ids = await self.apply_chat_template(
                        [{"role": "user", "content": nudge}],
                        tools=None,
                        remove_system_prompt=True,
                    )
                    if self.turn_separator:
                        nudge_ids = list(self.turn_separator) + list(nudge_ids)
                    if len(response_mask) + len(nudge_ids) >= self.response_length:
                        break
                    prompt_ids = prompt_ids + nudge_ids
                    response_mask.extend([0] * len(nudge_ids))
                    if response_logprobs:
                        response_logprobs.extend([0.0] * len(nudge_ids))
                    user_turns += 1
                    continue

                # No closed action — terminate.
                break
        finally:
            await self.search_tool.release(instance_id)

        response_ids = prompt_ids[-len(response_mask) :] if response_mask else []
        prompt_ids_out = prompt_ids[: len(prompt_ids) - len(response_mask)] if response_mask else prompt_ids

        # Store masked decode helpers for audit.
        # Fallback to 0 for sync smoke if generate never stamped versions
        # (empty gen / early break) — still must be int for metrics.
        if min_global_steps is None:
            min_global_steps = 0
        if max_global_steps is None:
            max_global_steps = min_global_steps
        extra_fields = {
            "turn_scores": [],
            "tool_rewards": [],
            "min_global_steps": min_global_steps,
            "max_global_steps": max_global_steps,
            "sample_id": create_kwargs["sample_id"],
            "search_queries": search_queries,
            "finish": int(finished or metrics.get("finish") or 0),
            "metrics": metrics,
        }

        return AgentLoopOutput(
            prompt_ids=prompt_ids_out,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            multi_modal_data={},
            num_turns=assistant_turns + user_turns + 1,
            metrics=metrics,
            extra_fields=extra_fields,
        )


# Re-export for convenience inside the container.
from src.rl.mask_audit import dump_mask_audit  # noqa: E402
