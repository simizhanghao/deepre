"""Frozen Phase-25 reasoning-checkpoint grammar.

This module is deliberately independent of veRL so parser semantics can be
unit-tested before any GPU rollout or AgentLoop integration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

OPEN = "<think>"
CLOSE = "</think>"
QUERY_LABEL = "Search query:"

_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_QUERY_RE = re.compile(r"(?:^|\n)\s*Search query:\s*(.*?)\s*$", re.IGNORECASE)
_FORBIDDEN_RE = re.compile(r"</?(?:search|answer|internal|observation|evidence)>", re.IGNORECASE)


@dataclass(frozen=True)
class StepCheckpoint:
    reasoning: str
    proposed_query: str
    block_text: str
    block_start: int
    block_end: int

    @property
    def query_is_none(self) -> bool:
        return self.proposed_query.upper() == "NONE"


class StepAction(str, Enum):
    CONTINUE = "continue"
    SEARCH = "search"
    ANSWER = "answer"


@dataclass(frozen=True)
class StepState:
    checkpoint_index: int
    searches_used: int
    max_checkpoints: int = 4
    max_searches: int = 3


def validate_action(action: StepAction, checkpoint: StepCheckpoint, state: StepState) -> None:
    """Raise when an external intervention violates the frozen state machine."""

    if state.checkpoint_index < 0 or state.checkpoint_index >= state.max_checkpoints:
        raise ValueError("checkpoint index is outside the trajectory budget")
    if state.searches_used < 0 or state.searches_used > state.max_searches:
        raise ValueError("search count is outside the trajectory budget")
    if action is StepAction.SEARCH:
        if checkpoint.query_is_none:
            raise ValueError("SEARCH is invalid for Search query: NONE")
        if state.searches_used >= state.max_searches:
            raise ValueError("SEARCH exceeds max_searches")
    if action is StepAction.CONTINUE and state.checkpoint_index + 1 >= state.max_checkpoints:
        raise ValueError("CONTINUE would exceed max_checkpoints; choose ANSWER")


def fixed_completion_action(checkpoint: StepCheckpoint, state: StepState) -> StepAction:
    """Frozen post-intervention completion policy used by both causal arms."""

    if state.checkpoint_index + 1 >= state.max_checkpoints:
        return StepAction.ANSWER
    if not checkpoint.query_is_none and state.searches_used < state.max_searches:
        return StepAction.SEARCH
    return StepAction.CONTINUE


def parse_step_checkpoint(text: str) -> StepCheckpoint | None:
    """Return the first valid complete checkpoint, otherwise ``None``.

    Frozen grammar::

        <think>
        one bounded reasoning step
        Search query: proposed external fact query
        </think>

    A checkpoint cannot itself execute a tool or emit a final answer. The
    external branch controller decides whether the proposed query is executed.
    """

    match = _BLOCK_RE.search(text)
    if match is None:
        return None
    body = match.group(1)
    if _FORBIDDEN_RE.search(body):
        return None
    query_match = _QUERY_RE.search(body)
    if query_match is None:
        return None
    query = " ".join(query_match.group(1).split())
    reasoning = body[: query_match.start()].strip()
    if not reasoning or not query:
        return None
    if "\n" in query or len(query) > 512:
        return None
    return StepCheckpoint(
        reasoning=reasoning,
        proposed_query=query,
        block_text=match.group(0),
        block_start=match.start(),
        block_end=match.end(),
    )


def checkpoint_prompt() -> str:
    """Exact assistant-output contract appended by the future S0 loop."""

    return (
        "Write exactly one bounded reasoning step inside <think>...</think>. "
        "End the block with one line `Search query: ...` naming the single "
        "external fact that would best resolve the current uncertainty, or "
        "write exactly `Search query: NONE` when no external fact is needed. "
        "Do not emit <search>, <answer>, <internal>, <observation>, or <evidence>."
    )
