from src.rl.step_checkpoint import (
    StepAction,
    StepState,
    checkpoint_prompt,
    fixed_completion_action,
    parse_step_checkpoint,
    validate_action,
)


def test_valid_checkpoint() -> None:
    parsed = parse_step_checkpoint(
        "<think>We need the birthplace of the director.\n"
        "Search query: director of Film X birthplace\n</think>"
    )
    assert parsed is not None
    assert parsed.reasoning == "We need the birthplace of the director."
    assert parsed.proposed_query == "director of Film X birthplace"


def test_decoded_prefix_and_first_complete_block() -> None:
    parsed = parse_step_checkpoint(
        "prefix <think>First fact is missing.\nSearch query: first fact</think> trailing"
    )
    assert parsed is not None
    assert parsed.block_text.endswith("</think>")
    assert parsed.proposed_query == "first fact"


def test_rejects_incomplete_empty_or_multiline_query() -> None:
    assert parse_step_checkpoint("<think>x\nSearch query: y") is None
    assert parse_step_checkpoint("<think>x\nSearch query:   </think>") is None
    assert parse_step_checkpoint("<think>\nSearch query: y\n</think>") is None


def test_rejects_embedded_actions() -> None:
    for action in ("<search>x</search>", "<answer>x</answer>", "<internal>x</internal>"):
        assert parse_step_checkpoint(
            f"<think>reason {action}\nSearch query: external fact</think>"
        ) is None


def test_prompt_freezes_no_tool_contract() -> None:
    prompt = checkpoint_prompt()
    assert "Search query:" in prompt
    assert "Do not emit <search>" in prompt


def test_none_query_is_not_answer() -> None:
    parsed = parse_step_checkpoint("<think>Need another step.\nSearch query: NONE</think>")
    assert parsed is not None and parsed.query_is_none
    state = StepState(checkpoint_index=0, searches_used=0)
    assert fixed_completion_action(parsed, state) is StepAction.CONTINUE
    try:
        validate_action(StepAction.SEARCH, parsed, state)
    except ValueError:
        pass
    else:
        raise AssertionError("SEARCH with a NONE query must fail")


def test_fixed_policy_searches_valid_query_and_answers_at_last_checkpoint() -> None:
    parsed = parse_step_checkpoint("<think>Missing birthplace.\nSearch query: X birthplace</think>")
    assert parsed is not None
    assert fixed_completion_action(parsed, StepState(0, 0)) is StepAction.SEARCH
    assert fixed_completion_action(parsed, StepState(3, 1)) is StepAction.ANSWER


def test_continue_does_not_permanently_disable_future_search() -> None:
    first = parse_step_checkpoint("<think>Can reason further.\nSearch query: NONE</think>")
    later = parse_step_checkpoint("<think>Now missing a date.\nSearch query: X date</think>")
    assert first is not None and later is not None
    validate_action(StepAction.CONTINUE, first, StepState(0, 0))
    validate_action(StepAction.SEARCH, later, StepState(1, 0))
