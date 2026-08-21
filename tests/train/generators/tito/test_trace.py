"""CPU tests for exact TITO trace bookkeeping."""

import pytest

from skyrl.train.generators.tito.trace import Trace
from skyrl.train.generators.tito.types import ModelTurnResult


def _result(
    prompt_ids,
    message_indices,
    completion_ids,
    assistant_content,
    *,
    reused_prefix_length=0,
):
    return ModelTurnResult(
        prompt_token_ids=tuple(prompt_ids),
        prompt_message_indices=tuple(message_indices),
        reused_prefix_length=reused_prefix_length,
        completion_ids=tuple(completion_ids),
        completion_logprobs=tuple(-0.1 for _ in completion_ids),
        assistant_message={"role": "assistant", "content": assistant_content},
        stop_reason="stop",
    )


def test_linear_bridge_preserves_exact_tokens():
    trace = Trace()
    first = trace.prepare_turn([{"role": "user", "content": "question"}])
    trace.commit(first, _result([10, 11, 12], [0, 0, -1], [20, 21], "answer"))

    second_messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
        {"role": "tool", "content": "observation", "tool_call_id": "call-1"},
    ]
    second = trace.prepare_turn(second_messages)
    assert second.bridge_anchor is not None
    assert second.bridge_anchor.previous_prompt_ids == (10, 11, 12)
    assert second.bridge_anchor.previous_completion_ids == (20, 21)
    assert second.new_messages == (second_messages[-1],)

    trace.commit(
        second,
        _result(
            [10, 11, 12, 20, 21, 30, 31],
            [-1, -1, -1, -1, -1, 2, -1],
            [40],
            "next",
            reused_prefix_length=5,
        ),
    )

    turns = trace.committed_turns()
    assert len(turns) == 2
    assert turns[1].prompt_token_ids == (10, 11, 12, 20, 21, 30, 31)
    assert turns[1].completion_ids == (40,)
    assert turns[1].full_token_ids == (10, 11, 12, 20, 21, 30, 31, 40)
    assert turns[1].is_exact_extension_of(turns[0])
    assert len(trace.branches()) == 1
    assert [node.token_ids for node in trace.nodes()] == [
        (10, 11),
        (12, 20, 21),
        (30,),
        (31, 40),
    ]
    assert trace.nodes()[-1].sampled_mask == (False, True)


def test_full_render_token_mismatch_creates_new_exact_path():
    trace = Trace()
    first = trace.prepare_turn([{"role": "user", "content": "same"}])
    trace.commit(first, _result([1, 2], [0, -1], [3], "answer"))

    rewritten = trace.prepare_turn(
        [
            {"role": "user", "content": "same"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "next"},
        ],
    )
    trace.commit(
        rewritten,
        _result(
            [9, 2, 3, 4, 5],
            [0, 0, 1, 2, -1],
            [6],
            "second",
        ),
    )

    assert len(trace.committed_turns()) == 2
    assert trace.committed_turns()[1].prompt_token_ids == (9, 2, 3, 4, 5)
    assert len(trace.branches()) == 2
    assert trace.nodes()[2].parent_id is None
    assert trace.nodes()[2].token_ids == (9, 2)


def test_full_render_reuses_best_exact_semantic_candidate():
    trace = Trace()
    user = {"role": "user", "content": "same"}
    assistant = {"role": "assistant", "content": "answer"}

    first = trace.prepare_turn([user])
    trace.commit(first, _result([1, 2], [0, -1], [3], "answer"))

    rewritten = trace.prepare_turn([user, assistant])
    trace.commit(
        rewritten,
        _result([9, 2, 3, 5], [0, 0, 1, -1], [6], "rewritten"),
    )
    assert len(trace.branches()) == 2

    exact = trace.prepare_turn(
        [user, assistant, {"role": "user", "content": "next"}],
    )
    trace.commit(
        exact,
        _result([1, 2, 3, 7, 8], [0, 1, 1, 2, -1], [10], "next-answer"),
    )

    assert len(trace.branches()) == 2
    assert trace.transition(2).prompt_token_ids == (1, 2, 3, 7, 8)


def test_identical_completed_attempts_create_distinct_transitions():
    trace = Trace()
    result = _result([1, 2], [0, -1], [3], "a")
    first = trace.commit(trace.prepare_turn([{"role": "user", "content": "q"}]), result)
    second = trace.commit(trace.prepare_turn([{"role": "user", "content": "q"}]), result)

    assert first != second
    assert len(trace.committed_turns()) == 2
    assert len(trace.branches()) == 2
    assert trace.nodes()[1].parent_id == trace.nodes()[2].parent_id == 0
    assert trace.nodes()[1].token_ids == trace.nodes()[2].token_ids


def test_later_request_reuses_the_retry_result_present_in_client_history():
    trace = Trace()
    user = {"role": "user", "content": "q"}
    first_assistant = {"role": "assistant", "content": "first"}
    retry_assistant = {"role": "assistant", "content": "retry"}

    trace.commit(trace.prepare_turn([user]), _result([1, 2], [0, -1], [3], "first"))
    trace.commit(trace.prepare_turn([user]), _result([1, 2], [0, -1], [4], "retry"))

    pending = trace.prepare_turn(
        [
            user,
            retry_assistant,
            {"role": "user", "content": "next"},
        ]
    )

    assert pending.bridge_anchor is not None
    assert pending.bridge_anchor.previous_completion_ids == (4,)
    assert pending.bridge_anchor.node_id == trace.transition(1).assistant_node_id
    assert trace.transition(0).assistant_message == first_assistant


def test_stale_pending_turn_is_rejected():
    trace = Trace()
    stale = trace.prepare_turn([{"role": "user", "content": "one"}])
    current = trace.prepare_turn([{"role": "user", "content": "two"}])
    trace.commit(current, _result([1, 2], [0, -1], [3], "a"))

    with pytest.raises(RuntimeError, match="Stale pending turn"):
        trace.commit(stale, _result([4, 5], [0, -1], [6], "b"))


def test_sealed_trace_rejects_new_work():
    trace = Trace()
    trace.seal()

    with pytest.raises(RuntimeError, match="sealed trace"):
        trace.prepare_turn([{"role": "user", "content": "q"}])


def test_transition_records_do_not_duplicate_cumulative_tokens():
    trace = Trace()
    first = trace.prepare_turn([{"role": "user", "content": "q"}])
    trace.commit(first, _result([1, 9], [0, -1], [2], "a"))
    second = trace.prepare_turn(
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "next"},
        ],
    )
    trace.commit(
        second,
        _result(
            [1, 9, 2, 3, 9],
            [-1, -1, -1, 2, -1],
            [4],
            "b",
            reused_prefix_length=3,
        ),
    )

    debug = trace.to_debug_dict()
    assert debug["storage"]["stored_token_ids"] == 6
    assert debug["storage"]["materialized_transition_token_ids"] == 9
    assert not hasattr(trace._transitions[0], "prompt_token_ids")


def test_harbor_style_summarization_produces_four_exact_branches():
    trace = Trace()
    user = {"role": "user", "content": "task"}
    main = {"role": "assistant", "content": "main", "reasoning_content": "main-think"}
    summary_prompt = {"role": "user", "content": "summarize"}
    summary = {"role": "assistant", "content": "summary", "reasoning_content": "summary-think"}
    questions_prompt = {"role": "user", "content": "questions prompt"}
    questions = {"role": "assistant", "content": "questions"}
    answers_prompt = {"role": "user", "content": "answers prompt"}
    handoff = {"role": "user", "content": "handoff"}

    pending = trace.prepare_turn([user])
    trace.commit(
        pending,
        ModelTurnResult(
            prompt_token_ids=(1, 9),
            prompt_message_indices=(0, -1),
            reused_prefix_length=0,
            completion_ids=(2,),
            completion_logprobs=(-0.1,),
            assistant_message=main,
            stop_reason="stop",
        ),
    )

    pending = trace.prepare_turn([user, main, summary_prompt])
    trace.commit(
        pending,
        ModelTurnResult(
            prompt_token_ids=(1, 9, 2, 3, 9),
            prompt_message_indices=(-1, -1, -1, 2, -1),
            reused_prefix_length=3,
            completion_ids=(4,),
            completion_logprobs=(-0.1,),
            assistant_message=summary,
            stop_reason="stop",
        ),
    )

    pending = trace.prepare_turn([questions_prompt])
    trace.commit(pending, _result([5, 9], [0, -1], [6], "questions"))

    # Harbor reconstructs summary history without the sampled reasoning fields.
    pending = trace.prepare_turn(
        [
            user,
            {"role": "assistant", "content": "main"},
            summary_prompt,
            {"role": "assistant", "content": "summary"},
            answers_prompt,
        ],
    )
    trace.commit(
        pending,
        _result(
            [1, 99, 2, 3, 4, 7, 9],
            [0, 1, 1, 2, 3, 4, -1],
            [8],
            "answers",
        ),
    )

    pending = trace.prepare_turn(
        [user, questions_prompt, questions, handoff],
    )
    trace.commit(
        pending,
        _result(
            [1, 5, 6, 10, 9],
            [0, 1, 2, 3, -1],
            [11],
            "main-post",
        ),
    )

    assert len(trace.transitions()) == 5
    assert [branch.transition_ids for branch in trace.branches()] == [
        (0, 1),
        (2,),
        (3,),
        (4,),
    ]
