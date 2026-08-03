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
    first = trace.prepare_turn(
        [{"role": "user", "content": "question"}],
        request_key="turn-1",
    )
    trace.commit(first, _result([10, 11, 12], [0, 0, -1], [20, 21], "answer"))

    second_messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
        {"role": "tool", "content": "observation", "tool_call_id": "call-1"},
    ]
    second = trace.prepare_turn(second_messages, request_key="turn-2")
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
    assert [node.token_ids for node in trace.nodes()] == [
        (10, 11),
        (12, 20, 21),
        (30,),
        (31, 40),
    ]
    assert trace.nodes()[-1].sampled_mask == (False, True)


def test_full_render_token_mismatch_creates_new_exact_path():
    trace = Trace()
    first = trace.prepare_turn([{"role": "user", "content": "same"}], request_key="turn-1")
    trace.commit(first, _result([1, 2], [0, -1], [3], "answer"))

    rewritten = trace.prepare_turn(
        [
            {"role": "user", "content": "same"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "next"},
        ],
        request_key="turn-2",
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
    assert trace.nodes()[2].parent_id is None
    assert trace.nodes()[2].token_ids == (9, 2)


def test_identical_request_key_is_idempotent():
    trace = Trace()
    pending = trace.prepare_turn([{"role": "user", "content": "q"}], request_key="same")
    result = _result([1, 2], [0, -1], [3], "a")
    first = trace.commit(pending, result)
    second = trace.commit(pending, result)

    assert first == second
    assert len(trace.committed_turns()) == 1


def test_stale_pending_turn_is_rejected():
    trace = Trace()
    stale = trace.prepare_turn([{"role": "user", "content": "one"}], request_key="stale")
    current = trace.prepare_turn([{"role": "user", "content": "two"}], request_key="current")
    trace.commit(current, _result([1, 2], [0, -1], [3], "a"))

    with pytest.raises(RuntimeError, match="Stale pending turn"):
        trace.commit(stale, _result([4, 5], [0, -1], [6], "b"))


def test_sealed_trace_rejects_new_work():
    trace = Trace()
    trace.seal()

    with pytest.raises(RuntimeError, match="sealed trace"):
        trace.prepare_turn([{"role": "user", "content": "q"}], request_key="turn")
