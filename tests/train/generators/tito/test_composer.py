"""CPU tests for converting exact TITO turns to GeneratorOutput."""

from skyrl.train.generators.base import TrajectoryID
from skyrl.train.generators.tito.composer import build_trace_generator_output
from skyrl.train.generators.tito.trace import Trace
from skyrl.train.generators.tito.types import ModelTurnResult, TraceOutcome


def _commit(trace, key, messages, prompt, indices, completion, content, reused=0, routed_experts=None):
    pending = trace.prepare_turn(messages, request_key=key)
    trace.commit(
        pending,
        ModelTurnResult(
            prompt_token_ids=tuple(prompt),
            prompt_message_indices=tuple(indices),
            reused_prefix_length=reused,
            completion_ids=tuple(completion),
            completion_logprobs=tuple(-0.1 for _ in completion),
            assistant_message={"role": "assistant", "content": content},
            stop_reason="stop",
            routed_experts=routed_experts,
        ),
    )


def test_linear_turns_become_one_training_row():
    trace = Trace()
    _commit(trace, "one", [{"role": "user", "content": "q"}], [1, 2], [0, -1], [3], "a")
    _commit(
        trace,
        "two",
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "tool", "content": "obs"},
        ],
        [1, 2, 3, 4, 5],
        [-1, -1, -1, 2, -1],
        [6],
        "b",
        reused=3,
    )
    output = build_trace_generator_output(
        [
            TraceOutcome(
                trace=trace,
                trajectory_id=TrajectoryID("instance", 0),
                reward=1.0,
                stop_reason="complete",
                generation_time=2.0,
                num_turns=2,
            )
        ],
        overlong_filtering=False,
        step_wise=False,
    )

    assert output["prompt_token_ids"] == [[1, 2]]
    assert output["response_ids"] == [[3, 4, 5, 6]]
    assert output["loss_masks"] == [[1, 0, 0, 1]]
    assert output["rewards"] == [1.0]


def test_discontinuity_uses_step_wise_rows_and_final_reward():
    trace = Trace()
    _commit(trace, "one", [{"role": "user", "content": "q"}], [1, 2], [0, -1], [3], "a")
    _commit(
        trace,
        "two",
        [{"role": "user", "content": "summary"}],
        [9, 10],
        [0, -1],
        [11],
        "b",
    )
    outcome = TraceOutcome(
        trace=trace,
        trajectory_id=TrajectoryID("instance", 0),
        reward=0.5,
        stop_reason="complete",
        generation_time=2.0,
        num_turns=2,
    )

    output = build_trace_generator_output([outcome], overlong_filtering=False, step_wise=True)

    assert output["prompt_token_ids"] == [[1, 2], [9, 10]]
    assert output["rewards"] == [0.0, 0.5]
    assert output["is_last_step"] == [False, True]
    assert output["trajectory_ids"] == [outcome.trajectory_id, outcome.trajectory_id]


def test_discontinuity_is_group_masked_without_step_wise():
    trace = Trace()
    _commit(trace, "one", [{"role": "user", "content": "q"}], [1, 2], [0, -1], [3], "a")
    _commit(
        trace,
        "two",
        [{"role": "user", "content": "summary"}],
        [9, 10],
        [0, -1],
        [11],
        "b",
    )
    outcome = TraceOutcome(
        trace=trace,
        trajectory_id=TrajectoryID("instance", 0),
        reward=0.5,
        stop_reason="complete",
        generation_time=2.0,
        num_turns=2,
    )

    output = build_trace_generator_output([outcome], overlong_filtering=False, step_wise=False)

    assert output["prompt_token_ids"] == [[0]]
    assert output["response_ids"] == [[0]]
    assert output["loss_masks"] == [[0]]
    assert output["rewards"] == [0.0]


def test_routed_experts_align_with_prompt_deltas():
    trace = Trace()
    _commit(
        trace,
        "one",
        [{"role": "user", "content": "q"}],
        [1, 2],
        [0, -1],
        [3],
        "a",
        routed_experts=(((1, 2),), ((3, 4),), ((5, 6),)),
    )
    _commit(
        trace,
        "two",
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
            {"role": "tool", "content": "obs"},
        ],
        [1, 2, 3, 4, 5],
        [-1, -1, -1, 2, -1],
        [6],
        "b",
        reused=3,
        routed_experts=(
            ((10, 11),),
            ((12, 13),),
            ((14, 15),),
            ((16, 17),),
            ((18, 19),),
            ((20, 21),),
        ),
    )
    outcome = TraceOutcome(
        trace=trace,
        trajectory_id=TrajectoryID("instance", 0),
        reward=1.0,
        stop_reason="complete",
        generation_time=2.0,
        num_turns=2,
    )

    output = build_trace_generator_output([outcome], overlong_filtering=False, step_wise=False)

    assert output["rollout_expert_indices"] == [
        [
            [[10, 11]],
            [[12, 13]],
            [[14, 15]],
            [[16, 17]],
            [[18, 19]],
            [[20, 21]],
        ]
    ]
