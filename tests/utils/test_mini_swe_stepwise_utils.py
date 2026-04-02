"""Tests for the Mini-SWE step-wise exact-token helpers.

uv run --isolated --extra dev --extra skyrl-train -- pytest tests/utils/test_mini_swe_stepwise_utils.py
"""

import pytest

from examples.train.mini_swe_agent.stepwise_utils import (
    build_stepwise_outputs_from_messages,
    build_stepwise_sampling_params,
)
from skyrl.train.generators.base import TrajectoryID


def test_build_stepwise_sampling_params_requests_exact_tokens_without_mutating_input():
    base_sampling_params = {
        "max_tokens": 128,
        "temperature": 0.7,
        "extra_body": {"skip_special_tokens": True},
    }

    sampling_params = build_stepwise_sampling_params(
        base_sampling_params,
        TrajectoryID(instance_id="instance-1", repetition_id=3),
    )

    assert sampling_params["logprobs"] is True
    assert sampling_params["top_logprobs"] == 1
    assert sampling_params["session_id"] == "instance-1_3"
    assert sampling_params["extra_body"] == {
        "skip_special_tokens": True,
        "return_token_ids": True,
        "return_tokens_as_token_ids": True,
    }
    assert base_sampling_params == {
        "max_tokens": 128,
        "temperature": 0.7,
        "extra_body": {"skip_special_tokens": True},
    }


def test_build_stepwise_outputs_uses_exact_token_ids_and_marks_last_step():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "first",
            "extra": {
                "response": {
                    "prompt_token_ids": [10, 11],
                    "choices": [
                        {
                            "token_ids": [101, 102],
                            "finish_reason": "stop",
                            "logprobs": {
                                "content": [
                                    {"token": 101, "logprob": -0.1},
                                    {"token": 102, "logprob": -0.2},
                                ]
                            },
                        }
                    ],
                }
            },
        },
        {
            "role": "assistant",
            "content": "second",
            "extra": {
                "response": {
                    "prompt_token_ids": [10, 11, 12],
                    "choices": [
                        {
                            "token_ids": [201, 202, 203],
                            "finish_reason": "stop",
                            "logprobs": {
                                "content": [
                                    {"token": 201, "logprob": -0.3},
                                    {"token": 202, "logprob": -0.4},
                                    {"token": 203, "logprob": -0.5},
                                ]
                            },
                        }
                    ],
                }
            },
        },
    ]

    outputs = build_stepwise_outputs_from_messages(
        messages=messages,
        reward=2.5,
        trajectory_id=TrajectoryID(instance_id="instance-2", repetition_id=1),
        max_seq_len=32,
    )

    assert len(outputs) == 2

    assert outputs[0].prompt_token_ids == [10, 11]
    assert outputs[0].response_ids == [101, 102]
    assert outputs[0].loss_mask == [1, 1]
    assert outputs[0].rollout_logprobs == [-0.1, -0.2]
    assert outputs[0].rewards == [0.0, 0.0]
    assert outputs[0].stop_reason == "stop"
    assert outputs[0].is_last_step is False

    assert outputs[1].prompt_token_ids == [10, 11, 12]
    assert outputs[1].response_ids == [201, 202, 203]
    assert outputs[1].loss_mask == [1, 1, 1]
    assert outputs[1].rollout_logprobs == [-0.3, -0.4, -0.5]
    assert outputs[1].rewards == [0.0, 0.0, 2.5]
    assert outputs[1].stop_reason == "stop"
    assert outputs[1].is_last_step is True
    assert outputs[1].trajectory_id.to_string() == "instance-2_1"


def test_build_stepwise_outputs_falls_back_to_token_ids_from_logprobs():
    messages = [
        {
            "role": "assistant",
            "content": "fallback",
            "extra": {
                "response": {
                    "choices": [
                        {
                            "prompt_token_ids": [1, 2, 3],
                            "finish_reason": "stop",
                            "logprobs": {
                                "content": [
                                    {"token": "token_id:321", "logprob": -0.7},
                                    {"token": "token_id:654", "logprob": -0.8},
                                ]
                            },
                        }
                    ]
                }
            },
        }
    ]

    outputs = build_stepwise_outputs_from_messages(
        messages=messages,
        reward=1.0,
        trajectory_id=TrajectoryID(instance_id="instance-3", repetition_id=0),
        max_seq_len=16,
    )

    assert len(outputs) == 1
    assert outputs[0].prompt_token_ids == [1, 2, 3]
    assert outputs[0].response_ids == [321, 654]
    assert outputs[0].rollout_logprobs == [-0.7, -0.8]
    assert outputs[0].rewards == [0.0, 1.0]


def test_build_stepwise_outputs_truncates_overlong_response_and_updates_stop_reason():
    messages = [
        {
            "role": "assistant",
            "content": "too long",
            "extra": {
                "response": {
                    "prompt_token_ids": [7, 8, 9],
                    "choices": [
                        {
                            "token_ids": [40, 41, 42, 43],
                            "finish_reason": "stop",
                            "logprobs": {
                                "content": [
                                    {"token": 40, "logprob": -0.1},
                                    {"token": 41, "logprob": -0.2},
                                    {"token": 42, "logprob": -0.3},
                                    {"token": 43, "logprob": -0.4},
                                ]
                            },
                        }
                    ],
                }
            },
        }
    ]

    outputs = build_stepwise_outputs_from_messages(
        messages=messages,
        reward=3.0,
        trajectory_id=TrajectoryID(instance_id="instance-4", repetition_id=0),
        max_seq_len=5,
    )

    assert len(outputs) == 1
    assert outputs[0].response_ids == [40, 41]
    assert outputs[0].loss_mask == [1, 1]
    assert outputs[0].rollout_logprobs == [-0.1, -0.2]
    assert outputs[0].rewards == [0.0, 3.0]
    assert outputs[0].stop_reason == "length"


def test_build_stepwise_outputs_requires_raw_response_metadata():
    with pytest.raises(ValueError, match="raw LiteLLM response"):
        build_stepwise_outputs_from_messages(
            messages=[{"role": "assistant", "content": "missing"}],
            reward=0.0,
            trajectory_id=TrajectoryID(instance_id="instance-5", repetition_id=0),
            max_seq_len=8,
        )
