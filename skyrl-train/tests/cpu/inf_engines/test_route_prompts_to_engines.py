"""
Test for route_prompts_to_engines function that routes prompts to inference engines in inference engine client.

Run with:
uv run --isolated --extra dev pytest tests/cpu/inf_engines/test_route_prompts_to_engines.py
"""

from unittest.mock import patch

import pytest

from skyrl_train.inference_engines.utils import route_prompts_to_engines, hash_with_sha256


def test_single_prompt_no_trajectory_random_engine():
    # Force deterministic random routing to engine index 1
    with patch("random.randint", return_value=1):
        mapping = route_prompts_to_engines(num_prompts=1, num_inference_engines=4, session_ids=None)
    assert mapping == {1: [0]}


def test_batched_even_split_exact_multiple():
    # 4 prompts, 2 engines => [0,1] and [2,3]
    num_prompts = 4
    num_engines = 2
    mapping = route_prompts_to_engines(num_prompts=num_prompts, num_inference_engines=num_engines, session_ids=None)
    assert mapping == {0: [0, 1], 1: [2, 3]}


def test_batched_uneven_split():
    # 5 prompts, 2 engines => ceil(5/2)=3 => [0,1,2] and [3,4]
    mapping = route_prompts_to_engines(num_prompts=5, num_inference_engines=2, session_ids=None)
    assert mapping == {0: [0, 1, 2], 1: [3, 4]}

    # 5 prompts, 3 engines => ceil(5/3)=2 => [0,1] and [2,3] and [4]
    mapping = route_prompts_to_engines(num_prompts=5, num_inference_engines=3, session_ids=None)
    assert mapping == {0: [0, 1], 1: [2, 3], 2: [4]}

    # 5 prompts, 4 engines => ceil(5/4)=2 => [0,1] and [2,3] and [4]
    mapping = route_prompts_to_engines(num_prompts=5, num_inference_engines=4, session_ids=None)
    assert mapping == {0: [0, 1], 1: [2, 3], 2: [4]}

    # 129 prompts, 4 engines => ceil(129/4)=33 => [0,1,2,...,32] and [33,34,35,...,65] and [66,67,68,...,99] and [100,101,102,...,128]
    mapping = route_prompts_to_engines(num_prompts=129, num_inference_engines=4, session_ids=None)
    assert mapping == {0: list(range(33)), 1: list(range(33, 66)), 2: list(range(66, 99)), 3: list(range(99, 129))}


def test_batched_more_engines_than_prompts():
    # 2 prompts, 4 engines => size=1 => {0:[0], 1:[1]}
    mapping = route_prompts_to_engines(num_prompts=2, num_inference_engines=4, session_ids=None)
    assert mapping == {0: [0], 1: [1]}


def test_with_session_ids_grouping_and_partition():
    num_engines = 4
    # Ensure same session IDs route to the same engine index
    sids = ["A", "A", "B", "C", "B"]
    # hash A ends in 45, B ends in 44, C ends in 69, with % 4 they become 1, 0, 1
    engine_idx = [hash_with_sha256(sid) % num_engines for sid in sids]  # what we do in route_prompts_to_engines
    assert engine_idx == [1, 1, 0, 1, 0]
    mapping = route_prompts_to_engines(num_prompts=5, num_inference_engines=num_engines, session_ids=sids)

    assert mapping == {1: [0, 1, 3], 0: [2, 4]}


def test_validation_errors():
    # num_prompts must be > 0
    with pytest.raises(AssertionError):
        route_prompts_to_engines(num_prompts=0, num_inference_engines=1, session_ids=None)

    # num_inference_engines must be > 0
    with pytest.raises(AssertionError):
        route_prompts_to_engines(num_prompts=1, num_inference_engines=0, session_ids=None)

    # session_ids length must match
    with pytest.raises(AssertionError):
        route_prompts_to_engines(num_prompts=2, num_inference_engines=1, session_ids=["x"])  # len 1 != 2

    # session_ids type checking
    with pytest.raises(AssertionError):
        route_prompts_to_engines(num_prompts=2, num_inference_engines=1, session_ids=[1, 2.0])  # float invalid

    # No error
    route_prompts_to_engines(num_prompts=2, num_inference_engines=1, session_ids=[1, 2])
    route_prompts_to_engines(num_prompts=2, num_inference_engines=1, session_ids=None)
    route_prompts_to_engines(num_prompts=1, num_inference_engines=1, session_ids=None)
