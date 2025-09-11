"""
Test for postprocess_completion_request function that postprocesses completion requests in inference engine client.

Run with:
uv run --isolated --extra dev pytest tests/cpu/inf_engines/test_utils.py
"""

from http import HTTPStatus

from skyrl_train.inference_engines.utils import postprocess_completion_request
from skyrl_train.inference_engines.inference_engine_client_http_endpoint import (
    ErrorResponse,
)

# -------------------------------------------
# tests for postprocess_completion_request
# --------------------------------------------


def test_postprocess_single_string_no_trajectory_id():
    prompt = "hello world"
    traj, processed = postprocess_completion_request(prompt, None)
    assert traj is None
    assert isinstance(processed, list)
    assert processed == [prompt]


def test_postprocess_single_string_scalar_trajectory_id():
    prompt = "hello world"
    traj, processed = postprocess_completion_request(prompt, 123)
    assert traj == [123]
    assert processed == [prompt]


def test_postprocess_single_string_list_trajectory_id_singleton():
    prompt = "hello world"
    traj, processed = postprocess_completion_request(prompt, ["abc"])  # accepts str ids
    assert traj == ["abc"]
    assert processed == [prompt]


def test_postprocess_single_string_list_trajectory_id_wrong_len():
    prompt = "hello world"
    traj, processed = postprocess_completion_request(prompt, [1, 2])
    assert isinstance(traj, ErrorResponse)
    assert processed == [prompt]
    assert traj.error.code == HTTPStatus.BAD_REQUEST.value


def test_postprocess_single_token_ids_no_trajectory_id():
    prompt = [1, 2, 3]
    traj, processed = postprocess_completion_request(prompt, None)
    assert traj is None
    assert processed == [prompt]


def test_postprocess_single_token_ids_scalar_trajectory_id():
    prompt = [1, 2, 3]
    traj, processed = postprocess_completion_request(prompt, 7)
    assert traj == [7]
    assert processed == [prompt]


def test_postprocess_single_token_ids_list_trajectory_id_singleton():
    prompt = [1, 2, 3]
    traj, processed = postprocess_completion_request(prompt, [8])
    assert traj == [8]
    assert processed == [prompt]


def test_postprocess_single_token_ids_list_trajectory_id_wrong_len():
    prompt = [1, 2, 3]
    traj, processed = postprocess_completion_request(prompt, [8, 9])
    assert isinstance(traj, ErrorResponse)
    assert processed == [prompt]
    assert traj.error.code == HTTPStatus.BAD_REQUEST.value


def test_postprocess_batched_token_ids_no_trajectory_id():
    prompt = [[1, 2], [3, 4, 5]]
    traj, processed = postprocess_completion_request(prompt, None)
    assert traj is None
    assert processed is prompt  # unchanged shape


def test_postprocess_batched_token_ids_with_matching_trajectory_ids():
    prompt = [[1, 2], [3, 4, 5]]
    traj, processed = postprocess_completion_request(prompt, ["a", "b"])  # accepts str ids too
    assert traj == ["a", "b"]
    assert processed is prompt


def test_postprocess_batched_token_ids_with_wrong_trajectory_ids_length():
    prompt = [[1, 2], [3, 4, 5]]
    traj, processed = postprocess_completion_request(prompt, [1])
    assert isinstance(traj, ErrorResponse)
    assert processed is prompt
    assert traj.error.code == HTTPStatus.BAD_REQUEST.value


def test_postprocess_batched_strings_no_trajectory_id():
    prompt = ["p0", "p1"]
    traj, processed = postprocess_completion_request(prompt, None)
    assert traj is None
    assert processed is prompt


def test_postprocess_batched_strings_with_matching_trajectory_ids():
    prompt = ["p0", "p1", "p2"]
    traj, processed = postprocess_completion_request(prompt, [10, 11, 12])
    assert traj == [10, 11, 12]
    assert processed is prompt


def test_postprocess_batched_strings_with_wrong_trajectory_ids_length():
    prompt = ["p0", "p1", "p2"]
    traj, processed = postprocess_completion_request(prompt, [10, 11])
    assert isinstance(traj, ErrorResponse)
    assert processed is prompt
    assert traj.error.code == HTTPStatus.BAD_REQUEST.value


def test_postprocess_batched_strings_with_wrong_trajectory_ids_length_2():
    prompt = ["p0", "p1", "p2"]
    traj, processed = postprocess_completion_request(prompt, 10)
    assert isinstance(traj, ErrorResponse)
    assert processed is prompt
    assert traj.error.code == HTTPStatus.BAD_REQUEST.value
