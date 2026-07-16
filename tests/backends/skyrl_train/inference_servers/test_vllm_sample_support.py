from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from skyrl.backends.skyrl_train.inference_servers.vllm_server_actor import (
    _sample_support_from_flat_logprobs,
)

pytestmark = pytest.mark.vllm


def test_flat_logprobs_extracts_sampled_scores_and_support_rows():
    flat_logprobs = SimpleNamespace(
        token_ids=[7, 7, 8, 9, 4, 3, 4, 5],
        logprobs=[-0.1, -0.1, -0.2, -0.3, -0.4, -0.2, -0.4, -0.6],
    )

    sampled, support = _sample_support_from_flat_logprobs(flat_logprobs, top_k=3)

    assert sampled == [{"logprob": -0.1}, {"logprob": -0.4}]
    assert support == [[7, 8, 9], [3, 4, 5]]


def test_flat_logprobs_replaces_top_p_masked_candidates():
    flat_logprobs = SimpleNamespace(
        token_ids=[7, 7, 8, 9],
        logprobs=[-0.1, -0.1, -0.2, float("-inf")],
    )

    _, support = _sample_support_from_flat_logprobs(flat_logprobs, top_k=3)

    assert support == [[7, 8, -1]]
