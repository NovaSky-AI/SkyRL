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


def test_flat_logprobs_repairs_sampled_token_absent_from_support():
    # Three rows, top_k=3 (row_width=4):
    #   Row A: sampled id (100) absent from a fully-valid support row -> repair.
    #   Row B: sampled id (7) already present -> unchanged.
    #   Row C: sampled id (5) absent from a support row that has trailing -1 padding.
    top_k = 3
    flat_logprobs = SimpleNamespace(
        token_ids=[100, 8, 9, 10, 7, 7, 8, 9, 5, 6, 7, 8],
        logprobs=[
            -0.1,
            -0.2,
            -0.3,
            -0.4,  # row A: all valid
            -0.1,
            -0.1,
            -0.2,
            -0.3,  # row B: all valid
            -0.4,
            -0.5,
            -0.6,
            float("-inf"),  # row C: last col filtered -> padding
        ],
    )

    _, support = _sample_support_from_flat_logprobs(flat_logprobs, top_k=top_k)
    sampled_ids = [100, 7, 5]

    # (b) each row keeps width == top_k
    assert all(len(row) == top_k for row in support)

    # (a) every row's support now contains its sampled id
    for sampled_id, row in zip(sampled_ids, support):
        assert sampled_id in row

    # (c) the sampled id appears exactly once per repaired row (no duplicate)
    assert support[0].count(100) == 1
    assert support[2].count(5) == 1

    # (d) trailing -1 padding preserved on the padded row
    assert support[2][-1] == -1

    # (e) the unaffected row (sampled already present) is unchanged
    assert support[1] == [7, 8, 9]

    # Concrete expected repair: weakest (trailing) valid member overwritten.
    assert support[0] == [8, 9, 100]
    assert support[2] == [6, 5, -1]


def test_flat_logprobs_top_k_one_repairs_single_support_column():
    # top_k == 1 (row_width == 2): a single support column that must hold the sampled id.
    flat_logprobs = SimpleNamespace(
        token_ids=[42, 9],
        logprobs=[-0.1, -0.2],
    )

    _, support = _sample_support_from_flat_logprobs(flat_logprobs, top_k=1)

    assert support == [[42]]
