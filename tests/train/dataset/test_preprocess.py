"""
uv run --isolated --extra dev pytest tests/train/dataset/test_preprocess.py
"""

import logging
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from skyrl.backends.skyrl_train.utils.routed_experts import (
    ROUTED_EXPERT_DTYPES,
    RoutedExpertTrace,
)
from skyrl.backends.skyrl_train.utils.sample_support import (
    SAMPLE_SUPPORT_DTYPE,
    SAMPLE_SUPPORT_PADDING,
    SAMPLE_SUPPORT_TORCH_DTYPE,
)
from skyrl.train.dataset import parallel_fill
from skyrl.train.dataset.preprocess import (
    ROUTED_EXPERT_TORCH_DTYPES,
    build_sample_support,
    convert_prompts_responses_to_batch_tensors,
    make_router_padding_mask,
)


# NOTE (sumanthrh): the tests in this file are hardcoded to use the below character-level tokenizer
@pytest.fixture
def tokenizer():
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 2

    # encode("abc") -> [97, 98, 99]
    def fake_encode(text):
        if isinstance(text, list):
            return [fake_encode(t) for t in text]
        return [ord(c) for c in text]

    mock_tokenizer.encode.side_effect = fake_encode

    # tokenizer("abc") -> {"input_ids": [...], "attention_mask": [...]}
    def fake_tokenizer_call(text, **kwargs):
        if isinstance(text, list):
            dicts = [fake_tokenizer_call(t, **kwargs) for t in text]
            return {
                "input_ids": [d["input_ids"] for d in dicts],
                "attention_mask": [d["attention_mask"] for d in dicts],
            }
        ids = [ord(c) for c in text]
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
        }

    mock_tokenizer.side_effect = fake_tokenizer_call

    def fake_tokenizer_decode(ids, **kwargs):
        return "".join([chr(i) for i in ids])

    mock_tokenizer.decode.side_effect = fake_tokenizer_decode

    def fake_tokenizer_decode_list(ids, **kwargs):
        return [fake_tokenizer_decode(i) for i in ids]

    mock_tokenizer.batch_decode.side_effect = fake_tokenizer_decode_list

    return mock_tokenizer


def test_router_padding_mask_marks_left_padding_and_uncaptured_suffix():
    attention_mask = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])

    mask = make_router_padding_mask(attention_mask, [2, 4])

    assert mask.tolist() == [[True, False, False, True], [False, False, False, False]]


def test_router_padding_mask_marks_the_tail_a_multi_turn_trace_never_captured():
    """A trace reports only the rows it captured, so the mask covers exactly the tail that
    collation dummy-fills. Two turns over a 3-token prompt and a 2-token observation capture
    8 rows of a 10-token sequence: no decode forward followed the last sampled token, and the
    synthetic EOS was never evaluated at all.
    """
    trace = RoutedExpertTrace()
    trace.record_generation(
        prompt_token_count=3,
        generated_token_count=2,
        routed_experts=np.zeros((4, 2, 2), dtype=np.int16),
    )
    trace.record_generation(
        prompt_token_count=7,
        generated_token_count=2,
        routed_experts=np.zeros((4, 2, 2), dtype=np.int16),
    )
    # The last two tokens are loss-masked, which is what the trace's own guard enforces.
    routes = trace.finalize(token_count=10, loss_mask=[0, 0, 0, 1, 1, 0, 0, 1, 1, 0])
    assert routes.shape[0] == 8

    attention_mask = torch.tensor([[0, 0] + [1] * 10])

    mask = make_router_padding_mask(attention_mask, [routes.shape[0]])

    assert mask.tolist() == [[True, True] + [False] * 8 + [True, True]]


def test_router_padding_mask_marks_the_last_token_for_batched_routes():
    """``generate_batched`` returns ``seq_len - 1`` rows, one short of its sequence."""
    attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1]])

    mask = make_router_padding_mask(attention_mask, [4])

    assert mask.tolist() == [[True, True, False, False, False, False, True]]


def test_routed_expert_tensor_uses_unique_dummy_routes(tokenizer):
    routes = [
        np.asarray(
            [
                [[2, 3], [4, 5]],
                [[6, 7], [0, 1]],
            ],
            dtype=np.uint8,
        ),
        np.asarray(
            [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 0]],
                [[2, 4], [6, 7]],
            ],
            dtype=np.uint8,
        ),
    ]

    *_, routed, _ = convert_prompts_responses_to_batch_tensors(
        tokenizer.pad_token_id,
        prompts=[[10], [20]],
        responses=[[11, 12], [21, 22]],
        rewards=[[0.0, 0.0], [0.0, 0.0]],
        loss_masks=[[1, 1], [1, 1]],
        rollout_expert_indices=routes,
    )

    assert routed.values.shape == (6, 2, 2)
    assert routed.cu_seqlens.tolist() == [0, 3, 6]
    assert routed.dtype == torch.uint8
    assert routed.segment(0)[2].tolist() == [[0, 1], [0, 1]]


@pytest.mark.parametrize(
    ("max_expert_id", "source_dtype", "expected_dtype"),
    [(2**8, np.int16, torch.int16), (2**15, np.int32, torch.int32)],
)
def test_routed_expert_tensor_promotes_mixed_batch_dtype(
    tokenizer,
    max_expert_id,
    source_dtype,
    expected_dtype,
):
    routes = [
        np.asarray([[[1, 2]]], dtype=np.uint8),
        np.asarray([[[max_expert_id, max_expert_id + 1]]], dtype=source_dtype),
    ]

    *_, routed, _ = convert_prompts_responses_to_batch_tensors(
        tokenizer.pad_token_id,
        prompts=[[10], [20]],
        responses=[[11], [21]],
        rewards=[[0.0], [0.0]],
        loss_masks=[[1], [1]],
        rollout_expert_indices=routes,
    )

    assert routed.dtype == expected_dtype
    assert routed.segment(1)[0].tolist() == [[max_expert_id, max_expert_id + 1]]


def test_routed_expert_tensor_accepts_read_only_arrays(tokenizer):
    routes = np.asarray([[[1, 2]], [[3, 4]]], dtype=np.uint8)
    routes.flags.writeable = False

    *_, routed, _ = convert_prompts_responses_to_batch_tensors(
        tokenizer.pad_token_id,
        prompts=[[10]],
        responses=[[11]],
        rewards=[[0.0]],
        loss_masks=[[1]],
        rollout_expert_indices=[routes],
    )

    assert routed.dtype == torch.uint8
    assert routed.segment(0).tolist() == [[[1, 2]], [[3, 4]]]


def test_routed_expert_tensor_rejects_nested_lists(tokenizer):
    with pytest.raises(TypeError, match="NumPy arrays"):
        convert_prompts_responses_to_batch_tensors(
            tokenizer.pad_token_id,
            prompts=[[10]],
            responses=[[11]],
            rewards=[[0.0]],
            loss_masks=[[1]],
            rollout_expert_indices=[[[[1, 2]], [[3, 4]]]],
        )


def test_routed_expert_tensor_accepts_non_contiguous_arrays(tokenizer):
    # Every other expert column, which leaves a non-contiguous view.
    base = np.asarray([[[1, 9, 2, 9]], [[3, 9, 4, 9]]], dtype=np.uint8)
    routes = base[:, :, ::2]
    assert not routes.flags.c_contiguous

    *_, routed, _ = convert_prompts_responses_to_batch_tensors(
        tokenizer.pad_token_id,
        prompts=[[10]],
        responses=[[11]],
        rewards=[[0.0]],
        loss_masks=[[1]],
        rollout_expert_indices=[routes],
    )

    assert routed.dtype == torch.uint8
    assert routed.segment(0).tolist() == [[[1, 2]], [[3, 4]]]


@pytest.mark.parametrize("dtype", [np.uint16, np.int64])
def test_routed_expert_tensor_rejects_non_canonical_dtypes(tokenizer, dtype):
    """The sender compacts to the canonical dtype, so collation validates instead of rescanning."""
    routes = np.asarray([[[1, 2]], [[3, 4]]], dtype=dtype)

    with pytest.raises(ValueError, match="canonical routed-expert dtype"):
        convert_prompts_responses_to_batch_tensors(
            tokenizer.pad_token_id,
            prompts=[[10]],
            responses=[[11]],
            rewards=[[0.0]],
            loss_masks=[[1]],
            rollout_expert_indices=[routes],
        )


def test_routed_expert_tensor_keeps_the_sender_dtype_after_truncation(tokenizer):
    # The dropped trailing value required int16 on the sender.
    routes = np.asarray([[[1, 2]], [[3, 4]], [[300, 5]]], dtype=np.int16)

    *_, routed, _ = convert_prompts_responses_to_batch_tensors(
        tokenizer.pad_token_id,
        prompts=[[10]],
        responses=[[11]],
        rewards=[[0.0]],
        loss_masks=[[1]],
        rollout_expert_indices=[routes[:2]],
    )

    assert routed.dtype == torch.int16
    assert routed.segment(0).tolist() == [[[1, 2]], [[3, 4]]]


@pytest.mark.parametrize(
    ("dtype", "expert_id", "expect_warning"),
    [(np.int16, 300, False), (np.int32, 2**16, True)],
)
def test_routed_expert_tensor_warns_only_on_an_int32_batch(tokenizer, caplog, dtype, expert_id, expect_warning):
    routes = np.asarray([[[expert_id, expert_id + 1]]], dtype=dtype)

    with caplog.at_level(logging.WARNING, logger="skyrl.train.dataset.preprocess"):
        *_, routed, _ = convert_prompts_responses_to_batch_tensors(
            tokenizer.pad_token_id,
            prompts=[[10]],
            responses=[[11]],
            rewards=[[0.0]],
            loss_masks=[[1]],
            rollout_expert_indices=[routes],
        )

    assert routed.dtype == ROUTED_EXPERT_TORCH_DTYPES[np.dtype(dtype)]
    assert ("not compacting its routes" in caplog.text) is expect_warning


def test_routed_expert_torch_dtype_map_covers_the_canonical_dtypes():
    assert set(ROUTED_EXPERT_TORCH_DTYPES) == set(ROUTED_EXPERT_DTYPES)


def _numpy_padded_routes(
    routes: List[np.ndarray],
    prompts: List[List[int]],
    responses: List[List[int]],
) -> np.ndarray:
    """Reference NumPy implementation of route collation."""
    max_total = max(len(prompt) + len(response) for prompt, response in zip(prompts, responses))
    num_layers, topk = routes[0].shape[1:]
    batch_dtype = max((sample.dtype for sample in routes), key=lambda dtype: dtype.itemsize)
    padded = np.empty((len(routes), max_total, num_layers, topk), dtype=batch_dtype)
    padded[...] = np.arange(topk, dtype=batch_dtype)
    for index, sample in enumerate(routes):
        left_pad = max_total - (len(prompts[index]) + len(responses[index]))
        padded[index, left_pad : left_pad + sample.shape[0]] = sample
    return padded


def test_routed_expert_tensor_is_bit_identical_to_numpy_collation(tokenizer):
    """Packed collation matches the real rows of the padded NumPy reference."""
    prompts = [[1, 2], [3, 4, 5, 6]]
    responses = [[10, 11, 12], [20, 21]]
    num_layers, topk = 2, 3
    # Sample 0 has 5 tokens but only 4 captured route rows, so its segment pads at the end.
    routes = [
        np.arange(4 * num_layers * topk, dtype=np.uint8).reshape(4, num_layers, topk),
        (np.arange(6 * num_layers * topk, dtype=np.int16) + 300).reshape(6, num_layers, topk),
    ]

    *_, routed, _ = convert_prompts_responses_to_batch_tensors(
        tokenizer.pad_token_id,
        prompts,
        responses,
        rewards=[[0.0] * 3, [0.0] * 2],
        loss_masks=[[1] * 3, [1] * 2],
        rollout_expert_indices=routes,
    )

    padded = _numpy_padded_routes(routes, prompts, responses)
    real_rows = np.concatenate(
        [
            padded[index, padded.shape[1] - (len(prompt) + len(response)) :]
            for index, (prompt, response) in enumerate(zip(prompts, responses))
        ]
    )
    assert routed.dtype == torch.int16
    assert routed.cu_seqlens.tolist() == [0, 5, 11]
    assert torch.equal(routed.values, torch.from_numpy(real_rows))
    # Padding routes use distinct experts for Megatron's dropless dispatcher.
    padding_row = [[0, 1, 2]] * num_layers
    assert routed.segment(0)[4].tolist() == padding_row
    assert not torch.equal(routed.segment(0)[4], torch.zeros_like(routed.segment(0)[4]))


def test_convert_prompts_responses_to_batch_tensors_exact(tokenizer):
    """
    Test with inputs of exact lengths.

    | [PAD]  [PAD]  [PAD]  [PAD]  prompt prompt prompt respon respon respon |
    | prompt prompt prompt prompt prompt respon respon respon respon respon |
                                         |<------- max_response_len ------->|
    """
    # prompts: "abc" (3 tokens), "12345" (5 tokens)
    # outputs: "def" (3 tokens), "67890" (5 tokens)
    prompts = ["abc", "12345"]
    outputs = ["def", "67890"]
    prompts = tokenizer(prompts)["input_ids"]
    outputs = tokenizer(outputs)["input_ids"]

    loss_masks = [[1, 1, 0], [1, 1, 1, 0, 0]]
    rewards = [torch.tensor([0, 1, 0]), torch.tensor([1, 0, 0, 0, 0])]

    sequences, attention_mask, response_mask, ret_rewards, ret_loss_masks, ret_log_probs, _, _ = (
        convert_prompts_responses_to_batch_tensors(
            tokenizer.pad_token_id,
            prompts,
            outputs,
            rewards,
            loss_masks,
        )
    )

    # max_total = max(3+3, 5+5) = 10, max_response = 5
    assert sequences.shape[0] == len(prompts)
    assert sequences.shape == (2, 10)
    assert response_mask.shape == ret_loss_masks.shape
    # Response data is RIGHT-ALIGNED within (batch, max_response)
    # Sample 0: response len=3, so 2 leading zeros then 3 values
    assert torch.equal(ret_loss_masks[0], torch.tensor([0, 0, 1, 1, 0]))
    assert torch.equal(ret_loss_masks[1], torch.tensor([1, 1, 1, 0, 0]))
    assert torch.equal(ret_rewards[0], torch.tensor([0, 0, 0, 1, 0]))
    assert torch.equal(ret_rewards[1], torch.tensor([1, 0, 0, 0, 0]))
    # max_total=10: sample 0 has total=6, so 4 left-pads; sample 1 has total=10, no padding
    assert torch.equal(attention_mask[0], torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]))
    assert torch.equal(attention_mask[1], torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))


def test_convert_prompts_responses_to_batch_tensors_different_lengths(tokenizer):
    # Test with inputs of different lengths
    # "Short" = 5 tokens, "This is a longer prompt" = 23 tokens
    # "Long response here" = 18 tokens, "Short" = 5 tokens
    prompts = ["Short", "This is a longer prompt"]
    outputs = ["Long response here", "Short"]
    prompts = tokenizer(prompts)["input_ids"]
    outputs = tokenizer(outputs)["input_ids"]
    rewards = [torch.tensor([1.0, 0.5, 0.3]), torch.tensor([0.8])]
    loss_masks = [[1, 1, 1], [1]]

    sequences, attention_mask, response_mask, ret_rewards, ret_loss_masks, ret_log_probs, _, _ = (
        convert_prompts_responses_to_batch_tensors(
            tokenizer.pad_token_id,
            prompts,
            outputs,
            rewards,
            loss_masks,
        )
    )

    max_response_len = max([len(output) for output in outputs])
    # max_total = max(5+18, 23+5) = 28
    max_total = max(len(p) + len(r) for p, r in zip(prompts, outputs))

    # Check shapes
    assert sequences.shape == (2, max_total)
    assert attention_mask.shape == sequences.shape
    assert response_mask.shape == (2, max_response_len)
    assert ret_rewards.shape == (2, max_response_len)
    assert ret_loss_masks.shape == (2, max_response_len)

    # Unified left-padding: shorter total gets left-padded
    # Sample 0: total=23, pad=28-23=5 left pads
    assert sequences[0, 0] == tokenizer.pad_token_id
    assert sequences[1, 0] != tokenizer.pad_token_id
    # All sequences end with real tokens (response at end), no right padding
    assert sequences[0, -1] != tokenizer.pad_token_id
    assert sequences[1, -1] != tokenizer.pad_token_id


def test_convert_prompts_responses_to_batch_tensors_empty_input(tokenizer):
    # Test with empty input
    prompts = []
    outputs = []
    rewards = []
    loss_masks = []

    with pytest.raises(AssertionError):
        convert_prompts_responses_to_batch_tensors(
            tokenizer.pad_token_id,
            prompts,
            outputs,
            rewards,
            loss_masks,
        )


def test_convert_prompts_responses_to_batch_tensors_mismatched_lengths(tokenizer):
    # Test with mismatched input lengths
    prompts = ["Hello", "World"]
    outputs = ["Response"]
    prompts = tokenizer(prompts)["input_ids"]
    outputs = tokenizer(outputs)["input_ids"]
    rewards = [torch.tensor([1.0])]
    loss_masks = [[1]]

    with pytest.raises(AssertionError):
        convert_prompts_responses_to_batch_tensors(
            tokenizer.pad_token_id,
            prompts,
            outputs,
            rewards,
            loss_masks,
        )


# ---------------------------------------------------------------------------
# Unified padding layout tests
# ---------------------------------------------------------------------------


def test_unified_left_padding_layout(tokenizer):
    """Sequences are laid out as [PAD ... PROMPT RESPONSE] with all padding on the left."""
    # Sample 0: prompt=[1,2], response=[10,11,12] -> total=5
    # Sample 1: prompt=[3,4,5,6], response=[20,21] -> total=6
    # max_total=6, max_response=3
    prompts = [[1, 2], [3, 4, 5, 6]]
    responses = [[10, 11, 12], [20, 21]]
    rewards = [[0.0] * 3, [0.0] * 2]
    loss_masks = [[1] * 3, [1] * 2]

    seq, attn, action, rew, lm, _, _, _ = convert_prompts_responses_to_batch_tensors(
        tokenizer.pad_token_id,
        prompts,
        responses,
        rewards,
        loss_masks,
    )
    assert seq.shape == (2, 6)

    # Sample 0: pad=1, then [1,2,10,11,12]
    assert seq[0].tolist() == [0, 1, 2, 10, 11, 12]
    assert attn[0].tolist() == [0, 1, 1, 1, 1, 1]
    # Response ends at the end of the sequence (no right-padding in sequences)
    assert seq[0, -1] == 12

    # Sample 1: no pad, [3,4,5,6,20,21]
    assert seq[1].tolist() == [3, 4, 5, 6, 20, 21]
    assert attn[1].tolist() == [1, 1, 1, 1, 1, 1]


def test_right_aligned_response_data(tokenizer):
    """Response-level tensors are right-aligned: actual values at the end, zeros at the start."""
    prompts = [[1, 2, 3], [4, 5]]
    responses = [[10], [20, 21, 22]]
    rewards = [[1.0], [0.5, 0.6, 0.7]]
    loss_masks = [[1], [1, 0, 1]]
    logprobs = [[-0.1], [-0.2, -0.3, -0.4]]
    prompts_copy = [p[:] for p in prompts]
    responses_copy = [r[:] for r in responses]

    seq, attn, action, rew, lm, lp, _, _ = convert_prompts_responses_to_batch_tensors(
        tokenizer.pad_token_id,
        prompts,
        responses,
        rewards,
        loss_masks,
        logprobs,
    )
    # max_response=3
    assert action.shape == (2, 3)

    # Sample 0: response_len=1, right-aligned -> [0, 0, 1]
    assert action[0].tolist() == [0, 0, 1]
    assert rew[0].tolist() == [0.0, 0.0, 1.0]
    assert lm[0].tolist() == [0.0, 0.0, 1.0]
    assert lp[0].tolist() == pytest.approx([0.0, 0.0, -0.1])

    # Sample 1: response_len=3, right-aligned -> [1, 1, 1] (no padding)
    assert action[1].tolist() == [1, 1, 1]
    assert rew[1].tolist() == pytest.approx([0.5, 0.6, 0.7])
    assert lm[1].tolist() == [1.0, 0.0, 1.0]
    assert lp[1].tolist() == pytest.approx([-0.2, -0.3, -0.4])

    # Test does not mutate inputs
    assert prompts == prompts_copy
    assert responses == responses_copy


def test_max_seq_len_warns_but_does_not_truncate(tokenizer):
    """max_seq_len only warns; no tokens are lost."""
    prompts = [[1] * 50, [2] * 10]
    responses = [[3] * 10, [4] * 50]
    rewards = [[0.0] * 10, [0.0] * 50]
    loss_masks = [[1] * 10, [1] * 50]

    seq, _, action, _, _, _, _, _ = convert_prompts_responses_to_batch_tensors(
        tokenizer.pad_token_id,
        prompts,
        responses,
        rewards,
        loss_masks,
        max_seq_len=30,
    )
    # max_total = max(60, 60) = 60, which exceeds max_seq_len=30
    # But no truncation: all tokens preserved
    assert seq.shape == (2, 60)
    assert action.shape == (2, 50)


def test_rollout_expert_indices_shape_padding_and_alignment(tokenizer):
    """Routes pack to [sum(seq_len), layers, topk] with one cu_seqlens segment per trajectory."""
    prompts = [[1, 2], [3, 4, 5, 6]]
    responses = [[10, 11, 12], [20, 21]]
    rewards = [[0.0] * 3, [0.0] * 2]
    loss_masks = [[1] * 3, [1] * 2]

    num_layers = 2
    topk = 2
    rei_0 = np.asarray([[[1, 2]] * num_layers for _ in range(5)], dtype=np.uint8)
    rei_1 = np.asarray([[[3, 4]] * num_layers for _ in range(6)], dtype=np.uint8)

    seq, attn, action, rew, lm, lp, rei_tensor, _ = convert_prompts_responses_to_batch_tensors(
        tokenizer.pad_token_id,
        prompts,
        responses,
        rewards,
        loss_masks,
        rollout_expert_indices=[rei_0, rei_1],
    )

    assert rei_tensor is not None
    assert rei_tensor.values.shape == (11, num_layers, topk)
    assert rei_tensor.cu_seqlens.tolist() == [0, 5, 11]
    assert rei_tensor.sequence_lengths.tolist() == attn.sum(dim=1).tolist()
    assert rei_tensor.segment(0).tolist() == [[[1, 2]] * num_layers] * 5
    assert rei_tensor.segment(1).tolist() == [[[3, 4]] * num_layers] * 6


def test_rollout_expert_indices_none_when_not_provided(tokenizer):
    """When rollout_expert_indices is not provided, the returned tensor should be None."""
    prompts = [[1, 2], [3, 4]]
    responses = [[10], [20]]
    rewards = [[0.0], [0.0]]
    loss_masks = [[1], [1]]

    *_, rei_tensor, _ = convert_prompts_responses_to_batch_tensors(
        tokenizer.pad_token_id,
        prompts,
        responses,
        rewards,
        loss_masks,
    )
    assert rei_tensor is None


def test_stepwise_anti_correlation_no_inflation(tokenizer):
    """Step-wise anti-correlated prompt/response lengths: seq_len = max(prompt_i + response_i),
    NOT max(prompt_i) + max(response_i)."""
    # Early turn: prompt=10, response=90 (total=100)
    # Late turn:  prompt=90, response=10 (total=100)
    prompts = [list(range(10)), list(range(90))]
    responses = [list(range(100, 190)), list(range(200, 210))]
    rewards = [[0.0] * 90, [0.0] * 10]
    loss_masks = [[1] * 90, [1] * 10]

    seq, attn, action, rew, lm, _, _, _ = convert_prompts_responses_to_batch_tensors(
        tokenizer.pad_token_id,
        prompts,
        responses,
        rewards,
        loss_masks,
    )
    # max(10+90, 90+10) = 100, NOT 90+90=180
    assert seq.shape == (2, 100)
    assert action.shape == (2, 90)

    # All real tokens are preserved (no truncation)
    assert seq[0].tolist() == list(range(10)) + list(range(100, 190))
    assert seq[1].tolist() == list(range(90)) + list(range(200, 210))

    # Response data right-aligned: sample 1 has 10 tokens -> [0]*80 + [1]*10
    assert action[1].tolist() == [0] * 80 + [1] * 10


# ---------------------------------------------------------------------------
# Sample support — packed to response tokens
# ---------------------------------------------------------------------------

SAMPLE_SUPPORT_TOP_K = 3
# Anti-correlated prompt/response lengths, so a prompt-region rectangle would be mostly filler.
SAMPLE_SUPPORT_LENGTHS = [(2, 3), (9, 2), (5, 5), (1, 1)]


def _make_sample_support(lengths: List[tuple], *, seed: int = 0) -> List[np.ndarray]:
    """One dense ``[response_len, top_k]`` support block per trajectory, some rows padded."""
    rng = np.random.default_rng(seed)
    support = []
    for _, response_len in lengths:
        rows = rng.integers(0, 32_000, size=(response_len, SAMPLE_SUPPORT_TOP_K), dtype=np.int64)
        # A padded row (nothing captured) and a partially-filled row, as the sampler emits.
        rows[0] = SAMPLE_SUPPORT_PADDING
        rows[-1, -1] = SAMPLE_SUPPORT_PADDING
        support.append(rows.astype(SAMPLE_SUPPORT_DTYPE))
    return support


def _convert_with_support(tokenizer, lengths: List[tuple], support: List[np.ndarray]):
    prompts = [list(range(1, prompt_len + 1)) for prompt_len, _ in lengths]
    responses = [list(range(100, 100 + response_len)) for _, response_len in lengths]
    return convert_prompts_responses_to_batch_tensors(
        tokenizer.pad_token_id,
        prompts,
        responses,
        rewards=[[0.0] * len(response) for response in responses],
        loss_masks=[[1] * len(response) for response in responses],
        rollout_sample_support=support,
    )


def test_sample_support_packs_to_the_response_tokens(tokenizer):
    """One segment per trajectory, holding exactly its response tokens."""
    support = _make_sample_support(SAMPLE_SUPPORT_LENGTHS)

    *_, packed = _convert_with_support(tokenizer, SAMPLE_SUPPORT_LENGTHS, support)

    response_lens = [response_len for _, response_len in SAMPLE_SUPPORT_LENGTHS]
    assert packed.values.shape == (sum(response_lens), SAMPLE_SUPPORT_TOP_K)
    assert packed.dtype == SAMPLE_SUPPORT_TORCH_DTYPE
    assert packed.sequence_lengths.tolist() == response_lens
    for index, rows in enumerate(support):
        assert torch.equal(packed.segment(index), torch.from_numpy(rows))


def test_sample_support_allocates_no_prompt_region_rectangle(tokenizer):
    """A ``[batch, seq_len, top_k]`` rectangle would be mostly prompt-region filler."""
    support = _make_sample_support(SAMPLE_SUPPORT_LENGTHS)

    *_, packed = _convert_with_support(tokenizer, SAMPLE_SUPPORT_LENGTHS, support)

    max_total = max(prompt_len + response_len for prompt_len, response_len in SAMPLE_SUPPORT_LENGTHS)
    rectangle_rows = len(SAMPLE_SUPPORT_LENGTHS) * max_total
    assert packed.values.shape[0] == sum(response_len for _, response_len in SAMPLE_SUPPORT_LENGTHS)
    assert packed.values.shape[0] < rectangle_rows
    assert packed.values.numel() == packed.values.shape[0] * SAMPLE_SUPPORT_TOP_K


def test_sample_support_preserves_padding_rows(tokenizer):
    """Padding stays ``-1``: not zeros, and not the route field's ``arange(topk)``."""
    support = _make_sample_support(SAMPLE_SUPPORT_LENGTHS)

    *_, packed = _convert_with_support(tokenizer, SAMPLE_SUPPORT_LENGTHS, support)

    first_rows = torch.stack([packed.segment(index)[0] for index in range(len(support))])
    assert torch.all(first_rows == SAMPLE_SUPPORT_PADDING)
    assert not torch.any(packed.values[packed.values >= 0] == SAMPLE_SUPPORT_PADDING)


def test_sample_support_pooled_fill_equals_serial_fill(monkeypatch, tokenizer):
    """The pooled fill writes disjoint segments, so it must be bit-identical to a serial one."""
    support = _make_sample_support(SAMPLE_SUPPORT_LENGTHS)

    monkeypatch.setattr(parallel_fill, "default_fill_workers", lambda: len(SAMPLE_SUPPORT_LENGTHS))
    *_, pooled = _convert_with_support(tokenizer, SAMPLE_SUPPORT_LENGTHS, support)
    monkeypatch.setattr(parallel_fill, "default_fill_workers", lambda: 1)
    *_, serial = _convert_with_support(tokenizer, SAMPLE_SUPPORT_LENGTHS, support)

    assert pooled == serial
    assert torch.equal(pooled.values, torch.from_numpy(np.concatenate(support, axis=0)))


def test_sample_support_accepts_read_only_and_sliced_arrays(tokenizer):
    """Wire arrays arrive read-only, and the single-turn path hands over a row slice."""
    lengths = [(2, 2)]
    rows = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=SAMPLE_SUPPORT_DTYPE)
    rows.flags.writeable = False

    *_, packed = _convert_with_support(tokenizer, lengths, [rows[:2]])

    assert packed.segment(0).tolist() == [[1, 2, 3], [4, 5, 6]]


def test_sample_support_rejects_a_row_count_that_is_not_the_response_length(tokenizer):
    support = _make_sample_support(SAMPLE_SUPPORT_LENGTHS)
    support[1] = support[1][:1]

    with pytest.raises(ValueError, match="support rows for"):
        _convert_with_support(tokenizer, SAMPLE_SUPPORT_LENGTHS, support)


def test_sample_support_rejects_a_ragged_width(tokenizer):
    support = _make_sample_support(SAMPLE_SUPPORT_LENGTHS)
    support[2] = support[2][:, :-1]

    with pytest.raises(ValueError, match="must share top_k"):
        _convert_with_support(tokenizer, SAMPLE_SUPPORT_LENGTHS, support)


def test_sample_support_rejects_nested_lists(tokenizer):
    support = [rows.tolist() for rows in _make_sample_support(SAMPLE_SUPPORT_LENGTHS)]

    with pytest.raises(TypeError, match="NumPy arrays"):
        _convert_with_support(tokenizer, SAMPLE_SUPPORT_LENGTHS, support)


@pytest.mark.parametrize("dtype", [np.int16, np.int64])
def test_sample_support_rejects_non_canonical_dtypes(tokenizer, dtype):
    """The sender establishes int32, so collation validates instead of rescanning."""
    support = [rows.astype(dtype) for rows in _make_sample_support(SAMPLE_SUPPORT_LENGTHS)]

    with pytest.raises(ValueError, match="canonical sample-support dtype"):
        _convert_with_support(tokenizer, SAMPLE_SUPPORT_LENGTHS, support)


def test_sample_support_rejects_a_batch_size_mismatch(tokenizer):
    support = _make_sample_support(SAMPLE_SUPPORT_LENGTHS)[:-1]

    with pytest.raises(ValueError, match="support for every trajectory"):
        _convert_with_support(tokenizer, SAMPLE_SUPPORT_LENGTHS, support)


def test_sample_support_none_when_not_provided(tokenizer):
    *_, packed = convert_prompts_responses_to_batch_tensors(
        tokenizer.pad_token_id,
        prompts=[[1, 2]],
        responses=[[10]],
        rewards=[[0.0]],
        loss_masks=[[1]],
    )
    assert packed is None


def test_build_sample_support_torch_dtype_matches_the_wire_dtype():
    assert torch.empty(0, dtype=SAMPLE_SUPPORT_TORCH_DTYPE).numpy().dtype == SAMPLE_SUPPORT_DTYPE
    packed = build_sample_support([np.zeros((2, 3), dtype=SAMPLE_SUPPORT_DTYPE)], np.asarray([2]))
    assert packed.dtype == SAMPLE_SUPPORT_TORCH_DTYPE
