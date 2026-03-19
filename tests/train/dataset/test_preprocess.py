"""
uv run --isolated --extra dev pytest tests/train/dataset/test_preprocess.py
"""

from unittest.mock import MagicMock

import pytest
import torch

from skyrl.train.dataset.preprocess import (
    convert_prompts_responses_to_batch_tensors,
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

    sequences, attention_mask, action_mask, ret_rewards, ret_loss_masks, ret_log_probs, _ = (
        convert_prompts_responses_to_batch_tensors(
            tokenizer,
            prompts,
            outputs,
            rewards,
            loss_masks,
        )
    )

    # max_total = max(3+3, 5+5) = 10, max_response = 5
    assert sequences.shape[0] == len(prompts)
    assert sequences.shape == (2, 10)
    assert action_mask.shape == ret_loss_masks.shape
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

    sequences, attention_mask, action_mask, ret_rewards, ret_loss_masks, ret_log_probs, _ = (
        convert_prompts_responses_to_batch_tensors(
            tokenizer,
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
    assert action_mask.shape == (2, max_response_len)
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
            tokenizer,
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
            tokenizer,
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

    seq, attn, action, rew, lm, _, _ = convert_prompts_responses_to_batch_tensors(
        tokenizer,
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

    seq, attn, action, rew, lm, lp, _ = convert_prompts_responses_to_batch_tensors(
        tokenizer,
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

    seq, _, action, _, _, _, _ = convert_prompts_responses_to_batch_tensors(
        tokenizer,
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


# ---------------------------------------------------------------------------
# R3 (Router Replay) — rollout_expert_indices padding tests
# ---------------------------------------------------------------------------


def test_rollout_expert_indices_shape_padding_and_alignment(tokenizer):
    """rollout_expert_indices tensor should have shape [batch, max_total, layers, topk]
    with left-padding aligned to the attention_mask."""
    # Sample 0: prompt=2, response=3  → total=5
    # Sample 1: prompt=4, response=2  → total=6
    # max_total=6
    prompts = [[1, 2], [3, 4, 5, 6]]
    responses = [[10, 11, 12], [20, 21]]
    rewards = [[0.0] * 3, [0.0] * 2]
    loss_masks = [[1] * 3, [1] * 2]

    num_layers = 2
    topk = 2
    # rollout_expert_indices[i] has shape [prompt_len_i + response_len_i, num_layers, topk]
    # Sample 0: 5 tokens, sample 1: 6 tokens
    rei_0 = [[[1, 2]] * num_layers for _ in range(5)]  # 5 tokens
    rei_1 = [[[3, 4]] * num_layers for _ in range(6)]  # 6 tokens

    seq, attn, action, rew, lm, lp, rei_tensor = convert_prompts_responses_to_batch_tensors(
        tokenizer,
        prompts,
        responses,
        rewards,
        loss_masks,
        rollout_expert_indices=[rei_0, rei_1],
    )

    assert rei_tensor is not None
    # Shape: [batch=2, max_total=6, layers=2, topk=2]
    assert rei_tensor.shape == (2, 6, num_layers, topk)

    # Sample 0 has total=5, so 1 left-pad position → first position should be zeros
    assert rei_tensor[0, 0].tolist() == [[0, 0]] * num_layers  # padding
    assert rei_tensor[0, 1].tolist() == [[1, 2]] * num_layers  # first real token

    # Sample 1 has total=6, no padding
    assert rei_tensor[1, 0].tolist() == [[3, 4]] * num_layers  # first real token

    # Non-zero positions in rei_tensor align exactly with attention_mask==1
    for i in range(2):
        for pos in range(6):
            if attn[i, pos] == 0:
                assert rei_tensor[i, pos].tolist() == [[0, 0]] * num_layers
            else:
                assert rei_tensor[i, pos].tolist() != [[0, 0]] * num_layers


def test_rollout_expert_indices_none_when_not_provided(tokenizer):
    """When rollout_expert_indices is not provided, the returned tensor should be None."""
    prompts = [[1, 2], [3, 4]]
    responses = [[10], [20]]
    rewards = [[0.0], [0.0]]
    loss_masks = [[1], [1]]

    *_, rei_tensor = convert_prompts_responses_to_batch_tensors(
        tokenizer,
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

    seq, attn, action, rew, lm, _, _ = convert_prompts_responses_to_batch_tensors(
        tokenizer,
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
