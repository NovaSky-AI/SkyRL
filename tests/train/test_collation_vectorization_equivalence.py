"""Equivalence tests for vectorized controller-side collation.

These tests compare the new NumPy-backed paths against the original loop
layouts for:

* RL: :func:`convert_prompts_responses_to_batch_tensors`
* SFT (unpacked): :func:`collate_sft_batch` / ``DefaultCollator``
* SFT (Megatron FFD packing): ``PackedDataCollator``

Run with:
  uv run --isolated --extra dev --extra megatron -- \
      pytest tests/train/test_collation_vectorization_equivalence.py
"""

from __future__ import annotations

import random
from typing import List
from unittest.mock import MagicMock

import pytest
import torch

from skyrl.train.dataset.collators import PackedDataCollator
from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl.train.sft_trainer import collate_sft_batch


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


# ---------------------------------------------------------------------------
# Reference implementations (the original, pre-vectorization loops)
# ---------------------------------------------------------------------------


def _ref_convert_prompts_responses(prompts, responses, rewards, loss_masks, logprobs, pad_token_id):
    """Original per-sample loop layout."""
    prompt_token_lens = [len(p) for p in prompts]
    response_token_lens = [len(r) for r in responses]
    max_response = max(response_token_lens)
    max_total = max(p + r for p, r in zip(prompt_token_lens, response_token_lens))

    sequences = []
    attention_masks = []
    action_masks = []
    for i in range(len(prompts)):
        total_real = prompt_token_lens[i] + response_token_lens[i]
        pad_len = max_total - total_real
        sequences.append([pad_token_id] * pad_len + prompts[i] + responses[i])
        attention_masks.append([0] * pad_len + [1] * total_real)
        resp_pad = max_response - response_token_lens[i]
        action_masks.append([0] * resp_pad + [1] * response_token_lens[i])

    sequences_t = torch.tensor(sequences)
    attention_mask_t = torch.tensor(attention_masks, dtype=torch.int64)
    action_mask_t = torch.tensor(action_masks, dtype=torch.int64)

    ret_loss_masks = torch.zeros(len(prompts), max_response, dtype=torch.float)
    for i, lm in enumerate(loss_masks):
        ret_loss_masks[i, max_response - len(lm) :] = torch.tensor(lm, dtype=torch.float)

    ret_rewards = torch.zeros(len(prompts), max_response, dtype=torch.float)
    for i, custom_reward in enumerate(rewards):
        cr = torch.tensor(custom_reward) if isinstance(custom_reward, list) else custom_reward
        ret_rewards[i, max_response - len(cr) :] = cr

    logprobs_tensor = None
    if logprobs:
        logprobs_tensor = torch.zeros(len(prompts), max_response, dtype=torch.float)
        for i, sample_logprobs in enumerate(logprobs):
            lp = torch.tensor(sample_logprobs, dtype=torch.float)
            logprobs_tensor[i, max_response - len(sample_logprobs) :] = lp

    return sequences_t, attention_mask_t, action_mask_t, ret_rewards, ret_loss_masks, logprobs_tensor


def _ref_collate_sft_batch(examples, pad_token_id):
    """Original per-example SFT collate loop."""
    max_len = max(len(ex["input_ids"]) for ex in examples)
    max_num_actions = max(ex["num_actions"] for ex in examples)

    sequences = []
    attention_masks = []
    loss_masks = []
    for ex in examples:
        pad_len = max_len - len(ex["input_ids"])
        sequences.append([pad_token_id] * pad_len + ex["input_ids"])
        attention_masks.append([0] * pad_len + ex["attention_mask"])
        action_pad = max_num_actions - ex["num_actions"]
        loss_masks.append([0] * action_pad + ex["loss_mask"])

    return (
        torch.tensor(sequences, dtype=torch.long),
        torch.tensor(attention_masks, dtype=torch.long),
        torch.tensor(loss_masks, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Random input generators
# ---------------------------------------------------------------------------


def _random_rl_inputs(rng: random.Random, with_logprobs: bool):
    n = rng.randint(1, 6)
    prompts, responses, rewards, loss_masks, logprobs = [], [], [], [], []
    tok = 10
    for _ in range(n):
        p_len = rng.randint(1, 7)
        r_len = rng.randint(1, 7)
        prompts.append([tok + j for j in range(p_len)])
        tok += p_len
        responses.append([tok + j for j in range(r_len)])
        tok += r_len
        rewards.append([round(rng.uniform(-1, 1), 3) for _ in range(r_len)])
        loss_masks.append([rng.randint(0, 1) for _ in range(r_len)])
        logprobs.append([round(rng.uniform(-5, 0), 4) for _ in range(r_len)])
    return prompts, responses, rewards, loss_masks, (logprobs if with_logprobs else None)


def _random_sft_examples(rng: random.Random):
    n = rng.randint(1, 8)
    examples = []
    base = 100
    for _ in range(n):
        seq_len = rng.randint(2, 12)
        num_actions = rng.randint(1, seq_len)
        examples.append(
            {
                "input_ids": [base + j for j in range(seq_len)],
                "attention_mask": [1] * seq_len,
                "num_actions": num_actions,
                "loss_mask": [rng.randint(0, 1) for _ in range(num_actions)],
            }
        )
        base += seq_len
    return examples


# ---------------------------------------------------------------------------
# RL: convert_prompts_responses_to_batch_tensors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_logprobs", [True, False])
@pytest.mark.parametrize("seed", range(30))
def test_rl_preprocess_bit_identical(seed, with_logprobs):
    rng = _rng(seed)
    prompts, responses, rewards, loss_masks, logprobs = _random_rl_inputs(rng, with_logprobs)

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0

    seq, attn, action, rew, lm, lp, _ = convert_prompts_responses_to_batch_tensors(
        tokenizer, prompts, responses, rewards, loss_masks, logprobs
    )
    r_seq, r_attn, r_action, r_rew, r_lm, r_lp = _ref_convert_prompts_responses(
        prompts, responses, rewards, loss_masks, logprobs, pad_token_id=0
    )

    assert torch.equal(seq, r_seq)
    assert torch.equal(attn, r_attn)
    assert torch.equal(action, r_action)
    assert torch.equal(rew, r_rew)
    assert torch.equal(lm, r_lm)
    if with_logprobs:
        assert torch.equal(lp, r_lp)
    else:
        assert lp is None and r_lp is None
    # torch.equal allows dtype-compatible matches, so pin dtypes explicitly.
    assert seq.dtype == r_seq.dtype
    assert attn.dtype == r_attn.dtype == torch.int64
    assert action.dtype == r_action.dtype == torch.int64
    assert rew.dtype == r_rew.dtype == torch.float32


def test_rl_preprocess_accepts_tensor_rewards():
    """Reward postprocessing hands per-token reward tensors, not lists."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    prompts = [[1, 2, 3], [4, 5]]
    responses = [[10], [20, 21, 22]]
    rewards = [torch.tensor([1.0]), torch.tensor([0.5, 0.6, 0.7])]
    loss_masks = [[1], [1, 0, 1]]

    _, _, _, rew, _, _, _ = convert_prompts_responses_to_batch_tensors(
        tokenizer, prompts, responses, rewards, loss_masks
    )
    _, _, _, r_rew, _, _ = _ref_convert_prompts_responses(prompts, responses, rewards, loss_masks, None, pad_token_id=0)
    assert torch.equal(rew, r_rew)


def test_rl_preprocess_accepts_grad_tensor_rewards():
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    prompts = [[1, 2, 3], [4, 5]]
    responses = [[10], [20, 21, 22]]
    rewards = [
        torch.tensor([1.0], requires_grad=True),
        torch.tensor([0.5, 0.6, 0.7], requires_grad=True),
    ]
    loss_masks = [[1], [1, 0, 1]]

    _, _, _, rew, _, _, _ = convert_prompts_responses_to_batch_tensors(
        tokenizer, prompts, responses, rewards, loss_masks
    )

    expected = torch.tensor([[0.0, 0.0, 1.0], [0.5, 0.6, 0.7]], dtype=torch.float32)
    assert torch.equal(rew, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_rl_preprocess_accepts_cuda_tensor_rewards():
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    prompts = [[1, 2, 3], [4, 5]]
    responses = [[10], [20, 21, 22]]
    rewards = [
        torch.tensor([1.0], device="cuda"),
        torch.tensor([0.5, 0.6, 0.7], device="cuda"),
    ]
    loss_masks = [[1], [1, 0, 1]]

    _, _, _, rew, _, _, _ = convert_prompts_responses_to_batch_tensors(
        tokenizer, prompts, responses, rewards, loss_masks
    )

    expected = torch.tensor([[0.0, 0.0, 1.0], [0.5, 0.6, 0.7]], dtype=torch.float32)
    assert torch.equal(rew, expected)


# ---------------------------------------------------------------------------
# SFT unpacked: collate_sft_batch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(30))
def test_sft_collate_bit_identical(seed):
    rng = _rng(seed)
    examples = _random_sft_examples(rng)

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0

    batch = collate_sft_batch(examples, tokenizer)
    r_seq, r_attn, r_lm = _ref_collate_sft_batch(examples, pad_token_id=0)

    assert torch.equal(batch["sequences"], r_seq)
    assert torch.equal(batch["attention_mask"], r_attn)
    assert torch.equal(batch["loss_mask"], r_lm)
    assert batch["sequences"].dtype == r_seq.dtype == torch.long
    assert batch["attention_mask"].dtype == r_attn.dtype == torch.long
    assert batch["loss_mask"].dtype == r_lm.dtype == torch.long
    assert batch.metadata["response_length"] == max(ex["num_actions"] for ex in examples)


# ---------------------------------------------------------------------------
# SFT packed: PackedDataCollator (Megatron FFD)
# ---------------------------------------------------------------------------


def _make_packed_collator(*, batch_size, tp, pp, cp, dp, bin_capacity):
    tok = MagicMock()
    tok.pad_token_id = 0
    return PackedDataCollator(
        tokenizer=tok,
        max_tokens_per_microbatch=bin_capacity,
        tp_size=tp,
        pp_size=pp,
        cp_size=cp,
        dp_size=dp,
        batch_size=batch_size,
        micro_train_batch_size_per_gpu=1,
    )


def _ref_packed_rows(collator: PackedDataCollator, examples, max_packed_len, flat_bins, seq_lengths):
    """Original packed-row loop."""

    def _round_up(x, m):
        return ((x + m - 1) // m) * m

    align_size = collator.tp_size * collator.cp_size * 2 if collator.cp_size > 1 else collator.tp_size
    full_loss_masks = []
    for ex in examples:
        n_pad = len(ex["input_ids"]) - ex["num_actions"]
        full_loss_masks.append([0] * n_pad + list(ex["loss_mask"]))

    pad_token_id = collator.tokenizer.pad_token_id
    num_bins = len(flat_bins)
    sequences = torch.full((num_bins, max_packed_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((num_bins, max_packed_len), dtype=torch.long)
    loss_mask = torch.zeros((num_bins, max_packed_len - 1), dtype=torch.float)
    total_nonpad = 0
    for row_idx, bin_indices in enumerate(flat_bins):
        row_offset = 0
        for ex_idx in bin_indices:
            ex = examples[ex_idx]
            s = seq_lengths[ex_idx]
            sequences[row_idx, row_offset : row_offset + s] = torch.tensor(ex["input_ids"], dtype=torch.long)
            attention_mask[row_idx, row_offset : row_offset + s] = 1
            full_mask = full_loss_masks[ex_idx]
            for p_local in range(s - 1):
                row_p = row_offset + p_local
                if row_p < max_packed_len - 1:
                    loss_mask[row_idx, row_p] = float(full_mask[p_local + 1])
                    if full_mask[p_local + 1]:
                        total_nonpad += 1
            row_offset += _round_up(s, align_size)
    if total_nonpad != int(loss_mask.sum().item()):
        total_nonpad = int(loss_mask.sum().item())
    # Same normalization as production: sum loss in the workers, so scale the
    # loss mask by total non-padding tokens (see #1893).
    scale = 1 / max(total_nonpad, 1)
    loss_mask.mul_(scale)
    return sequences, attention_mask, loss_mask


@pytest.mark.parametrize("tp,pp,cp,dp", [(1, 1, 1, 1), (2, 1, 1, 2), (1, 2, 1, 2), (2, 1, 2, 1), (4, 1, 1, 4)])
@pytest.mark.parametrize("seed", range(8))
def test_packed_collator_bit_identical(seed, tp, pp, cp, dp):
    """PackedDataCollator matches the loop reference."""
    rng = _rng(seed)
    batch_size = dp * rng.randint(2, 4)
    bin_capacity = 64
    collator = _make_packed_collator(batch_size=batch_size, tp=tp, pp=pp, cp=cp, dp=dp, bin_capacity=bin_capacity)

    examples: List[dict] = []
    base = 100
    for _ in range(batch_size):
        seq_len = rng.randint(2, 10)
        num_actions = rng.randint(1, seq_len)
        examples.append(
            {
                "input_ids": [base + j for j in range(seq_len)],
                "attention_mask": [1] * seq_len,
                "num_actions": num_actions,
                "loss_mask": [rng.randint(0, 1) for _ in range(num_actions)],
            }
        )
        base += seq_len

    batch = collator(examples, batch_size=batch_size)

    # Re-derive FFD rows independently before running the loop reference.
    from skyrl.train.dataset.bin_packing import make_seq_packer

    seq_lengths = [len(ex["input_ids"]) for ex in examples]
    packer = make_seq_packer("first_fit_decreasing", bin_capacity=bin_capacity, min_bin_count=dp, bin_count_multiple=dp)
    bins = packer.pack(seq_lengths)
    shard_bins: List[List[List[int]]] = [[] for _ in range(dp)]
    for bin_idx, bi in enumerate(bins):
        shard_bins[bin_idx % dp].append(bi)
    flat_bins: List[List[int]] = []
    for shard_idx in range(dp):
        flat_bins.extend(shard_bins[shard_idx])

    def _round_up(x, m):
        return ((x + m - 1) // m) * m

    align_size = tp * cp * 2 if cp > 1 else tp
    bin_packed_lengths = [sum(_round_up(seq_lengths[idx], align_size) for idx in bi) for bi in flat_bins]
    if pp > 1:
        max_packed_len = _round_up(max(bin_packed_lengths), align_size)
    else:
        max_packed_len = max(bin_packed_lengths)

    r_seq, r_attn, r_loss = _ref_packed_rows(collator, examples, max_packed_len, flat_bins, seq_lengths)

    assert torch.equal(batch["sequences"], r_seq)
    assert torch.equal(batch["attention_mask"], r_attn)
    assert torch.equal(batch["loss_mask"], r_loss)
    assert batch["sequences"].dtype == r_seq.dtype == torch.long
    assert batch["attention_mask"].dtype == r_attn.dtype == torch.long
    assert batch["loss_mask"].dtype == r_loss.dtype == torch.float32
