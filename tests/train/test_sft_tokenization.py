"""
CPU tests for SFT tokenization and collation helpers.

uv run --isolated --extra dev --extra fsdp pytest tests/train/test_sft_tokenization.py -v
"""

from dataclasses import dataclass

import pytest
import torch
from transformers import AutoTokenizer

from skyrl.train.sft_trainer import (
    collate_sft_batch,
    tokenize_chat_example,
    tokenize_sft_example,
)


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ---------------------------------------------------------------------------
# tokenize_chat_example
# ---------------------------------------------------------------------------


def test_chat_basic(tokenizer):
    """Single user+assistant conversation returns correct format."""
    example = {
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
    }
    result = tokenize_chat_example(example, tokenizer)

    assert result is not None
    assert isinstance(result["input_ids"], list)
    assert all(isinstance(t, int) for t in result["input_ids"])
    assert result["attention_mask"] == [1] * len(result["input_ids"])
    assert result["num_actions"] > 0
    assert result["num_actions"] < len(result["input_ids"])


def test_chat_multi_turn_no_thinking(tokenizer):
    """Multi-turn conversation: only last assistant turn counted in num_actions."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
        {"role": "assistant", "content": "6"},
    ]
    # with enable_thinking=False - <think></think> are expected to be included at the end of the user message directly
    expected_assistant_sequence = "6<|im_end|>\n"
    result = tokenize_chat_example({"messages": messages}, tokenizer, enable_thinking=False)
    assert result is not None
    assert result["num_actions"] > 0

    assert result["num_actions"] == len(tokenizer.encode(expected_assistant_sequence))


def test_chat_multi_turn_thinking(tokenizer):
    """Multi-turn conversation: only last assistant turn counted in num_actions."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "<think>\nThat's a tough one\n</think>\n\n4"},
        {"role": "user", "content": "And 3+3?"},
        {"role": "assistant", "content": "<think>\nThat's a tough one\n</think>\n\n6"},
    ]
    expected_assistant_sequence = "<think>\nThat's a tough one\n</think>\n\n6<|im_end|>\n"
    result = tokenize_chat_example({"messages": messages}, tokenizer, max_length=10000)

    assert result is not None
    assert result["num_actions"] > 0

    assert result["num_actions"] == len(tokenizer.encode(expected_assistant_sequence))


def test_chat_last_not_assistant(tokenizer):
    """Returns None when last message is not from assistant."""
    example = {
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Bye"},
        ]
    }
    assert tokenize_chat_example(example, tokenizer) is None


def test_chat_empty_messages(tokenizer):
    """Returns None for empty messages list."""
    assert tokenize_chat_example({"messages": []}, tokenizer) is None


def test_chat_truncation(tokenizer):
    """Truncation: len(input_ids) <= max_length. Returns None if response fully truncated."""
    messages = [
        {"role": "user", "content": "Tell me a very long story about " + "dragons " * 200},
        {"role": "assistant", "content": "Once upon a time " + "in a land " * 200},
    ]
    result = tokenize_chat_example({"messages": messages}, tokenizer, max_length=32)

    if result is not None:
        assert len(result["input_ids"]) <= 32
        assert result["num_actions"] > 0
    # If the prompt alone fills the budget, result is None -- also acceptable


# ---------------------------------------------------------------------------
# tokenize_sft_example
# ---------------------------------------------------------------------------


def test_alpaca_basic(tokenizer):
    """Instruction + output returns correct format with num_actions > 0."""
    example = {
        "instruction": "Summarize the following text.",
        "output": "This is the summary.",
    }
    result = tokenize_sft_example(example, tokenizer)

    assert result is not None
    assert isinstance(result["input_ids"], list)
    assert result["attention_mask"] == [1] * len(result["input_ids"])
    assert result["num_actions"] > 0
    assert result["num_actions"] < len(result["input_ids"])


def test_alpaca_with_input(tokenizer):
    """Instruction + input + output tests the '\\n\\n' join path."""
    example = {
        "instruction": "Translate to French.",
        "input": "Good morning.",
        "output": "Bonjour.",
    }
    result = tokenize_sft_example(example, tokenizer)

    assert result is not None
    assert result["num_actions"] > 0


def test_alpaca_truncated_response(tokenizer):
    """Prompt fills entire max_length, response fully truncated -> None."""
    example = {
        "instruction": "Describe the universe in detail. " * 100,
        "output": "The universe is vast.",
    }
    result = tokenize_sft_example(example, tokenizer, max_length=32)

    # The prompt is so long that after truncation there are no response tokens
    assert result is None


# ---------------------------------------------------------------------------
# collate_sft_batch
# ---------------------------------------------------------------------------


def _make_example(input_ids, num_actions):
    """Helper to create a tokenized example dict for collation tests."""
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "num_actions": num_actions,
    }


def test_collate_shapes(tokenizer):
    """3 examples of different lengths produce correct tensor shapes."""
    examples = [
        _make_example([1, 2, 3, 4, 5], 2),  # len=5, actions=2
        _make_example([10, 20, 30], 1),  # len=3, actions=1
        _make_example([100, 200, 300, 400], 3),  # len=4, actions=3
    ]
    batch = collate_sft_batch(examples, tokenizer)

    max_len = 5
    max_num_actions = 3
    assert batch["sequences"].shape == (3, max_len)
    assert batch["attention_mask"].shape == (3, max_len)
    assert batch["loss_mask"].shape == (3, max_num_actions)


def test_collate_left_padding(tokenizer):
    """Shorter sequences have pad_token_id on the left, zeros in attention_mask."""
    examples = [
        _make_example([1, 2, 3, 4, 5], 2),
        _make_example([10, 20, 30], 1),
    ]
    batch = collate_sft_batch(examples, tokenizer)

    # Example 0: len=5 (max_len=5), no padding
    assert batch["sequences"][0].tolist() == [1, 2, 3, 4, 5]
    assert batch["attention_mask"][0].tolist() == [1, 1, 1, 1, 1]

    # Example 1: len=3, padded to 5 on the left
    pad_id = tokenizer.pad_token_id
    assert batch["sequences"][1].tolist() == [pad_id, pad_id, 10, 20, 30]
    assert batch["attention_mask"][1].tolist() == [0, 0, 1, 1, 1]


def test_collate_loss_mask_alignment(tokenizer):
    """Loss mask has 1s right-aligned for response tokens, 0 padding on the left."""
    examples = [
        _make_example([1, 2, 3, 4, 5], 2),  # actions=2
        _make_example([10, 20, 30], 3),  # actions=3
        _make_example([100, 200, 300, 400], 1),  # actions=1
    ]
    batch = collate_sft_batch(examples, tokenizer)

    max_num_actions = 3
    assert batch["loss_mask"].shape == (3, max_num_actions)

    # Example 0: 2 actions -> [0, 1, 1]
    assert batch["loss_mask"][0].tolist() == [0, 1, 1]
    # Example 1: 3 actions -> [1, 1, 1]
    assert batch["loss_mask"][1].tolist() == [1, 1, 1]
    # Example 2: 1 action  -> [0, 0, 1]
    assert batch["loss_mask"][2].tolist() == [0, 0, 1]


def test_collate_single_example(tokenizer):
    """Batch of one: no padding needed."""
    examples = [_make_example([1, 2, 3], 2)]
    batch = collate_sft_batch(examples, tokenizer)

    assert batch["sequences"].shape == (1, 3)
    assert batch["attention_mask"].shape == (1, 3)
    assert batch["loss_mask"].shape == (1, 2)
    assert batch["sequences"][0].tolist() == [1, 2, 3]
    assert batch["attention_mask"][0].tolist() == [1, 1, 1]
    assert batch["loss_mask"][0].tolist() == [1, 1]


def test_collate_metadata(tokenizer):
    """batch.metadata['response_length'] equals max_num_actions."""
    examples = [
        _make_example([1, 2, 3, 4], 3),
        _make_example([10, 20], 1),
    ]
    batch = collate_sft_batch(examples, tokenizer)

    assert batch.metadata["response_length"] == 3


# ---------------------------------------------------------------------------
# tokenize_sft_example uses chat template (Fix 2 verification)
# ---------------------------------------------------------------------------


def test_alpaca_uses_chat_template(tokenizer):
    """tokenize_sft_example now uses apply_chat_template, producing the same
    output as tokenize_chat_example with equivalent messages."""
    example = {
        "instruction": "Translate to French.",
        "input": "Good morning.",
        "output": "Bonjour.",
    }
    alpaca_result = tokenize_sft_example(example, tokenizer)

    # Build equivalent chat example
    chat_example = {
        "messages": [
            {"role": "user", "content": "Translate to French.\n\nGood morning."},
            {"role": "assistant", "content": "Bonjour."},
        ]
    }
    chat_result = tokenize_chat_example(chat_example, tokenizer)

    assert alpaca_result is not None
    assert chat_result is not None
    assert alpaca_result["input_ids"] == chat_result["input_ids"]
    assert alpaca_result["num_actions"] == chat_result["num_actions"]


def test_alpaca_no_input_uses_chat_template(tokenizer):
    """tokenize_sft_example without input field also uses chat template."""
    example = {
        "instruction": "Say hello.",
        "output": "Hello!",
    }
    alpaca_result = tokenize_sft_example(example, tokenizer)

    chat_example = {
        "messages": [
            {"role": "user", "content": "Say hello."},
            {"role": "assistant", "content": "Hello!"},
        ]
    }
    chat_result = tokenize_chat_example(chat_example, tokenizer)

    assert alpaca_result is not None
    assert chat_result is not None
    assert alpaca_result["input_ids"] == chat_result["input_ids"]
    assert alpaca_result["num_actions"] == chat_result["num_actions"]


# ---------------------------------------------------------------------------
# Loss normalization (Fix 1 verification)
# ---------------------------------------------------------------------------


@dataclass
class _FakeSFTConfig:
    """Minimal stand-in for SFTConfig used by SFTTrainer.collate_batch."""

    batch_size: int = 4
    micro_train_batch_size_per_gpu: int = 2


def test_loss_norm_sums_to_expected(tokenizer):
    """After collate_batch normalization, loss_mask encodes the correct
    per-non-pad-token scaling factor.

    For a batch with ``total_nonpad`` non-pad loss tokens:
      loss_mask.sum() == batch_size / micro_batch_size

    This ensures that reduce_loss (sum over micro-batch) combined with
    microbatch_weight (mbs/pgb) produces a mean-per-non-pad-token loss.
    """
    from skyrl.train.sft_trainer import SFTTrainer

    cfg = _FakeSFTConfig(batch_size=4, micro_train_batch_size_per_gpu=2)

    # Build a batch with known non-pad counts:
    # Ex 0: 2 actions out of max 4 -> [0, 0, 1, 1]
    # Ex 1: 4 actions out of max 4 -> [1, 1, 1, 1]
    # Ex 2: 1 action out of max 4  -> [0, 0, 0, 1]
    # Ex 3: 3 actions out of max 4 -> [0, 1, 1, 1]
    # total_nonpad = 2+4+1+3 = 10
    examples = [
        _make_example([1, 2, 3, 4, 5], 2),
        _make_example([1, 2, 3, 4, 5], 4),
        _make_example([1, 2, 3, 4, 5], 1),
        _make_example([1, 2, 3, 4, 5], 3),
    ]

    # Create a minimal trainer-like object to call collate_batch
    trainer = object.__new__(SFTTrainer)
    trainer.sft_cfg = cfg
    trainer.tokenizer = tokenizer

    batch = trainer.collate_batch(examples)

    total_nonpad = 2 + 4 + 1 + 3  # = 10
    expected_scaling = cfg.batch_size / (cfg.micro_train_batch_size_per_gpu * total_nonpad)

    # Each non-pad position should have value = expected_scaling
    # Total sum = total_nonpad * expected_scaling = batch_size / micro_batch_size
    expected_sum = cfg.batch_size / cfg.micro_train_batch_size_per_gpu
    assert abs(batch["loss_mask"].sum().item() - expected_sum) < 1e-5

    # Verify individual non-zero values equal expected_scaling
    nonzero_vals = batch["loss_mask"][batch["loss_mask"] > 0]
    assert torch.allclose(nonzero_vals, torch.tensor(expected_scaling, dtype=torch.float32))


def test_loss_norm_all_nonpad(tokenizer):
    """When all tokens are non-pad, loss_mask values equal
    batch_size / (micro_batch_size * batch_size * num_actions)
    = 1 / (micro_batch_size * num_actions)."""
    from skyrl.train.sft_trainer import SFTTrainer

    cfg = _FakeSFTConfig(batch_size=2, micro_train_batch_size_per_gpu=1)

    examples = [
        _make_example([1, 2, 3], 2),  # 2 actions
        _make_example([4, 5, 6], 2),  # 2 actions
    ]

    trainer = object.__new__(SFTTrainer)
    trainer.sft_cfg = cfg
    trainer.tokenizer = tokenizer

    batch = trainer.collate_batch(examples)

    total_nonpad = 4  # 2 + 2
    expected_scaling = cfg.batch_size / (cfg.micro_train_batch_size_per_gpu * total_nonpad)
    # = 2 / (1 * 4) = 0.5

    # All loss_mask values should be either 0 or expected_scaling
    nonzero_vals = batch["loss_mask"][batch["loss_mask"] > 0]
    assert torch.allclose(nonzero_vals, torch.tensor(expected_scaling, dtype=torch.float32))

    # Sum should be batch_size / micro_batch_size = 2
    assert abs(batch["loss_mask"].sum().item() - 2.0) < 1e-5
