"""
CPU tests for SFT tokenization and collation helpers.

uv run --isolated --extra dev --extra fsdp pytest tests/train/test_sft_tokenization.py -v
"""

import pytest
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


def test_chat_return_dict_false(tokenizer):
    """input_ids must be a plain list[int], not a BatchEncoding."""
    example = {
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
    }
    result = tokenize_chat_example(example, tokenizer)
    assert result is not None
    assert isinstance(result["input_ids"], list)
    assert not hasattr(result["input_ids"], "input_ids")  # not a BatchEncoding


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
