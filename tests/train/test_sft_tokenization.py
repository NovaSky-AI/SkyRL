"""
CPU tests for SFT tokenization and collation helpers.

uv run --isolated --extra dev --extra fsdp pytest tests/train/test_sft_tokenization.py -v
"""

import json

import pytest
import torch
from transformers import AutoProcessor, AutoTokenizer

from skyrl.train.config.sft_config import SFTConfig, TrainOnWhat
from skyrl.train.generators.utils import get_generation_prompt_ids
from skyrl.train.sft_trainer import (
    _normalize_chat_messages,
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


@pytest.fixture(scope="module")
def vlm_processor():
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    if proc.tokenizer.pad_token is None:
        proc.tokenizer.pad_token = proc.tokenizer.eos_token
    return proc


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


# NOTE: This test requires torchvision and thus torch - we use the vllm marker so that the test is run with the appropriate extras
@pytest.mark.vllm
def test_chat_vlm(vlm_processor):
    """A user+image+assistant conversation tokenizes with vision tokens."""
    example = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hi"},
                    {
                        "type": "image",
                        "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                    },
                ],
            },
            {"role": "assistant", "content": "Hello!"},
        ]
    }
    result = tokenize_chat_example(example, vlm_processor.tokenizer, processor=vlm_processor)

    assert result is not None
    ids = result["input_ids"]
    assert isinstance(ids, list)
    assert all(isinstance(t, int) for t in ids)

    assert vlm_processor.tokenizer.convert_tokens_to_ids("<|vision_start|>") in ids
    assert vlm_processor.tokenizer.convert_tokens_to_ids("<|image_pad|>") in ids
    assert vlm_processor.tokenizer.convert_tokens_to_ids("<|vision_end|>") in ids
    assert "pixel_values" in result
    assert "image_grid_thw" in result


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


def _make_example(input_ids, num_actions, image_grid=None):
    """Helper to create a tokenized example dict for collation tests.

    Mirrors ``_tokenize_chat_last_assistant``: loss_mask is num_actions long,
    all 1s (train on every response token). When ``image_grid`` is given, attach
    matching ``image_grid_thw`` / ``pixel_values`` tensors.
    """
    example = {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "num_actions": num_actions,
        "loss_mask": [1] * num_actions,
    }
    if image_grid:
        example["image_grid_thw"] = torch.tensor(image_grid)
        patches = sum(int(thw.prod()) for thw in example["image_grid_thw"])
        example["pixel_values"] = torch.ones(patches, patches * 3)
    return example


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


@pytest.mark.vllm
def test_collate_vlm(vlm_processor):
    """Examples with vision tensors collate into the expected TensorList shapes."""
    examples = [
        _make_example([1, 2, 3, 4, 5], 2, image_grid=[[1, 3, 3]]),
        _make_example(list(range(7)), 3, image_grid=[[1, 2, 2]]),
    ]
    batch = collate_sft_batch(examples, vlm_processor.tokenizer)

    assert (batch["image_grid_thw"][0] == torch.tensor([[1, 3, 3]])).all().item()
    assert (batch["image_grid_thw"][1] == torch.tensor([[1, 2, 2]])).all().item()
    assert batch["pixel_values"][0].shape == (9, 27)
    assert batch["pixel_values"][1].shape == (4, 12)


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


def test_loss_norm_sums_to_expected(tokenizer):
    """After collate_batch normalization, loss_mask encodes the correct
    per-non-pad-token scaling factor.

    For a batch with ``total_nonpad`` non-pad loss tokens, non-zero mask
    entries are ``1 / total_nonpad`` and ``loss_mask.sum() == 1``.
    """
    from skyrl.train.dataset.collators import DefaultCollator

    cfg = SFTConfig(batch_size=4, micro_train_batch_size_per_gpu=2)

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

    collator = DefaultCollator(tokenizer, micro_train_batch_size_per_gpu=cfg.micro_train_batch_size_per_gpu)

    batch = collator(examples, batch_size=cfg.batch_size)

    total_nonpad = 2 + 4 + 1 + 3  # = 10
    expected_scaling = 1 / total_nonpad

    # Each non-pad position should have value = expected_scaling
    assert abs(batch["loss_mask"].sum().item() - 1.0) < 1e-5

    # Verify individual non-zero values equal expected_scaling
    nonzero_vals = batch["loss_mask"][batch["loss_mask"] > 0]
    assert torch.allclose(nonzero_vals, torch.tensor(expected_scaling, dtype=torch.float32))


def test_loss_norm_all_nonpad(tokenizer):
    """When all tokens are non-pad, loss_mask values equal ``1 / total_nonpad``."""
    from skyrl.train.dataset.collators import DefaultCollator

    cfg = SFTConfig(batch_size=2, micro_train_batch_size_per_gpu=1)

    examples = [
        _make_example([1, 2, 3], 2),  # 2 actions
        _make_example([4, 5, 6], 2),  # 2 actions
    ]

    collator = DefaultCollator(tokenizer, micro_train_batch_size_per_gpu=cfg.micro_train_batch_size_per_gpu)

    batch = collator(examples, batch_size=cfg.batch_size)

    total_nonpad = 4  # 2 + 2
    expected_scaling = 1 / total_nonpad

    # All loss_mask values should be either 0 or expected_scaling
    nonzero_vals = batch["loss_mask"][batch["loss_mask"] > 0]
    assert torch.allclose(nonzero_vals, torch.tensor(expected_scaling, dtype=torch.float32))

    assert abs(batch["loss_mask"].sum().item() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# TrainOnWhat: LAST_ASSISTANT_MESSAGE tests
# ---------------------------------------------------------------------------


def test_train_on_what_single_turn_last_assistant(tokenizer):
    """Single-turn with LAST_ASSISTANT_MESSAGE: returns loss_mask covering last assistant turn."""
    example = {
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
    }
    result = tokenize_chat_example(
        example,
        tokenizer,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    assert result is not None
    assert result["num_actions"] > 0
    assert "loss_mask" in result
    assert len(result["loss_mask"]) == result["num_actions"]
    assert result["loss_mask"] == [1] * result["num_actions"]


def test_train_on_what_multi_turn_last_assistant_only(tokenizer):
    """Multi-turn with LAST_ASSISTANT_MESSAGE: only last assistant counted, loss_mask reflects that."""
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
        {"role": "assistant", "content": "6"},
    ]
    result = tokenize_chat_example(
        {"messages": messages},
        tokenizer,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        enable_thinking=False,
    )
    assert result is not None

    # Only the last assistant's tokens should be counted
    expected_last = "6<|im_end|>\n"
    assert result["num_actions"] == len(tokenizer.encode(expected_last))
    assert "loss_mask" in result
    assert len(result["loss_mask"]) == result["num_actions"]
    assert result["loss_mask"] == [1] * result["num_actions"]


# ---------------------------------------------------------------------------
# TrainOnWhat: ALL_ASSISTANT_MESSAGES tests
# ---------------------------------------------------------------------------


def test_train_on_what_multi_turn_all_assistants(tokenizer):
    """Multi-turn with ALL_ASSISTANT_MESSAGES: both assistant msgs get loss=1, user tokens masked."""
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
        {"role": "assistant", "content": "6"},
    ]
    result = tokenize_chat_example(
        {"messages": messages},
        tokenizer,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    assert result is not None
    assert "loss_mask" in result
    assert len(result["loss_mask"]) == result["num_actions"]

    loss_mask = result["loss_mask"]
    # There must be some 1s (assistant tokens)
    assert sum(loss_mask) > 0
    # There must be some 0s (user tokens between the two assistant turns)
    assert 0 in loss_mask
    # The mask should have a non-contiguous pattern: 1s, then 0s, then 1s
    # Find the transition points
    first_zero_after_one = None
    second_one_after_zero = None
    for i in range(1, len(loss_mask)):
        if loss_mask[i] == 0 and loss_mask[i - 1] == 1 and first_zero_after_one is None:
            first_zero_after_one = i
        if (
            loss_mask[i] == 1
            and loss_mask[i - 1] == 0
            and first_zero_after_one is not None
            and second_one_after_zero is None
        ):
            second_one_after_zero = i
    assert first_zero_after_one is not None, "Expected 0s between assistant turns"
    assert second_one_after_zero is not None, "Expected 1s for second assistant turn"


def test_train_on_what_non_contiguous_assistants(tokenizer):
    """assistant->user->assistant with ALL_ASSISTANT_MESSAGES: interior user masked out."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "Tell me a joke"},
        {"role": "assistant", "content": "Why did the chicken cross the road?"},
    ]
    result = tokenize_chat_example(
        {"messages": messages},
        tokenizer,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    assert result is not None
    loss_mask = result["loss_mask"]

    # The num_actions window starts at the first assistant token
    # and extends to the end. Within it, user/system tokens are 0.
    total_ones = sum(loss_mask)
    total_zeros = len(loss_mask) - total_ones
    assert total_ones > 0, "Must have assistant tokens"
    assert total_zeros > 0, "Must have masked user tokens between assistant turns"

    # First few elements are generation prompts - loss mask first assistant token must be 1
    num_generation_tokens = len(get_generation_prompt_ids(tokenizer))
    assert loss_mask[:num_generation_tokens] == [0] * num_generation_tokens
    assert loss_mask[num_generation_tokens] == 1


def test_train_on_what_all_assistants_truncation(tokenizer):
    """Truncation cutting into assistant turns with ALL_ASSISTANT_MESSAGES."""
    messages = [
        {"role": "user", "content": "Short question"},
        {"role": "assistant", "content": "First answer " + "word " * 50},
        {"role": "user", "content": "Another question"},
        {"role": "assistant", "content": "Second answer " + "word " * 50},
    ]
    # Use a small max_length to force truncation
    result = tokenize_chat_example(
        {"messages": messages},
        tokenizer,
        max_length=64,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    if result is not None:
        assert len(result["input_ids"]) <= 64
        assert result["num_actions"] > 0
        assert len(result["loss_mask"]) == result["num_actions"]
        # All loss_mask values should be 0 or 1
        assert all(v in (0, 1) for v in result["loss_mask"])


# ---------------------------------------------------------------------------
# Tool-calling datasets (APIGen-MT, xLAM, ToolACE, ...).
# These ship per-row ``tools`` (function schemas), an optional per-row
# ``system`` prompt, and ``messages`` containing assistant-with-tool_calls
# and tool-result turns. ``tokenize_chat_example`` should:
#   - normalize string-encoded ``tool_calls`` into the OpenAI list shape,
#   - forward the per-row ``tools`` to ``apply_chat_template``,
#   - prepend the per-row system message,
#   - mask tool-result tokens to 0 and assistant-generated tokens to 1.
# ---------------------------------------------------------------------------


_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def _tool_call_messages():
    return [
        {"role": "user", "content": "What's the weather in Paris?", "tool_calls": "[]"},
        # APIGen-MT shape: tool_calls is a JSON-encoded single dict, content is "".
        {
            "role": "assistant",
            "content": "",
            "tool_calls": '{"name": "get_weather", "arguments": {"city": "Paris"}}',
        },
        {"role": "tool", "content": '{"temp_c": 21, "conditions": "sunny"}'},
        {"role": "assistant", "content": "It's 21C and sunny in Paris.", "tool_calls": "[]"},
    ]


@pytest.mark.parametrize(
    "tools_value",
    [
        _TOOLS_SCHEMA,
        json.dumps(_TOOLS_SCHEMA),
    ],
    ids=["list", "json-string"],
)
def test_chat_tool_calling_apigen_shape(tokenizer, tools_value):
    """APIGen-MT-style row with stringified tool_calls and a separate
    ``system``/``tools`` column tokenizes end-to-end. ``tools`` is accepted as
    either a list[dict] (parquet-typed) or a JSON-encoded string."""
    example = {
        "messages": _tool_call_messages(),
        "tools": tools_value,
        "system": "You are a weather assistant.",
    }
    result = tokenize_chat_example(
        example,
        tokenizer,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    assert result is not None
    assert result["num_actions"] > 0
    assert sum(result["loss_mask"]) > 0
    decoded = tokenizer.decode(result["input_ids"])
    assert "get_weather" in decoded
    assert "You are a weather assistant." in decoded
    assert "<tool_call>" in decoded
    assert "<tool_response>" in decoded


def test_chat_tool_calling_no_tools_column(tokenizer):
    """A chat dataset without a ``tools`` or ``system`` column still tokenizes
    when ``tools_key``/``system_key`` are explicitly disabled."""
    example = {"messages": _tool_call_messages()}
    result = tokenize_chat_example(
        example,
        tokenizer,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        tools_key=None,
        system_key=None,
    )
    assert result is not None
    assert result["num_actions"] > 0


def test_chat_tool_calling_trailing_tool_turn_trimmed(tokenizer):
    """Rows that end with a trailing ``tool`` (or ``user``) turn — common in
    APIGen-MT — should still tokenize: the trailing non-assistant turns are
    trimmed so the conversation ends on the last assistant message."""
    msgs = _tool_call_messages()
    # Append a trailing tool turn with no follow-up assistant response.
    msgs.append({"role": "tool", "content": '{"ok": true}'})
    example = {"messages": msgs, "tools": _TOOLS_SCHEMA, "system": "You are a weather assistant."}
    result = tokenize_chat_example(
        example,
        tokenizer,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    assert result is not None
    assert result["num_actions"] > 0
    # The trailing observation must not appear in the tokenized stream.
    assert '{"ok": true}' not in tokenizer.decode(result["input_ids"])


def test_chat_tool_calling_tool_turn_masked(tokenizer):
    """Tokens from the ``tool`` observation turn must be fully masked (loss 0).

    We compute the byte range the tool turn would occupy and verify all
    overlapping loss-mask positions are 0.
    """
    example = {
        "messages": _tool_call_messages(),
        "tools": _TOOLS_SCHEMA,
        "system": "You are a weather assistant.",
    }
    result = tokenize_chat_example(
        example,
        tokenizer,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    assert result is not None

    decoded = tokenizer.decode(result["input_ids"])
    # The tool result is rendered inside <tool_response>…</tool_response>.
    assert "21" in decoded
    # Every tool-response token must be masked to 0. We re-tokenize the
    # observation segment in isolation and assert that all matching positions
    # in input_ids share the same mask=0 region.
    obs_marker_ids = tokenizer.encode("<tool_response>", add_special_tokens=False)
    input_ids = result["input_ids"]
    action_start = len(input_ids) - len(result["loss_mask"])
    # Find the index of the tool_response marker in the input_ids stream.
    found = -1
    for i in range(len(input_ids) - len(obs_marker_ids) + 1):
        if input_ids[i : i + len(obs_marker_ids)] == obs_marker_ids:
            found = i
            break
    assert found >= 0
    # Every position from the marker through the closing </tool_response>
    # marker must be masked to 0.
    close_ids = tokenizer.encode("</tool_response>", add_special_tokens=False)
    close = -1
    for i in range(found, len(input_ids) - len(close_ids) + 1):
        if input_ids[i : i + len(close_ids)] == close_ids:
            close = i + len(close_ids)
            break
    assert close > found
    for k in range(found, close):
        if k >= action_start:
            assert result["loss_mask"][k - action_start] == 0, f"tool-response position {k} must be masked to 0"


# ---------------------------------------------------------------------------
# _normalize_chat_messages: preserve fields like ``reasoning_content`` so
# thinking-model templates (Qwen3, Nemotron-3) keep working.
# ---------------------------------------------------------------------------


def test_normalize_preserves_unknown_extras_on_all_roles():
    msgs = [
        {"role": "system", "content": "sys", "metadata": {"k": 1}},
        {"role": "user", "content": "u", "name": "alice"},
        {
            "role": "assistant",
            "content": "a",
            "reasoning_content": "thought",
            "custom_field": "x",
        },
        {
            "role": "tool",
            "content": '{"result": 42}',
            "name": "calculator",
            "tool_call_id": "call_0",
        },
    ]
    out = _normalize_chat_messages(msgs)
    assert out[0]["metadata"] == {"k": 1}
    assert out[1]["name"] == "alice"
    assert out[2]["reasoning_content"] == "thought"
    assert out[2]["custom_field"] == "x"
    assert out[3]["name"] == "calculator"
    assert out[3]["tool_call_id"] == "call_0"


def test_normalize_drops_placeholder_tool_calls_but_keeps_extras():
    """Placeholder ``tool_calls="[]"`` is dropped on assistant messages even
    when other extras are present on the same row."""
    msgs = [
        {
            "role": "assistant",
            "content": "ok",
            "reasoning_content": "thinking...",
            "tool_calls": "[]",
        },
    ]
    out = _normalize_chat_messages(msgs)
    assert "tool_calls" not in out[0]
    assert out[0]["reasoning_content"] == "thinking..."


def test_normalize_strips_tool_calls_off_non_assistant_roles():
    """``tool_calls`` on user/tool/system rows is stripped (would fight the
    template) while other extras survive."""
    msgs = [
        {"role": "user", "content": "u", "tool_calls": "[]", "name": "alice"},
        {"role": "tool", "content": "t", "tool_calls": "[]", "tool_call_id": "c0"},
    ]
    out = _normalize_chat_messages(msgs)
    assert "tool_calls" not in out[0]
    assert out[0]["name"] == "alice"
    assert "tool_calls" not in out[1]
    assert out[1]["tool_call_id"] == "c0"


# ---------------------------------------------------------------------------
# _load_and_tokenize: num_workers > 0 (parallel path) and cache-hit behavior.
#
# The parallel-path test runs real multiprocessing -- no stubs on
# ``mp.get_context``, ``Pool``, or the slice worker. The dataset is written to
# disk as JSONL inside ``tmp_path`` so spawned workers (which re-import the
# module and lose any in-process monkeypatches) can ``load_dataset`` it via the
# directory path. The cache-hit test stays in-process with ``num_workers=0``
# and monkeypatches ``load_dataset`` to return an in-memory Dataset.
# ---------------------------------------------------------------------------


def _make_inmem_dataset():
    """Tiny chat-format Dataset used by parallel/cache tests. Six short rows so
    a 2-worker split is non-trivial (3 + 3) without spending real tokenizer time."""
    from datasets import Dataset

    rows = [
        {
            "messages": [
                {"role": "user", "content": f"Q{i}: what is {i} + {i}?"},
                {"role": "assistant", "content": f"{2 * i}"},
            ]
        }
        for i in range(6)
    ]
    return Dataset.from_list(rows)


def _write_inmem_dataset_to_dir(dataset_dir):
    """Materialize the in-memory chat fixture as JSONL inside ``dataset_dir``.

    The directory is structured so HF ``load_dataset(<dir>, split='train')``
    autodetects the JSON builder. Required for the real-mp parallel test --
    spawned workers re-import the module and cannot see in-process patches, so
    the data must live on disk where they can load it independently."""
    import json
    import os

    os.makedirs(dataset_dir, exist_ok=True)
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"Q{i}: what is {i} + {i}?"},
                {"role": "assistant", "content": f"{2 * i}"},
            ]
        }
        for i in range(6)
    ]
    with open(os.path.join(dataset_dir, "train.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _build_trainer(tokenizer, sft_cfg):
    """Build a bare SFTTrainer with just the attributes ``_load_and_tokenize`` reads."""
    from skyrl.train.sft_trainer import SFTTrainer

    trainer = object.__new__(SFTTrainer)
    trainer.sft_cfg = sft_cfg
    trainer.tokenizer = tokenizer
    trainer.is_vlm = False
    trainer.processor = None
    return trainer


def test_load_and_tokenize_num_workers_uses_parallel_path(tokenizer, tmp_path):
    """``num_workers > 0`` actually runs the real ``mp.Pool`` parallel path and
    produces bit-exact output vs ``num_workers=0``.

    The implementation uses ``mp.get_context("spawn")`` -- spawned workers
    re-import ``skyrl.train.sft_trainer``, so we can't monkeypatch around them.
    Instead the dataset is materialized as JSONL on disk and workers re-load it
    via ``load_dataset(<dir>, split=...)``. The serial reference run uses the
    same directory; equality of (input_ids, attention_mask, num_actions,
    loss_mask) across both paths verifies the parallel implementation."""
    dataset_dir = str(tmp_path / "ds")
    _write_inmem_dataset_to_dir(dataset_dir)

    # Serial reference: num_workers=0 path.
    serial_cfg = SFTConfig(num_workers=0, cache_dir=str(tmp_path / "ser_cache"))
    serial_trainer = _build_trainer(tokenizer, serial_cfg)
    serial_out = serial_trainer._load_and_tokenize(dataset_dir, "train")

    # Real-multiprocessing run: num_workers=2 path. With 6 rows that's a 3+3
    # split across two spawned worker processes.
    parallel_cfg = SFTConfig(num_workers=2, cache_dir=str(tmp_path / "par_cache"))
    parallel_trainer = _build_trainer(tokenizer, parallel_cfg)
    parallel_out = parallel_trainer._load_and_tokenize(dataset_dir, "train")

    assert len(parallel_out) == len(serial_out) == 6
    for par_ex, ser_ex in zip(parallel_out, serial_out):
        assert par_ex["input_ids"] == ser_ex["input_ids"]
        assert par_ex["attention_mask"] == ser_ex["attention_mask"]
        assert par_ex["num_actions"] == ser_ex["num_actions"]
        assert par_ex["loss_mask"] == ser_ex["loss_mask"]


def test_load_and_tokenize_cache_hit_skips_retokenization(tokenizer, tmp_path, monkeypatch):
    """With caching enabled (default), a second call hits the on-disk HF
    dataset cache and does NOT re-tokenize:
      - the inner per-row tokenizer is not called on the second run,
      - the cache directory exists at the resolved ``_get_cache_path`` path
        and contains an HF ``Dataset`` (``dataset_info.json`` is present),
      - the two calls return bit-exact equal output."""
    import os
    from unittest.mock import MagicMock

    from datasets import Dataset

    from skyrl.train import sft_trainer as sft_mod

    dataset = _make_inmem_dataset()
    monkeypatch.setattr(sft_mod, "load_dataset", lambda *a, **kw: dataset)

    cfg = SFTConfig(
        num_workers=0,  # serial path -- exercises tokenize_chat_example directly
        cache_dir=str(tmp_path),
        disable_cache=False,
        force_recache=False,
    )
    trainer = _build_trainer(tokenizer, cfg)

    # First call: cache miss -> tokenizes and writes the HF dataset to disk.
    first_out = trainer._load_and_tokenize("dummy/ds", "train")
    assert len(first_out) == 6

    # The cache directory must exist at the path derived from cache_dir + cache_key.
    cache_key = sft_mod._compute_cache_key(
        dataset_name="dummy/ds",
        dataset_split="train",
        model_path=cfg.model.path,
        max_length=cfg.max_length,
        messages_key=cfg.messages_key,
        train_on_what=cfg.train_on_what.value,
        tools_key=cfg.tools_key,
        system_key=cfg.system_key,
    )
    expected_cache_path = sft_mod._get_cache_path(str(tmp_path), cache_key)
    assert os.path.isdir(expected_cache_path), f"cache directory missing at {expected_cache_path}"
    # ``dataset_info.json`` is part of every HF dataset
    # written this way, and ``load_from_disk`` round-trips the rows verbatim.
    assert os.path.isfile(
        os.path.join(expected_cache_path, "dataset_info.json")
    ), f"cache at {expected_cache_path} is not an HF dataset"
    cached_rows = Dataset.load_from_disk(expected_cache_path).to_list()
    assert len(cached_rows) == len(first_out)

    # Second call: cache hit. Spy on tokenize_chat_example AND on _load_from_cache
    # so we can prove (a) no re-tokenization happens and (b) the cached dataset
    # is what served the result.
    tok_spy = MagicMock(side_effect=AssertionError("tokenize_chat_example must not be called on cache hit"))
    monkeypatch.setattr(sft_mod, "tokenize_chat_example", tok_spy)
    load_spy = MagicMock(wraps=sft_mod._load_from_cache)
    monkeypatch.setattr(sft_mod, "_load_from_cache", load_spy)

    second_out = trainer._load_and_tokenize("dummy/ds", "train")

    assert tok_spy.call_count == 0, "cache hit must not re-tokenize"
    assert load_spy.call_count == 1, "cache hit must go through _load_from_cache exactly once"
    assert load_spy.call_args[0][0] == expected_cache_path

    # Bit-exact equality between fresh-tokenize and cache-hit outputs.
    assert len(second_out) == len(first_out)
    for a, b in zip(first_out, second_out):
        assert a["input_ids"] == b["input_ids"]
        assert a["attention_mask"] == b["attention_mask"]
        assert a["num_actions"] == b["num_actions"]
        assert a["loss_mask"] == b["loss_mask"]
