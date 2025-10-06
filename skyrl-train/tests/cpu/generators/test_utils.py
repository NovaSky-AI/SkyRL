"""
uv run --extra dev --isolated pytest tests/cpu/generators/test_utils.py
"""

import pytest
from skyrl_train.generators.utils import apply_overlong_filtering, encode_messages_subset
from transformers import AutoTokenizer


@pytest.mark.parametrize(
    "loss_masks,response_ids,eos_token_id,expected_masks",
    [
        # Test case 1: All responses end with eos token - masks should remain unchanged
        (
            [[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1]],
            [[1, 2, 3, 4], [5, 6, 7, 4], [8, 9, 4]],  # All end with eos_token_id=4
            4,
            [[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1]],
        ),
        # Test case 2: No responses end with eos token - all masks should be zeroed
        (
            [[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1]],
            [[1, 2, 3, 5], [5, 6, 7, 8], [8, 9, 10]],  # None end with eos_token_id=4
            4,
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0]],
        ),
        # Test case 3: Mixed responses - only non-eos ending masks should be zeroed
        (
            [[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 0, 1]],
            [[1, 2, 3, 4], [5, 6, 7, 8], [8, 9, 10, 11, 4]],  # First and third end with eos_token_id=4
            4,
            [[1, 1, 0, 1], [0, 0, 0, 0], [1, 0, 1, 0, 1]],
        ),
        # Test case 4: Empty responses should be zeroed
        (
            [[1, 1], [1, 0, 1], [0, 1, 1, 1]],
            [[], [1, 2, 3], [4, 5, 6, 7]],  # Empty, no eos, no eos (eos_token_id=4)
            4,
            [[0, 0], [0, 0, 0], [0, 0, 0, 0]],
        ),
        # Test case 5: Empty lists
        ([], [], 4, []),
        # Test case 6: Different eos token id
        (
            [[1, 1], [1, 0, 1], [0, 1, 1, 1]],
            [[1, 2], [3, 4, 99], [5, 6, 7, 99]],  # Second and third end with eos_token_id=99
            99,
            [[0, 0], [1, 0, 1], [0, 1, 1, 1]],
        ),
    ],
)
def test_apply_overlong_filtering(loss_masks, response_ids, eos_token_id, expected_masks):
    """
    Test the apply_overlong_filtering function which implements DAPO Overlong Filtering.

    This function should zero-out every token's mask whenever the response does not end
    with the eos token id (i.e. truncated), while leaving other masks unchanged.
    """
    result = apply_overlong_filtering(loss_masks, response_ids, eos_token_id)

    assert result == expected_masks, f"Expected {expected_masks}, but got {result}"

    # Verify that the original inputs are not modified (immutability check)
    assert len(result) == len(loss_masks), "Result should have same length as input"

    # Check that each individual mask is processed correctly
    for i, (original_mask, response, expected_mask) in enumerate(zip(loss_masks, response_ids, expected_masks)):
        if len(response) == 0 or response[-1] != eos_token_id:
            # Should be all zeros with same length as original
            assert result[i] == [0] * len(original_mask), f"Mask {i} should be all zeros for truncated response"
        else:
            # Should be unchanged
            assert result[i] == original_mask, f"Mask {i} should be unchanged for response ending with eos token"


def test_apply_overlong_filtering_immutability():
    """
    Test that apply_overlong_filtering doesn't modify the original input lists.
    """
    original_loss_masks = [[1, 1, 0, 1], [0, 1, 1]]
    original_response_ids = [[1, 2, 3, 4], [5, 6, 7]]  # First ends with eos=4, second doesn't
    eos_token_id = 4

    # Create copies to compare against later
    loss_masks_copy = [mask[:] for mask in original_loss_masks]  # Deep copy of lists
    response_ids_copy = [response[:] for response in original_response_ids]  # Deep copy of lists

    result = apply_overlong_filtering(original_loss_masks, original_response_ids, eos_token_id)

    # Verify original inputs are unchanged
    assert original_loss_masks == loss_masks_copy, "Original loss_masks should not be modified"
    assert original_response_ids == response_ids_copy, "Original response_ids should not be modified"

    # Verify result is correct
    expected = [[1, 1, 0, 1], [0, 0, 0]]  # Second mask zeroed due to not ending with eos
    assert result == expected, f"Expected {expected}, got {result}"


@pytest.mark.parametrize(
    "loss_masks,response_ids",
    [
        # Test case 1: More loss_masks than response_ids
        ([[1, 1], [0, 1]], [[1, 2]]),
        # Test case 2: More response_ids than loss_masks
        ([[1, 1]], [[1, 2], [3, 4]]),
        # Test case 3: Empty loss_masks but non-empty response_ids
        ([], [[1, 2]]),
        # Test case 4: Non-empty loss_masks but empty response_ids
        ([[1, 0]], []),
    ],
)
def test_apply_overlong_filtering_length_mismatch_assertion(loss_masks, response_ids):
    """
    Test that apply_overlong_filtering raises AssertionError when loss_masks and response_ids
    have different lengths.
    """
    eos_token_id = 4
    with pytest.raises(AssertionError, match="loss_masks and response_ids must have the same length"):
        apply_overlong_filtering(loss_masks, response_ids, eos_token_id)


dummy_chat_template = (
    "{%- for message in messages %}"
    "{%- if message['role'] == 'user' %}"
    "<USER>{{ message['content'] }}</s>\n"
    "{%- elif message['role'] == 'assistant' %}"
    "<ASSISTANT>{{ message['content'] }}</s>\n"
    "{%- elif message['role'] == 'system' %}"
    "<SYSTEM>{{ message['content'] }}</s>\n"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "<ASSISTANT>"
    "{%- endif %}"
)


@pytest.fixture
def tokenizer_w_dummy_template():
    tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-2-7b")
    tokenizer.chat_template = dummy_chat_template
    return tokenizer


@pytest.mark.parametrize(
    "messages",
    [
        # Test case 1: Single assistant message
        [{"role": "assistant", "content": "Hello, I can help you."}],
        # Test case 2: Single user message
        [{"role": "user", "content": "What is the weather today?"}],
        # Test case 3: Multiple messages (user-assistant exchange)
        [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "The answer is 4."}],
        # Test case 4: Multiple messages starting with assistant
        [
            {"role": "assistant", "content": "I'm here to help."},
            {"role": "user", "content": "Can you explain Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
        ],
    ],
)
def test_encode_messages(messages, tokenizer_w_dummy_template):
    expected_token_ids = tokenizer_w_dummy_template.apply_chat_template(messages)
    actual_token_ids = encode_messages_subset(messages, tokenizer_w_dummy_template)
    assert expected_token_ids == actual_token_ids


@pytest.fixture
def qwen_tokenizer():
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")


@pytest.mark.parametrize(
    "messages, expected_str",
    [
        # Test case 1: Single assistant message
        (
            [{"role": "assistant", "content": "Hello, I can help you."}],
            "<|im_start|>assistant\nHello, I can help you.<|im_end|>",
        ),
        # Test case 2: Single user message - additional \n because the expectation is that there is a previous assistant turn
        (
            [{"role": "user", "content": "What is the weather today?"}],
            "\n<|im_start|>user\nWhat is the weather today?<|im_end|>\n",
        ),
        # Test case 3: Multiple messages (user-assistant exchange)
        (
            [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "The answer is 4."}],
            # NOTE: Additional \n because the expectation is that there is a previous assistant turn.
            # All tokens after EOS in the previous turn get pushed into the next user/tool message.
            "\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\nThe answer is 4.<|im_end|>",
        ),
        # Test case 4: Multiple messages starting with assistant
        (
            [
                {"role": "assistant", "content": "I'm here to help."},
                {"role": "user", "content": "Can you explain Python?"},
                {"role": "assistant", "content": "Python is a programming language."},
            ],
            "<|im_start|>assistant\nI'm here to help.<|im_end|>\n<|im_start|>user\nCan you explain Python?<|im_end|>\n<|im_start|>assistant\nPython is a programming language.<|im_end|>",
        ),
    ],
)
def test_encode_messages_qwen(messages, expected_str, qwen_tokenizer):
    expected_token_ids = qwen_tokenizer.encode(expected_str)
    actual_token_ids = encode_messages_subset(messages, qwen_tokenizer)
    assert expected_token_ids == actual_token_ids, f"Got actual tokens: {qwen_tokenizer.decode(actual_token_ids)}"
