"""Message history management and conversation utilities."""

import json
from typing import Any, Dict, List, Optional, Tuple
import copy

from skyrl_agent.functional.function_calling import (
    convert_fncall_messages_to_non_fncall_messages,
    convert_non_fncall_messages_to_fncall_messages,
    FunctionCallConversionError,
    FunctionCallValidationError,
)
from skyrl_agent.functional.chat_template import get_templates_path


class MessageHistory:
    """Manages conversation message history with convenient add methods."""

    def __init__(self):
        self.messages: List[Dict] = []
        self._was_reset: bool = False

    def add_assistant(self, content: str) -> None:
        """Add assistant message."""
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_error(self, error: str, tool_call_id: Optional[str] = None) -> None:
        """Add tool error message."""
        msg = {
            "role": "tool",
            "content": json.dumps({"error": error}),
        }
        if tool_call_id:
            msg["tool_call_id"] = tool_call_id
        self.messages.append(msg)

    def add_tool_response(self, output: Any, tool_call_id: str) -> None:
        """Add tool response message."""
        self.messages.append(
            {
                "role": "tool",
                "content": json.dumps(output),
                "tool_call_id": tool_call_id,
            }
        )

    def add_user_guidance(self, guidance: str) -> None:
        """Add user guidance message."""
        self.messages.append({"role": "user", "content": guidance})

    def append_to_last_message(self, content: str) -> None:
        """Append content to the last message in history.

        Args:
            content: Content to append to the last message
        """
        if not self.messages:
            raise ValueError("Cannot append to last message: history is empty")
        self.messages[-1]["content"] += content

    def add_turn_reminder(self, reminder_text: str) -> None:
        """Add a turn reminder to the last message in history.

        This is a convenience method that appends reminder text to the last message.

        Args:
            reminder_text: Reminder text to append to the last message
        """
        self.append_to_last_message(reminder_text)

    def initialize(self, messages: List[Dict]) -> None:
        """Initialize history with messages.

        Sets the _was_reset flag to True to indicate history was reset.
        This flag can be checked to determine if state like prompt_token_len
        needs to be reset.
        """
        # deep copy the messages
        self.messages = copy.deepcopy(messages)
        self._was_reset = True

    def clear_reset_flag(self) -> None:
        """Clear the reset flag after it has been handled."""
        self._was_reset = False

    def was_reset(self) -> bool:
        """Check if history was recently reset."""
        return self._was_reset

    def get_messages(self) -> List[Dict]:
        """Get all messages."""
        return self.messages

    def __len__(self) -> int:
        """Get number of messages."""
        return len(self.messages)


class MessageEncoder:
    """Handles encoding messages to input_ids for LLM."""

    def __init__(self, tokenizer, qwen3_enable_thinking: bool = False, qwen3_acc_thinking: bool = False):
        self.tokenizer = tokenizer
        self.qwen3_enable_thinking = qwen3_enable_thinking
        self.chat_template = get_templates_path() / "qwen3_acc_thinking.jinja2" if qwen3_acc_thinking else None

    def encode_messages(
        self,
        messages: List[Dict],
        tool_params: List[Dict],
        is_first_message: bool = False,
        add_generation: bool = True,
    ) -> List[int]:
        """Convert messages to input_ids.

        Args:
            messages: List of conversation messages
            tool_params: List of tool parameter definitions
            is_first_message: Whether this is the first message in the conversation
            add_generation: Whether to add generation prompt

        Returns:
            List of token IDs

        Examples:
            Case 1 - First message (is_first_message=True):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"}
                ]
                # Encodes the full conversation from scratch with generation prompt
                # Formatted as:
                # <im_start>system\nYou are a helpful assistant.<im_end>\n<im_start>user\nHello!<im_end>\n<im_start>assistant\n
                input_ids = encoder.encode_messages(messages, tool_params, is_first_message=True)

            Case 2 - Not first message, starting from assistant (is_first_message=False):
                messages = [
                    {"role": "assistant", "content": "Hi! How can I help you?"}
                ]
                # Performs incremental encoding for assistant response
                # Base conversation: <im_start>system\nYou are a helpful assistant.<im_end>\n<im_start>user\nI am a user.<im_end>\n<im_start>assistant\n
                # assistant response: Hi! How can I help you?<im_end>\n
                # Returns only the new tokens after base conversation

            Case 3 - Not first message, starting from user (is_first_message=False):
                messages = [
                    {"role": "user", "content": "Tell me a joke"}
                ]
                # Base conversation: <im_start>system\nYou are a helpful assistant.<im_end>\n<im_start>user\nI am a user.<im_end>\n
                # (Note: For user messages, tokens after last EOS are removed from base)
                # user message: \n<im_start>user\nTell me a joke<im_end>\n<im_start>assistant\n
        """
        formatted_messages = convert_fncall_messages_to_non_fncall_messages(messages, tool_params)
        kwargs = {
            "add_generation_prompt": add_generation,
            "tokenize": True,
        }
        if self.chat_template:
            kwargs["chat_template"] = self.chat_template.read_text()
        kwargs["enable_thinking"] = self.qwen3_enable_thinking
        if is_first_message:
            input_ids = self.tokenizer.apply_chat_template(formatted_messages, **kwargs)
        else:
            # do incremental encoding,
            # for assistant messages, we assume the generation prompt is already added in the previous message
            base_conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "I am a user."},
            ]
            is_assistant_message = formatted_messages[0]["role"] == "assistant"
            kwargs["add_generation_prompt"] = True if is_assistant_message else False
            base_conversation_token_ids = self.tokenizer.apply_chat_template(base_conversation, **kwargs)

            if not is_assistant_message:
                # remove tokens after the last EOS
                last_eos_token_index = (
                    len(base_conversation_token_ids)
                    - 1
                    - base_conversation_token_ids[::-1].index(self.tokenizer.eos_token_id)
                )
                base_conversation_token_ids = base_conversation_token_ids[: last_eos_token_index + 1]
            kwargs["add_generation_prompt"] = add_generation
            full_conversation = base_conversation + formatted_messages
            full_conversation_token_ids = self.tokenizer.apply_chat_template(full_conversation, **kwargs)
            input_ids = full_conversation_token_ids[len(base_conversation_token_ids) :]

        return input_ids


# Tool call parsing and extraction


def parse_tool_call(
    response_str: str,
    tool_params: List[Dict],
) -> Tuple[Optional[Dict], Optional[str]]:
    """Parse tool call from LLM response.

    Args:
        response_str: Raw LLM response text
        tool_params: List of tool parameter definitions

    Returns:
        Tuple of (tool_call_dict, error_message)
        - If successful: (tool_call, None)
        - If failed: (None, error_message)
    """
    assistant_msg = {"role": "assistant", "content": response_str}

    try:
        fncall_messages = convert_non_fncall_messages_to_fncall_messages([assistant_msg], tool_params)
        tool_call = fncall_messages[0].get("tool_calls", [None])[0]
        return tool_call, None

    except (FunctionCallValidationError, FunctionCallConversionError) as e:
        return None, str(e)


def extract_tool_info(tool_call: Optional[Dict]) -> Tuple[Optional[str], Optional[Dict]]:
    """Extract tool name and arguments from tool call.

    Args:
        tool_call: Parsed tool call dictionary

    Returns:
        Tuple of (tool_name, tool_args)
    """
    if not tool_call:
        return None, None

    tool_name = tool_call.get("function", {}).get("name")
    tool_args = tool_call.get("function", {}).get("arguments")
    return tool_name, tool_args


def check_truncated_tool_call(response_str: str) -> bool:
    """Check if response contains a truncated tool call.

    Args:
        response_str: Raw LLM response text

    Returns:
        True if tool call appears truncated, False otherwise
    """
    return "<function=" in response_str and not response_str.strip().endswith("</function>")


def format_output_preview(output: Any, max_length: int = 400) -> str:
    """Format tool output for logging/preview.

    Args:
        output: Tool output to format
        max_length: Maximum length of preview

    Returns:
        Formatted string preview
    """
    try:
        out_str = json.dumps(output)
    except Exception:
        out_str = str(output)

    out_str = out_str.replace("\n", " ")
    if len(out_str) > max_length:
        out_str = out_str[:max_length] + "..."

    return out_str
