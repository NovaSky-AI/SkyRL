"""Tests for vLLM-owned OpenAI protocol handling."""

import msgspec
import pytest
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionResponse
from vllm.entrypoints.openai.engine.protocol import ErrorResponse

from skyrl.train.generators.tito.vllm_openai import (
    OpenAIProtocolError,
    build_chat_response,
    build_sampling_params,
    parse_chat_request,
)


def test_vllm_normalizes_messages_tools_and_supported_sampling_fields():
    parsed = parse_chat_request(
        {
            "model": "model",
            "messages": [
                {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "thinking",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Look up a value",
                        "parameters": {
                            "type": "object",
                            "properties": {"key": {"type": "string"}},
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "min_p": 0.2,
            "seed": 7,
        },
        registered_model="model",
    )

    assert parsed.messages == (
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "thinking",
        },
    )
    assert parsed.tools[0]["function"]["name"] == "lookup"

    sampling = build_sampling_params(
        parsed,
        prompt_token_count=4,
        max_model_len=32,
        renderer_stop_token_ids=[2],
    )
    expected = msgspec.to_builtins(
        parsed.request.to_sampling_params(
            max_tokens=28,
            default_sampling_params={},
        )
    )
    expected["logprobs"] = 1
    expected["stop_token_ids"] = [2]
    assert sampling == expected
    assert sampling["min_p"] == 0.2
    assert sampling["seed"] == 7


def test_invalid_request_uses_vllm_error_response_shape():
    with pytest.raises(OpenAIProtocolError) as exc_info:
        parse_chat_request(
            {
                "model": "model",
                "messages": [{"role": "user", "content": "hello"}],
                "n": 2,
            },
            registered_model="model",
        )

    error = ErrorResponse.model_validate(exc_info.value.body)
    assert error.error.code == 400
    assert "n=1" in error.error.message


def test_empty_tools_follow_vllm_validation():
    with pytest.raises(OpenAIProtocolError, match="empty array"):
        parse_chat_request(
            {
                "model": "model",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [],
            },
            registered_model="model",
        )


def test_sampling_rejects_prompt_larger_than_model_limit():
    parsed = parse_chat_request(
        {
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
        },
        registered_model="model",
    )

    with pytest.raises(OpenAIProtocolError, match="model"):
        build_sampling_params(
            parsed,
            prompt_token_count=9,
            max_model_len=8,
            renderer_stop_token_ids=[],
        )


def test_sampling_uses_skyrl_default_when_request_omits_token_limit():
    parsed = parse_chat_request(
        {
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
        },
        registered_model="model",
    )

    sampling = build_sampling_params(
        parsed,
        prompt_token_count=4,
        max_model_len=32,
        renderer_stop_token_ids=[],
        default_max_tokens=6,
    )

    assert sampling["max_tokens"] == 6


def test_response_validates_with_vllm_and_preserves_tito_fields():
    parsed = parse_chat_request(
        {
            "model": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "logprobs": True,
            "return_token_ids": True,
        },
        registered_model="model",
    )
    response = build_chat_response(
        parsed=parsed,
        assistant_message={
            "role": "assistant",
            "content": "done",
            "reasoning_content": "thinking",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"key":"x"}'},
                }
            ],
        },
        prompt_token_ids=[10, 11],
        completion_ids=[50],
        completion_logprobs=[-0.2],
        finish_reason="stop",
        decode_token=lambda token_id: str(token_id),
    )

    validated = ChatCompletionResponse.model_validate(response)
    choice = validated.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.reasoning == "thinking"
    assert choice.message.tool_calls[0].function.name == "lookup"
    assert choice.token_ids == [50]
    assert response["choices"][0]["message"]["reasoning_content"] == "thinking"
    assert response["prompt_token_ids"] == [10, 11]
