"""
Tests for OpenAI request utils.

uv run --extra dev --isolated pytest tests/cpu/http/test_openai_request_utils.py
"""

import pytest
from skyrl_train.inference_engines.openai_api_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    check_unsupported_fields,
)
from skyrl_train.inference_engines.base import InferenceEngineOutput
from skyrl_train.inference_engines.inference_engine_client_http_endpoint import (
    convert_openai_to_inference_input,
    convert_inference_output_to_openai,
)


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def _basic_request(**kwargs):
    return ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[ChatMessage(role="user", content="hi")],
        **kwargs,
    )


def test_check_unsupported_fields():
    req = _basic_request(tools=[{"type": "function", "function": {"name": "t"}}])
    with pytest.raises(ValueError):
        check_unsupported_fields(req)

    req_ok = _basic_request()
    check_unsupported_fields(req_ok)


def test_convert_openai_to_inference_input():
    """Test conversion with multiple messages in conversation."""
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="What is the capital of France?"),
        ChatMessage(role="assistant", content="The capital of France is Paris."),
    ]
    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=10,
        temperature=0.5,
        trajectory_id="test_trajectory_123",
    )

    for backend in ["vllm", "sglang"]:
        result = convert_openai_to_inference_input(req, backend)

        expected_conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ]

        assert result["prompts"] == [expected_conversation]
        assert result["trajectory_ids"] == ["test_trajectory_123"]
        assert result["sampling_params"] is not None
        assert result["sampling_params"]["temperature"] == 0.5
        if backend == "vllm":
            assert result["sampling_params"]["max_tokens"] == 10
        elif backend == "sglang":
            assert result["sampling_params"]["max_new_tokens"] == 10


def test_convert_inference_output_to_openai():
    """Test basic conversion from InferenceEngineOutput to OpenAI response."""
    response = "Hello! How can I help you today?"
    stop_reason = "stop"
    engine_output = InferenceEngineOutput(
        responses=[response],
        stop_reasons=[stop_reason],
    )

    result = convert_inference_output_to_openai(engine_output, MODEL_NAME)
    assert result.model == MODEL_NAME

    # Check response structure
    assert isinstance(result, ChatCompletionResponse)
    assert result.object == "chat.completion"
    assert isinstance(result.created, int)
    assert result.id.startswith("chatcmpl-")

    # Check choices
    assert len(result.choices) == 1
    choice = result.choices[0]
    assert choice.index == 0
    assert choice.message.role == "assistant"
    assert choice.message.content == response
    assert choice.finish_reason == stop_reason
