import pytest
from skyrl_train.inference_engines.openai_api_protocol import (
    ChatCompletionRequest,
    ChatMessage,
    check_unsupported_fields,
    build_sampling_params,
)


def _basic_request(**kwargs):
    return ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="hi")],
        **kwargs,
    )


def test_check_unsupported_fields():
    req = _basic_request(tools=[{"type": "function", "function": {"name": "t"}}])
    with pytest.raises(ValueError):
        check_unsupported_fields(req)

    req_ok = _basic_request()
    check_unsupported_fields(req_ok)


def test_build_sampling_params():
    req = _basic_request(max_tokens=5, temperature=0.5)
    params_vllm = build_sampling_params(req, "vllm")
    assert params_vllm["max_tokens"] == 5
    assert params_vllm["temperature"] == 0.5

    params_sglang = build_sampling_params(req, "sglang")
    assert params_sglang["max_new_tokens"] == 5
    assert params_sglang["temperature"] == 0.5
