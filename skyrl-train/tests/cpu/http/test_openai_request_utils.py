"""
# Run only vllm tests (requires vllm extra):
uv run --isolated --extra dev --extra vllm pytest tests/cpu/http/test_openai_request_utils.py -m "vllm"
# Run only sglang tests (requires sglang extra):
uv run --isolated --extra dev --extra sglang pytest tests/cpu/http/test_openai_request_utils.py -m "sglang"
"""

import pytest
from skyrl_train.inference_engines.openai_api_protocol import (
    ChatCompletionRequest,
    ChatMessage,
    check_unsupported_fields,
    build_sampling_params,
)


def _basic_request(**kwargs):
    return ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="hi")],
        **kwargs,
    )

def _full_request():
    return ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="hi")],
        max_tokens=10,
        temperature=0.5,
        top_p=0.9,
        top_k=40,
        min_p=0.0,
        repetition_penalty=1.0,
        seed=42,
        stop=["\n"],
        stop_token_ids=[2, 3],
        presence_penalty=0.0,
        frequency_penalty=0.0,
        ignore_eos=True,
        skip_special_tokens=True,
        include_stop_str_in_output=True,
        min_tokens=1,
        n=1,
        trajectory_id="test_trajectory_id",
    )


def test_check_unsupported_fields():
    req = _basic_request(tools=[{"type": "function", "function": {"name": "t"}}])
    with pytest.raises(ValueError):
        check_unsupported_fields(req)

    req_ok = _basic_request()
    check_unsupported_fields(req_ok)


def test_basic_build_sampling_params():
    req = _basic_request(max_tokens=5, temperature=0.5)
    params_vllm = build_sampling_params(req, "vllm")
    assert params_vllm["max_tokens"] == 5
    assert params_vllm["temperature"] == 0.5

    params_sglang = build_sampling_params(req, "sglang")
    assert params_sglang["max_new_tokens"] == 5
    assert params_sglang["temperature"] == 0.5

@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("vllm", marks=pytest.mark.vllm),
        pytest.param("sglang", marks=pytest.mark.sglang),
    ]
)
def test_full_build_sampling_params(backend: str):
    full_req = _full_request()
    if backend == "vllm":
        from vllm import SamplingParams as VLLMSamplingParams
        full_params_vllm = build_sampling_params(full_req, "vllm")
        vllm_sampling_params = VLLMSamplingParams(**full_params_vllm)  # has __post_init__ to check validity
        assert vllm_sampling_params is not None
    elif backend == "sglang":
        from sglang.srt.sampling.sampling_params import SamplingParams as SGLangSamplingParams
        # makes sure that the inclusion of `include_stop_str_in_output` will raise an error
        with pytest.raises(ValueError):
            full_params_sglang = build_sampling_params(full_req, "sglang")
        full_req.include_stop_str_in_output = None

        # makes sure that the inclusion of `seed` will raise an error
        with pytest.raises(ValueError):
            # makes sure that the inclusion of `seed` will raise an error
            full_params_sglang = build_sampling_params(full_req, "sglang")
        full_req.seed = None

        # makes sure that the inclusion of `min_tokens` will raise an error
        with pytest.raises(ValueError):
            full_params_sglang = build_sampling_params(full_req, "sglang")
        full_req.min_tokens = None

        # Now no errors should be raised
        full_params_sglang = build_sampling_params(full_req, "sglang")
        sglang_sampling_params = SGLangSamplingParams(**full_params_sglang)
        sglang_sampling_params.verify()  # checks validty
        assert sglang_sampling_params is not None
    else:
        raise ValueError(f"Unsupported backend: {backend}")
