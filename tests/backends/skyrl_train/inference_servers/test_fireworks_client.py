"""Tests for FireworksInferenceClient and the fireworks sampling-params conversion.

CPU-only and offline: an ``httpx.MockTransport`` is injected into the real fireworks SDK via the
client's ``http_client`` parameter, so the tests exercise the full request path (URL construction,
auth header, JSON body serialization) and the full response path (SDK typed parsing -> the
``InferenceEngineOutput`` contract).
"""

import json
import sys

import httpx
import pytest
from omegaconf import OmegaConf

fireworks = pytest.importorskip("fireworks", reason="requires the `fireworks` extra")

from skyrl.backends.skyrl_train.inference_servers.base import (  # noqa: E402
    InferenceEngineInput,
)
from skyrl.backends.skyrl_train.inference_servers.engine_utils import (  # noqa: E402
    get_fireworks_sampling_params,
    get_sampling_params_for_backend,
)
from skyrl.backends.skyrl_train.inference_servers.fireworks_client import (  # noqa: E402
    DEFAULT_FIREWORKS_BASE_URL,
    FireworksInferenceClient,
)
from skyrl.train.config import SamplingParams, SkyRLTrainConfig  # noqa: E402
from skyrl.train.entrypoints.main_base import BasePPOExp  # noqa: E402
from skyrl.train.entrypoints.main_generate import EvalOnlyEntrypoint  # noqa: E402


class _FakeTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "T:" + ",".join(map(str, ids))


def _payload(choices):
    return {"id": "cmpl-1", "object": "text_completion", "created": 1, "model": "m", "choices": choices}


def _make_client(payload, captured, status=200, api_key="fw-x"):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["json"] = json.loads(request.content)
        return httpx.Response(status, json=payload)

    return FireworksInferenceClient(
        model_name="accounts/fw/qwen3-4b",
        tokenizer=_FakeTokenizer(),
        base_url="https://api.fireworks.ai/inference",
        api_key=api_key,
        max_retries=0,
        _http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def test_fireworks_params_drop_vllm_keys_and_nones():
    params = get_fireworks_sampling_params(
        SamplingParams(
            max_generate_length=64,
            top_k=-1,
            min_p=0.0,
            additional_kwargs={"min_tokens": 1, "skip_special_tokens": True, "include_stop_str_in_output": True},
        )
    )
    assert "min_tokens" not in params and "skip_special_tokens" not in params
    assert "include_stop_str_in_output" not in params
    assert "top_k" not in params and "min_p" not in params
    assert params["max_tokens"] == 64
    assert None not in params.values()  # forwarded via extra_body, so no JSON nulls
    assert get_sampling_params_for_backend("fireworks", SamplingParams()) == get_fireworks_sampling_params(
        SamplingParams()
    )


def test_fireworks_params_in_range_and_passthrough():
    params = get_fireworks_sampling_params(SamplingParams(top_k=40, min_p=0.05, stop=["</s>"]))
    assert params["top_k"] == 40 and params["min_p"] == 0.05 and params["stop"] == ["</s>"]


def test_fireworks_params_out_of_range_top_k_dropped():
    assert "top_k" not in get_fireworks_sampling_params(SamplingParams(top_k=500))


def test_fireworks_params_dictconfig_sanitized_after_merge():
    cfg = OmegaConf.create(
        {
            "max_generate_length": 32,
            "temperature": 0.5,
            "top_p": 1.0,
            "min_p": 0.0,
            "top_k": -1,
            "logprobs": None,
            "stop": None,
            "skip_special_tokens": True,
            "repetition_penalty": 1.1,
        }
    )
    params = get_fireworks_sampling_params(cfg)
    assert "top_k" not in params and "min_p" not in params
    assert "logprobs" not in params and "stop" not in params
    assert "skip_special_tokens" not in params  # sanitized after the DictConfig merge
    assert params["repetition_penalty"] == 1.1  # supported keys pass through
    assert params["max_tokens"] == 32


@pytest.mark.asyncio
async def test_generate_auth_tokenids_model_url():
    captured = {}
    eng = _make_client(
        _payload(
            [
                {"index": 0, "text": "a", "finish_reason": "stop", "token_ids": [7, 8], "prompt_token_ids": [1, 2]},
                {"index": 1, "text": "b", "finish_reason": "length", "token_ids": [9], "prompt_token_ids": [3]},
            ]
        ),
        captured,
    )
    out = await eng.generate(
        InferenceEngineInput(
            prompt_token_ids=[[1, 2], [3]],
            # logprobs=None: the mock payload carries no logprobs, and requesting them without a
            # logprobs field in the response is a hard error (see test_requested_logprobs_missing_raises).
            sampling_params=get_fireworks_sampling_params(SamplingParams(max_generate_length=16, logprobs=None)),
        ),
        model="accounts/fw/qwen3-4b",
    )
    # The SDK appends /v1/completions to the configured server root.
    assert captured["url"] == "https://api.fireworks.ai/inference/v1/completions"
    assert captured["auth"] == "Bearer fw-x"
    assert captured["json"]["model"] == "accounts/fw/qwen3-4b"
    assert captured["json"]["return_token_ids"] is True
    assert captured["json"]["prompt"] == [[1, 2], [3]]
    assert captured["json"]["max_tokens"] == 16  # extra_body merged into the request body
    assert out["response_ids"] == [[7, 8], [9]]  # provider token ids, verbatim
    assert out["responses"] == ["T:7,8", "T:9"]  # locally decoded from the returned ids
    assert out["stop_reasons"] == ["stop", "length"]


@pytest.mark.asyncio
async def test_missing_token_ids_errors_out():
    eng = _make_client(_payload([{"index": 0, "text": "abc", "finish_reason": "stop"}]), {})
    with pytest.raises(AssertionError, match="token_ids"):
        await eng.generate(InferenceEngineInput(prompt_token_ids=[[1]], sampling_params={}))


@pytest.mark.asyncio
async def test_generate_extracts_legacy_logprobs():
    # Verified against the live endpoint (gpt-oss-20b): integer `logprobs` yields the legacy
    # LogProbs shape with `token_logprobs` aligned 1:1 with `token_ids`.
    eng = _make_client(
        _payload(
            [
                {
                    "index": 0,
                    "text": "a",
                    "finish_reason": "stop",
                    "token_ids": [7, 8, 9],
                    "logprobs": {
                        "tokens": ["a", "b", "c"],
                        "token_ids": [7, 8, 9],
                        # a null entry must map to 0.0 (not be dropped) to stay aligned with token_ids
                        "token_logprobs": [-0.5, None, -1.25],
                    },
                }
            ]
        ),
        {},
    )
    out = await eng.generate(InferenceEngineInput(prompt_token_ids=[[1]], sampling_params={"logprobs": 1}))
    assert out["response_logprobs"] == [[-0.5, 0.0, -1.25]]


@pytest.mark.asyncio
async def test_generate_extracts_openai_style_logprobs():
    # The OpenAI chat-style shape (NewLogProbs): `content` items each carrying `.logprob`.
    content = [
        {"token": "a", "bytes": [97], "logprob": -0.1, "text_offset": 0, "token_id": 7},
        {"token": "b", "bytes": [98], "logprob": -0.2, "text_offset": 1, "token_id": 8},
    ]
    eng = _make_client(
        _payload(
            [
                {
                    "index": 0,
                    "text": "ab",
                    "finish_reason": "stop",
                    "token_ids": [7, 8],
                    "logprobs": {"content": content},
                }
            ]
        ),
        {},
    )
    out = await eng.generate(InferenceEngineInput(prompt_token_ids=[[1]], sampling_params={"logprobs": 1}))
    assert out["response_logprobs"] == [[-0.1, -0.2]]


@pytest.mark.asyncio
async def test_requested_logprobs_missing_raises():
    # A silent None here would crash much later in GeneratorOutput validation; fail at the source.
    eng = _make_client(
        _payload([{"index": 0, "text": "a", "finish_reason": "stop", "token_ids": [7]}]),
        {},
    )
    with pytest.raises(RuntimeError, match="requested logprobs"):
        await eng.generate(InferenceEngineInput(prompt_token_ids=[[1]], sampling_params={"logprobs": 1}))


@pytest.mark.asyncio
async def test_logprobs_not_requested_stay_none():
    eng = _make_client(
        _payload([{"index": 0, "text": "a", "finish_reason": "stop", "token_ids": [7]}]),
        {},
    )
    out = await eng.generate(InferenceEngineInput(prompt_token_ids=[[1]], sampling_params={}))
    assert out["response_logprobs"] is None


@pytest.mark.asyncio
async def test_keyless_sends_placeholder_not_env(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "must-not-be-used")
    captured = {}
    eng = _make_client(
        _payload([{"index": 0, "text": "a", "finish_reason": "stop", "token_ids": [1]}]), captured, api_key=None
    )
    await eng.generate(InferenceEngineInput(prompt_token_ids=[[1]], sampling_params={}))
    assert captured["auth"] == "Bearer EMPTY"


@pytest.mark.asyncio
async def test_auth_error_propagates():
    eng = _make_client({"error": "bad key"}, {}, status=401)
    with pytest.raises(fireworks.AuthenticationError):
        await eng.generate(InferenceEngineInput(prompt_token_ids=[[1]], sampling_params={}))


@pytest.mark.asyncio
async def test_empty_choices_raises():
    eng = _make_client(_payload([]), {})
    with pytest.raises(RuntimeError, match="no choices"):
        await eng.generate(InferenceEngineInput(prompt_token_ids=[[1]], sampling_params={}))


@pytest.mark.asyncio
async def test_input_validation():
    eng = _make_client(_payload([]), {})
    with pytest.raises(ValueError, match="prompt_token_ids"):
        await eng.generate(InferenceEngineInput(prompts=[[{"role": "user", "content": "x"}]]))
    with pytest.raises(ValueError, match="n > 1"):
        await eng.generate(InferenceEngineInput(prompt_token_ids=[[1]], sampling_params={"n": 2}))
    with pytest.raises(NotImplementedError, match="multi-modal"):
        await eng.generate(InferenceEngineInput(prompt_token_ids=[[1]], mm_features=[{"mm_hashes": {}}]))


@pytest.mark.asyncio
async def test_control_plane_noops_and_weight_sync_raises():
    eng = _make_client(_payload([]), {})
    assert await eng.wake_up() == {}
    assert await eng.sleep() == {}
    assert await eng.reset_prefix_cache() == {}
    await eng.finish_session("s")
    await eng.pause_generation()
    await eng.resume_generation()
    with pytest.raises(NotImplementedError):
        await eng.get_world_size()
    with pytest.raises(NotImplementedError):
        await eng.update_named_weights(None)
    with pytest.raises(NotImplementedError):
        await eng.init_weight_update_communicator(None)
    await eng.teardown()


def test_default_base_url():
    eng = FireworksInferenceClient(model_name="m", tokenizer=_FakeTokenizer())
    assert eng.get_endpoint_url() == DEFAULT_FIREWORKS_BASE_URL
    assert eng.model_name == "m"


def _fireworks_cfg() -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.generator.inference_engine.backend = "fireworks"
    cfg.generator.inference_engine.run_engines_locally = False
    cfg.generator.inference_engine.served_model_name = "accounts/fw/qwen3-4b"
    cfg.generator.inference_engine.api_key = "fw-x"
    cfg.generator.inference_engine.hf_tokenizer_name = "org/some-tokenizer"
    return cfg


def test_base_exp_rejects_non_vllm_backend():
    exp = object.__new__(BasePPOExp)  # gating happens before any other attribute access
    exp.cfg = _fireworks_cfg()
    with pytest.raises(ValueError, match="generation/eval-only"):
        BasePPOExp.get_inference_client(exp)


def test_eval_entrypoint_builds_fireworks_client_without_vllm():
    exp = object.__new__(EvalOnlyEntrypoint)
    exp.cfg = _fireworks_cfg()
    exp.tokenizer = _FakeTokenizer()
    client = EvalOnlyEntrypoint.get_inference_client(exp)
    assert isinstance(client, FireworksInferenceClient)
    assert client.model_name == "accounts/fw/qwen3-4b"
    assert "vllm" not in sys.modules


def test_eval_entrypoint_tokenizer_from_hf_tokenizer_name(monkeypatch):
    captured = {}

    def fake_get_tokenizer(name, **kwargs):
        captured["name"] = name
        return _FakeTokenizer()

    monkeypatch.setattr("skyrl.train.entrypoints.main_generate.get_tokenizer", fake_get_tokenizer)
    exp = object.__new__(EvalOnlyEntrypoint)
    exp.cfg = _fireworks_cfg()
    tokenizer = EvalOnlyEntrypoint.get_tokenizer(exp)
    assert captured["name"] == "org/some-tokenizer"
    assert isinstance(tokenizer, _FakeTokenizer)
