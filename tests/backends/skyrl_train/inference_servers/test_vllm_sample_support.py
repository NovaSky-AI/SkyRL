"""Sample-support capture out of vLLM's flat-logprobs rows."""

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytest.importorskip("vllm")

from vllm.lora.request import LoRARequest

from skyrl.backends.skyrl_train.inference_servers.generate_wire import (
    PackedField,
    decode_packed_sample_support,
)
from skyrl.backends.skyrl_train.inference_servers.vllm_server_actor import (
    VLLMServerActor,
    _sample_support_from_flat_logprobs,
)
from tests.backends.skyrl_train.sample_support_helpers import dense

pytestmark = pytest.mark.vllm


def test_flat_logprobs_extracts_sampled_scores_and_support_rows():
    flat_logprobs = SimpleNamespace(
        token_ids=[7, 7, 8, 9, 4, 3, 4, 5],
        logprobs=[-0.1, -0.1, -0.2, -0.3, -0.4, -0.2, -0.4, -0.6],
    )

    sampled, support = _sample_support_from_flat_logprobs(flat_logprobs, top_k=3)

    assert sampled == [{"logprob": -0.1}, {"logprob": -0.4}]
    assert support.token_ids.dtype == np.int32
    assert support.top_k == 3
    np.testing.assert_array_equal(dense(support), [[7, 8, 9], [3, 4, 5]])


def test_flat_logprobs_replaces_top_p_masked_candidates():
    flat_logprobs = SimpleNamespace(
        token_ids=[7, 7, 8, 9],
        logprobs=[-0.1, -0.1, -0.2, float("-inf")],
    )

    _, support = _sample_support_from_flat_logprobs(flat_logprobs, top_k=3)

    np.testing.assert_array_equal(dense(support), [[7, 8, -1]])


def test_flat_logprobs_compacts_nonfinite_candidates():
    flat_logprobs = SimpleNamespace(
        token_ids=[7, 7, 8, 9, 10],
        logprobs=[-0.1, -0.1, float("nan"), -0.3, float("inf")],
    )

    sampled, support = _sample_support_from_flat_logprobs(flat_logprobs, top_k=4)

    assert sampled == [{"logprob": -0.1}]
    np.testing.assert_array_equal(dense(support), [[7, 9, -1, -1]])


def test_flat_logprobs_repairs_sampled_token_absent_from_support():
    top_k = 3
    flat_logprobs = SimpleNamespace(
        token_ids=[100, 8, 9, 10, 7, 7, 8, 9, 5, 6, 7, 8],
        logprobs=[
            -0.1,
            -0.2,
            -0.3,
            -0.4,
            -0.1,
            -0.1,
            -0.2,
            -0.3,
            -0.4,
            -0.5,
            -0.6,
            float("-inf"),
        ],
    )

    _, support = _sample_support_from_flat_logprobs(flat_logprobs, top_k=top_k)
    np.testing.assert_array_equal(dense(support), [[8, 9, 100], [7, 8, 9], [6, 5, -1]])


def test_flat_logprobs_keeps_an_all_filtered_row_empty():
    flat_logprobs = SimpleNamespace(
        token_ids=[7, 8, 9, 7, 7, 8],
        logprobs=[-0.1, float("-inf"), float("-inf"), -0.1, -0.1, -0.2],
    )

    _, support = _sample_support_from_flat_logprobs(flat_logprobs, top_k=2)

    assert support.row_lengths.tolist() == [0, 2]
    np.testing.assert_array_equal(support.token_ids, [7, 8])


def test_flat_logprobs_top_k_one_repairs_single_support_column():
    flat_logprobs = SimpleNamespace(
        token_ids=[42, 9],
        logprobs=[-0.1, -0.2],
    )

    _, support = _sample_support_from_flat_logprobs(flat_logprobs, top_k=1)

    np.testing.assert_array_equal(dense(support), [[42]])


def test_flat_logprobs_keeps_member_logprobs_in_member_order():
    flat_logprobs = SimpleNamespace(
        token_ids=[7, 7, 8, 9, 4, 3, 4, 5],
        logprobs=[-0.1, -0.1, -0.2, float("-inf"), -0.4, -0.2, -0.4, -0.6],
    )

    _, support = _sample_support_from_flat_logprobs(flat_logprobs, top_k=3, with_member_logprobs=True)

    assert support.logprobs.dtype == np.float32
    np.testing.assert_array_equal(support.logprobs, np.float32([-0.1, -0.2, -0.2, -0.4, -0.6]))


def test_flat_logprobs_repair_gives_the_sampled_member_its_own_logprob():
    flat_logprobs = SimpleNamespace(
        token_ids=[100, 8, 9, 10],
        logprobs=[-0.7, -0.2, -0.3, -0.4],
    )

    _, support = _sample_support_from_flat_logprobs(flat_logprobs, top_k=3, with_member_logprobs=True)

    np.testing.assert_array_equal(dense(support), [[8, 9, 100]])
    np.testing.assert_array_equal(support.logprobs, np.float32([-0.2, -0.3, -0.7]))


def test_flat_logprobs_omit_member_logprobs_unless_asked():
    flat_logprobs = SimpleNamespace(token_ids=[7, 7, 8], logprobs=[-0.1, -0.1, -0.2])

    _, support = _sample_support_from_flat_logprobs(flat_logprobs, top_k=2)

    assert support.logprobs is None


class FakeEngine:
    sampling_params = None
    lora_request = None

    async def generate(self, prompt, sampling_params, request_id, lora_request=None):
        self.sampling_params = sampling_params
        self.lora_request = lora_request
        yield SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    token_ids=[7],
                    finish_reason="stop",
                    logprobs=SimpleNamespace(
                        token_ids=[7, 7, 8],
                        logprobs=[-0.1, -0.1, -0.2],
                    ),
                    routed_experts=None,
                )
            ]
        )


@pytest.mark.parametrize("sampling_params", [{"temperature": 1.0}, {"temperature": 0.0, "top_k": -1}, {"top_k": 1}])
def test_skyrl_generate_rejects_sample_support_without_a_bounded_support(sampling_params):
    app = FastAPI()
    engine = FakeEngine()
    VLLMServerActor._add_custom_endpoints(app, engine, SimpleNamespace(enable_lora=False))

    with TestClient(app) as client:
        response = client.post(
            "/skyrl/v1/generate",
            json={"token_ids": [1, 2], "sampling_params": sampling_params, "return_sample_support": True},
        )

    assert response.status_code == 400
    assert "top_k > 1" in response.json()["detail"]
    assert engine.sampling_params is None


def test_skyrl_generate_returns_packed_sample_support():
    app = FastAPI()
    engine = FakeEngine()
    VLLMServerActor._add_custom_endpoints(app, engine, SimpleNamespace(enable_lora=False))

    with TestClient(app) as client:
        response = client.post(
            "/skyrl/v1/generate",
            json={
                "token_ids": [1, 2],
                "sampling_params": {"temperature": 1.0, "top_k": 2},
                "return_sample_support": True,
            },
        )

    assert response.status_code == 200
    assert engine.sampling_params.flat_logprobs is True
    assert engine.sampling_params.logprobs == 2
    packed = response.json()["choices"][0][PackedField.ROLLOUT_SAMPLE_SUPPORT]
    np.testing.assert_array_equal(dense(decode_packed_sample_support(packed)), [[7, 8]])


@pytest.mark.parametrize(
    "extra_params, expected_detokenize",
    [({}, False), ({"stop": ["</answer>"]}, True), ({"detokenize": True}, True)],
)
def test_skyrl_generate_skips_detokenize_for_sample_support(extra_params, expected_detokenize):
    app = FastAPI()
    engine = FakeEngine()
    VLLMServerActor._add_custom_endpoints(app, engine, SimpleNamespace(enable_lora=False))

    with TestClient(app) as client:
        response = client.post(
            "/skyrl/v1/generate",
            json={
                "token_ids": [1, 2],
                "sampling_params": {"temperature": 1.0, "top_k": 2, **extra_params},
                "return_sample_support": True,
            },
        )

    assert response.status_code == 200
    assert engine.sampling_params.detokenize is expected_detokenize


def test_skyrl_generate_returns_packed_sample_support_logprobs():
    app = FastAPI()
    engine = FakeEngine()
    VLLMServerActor._add_custom_endpoints(app, engine, SimpleNamespace(enable_lora=False))

    with TestClient(app) as client:
        response = client.post(
            "/skyrl/v1/generate",
            json={
                "token_ids": [1, 2],
                "sampling_params": {"temperature": 1.0, "top_k": 2},
                "return_sample_support": True,
                "return_sample_support_logprobs": True,
            },
        )

    assert response.status_code == 200
    choice = response.json()["choices"][0]
    decoded = decode_packed_sample_support(
        choice[PackedField.ROLLOUT_SAMPLE_SUPPORT], choice[PackedField.ROLLOUT_SAMPLE_SUPPORT_LOGPROBS]
    )
    np.testing.assert_array_equal(decoded.logprobs, np.float32([-0.1, -0.2]))


def test_skyrl_generate_rejects_member_logprobs_without_sample_support():
    app = FastAPI()
    engine = FakeEngine()
    VLLMServerActor._add_custom_endpoints(app, engine, SimpleNamespace(enable_lora=False))

    with TestClient(app) as client:
        response = client.post(
            "/skyrl/v1/generate",
            json={
                "token_ids": [1, 2],
                "sampling_params": {"temperature": 1.0, "top_k": 2},
                "return_sample_support_logprobs": True,
            },
        )

    assert response.status_code == 400
    assert "requires return_sample_support" in response.json()["detail"]
    assert engine.sampling_params is None


class FakeLoraEngine(FakeEngine):
    async def generate(self, prompt, sampling_params, request_id, lora_request=None):
        self.sampling_params = sampling_params
        self.lora_request = lora_request
        yield SimpleNamespace(
            outputs=[SimpleNamespace(token_ids=[7], finish_reason="stop", logprobs=None, routed_experts=None)]
        )


class FakeServingModels:
    def __init__(self, base_model, lora_requests):
        self.base_model = base_model
        self.lora_requests = lora_requests

    def is_base_model(self, model_name):
        return model_name == self.base_model


def _lora_app(engine):
    app = FastAPI()
    VLLMServerActor._add_custom_endpoints(app, engine, SimpleNamespace(enable_lora=True))
    app.state.openai_serving_models = FakeServingModels(
        base_model="base-model",
        lora_requests={"skyrl-lora": LoRARequest(lora_name="skyrl-lora", lora_int_id=1, lora_path="/tmp/lora")},
    )
    return app


def test_skyrl_generate_resolves_lora_adapter_from_model():
    engine = FakeLoraEngine()

    with TestClient(_lora_app(engine)) as client:
        response = client.post(
            "/skyrl/v1/generate",
            json={"model": "skyrl-lora", "token_ids": [1, 2], "sampling_params": {"temperature": 1.0}},
        )

    assert response.status_code == 200
    assert engine.lora_request.lora_name == "skyrl-lora"
    assert engine.lora_request.lora_int_id == 1


def test_skyrl_generate_base_model_skips_lora_with_lora_enabled():
    engine = FakeLoraEngine()

    with TestClient(_lora_app(engine)) as client:
        response = client.post(
            "/skyrl/v1/generate",
            json={"model": "base-model", "token_ids": [1, 2], "sampling_params": {"temperature": 1.0}},
        )

    assert response.status_code == 200
    assert engine.lora_request is None


def test_skyrl_generate_rejects_unknown_model_with_lora_enabled():
    engine = FakeLoraEngine()

    with TestClient(_lora_app(engine)) as client:
        response = client.post(
            "/skyrl/v1/generate",
            json={"model": "missing-lora", "token_ids": [1, 2], "sampling_params": {"temperature": 1.0}},
        )

    assert response.status_code == 404
    assert "missing-lora" in response.json()["detail"]
    assert engine.sampling_params is None
