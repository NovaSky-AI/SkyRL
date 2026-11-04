"""``/skyrl/v1/generate`` with multimodal ``features``, as vLLM's ``/inference/v1/generate`` takes them."""

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytest.importorskip("vllm")

import torch
from vllm.entrypoints.scale_out.token_in_token_out.mm_serde import (
    encode_mm_kwargs_item,
)
from vllm.multimodal.inputs import MultiModalKwargsItem

from skyrl.backends.skyrl_train.inference_servers.generate_wire import (
    PackedField,
    decode_packed_sample_support,
)
from skyrl.backends.skyrl_train.inference_servers.vllm_server_actor import (
    VLLMServerActor,
)
from tests.backends.skyrl_train.sample_support_helpers import dense

pytestmark = pytest.mark.vllm


class FakeEngine:
    prompt = None
    sampling_params = None

    async def generate(self, prompt, sampling_params, request_id, lora_request=None):
        self.prompt, self.sampling_params = prompt, sampling_params
        yield SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    token_ids=[7],
                    finish_reason="stop",
                    # Flat rows only when sample support asked for them.
                    logprobs=(
                        SimpleNamespace(token_ids=[7, 7, 8], logprobs=[-0.1, -0.1, -0.2])
                        if sampling_params.flat_logprobs
                        else None
                    ),
                    routed_experts=None,
                )
            ]
        )


def _post(body):
    app = FastAPI()
    engine = FakeEngine()
    VLLMServerActor._add_custom_endpoints(app, engine, SimpleNamespace(enable_lora=False))
    with TestClient(app) as client:
        return client.post("/skyrl/v1/generate", json=body), engine


def test_features_reach_the_engine_as_a_multimodal_prompt_with_sample_support():
    item = MultiModalKwargsItem.dummy(4)
    features = {
        "mm_hashes": {"image": ["abc"]},
        "mm_placeholders": {"image": [{"offset": 1, "length": 3}]},
        "kwargs_data": {"image": [encode_mm_kwargs_item(item)]},
    }
    response, engine = _post(
        {
            "token_ids": [1, 9, 9, 9, 2],
            "sampling_params": {"temperature": 1.0, "top_k": 2},
            "return_sample_support": True,
            "cache_salt": "salt",
            "features": features,
        }
    )

    assert response.status_code == 200
    prompt = engine.prompt
    assert prompt["type"] == "multimodal"
    assert prompt["prompt_token_ids"] == [1, 9, 9, 9, 2]
    assert prompt["mm_hashes"] == {"image": ["abc"]}
    (placeholder,) = prompt["mm_placeholders"]["image"]
    assert (placeholder.offset, placeholder.length) == (1, 3)
    assert prompt["cache_salt"] == "salt"
    (decoded,) = prompt["mm_kwargs"]["image"]
    assert torch.equal(decoded.get_data()["dummy"], item.get_data()["dummy"])
    # The packed side channels still come back with images.
    packed = response.json()["choices"][0][PackedField.ROLLOUT_SAMPLE_SUPPORT]
    np.testing.assert_array_equal(dense(decode_packed_sample_support(packed)), [[7, 8]])


def test_a_request_without_features_keeps_a_token_prompt():
    response, engine = _post({"token_ids": [1, 2], "sampling_params": {"temperature": 1.0}})

    assert response.status_code == 200
    assert engine.prompt == {"prompt_token_ids": [1, 2]}


@pytest.mark.parametrize(
    "features",
    [
        {"mm_hashes": {"image": ["abc"]}},
        [],
        "",
        {
            "mm_hashes": {"image": ["abc"]},
            "mm_placeholders": {"image": [{"offset": 0, "length": 1}]},
            # Valid base64, but not a serialized item.
            "kwargs_data": {"image": ["aGVsbG8="]},
        },
    ],
    ids=["no-placeholders", "empty-list", "empty-string", "not-an-item"],
)
def test_malformed_features_are_refused(features):
    response, engine = _post({"token_ids": [1, 2], "sampling_params": {}, "features": features})

    assert response.status_code == 400
    assert "invalid features" in response.json()["detail"]
    assert engine.prompt is None
