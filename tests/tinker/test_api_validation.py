import base64

import pytest
from pydantic import TypeAdapter, ValidationError

from skyrl.tinker import api, types

_B64_PNG = base64.b64encode(b"\x89PNG").decode()


def _make_datum() -> api.Datum:
    return api.Datum(
        model_input=api.ModelInput(chunks=[api.EncodedTextChunk(tokens=[1, 2, 3])]),
        loss_fn_inputs={
            "target_tokens": api.TensorData(data=[2, 3, 4]),
            "weights": api.TensorData(data=[1.0, 1.0, 1.0]),
        },
    )


def test_forward_backward_input_accepts_ppo_threshold_keys():
    req = api.ForwardBackwardInput(
        data=[_make_datum()],
        loss_fn="ppo",
        loss_fn_config={"clip_low_threshold": 0.9, "clip_high_threshold": 1.1},
    )
    assert req.loss_fn_config == {"clip_low_threshold": 0.9, "clip_high_threshold": 1.1}


def test_forward_backward_input_accepts_ppo_value_clip():
    req = api.ForwardBackwardInput(
        data=[_make_datum()],
        loss_fn="ppo",
        loss_fn_config={"value_clip": 0.2},
    )
    assert req.loss_fn_config == {"value_clip": 0.2}


def test_forward_backward_input_accepts_ppo_critic_value_clip():
    req = api.ForwardBackwardInput(
        data=[_make_datum()],
        loss_fn="ppo_critic",
        loss_fn_config={"value_clip": 0.2},
    )
    assert req.loss_fn_config == {"value_clip": 0.2}


def test_forward_backward_input_accepts_dppo_delta_keys():
    req = api.ForwardBackwardInput(
        data=[_make_datum()],
        loss_fn="dppo",
        loss_fn_config={"delta_low": 0.2, "delta_high": 0.2},
    )
    assert req.loss_fn_config == {"delta_low": 0.2, "delta_high": 0.2}


def test_forward_backward_input_rejects_invalid_dppo_loss_fn_config_keys():
    with pytest.raises(ValidationError, match="Invalid loss_fn_config keys"):
        api.ForwardBackwardInput(
            data=[_make_datum()],
            loss_fn="dppo",
            loss_fn_config={"clip_low_threshold": 0.9},
        )


def test_forward_backward_input_rejects_invalid_ppo_loss_fn_config_keys():
    with pytest.raises(ValidationError, match="Invalid loss_fn_config keys"):
        api.ForwardBackwardInput(
            data=[_make_datum()],
            loss_fn="ppo",
            loss_fn_config={"clip_ratio": 0.2},
        )


def test_forward_backward_input_rejects_loss_fn_config_for_cross_entropy():
    with pytest.raises(ValidationError, match="does not accept loss_fn_config keys"):
        api.ForwardBackwardInput(
            data=[_make_datum()],
            loss_fn="cross_entropy",
            loss_fn_config={"clip_low_threshold": 0.9},
        )


def test_datum_to_types_defaults_values_and_returns_to_empty():
    datum = _make_datum().to_types()
    assert datum.loss_fn_inputs.values.data == []
    assert datum.loss_fn_inputs.returns.data == []


def test_datum_to_types_preserves_values_and_returns():
    datum = api.Datum(
        model_input=api.ModelInput(chunks=[api.EncodedTextChunk(tokens=[1, 2, 3])]),
        loss_fn_inputs={
            "target_tokens": api.TensorData(data=[2, 3, 4]),
            "weights": api.TensorData(data=[1.0, 1.0, 1.0]),
            "values": api.TensorData(data=[0.1, 0.2, 0.3]),
            "returns": api.TensorData(data=[0.4, 0.5, 0.6]),
        },
    ).to_types()

    assert datum.loss_fn_inputs.values.data == [0.1, 0.2, 0.3]
    assert datum.loss_fn_inputs.returns.data == [0.4, 0.5, 0.6]


@pytest.mark.parametrize("optimizer", [False, True])
def test_load_weights_request_preserves_optimizer_choice(optimizer):
    request = api.LoadWeightsRequest(
        model_id="model_test",
        path="tinker://source_model/weights/checkpoint",
        optimizer=optimizer,
    )

    assert request.optimizer is optimizer


# --- ModelInputChunk discriminator tests (api) ---

_api_adapter = TypeAdapter(api.ModelInputChunk)


class TestApiChunkDiscriminatorWithoutType:
    """Chunks resolved when ``type`` is absent (exclude_unset case)."""

    def test_encoded_text(self):
        obj = _api_adapter.validate_python({"tokens": [1, 2]})
        assert isinstance(obj, api.EncodedTextChunk)

    def test_image(self):
        obj = _api_adapter.validate_python({"data": _B64_PNG, "format": "png"})
        assert isinstance(obj, api.ImageChunk)

    def test_image_asset_pointer(self):
        obj = _api_adapter.validate_python({"format": "png", "location": "s3://bucket/img.png"})
        assert isinstance(obj, api.ImageAssetPointerChunk)


def test_api_chunk_discriminator_rejects_ambiguous_payload():
    with pytest.raises(ValueError, match="Ambiguous model chunk type"):
        _api_adapter.validate_python({"tokens": [1, 2], "data": _B64_PNG, "format": "png"})


def test_api_chunk_discriminator_rejects_unrecognised_payload():
    with pytest.raises(ValidationError):
        _api_adapter.validate_python({"format": "png"})


# --- ImageChunk base64 round-trip tests ---

_RAW_PNG = b"\x89PNG\r\n\x1a\nfake_image_data"
_B64_PNG_FULL = base64.b64encode(_RAW_PNG).decode()


class TestImageChunkBase64RoundTrip:
    """Verify that image data survives the api -> types -> JSON -> types cycle."""

    def test_to_types_preserves_bytes(self):
        api_chunk = api.ImageChunk.model_validate({"data": _B64_PNG_FULL, "format": "png"})
        assert api_chunk.data == _RAW_PNG

    def test_json_round_trip(self):

        api_chunk = api.ImageChunk.model_validate({"data": _B64_PNG_FULL, "format": "png"})
        types_chunk = api_chunk.to_types()
        assert types_chunk.data == _RAW_PNG

        json_dict = types_chunk.model_dump(mode="json")
        assert json_dict["data"] == _B64_PNG_FULL

        recovered = types.ImageChunk.model_validate(json_dict)
        assert recovered.data == _RAW_PNG

    def test_nested_in_model_input(self):

        api_chunk = api.ImageChunk.model_validate({"data": _B64_PNG_FULL, "format": "png"})
        model_input = api.ModelInput(chunks=[api_chunk])
        types_input = model_input.to_types()

        json_dict = types_input.model_dump(mode="json")
        recovered = types.ModelInput.model_validate(json_dict)
        assert recovered.chunks[0].data == _RAW_PNG


# --- SDK >= 0.32 request fields SkyRL does not implement -------------------


def _sample_request(**extra) -> api.SampleRequest:
    return api.SampleRequest(
        base_model="m",
        prompt=api.ModelInput(chunks=[api.EncodedTextChunk(tokens=[1, 2])]),
        sampling_params=api.SamplingParams(max_tokens=1),
        **extra,
    )


def test_sample_request_accepts_default_new_options():
    req = _sample_request(topk_sample_logprobs=0, prompt_alt_tokens_k=0, target_prompt_logprobs=None)
    assert req.topk_sample_logprobs == 0


@pytest.mark.parametrize(
    "extra",
    [
        {"topk_sample_logprobs": 4},
        {"prompt_alt_tokens_k": 2},
        {"target_prompt_logprobs": {"data": [1], "dtype": "int64", "shape": [1, 1]}},
        {"prompt_logprobs_last_n": 3},
    ],
)
def test_sample_request_rejects_unsupported_options(extra):
    """Rejecting beats ignoring: the SDK would otherwise return None for data the caller asked for."""
    with pytest.raises(ValidationError, match="Unsupported sampling options"):
        _sample_request(**extra)


def test_create_model_accepts_adamw_optimizer_config():
    req = api.CreateModelRequest(
        session_id="s", base_model="m", lora_config=api.LoRAConfig(rank=1), optimizer_config={"type": "adamw"}
    )
    assert req.optimizer_config == {"type": "adamw"}


def test_create_model_rejects_non_adam_optimizer_config():
    with pytest.raises(ValidationError, match="only 'adamw' is supported"):
        api.CreateModelRequest(
            session_id="s",
            base_model="m",
            lora_config=api.LoRAConfig(rank=1),
            optimizer_config={"type": "dimuon", "version": 1},
        )


def test_load_weights_rejects_non_adam_optimizer_config():
    with pytest.raises(ValidationError, match="only 'adamw' is supported"):
        api.LoadWeightsRequest(model_id="m", path="tinker://m/weights/c", optimizer_config={"type": "dimuon"})


def test_optim_step_legacy_adam_params_wire_key():
    req = api.OptimStepRequest(model_id="m", adam_params={"learning_rate": 1e-4})
    assert req.adam_params is not None and req.adam_params.learning_rate == 1e-4


def test_optim_step_rejects_non_adam_optimizer_params():
    """SDK >= 0.32 sends non-Adam families under ``optimizer_params``."""
    with pytest.raises(ValidationError, match="only Adam"):
        api.OptimStepRequest(model_id="m", optimizer_params={"type": "dimuon", "learning_rate": 1e-3})


def test_optim_step_requires_adam_params():
    with pytest.raises(ValidationError, match="adam_params is required"):
        api.OptimStepRequest(model_id="m")
