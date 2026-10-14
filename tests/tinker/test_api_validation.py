import base64
from unittest import mock

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


class TestLoRAConfigFieldRoundTrip:
    """Every field the Tinker SDK sends must survive the wire -> internal hop.

    The wire model is a plain ``BaseModel``, so Pydantic's default ``extra="ignore"`` drops an
    undeclared field without telling the client.
    """

    def test_fields_and_defaults_match_the_installed_sdk(self):
        """A field the SDK adds, drops or re-defaults must show up here.

        The SDK defaults train_unembed to True, so every client sends it.
        """
        import tinker

        sdk = tinker.LoraConfig.model_fields
        assert set(sdk) == set(api.LoRAConfig.model_fields)
        optional = {name: field for name, field in sdk.items() if not field.is_required()}
        ours = {name: api.LoRAConfig.model_fields[name].default for name in optional}
        assert ours == {name: field.default for name, field in optional.items()}

    def test_to_types_applies_the_resolved_seed(self):
        """A client that omits `seed` gets the server-generated one, not None."""
        internal = api.LoRAConfig(rank=8).to_types(seed=12345, alpha=32.0)
        assert internal.seed == 12345

    def test_from_types_preserves_the_resolved_seed(self):
        """The read-back path must not drop the seed, or runs are unreproducible."""
        internal = types.LoraConfig(rank=8, alpha=32.0, seed=4242, train_attn=False, train_mlp=True)
        wire = api.LoRAConfig.from_types(internal)
        assert wire.seed == 4242
        assert (wire.train_attn, wire.train_mlp, wire.train_unembed) == (False, True, False)

    def test_round_trip_is_lossless(self):
        """Converting to the internal type and back loses no field."""
        original = types.LoraConfig(rank=32, alpha=32.0, seed=99, train_attn=False, train_mlp=True, train_unembed=True)
        recovered = api.LoRAConfig.from_types(original).to_types(seed=original.seed, alpha=original.alpha)
        assert recovered == original


class TestSupportedLoraConfig:
    """What the API stores, sends to the engine and echoes is what the backend builds."""

    def test_train_unembed_is_dropped(self):
        """The SDK's default train_unembed=True is dropped with one warning."""
        with mock.patch.object(api, "logger") as log:
            lora = api._supported_lora_config(api.LoRAConfig(rank=8))
        assert (lora.train_attn, lora.train_mlp, lora.train_unembed) == (True, True, False)
        log.warning.assert_called_once()

    def test_no_warning_when_not_requested(self):
        """A client that turns train_unembed off gets no warning."""
        with mock.patch.object(api, "logger") as log:
            lora = api._supported_lora_config(api.LoRAConfig(rank=8, train_unembed=False))
        assert lora.train_unembed is False
        log.warning.assert_not_called()

    def test_warning_names_only_the_adapted_groups(self):
        """The warning describes the groups the adapter actually covers."""
        with mock.patch.object(api, "logger") as log:
            api._supported_lora_config(api.LoRAConfig(rank=8, train_mlp=False))
        message = log.warning.call_args.args[0]
        assert "covers the attention projections only" in message
        assert "MLP" not in message

    def test_nothing_left_to_train_is_rejected(self):
        """Both groups off is a 400 at the API, before anything is stored."""
        with pytest.raises(api.HTTPException) as exc:
            api._supported_lora_config(api.LoRAConfig(rank=8, train_attn=False, train_mlp=False))
        assert exc.value.status_code == 400

    def test_full_fine_tuning_ignores_the_group_flags(self):
        """Full fine-tuning has no adapter, so its config is kept as sent and nothing is logged."""
        with mock.patch.object(api, "logger") as log:
            lora = api._supported_lora_config(api.LoRAConfig(rank=0, train_attn=False, train_mlp=False))
        assert (lora.rank, lora.train_unembed) == (0, True)
        log.warning.assert_not_called()


class TestWeightsInfoResponse:
    """The SDK resumes from state by recreating the adapter from this response."""

    def test_reports_the_group_flags(self):
        """A LoRA model reports its rank and group flags, so a resume rebuilds the same adapter."""
        lora = types.LoraConfig(rank=8, alpha=32.0, seed=0, train_attn=True, train_mlp=False, train_unembed=False)
        info = api.WeightsInfoResponse.from_types("base", lora)
        assert (info.is_lora, info.lora_rank) == (True, 8)
        assert (info.train_attn, info.train_mlp, info.train_unembed) == (True, False, False)

    def test_matches_the_sdk_fields(self):
        """The response declares every field the installed SDK reads."""
        import tinker

        assert set(tinker.types.WeightsInfoResponse.model_fields) <= set(api.WeightsInfoResponse.model_fields)

    def test_full_fine_tuning_has_no_flags(self):
        """A full fine-tuning model reports is_lora=False and no group flags."""
        info = api.WeightsInfoResponse.from_types("base", types.LoraConfig(rank=0, alpha=32.0, seed=0))
        assert (info.is_lora, info.train_attn, info.train_mlp) == (False, None, None)
