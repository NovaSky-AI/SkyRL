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
