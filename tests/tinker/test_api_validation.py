import base64

import pytest
from pydantic import TypeAdapter, ValidationError

from skyrl.tinker import api


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
_B64_PNG = base64.b64encode(b"\x89PNG").decode()


class TestApiChunkDiscriminatorWithType:
    """Chunks resolved when the ``type`` field is present."""

    def test_encoded_text(self):
        obj = _api_adapter.validate_python({"type": "encoded_text", "tokens": [1, 2]})
        assert isinstance(obj, api.EncodedTextChunk)

    def test_image(self):
        obj = _api_adapter.validate_python({"type": "image", "data": _B64_PNG, "format": "png"})
        assert isinstance(obj, api.ImageChunk)

    def test_image_asset_pointer(self):
        obj = _api_adapter.validate_python(
            {"type": "image_asset_pointer", "format": "png", "location": "s3://bucket/img.png"}
        )
        assert isinstance(obj, api.ImageAssetPointerChunk)


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


def test_api_chunk_discriminator_rejects_unrecognised_payload():
    with pytest.raises(ValidationError):
        _api_adapter.validate_python({"format": "png"})


"""
# --- ModelInputChunk discriminator tests (types) ---

_types_adapter = TypeAdapter(types.ModelInputChunk)


class TestTypesChunkDiscriminatorWithType:
    def test_encoded_text(self):
        obj = _types_adapter.validate_python({"type": "encoded_text", "tokens": [1, 2]})
        assert isinstance(obj, types.EncodedTextChunk)

    def test_image(self):
        obj = _types_adapter.validate_python(
            {"type": "image", "data": b"\x89PNG", "format": "png"}
        )
        assert isinstance(obj, types.ImageChunk)

    def test_image_asset_pointer(self):
        obj = _types_adapter.validate_python(
            {"type": "image_asset_pointer", "format": "png", "location": "s3://bucket/img.png"}
        )
        assert isinstance(obj, types.ImageAssetPointerChunk)


class TestTypesChunkDiscriminatorWithoutType:
    def test_encoded_text(self):
        obj = _types_adapter.validate_python({"tokens": [1, 2]})
        assert isinstance(obj, types.EncodedTextChunk)

    def test_image(self):
        obj = _types_adapter.validate_python({"data": b"\x89PNG", "format": "png"})
        assert isinstance(obj, types.ImageChunk)

    def test_image_asset_pointer(self):
        obj = _types_adapter.validate_python(
            {"format": "png", "location": "s3://bucket/img.png"}
        )
        assert isinstance(obj, types.ImageAssetPointerChunk)


def test_types_chunk_discriminator_rejects_unrecognised_payload():
    with pytest.raises(ValidationError):
        _types_adapter.validate_python({"format": "png"})
"""
