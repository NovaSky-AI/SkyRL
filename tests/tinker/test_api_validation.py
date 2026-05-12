import base64

import pytest
from pydantic import TypeAdapter, ValidationError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession

from skyrl.tinker import api, types
from skyrl.tinker.db_models import FutureDB, ModelDB, SessionDB

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


def test_lora_config_accepts_tinker_train_flags():
    cfg = api.LoRAConfig(rank=8, seed=123, train_attn=False, train_mlp=True, train_unembed=True)

    assert cfg.seed == 123
    assert cfg.train_attn is False
    assert cfg.train_mlp is True
    assert cfg.train_unembed is True


def test_api_lora_config_defaults_match_tinker_sdk():
    api_cfg = api.LoRAConfig(rank=8)

    assert api_cfg.train_attn is True
    assert api_cfg.train_mlp is True
    assert api_cfg.train_unembed is True


def test_lora_config_rejects_rank_with_no_train_targets():
    with pytest.raises(ValidationError, match="At least one"):
        api.LoRAConfig(rank=8, train_attn=False, train_mlp=False, train_unembed=False)


def test_weights_info_response_accepts_lora_train_flags():
    response = api.WeightsInfoResponse(
        base_model="base",
        is_lora=True,
        lora_rank=8,
        train_attn=False,
        train_mlp=True,
        train_unembed=True,
    )

    assert response.train_attn is False
    assert response.train_mlp is True
    assert response.train_unembed is True


@pytest.mark.asyncio
async def test_create_model_persists_lora_flags_and_seed_provenance():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

        async with AsyncSession(engine) as session:
            session.add(SessionDB(session_id="session", sdk_version="test"))
            await session.commit()

            response = await api.create_model(
                api.CreateModelRequest(
                    session_id="session",
                    base_model="base",
                    lora_config=api.LoRAConfig(
                        rank=8,
                        train_attn=False,
                        train_mlp=True,
                        train_unembed=True,
                    ),
                ),
                session,
            )

            assert response.lora_config is not None
            response_lora_config = response.lora_config
            assert response_lora_config == api.LoRAConfig(
                rank=8,
                seed=response_lora_config.seed,
                train_attn=False,
                train_mlp=True,
                train_unembed=True,
            )
            assert response_lora_config.seed is not None

            future = (await session.exec(select(FutureDB))).one()
            queued_input = types.CreateModelInput.model_validate(future.request_data)
            assert queued_input.seed_was_provided is False
            assert queued_input.lora_config.seed == response_lora_config.seed
            assert queued_input.lora_config.train_attn is False
            assert queued_input.lora_config.train_mlp is True
            assert queued_input.lora_config.train_unembed is True

            model = (await session.exec(select(ModelDB))).one()
            stored_lora = types.LoraConfig.model_validate(model.lora_config)
            assert stored_lora == queued_input.lora_config
    finally:
        await engine.dispose()


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
