import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from skyrl.tinker import api
from skyrl.tinker.db_models import SamplingSessionDB


class _SamplingSession:
    base_model = None
    model_path = "tinker://model_a/sampler_weights/checkpoint_a"


class _Session:
    def __init__(self, reads: list[str]):
        self._reads = reads

    async def get(self, model, sampling_session_id):
        assert model is SamplingSessionDB
        self._reads.append(sampling_session_id)
        await asyncio.sleep(0.01)
        return _SamplingSession()


def _request():
    return SimpleNamespace(
        sampling_session_id="sampling_a",
        base_model=None,
        model_path=None,
    )


def _server_request(external_inference_client=object()):
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                external_inference_client=external_inference_client,
                external_sampling_targets={},
                external_sampling_target_locks={},
            )
        )
    )


@pytest.mark.asyncio
async def test_external_sampling_target_resolves_once_under_concurrency(monkeypatch):
    reads = []
    get_model = AsyncMock()
    validate_checkpoint = AsyncMock()
    monkeypatch.setattr(api, "get_model", get_model)
    monkeypatch.setattr(api, "validate_checkpoint", validate_checkpoint)
    request = _request()
    server_request = _server_request()

    targets = await asyncio.gather(
        *(api.get_sampling_target(request, server_request, _Session(reads)) for _ in range(512))
    )

    assert targets == [api.SamplingTarget(None, "model_a", "checkpoint_a")] * 512
    assert reads == ["sampling_a"]
    get_model.assert_awaited_once()
    validate_checkpoint.assert_awaited_once()


@pytest.mark.asyncio
async def test_external_sampling_target_caches_only_successful_validation(monkeypatch):
    reads = []
    monkeypatch.setattr(api, "get_model", AsyncMock())
    validate_checkpoint = AsyncMock(
        side_effect=[
            HTTPException(status_code=425, detail="Checkpoint is still being created"),
            None,
        ]
    )
    monkeypatch.setattr(api, "validate_checkpoint", validate_checkpoint)
    request = _request()
    server_request = _server_request()

    with pytest.raises(HTTPException, match="Checkpoint is still being created"):
        await api.get_sampling_target(request, server_request, _Session(reads))
    target = await api.get_sampling_target(request, server_request, _Session(reads))
    cached_target = await api.get_sampling_target(request, server_request, _Session(reads))

    assert target == cached_target == api.SamplingTarget(None, "model_a", "checkpoint_a")
    assert reads == ["sampling_a", "sampling_a"]
    assert validate_checkpoint.await_count == 2


@pytest.mark.asyncio
async def test_internal_sampling_target_preserves_database_validation(monkeypatch):
    reads = []
    get_model = AsyncMock()
    validate_checkpoint = AsyncMock()
    monkeypatch.setattr(api, "get_model", get_model)
    monkeypatch.setattr(api, "validate_checkpoint", validate_checkpoint)
    request = _request()
    server_request = _server_request(external_inference_client=None)

    await api.get_sampling_target(request, server_request, _Session(reads))
    await api.get_sampling_target(request, server_request, _Session(reads))

    assert reads == ["sampling_a", "sampling_a"]
    assert get_model.await_count == 2
    assert validate_checkpoint.await_count == 2
