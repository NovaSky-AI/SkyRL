from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from skyrl.tinker import api


@pytest.mark.asyncio
async def test_sample_promises_identify_each_sequence(monkeypatch):
    monkeypatch.setattr(api, "get_sampling_model", AsyncMock(return_value=("base", None)))
    monkeypatch.setattr(api, "create_future", AsyncMock(side_effect=[1, 2]))
    request = api.SampleRequest(
        base_model="base",
        prompt=api.ModelInput(chunks=[api.EncodedTextChunk(tokens=[1])]),
        sampling_params=api.SamplingParams(max_tokens=1),
        num_samples=3,
    )
    req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(external_future_store=None)))
    session = AsyncMock()

    first = await api.asample(request, req, session)
    second = await api.asample(request, req, session)

    assert len(first.sample_sequence_ids) == 3
    assert len(second.sample_sequence_ids) == 3
    assert len(set(first.sample_sequence_ids + second.sample_sequence_ids)) == 6
    assert first.model_dump()["sample_sequence_ids"] == first.sample_sequence_ids
