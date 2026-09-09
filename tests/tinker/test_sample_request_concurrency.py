import asyncio
from types import SimpleNamespace

import pytest

from skyrl.tinker import api


class _Session:
    def __init__(self) -> None:
        self.commits = 0

    async def commit(self) -> None:
        self.commits += 1


@pytest.mark.asyncio
async def test_asample_serializes_future_row_transactions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_transactions = 0
    max_active_transactions = 0

    async def get_sampling_model(
        request: api.SampleRequest,
        http_request: SimpleNamespace,
        session: _Session,
    ) -> tuple[str, None]:
        del request, http_request, session
        return "test-model", None

    async def create_future(**kwargs: object) -> int:
        nonlocal active_transactions, max_active_transactions
        nonlocal next_request_id
        del kwargs
        active_transactions += 1
        max_active_transactions = max(max_active_transactions, active_transactions)
        await asyncio.sleep(0.01)
        active_transactions -= 1
        next_request_id += 1
        return next_request_id

    next_request_id = 0

    monkeypatch.setattr(api, "get_sampling_model", get_sampling_model)
    monkeypatch.setattr(api, "create_future", create_future)

    state = SimpleNamespace(
        external_inference_client=None,
        external_future_store=None,
        sample_request_db_lock=asyncio.Lock(),
    )
    http_request = SimpleNamespace(app=SimpleNamespace(state=state))
    sample_request = api.SampleRequest(
        prompt=api.ModelInput(chunks=[api.EncodedTextChunk(tokens=[1])]),
        sampling_params=api.SamplingParams(max_tokens=1, seed=0),
        base_model="test-model",
    )
    sessions = [_Session() for _ in range(64)]

    responses = await asyncio.gather(
        *(api.asample(sample_request, http_request, session) for session in sessions)
    )

    assert max_active_transactions == 1
    assert [response.request_id for response in responses] == [
        str(i) for i in range(1, 65)
    ]
    assert all(session.commits == 1 for session in sessions)
