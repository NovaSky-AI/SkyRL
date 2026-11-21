import asyncio
import json
from collections import Counter

import pytest
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel, select

from skyrl.tinker import types
from skyrl.tinker.db_models import FutureDB, RequestStatus, enable_sqlite_wal
from skyrl.tinker.engine import TinkerEngine
from skyrl.utils.adaptive_concurrency import (
    EngineLoadConcurrencyPolicy,
    FixedConcurrencyPolicy,
    SamplingConcurrencyController,
    VLLMEngineLoad,
    VLLMEngineSamplingFeedback,
)


def _sample_payload(seed: int) -> dict:
    return types.SampleInput(
        prompt=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[1, 2])]),
        sampling_params=types.SamplingParams(temperature=1.0, max_tokens=4, seed=seed),
        num_samples=1,
        checkpoint_id="ckpt",
        prompt_logprobs=False,
    ).model_dump(mode="json")


def _add_external_samples(engine: TinkerEngine, count: int) -> list[int]:
    with Session(engine.db_engine) as session:
        rows = [
            FutureDB(
                request_type=types.RequestType.EXTERNAL,
                model_id=f"model-{index % 3}",
                request_data=_sample_payload(index),
                status=RequestStatus.PENDING,
            )
            for index in range(count)
        ]
        session.add_all(rows)
        session.commit()
        return [row.request_id for row in rows]


def _statuses(engine: TinkerEngine) -> dict[int, RequestStatus]:
    with Session(engine.db_engine) as session:
        rows = session.exec(select(FutureDB).order_by(FutureDB.request_id)).all()
        return {row.request_id: row.status for row in rows}


def _dispatch_once(engine: TinkerEngine) -> int:
    with Session(engine.db_engine) as session:
        requests = engine.find_dispatchable_external_samples(session)
        requests = engine.dispatch_external_samples(session, requests)
    for request_id, (model_id, request_data) in requests.items():
        engine._submit_external_sample(request_id, model_id, request_data)
    return len(requests)


async def _drive_until_complete(engine: TinkerEngine, request_ids: list[int], timeout_s: float = 30.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while True:
        _dispatch_once(engine)
        # Read one snapshot per poll rather than rescanning the entire database
        # once per request, which can starve the forwarding tasks.
        statuses = _statuses(engine)
        if (
            all(statuses.get(request_id) == RequestStatus.COMPLETED for request_id in request_ids)
            and not engine._external_samples
        ):
            return
        if asyncio.get_running_loop().time() >= deadline:
            counts = Counter(statuses.values())
            pytest.fail(f"timed out completing simulated Tinker samples: {dict(counts)}")
        await asyncio.sleep(0.001)


class _SimulatedForwardingClient:
    """Fault-injectable inference lane that persists results like the real client."""

    def __init__(
        self,
        engine: TinkerEngine,
        *,
        gate: asyncio.Event | None = None,
        crash_once: set[int] | None = None,
    ) -> None:
        self._engine = engine
        self._gate = gate
        self._crash_once = crash_once or set()
        self.calls: Counter[int] = Counter()
        self.cancelled: set[int] = set()
        self.active = 0
        self.max_active = 0

    async def call_and_store_result(
        self,
        request_id: int,
        request_data: types.SampleInput,
        model_id: str,
        checkpoint_id: str,
        *,
        base_model: str | None,
    ) -> bool:
        del request_data, model_id, checkpoint_id, base_model
        self.calls[request_id] += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self._gate is not None:
                await self._gate.wait()
            else:
                await asyncio.sleep((request_id % 5) * 0.0002)
            if request_id in self._crash_once and self.calls[request_id] == 1:
                raise RuntimeError("injected forwarding-task crash")
            with Session(self._engine.db_engine) as session:
                row = session.get(FutureDB, request_id)
                assert row is not None
                assert row.status == RequestStatus.DISPATCHED
                row.status = RequestStatus.COMPLETED
                row.result_data = json.dumps({"request_id": request_id})
                session.add(row)
                session.commit()
            return True
        except asyncio.CancelledError:
            self.cancelled.add(request_id)
            raise
        finally:
            self.active -= 1


def _make_engine(tmp_path, *, limit: int, policy=None, gate=None, crash_once=None):
    engine = object.__new__(TinkerEngine)
    engine.db_engine = create_engine(f"sqlite:///{tmp_path / 'stress.db'}", echo=False)
    enable_sqlite_wal(engine.db_engine)
    SQLModel.metadata.create_all(engine.db_engine)
    engine._external_samples = {}
    engine._forwarding_loop = asyncio.get_running_loop()
    engine.sampling_concurrency_controller = SamplingConcurrencyController(
        policy=policy or FixedConcurrencyPolicy(),
        initial_limit=limit,
    )
    client = _SimulatedForwardingClient(engine, gate=gate, crash_once=crash_once)
    engine._forwarding_client = client
    return engine, client


@pytest.mark.asyncio
async def test_fixed_window_stress_completes_every_request_once(tmp_path):
    engine, client = _make_engine(tmp_path, limit=7)
    request_ids = _add_external_samples(engine, 200)

    await _drive_until_complete(engine, request_ids)

    assert client.max_active == 7
    assert client.calls == Counter({request_id: 1 for request_id in request_ids})
    assert set(_statuses(engine).values()) == {RequestStatus.COMPLETED}


@pytest.mark.asyncio
async def test_vllm_feedback_hard_cut_requeues_youngest_then_retries(tmp_path):
    feedback = VLLMEngineSamplingFeedback(
        engine_loads=(
            VLLMEngineLoad(
                engine_id="managed-vllm-0",
                role="decode",
                kv_capacity_tokens=131_072,
                max_model_len=32_768,
                kv_usage=0.5,
                running=8,
                waiting=0,
                waiting_capacity=0,
                preemptions_delta=1,
            ),
        )
    )

    gate = asyncio.Event()
    policy = EngineLoadConcurrencyPolicy(min_limit=1, max_limit=32)
    engine, client = _make_engine(tmp_path, limit=8, policy=policy, gate=gate)
    controller = engine.sampling_concurrency_controller
    controller.set_decision_handler(engine._on_sampling_concurrency_decision)
    request_ids = _add_external_samples(engine, 40)

    assert _dispatch_once(engine) == 8
    for _ in range(1000):
        if client.active == 8:
            break
        await asyncio.sleep(0.001)
    assert client.active == 8

    decision = await controller.on_feedback(feedback)
    for _ in range(1000):
        status_counts = Counter(_statuses(engine).values())
        if len(client.cancelled) == 2 and status_counts[RequestStatus.DISPATCHED] == 6:
            break
        await asyncio.sleep(0.001)

    assert decision is not None
    assert decision.desired_limit == 6
    assert decision.reason == "engine_preemptions"
    assert decision.shed_count == 2
    assert controller.current_limit == 6
    assert client.cancelled == set(request_ids[6:8])
    assert Counter(_statuses(engine).values()) == {
        RequestStatus.PENDING: 34,
        RequestStatus.DISPATCHED: 6,
    }
    assert _dispatch_once(engine) == 0

    gate.set()
    await _drive_until_complete(engine, request_ids)

    assert client.max_active == 8
    assert all(client.calls[request_id] == (2 if request_id in client.cancelled else 1) for request_id in request_ids)
    assert set(_statuses(engine).values()) == {RequestStatus.COMPLETED}


@pytest.mark.asyncio
async def test_forwarding_crash_stress_requeues_without_losing_or_duplicating_results(tmp_path):
    request_ids = list(range(1, 121))
    crash_once = {request_id for request_id in request_ids if request_id % 11 == 0}
    engine, client = _make_engine(tmp_path, limit=9, crash_once=crash_once)
    assert _add_external_samples(engine, len(request_ids)) == request_ids

    await _drive_until_complete(engine, request_ids)

    assert client.max_active <= 9
    assert all(client.calls[request_id] == (2 if request_id in crash_once else 1) for request_id in request_ids)
    with Session(engine.db_engine) as session:
        rows = session.exec(select(FutureDB).order_by(FutureDB.request_id)).all()
        assert [json.loads(row.result_data) for row in rows] == [
            {"request_id": request_id} for request_id in request_ids
        ]
