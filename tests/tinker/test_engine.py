from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from cloudpathlib import AnyPath
from sqlmodel import Session, SQLModel

from skyrl.tinker import types
from skyrl.tinker.config import EngineConfig
from skyrl.tinker.db_models import FutureDB, ModelDB, RequestStatus, SessionDB
from skyrl.tinker.engine import (
    TinkerEngine,
    prepare_model_pass_batch,
    prepare_sample_batch,
)

BASE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"


@pytest.mark.parametrize("load_optimizer", [False, True])
def test_process_load_weights_forwards_optimizer_choice(load_optimizer):
    engine = object.__new__(TinkerEngine)
    engine.config = SimpleNamespace(checkpoints_base=AnyPath("/checkpoints"))
    engine.backend = MagicMock()
    engine.backend.has_model.return_value = True

    result = engine.process_load_weights(
        "target_model",
        types.LoadWeightsInput(
            source_model_id="source_model",
            checkpoint_id="checkpoint",
            load_optimizer=load_optimizer,
        ),
    )

    engine.backend.load_checkpoint.assert_called_once_with(
        AnyPath("/checkpoints/source_model/checkpoint.tar.gz"),
        "target_model",
        load_optimizer=load_optimizer,
    )
    assert result.type == "load_weights"


def test_process_unload_model():
    """Test that process_unload_model removes model from backend."""
    config = EngineConfig(
        base_model=BASE_MODEL,
        checkpoints_base=AnyPath(""),
        backend_config={"max_lora_adapters": 4, "max_lora_rank": 32},
    )
    engine = TinkerEngine(config)
    SQLModel.metadata.create_all(engine.db_engine)

    model_id = "test_model"
    _ = engine.process_single_request(
        types.RequestType.CREATE_MODEL, model_id, {"lora_config": {"rank": 8, "alpha": 16, "seed": 0}}
    )
    assert engine.backend.has_model(model_id)

    result = engine.process_unload_model(model_id, types.UnloadModelInput())
    assert result.status == "unloaded"
    assert not engine.backend.has_model(model_id)


def test_cleanup_stale_sessions():
    """Test that cleanup_stale_sessions unloads models from expired sessions."""
    config = EngineConfig(
        base_model=BASE_MODEL,
        checkpoints_base=AnyPath(""),
        backend_config={"max_lora_adapters": 4, "max_lora_rank": 32},
        session_timeout_sec=60,
        database_url="sqlite:///:memory:",  # Use in-memory DB for test isolation
    )
    engine = TinkerEngine(config)
    SQLModel.metadata.create_all(engine.db_engine)

    model_id = "stale_model"
    session_id = "stale_session"

    # Create model in backend
    _ = engine.process_single_request(
        types.RequestType.CREATE_MODEL, model_id, {"lora_config": {"rank": 8, "alpha": 16, "seed": 0}}
    )
    assert engine.backend.has_model(model_id)

    # Insert stale session and model into DB
    stale_heartbeat = datetime.now(timezone.utc) - timedelta(seconds=120)
    with Session(engine.db_engine) as session:
        session.add(
            SessionDB(
                session_id=session_id,
                sdk_version="test",
                status="active",
                last_heartbeat_at=stale_heartbeat,
            )
        )
        session.add(
            ModelDB(
                model_id=model_id,
                base_model=BASE_MODEL,
                lora_config=types.LoraConfig(rank=8, alpha=16, seed=0).model_dump(),
                status="ready",
                request_id=1,
                session_id=session_id,
            )
        )
        session.commit()

    # Run cleanup and assert one model was unloaded
    assert engine.cleanup_stale_sessions() == 1
    assert not engine.backend.has_model(model_id)


@pytest.mark.parametrize(
    ("loss_fn", "loss_fn_config", "advantages", "logprobs", "values", "returns"),
    [
        pytest.param(
            "ppo",
            {"clip_low_threshold": 0.7, "clip_high_threshold": 1.3},
            [],
            [],
            [],
            [],
            id="ppo_with_loss_fn_config",
        ),
        pytest.param("ppo", {"value_clip": 0.2}, [], [], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], id="ppo_with_value_clip"),
        pytest.param("cross_entropy", None, [], [], [], [], id="cross_entropy_default_config"),
        pytest.param(
            "cispo",
            {"clip_low_threshold": 0.7, "clip_high_threshold": 1.3},
            [0.1, 0.2, 0.3],
            [-1.1, -1.0, -0.9],
            [],
            [],
            id="cispo",
        ),
        pytest.param("ppo_critic", {"value_clip": 0.2}, [], [], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], id="ppo_critic"),
        pytest.param(
            "dppo",
            {"delta_low": 0.2, "delta_high": 0.2},
            [0.1, 0.2, 0.3],
            [-1.1, -1.0, -0.9],
            [],
            [],
            id="dppo",
        ),
    ],
)
def test_prepare_model_pass_batch_loss_fn_and_config(
    loss_fn: str,
    loss_fn_config: dict[str, float] | None,
    advantages: list[float],
    logprobs: list[float],
    values: list[float],
    returns: list[float],
):
    """Test that prepare_model_pass_batch preserves loss_fn and loss_fn_config values."""
    datum = types.Datum(
        model_input=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[1, 2, 3])]),
        loss_fn_inputs=types.LossFnInputs(
            target_tokens=types.TensorData(data=[2, 3, 4]),
            weights=types.TensorData(data=[1.0, 1.0, 1.0]),
            advantages=types.TensorData(data=advantages),
            logprobs=types.TensorData(data=logprobs),
            values=types.TensorData(data=values),
            returns=types.TensorData(data=returns),
        ),
    )

    requests = {
        "req1": (
            "model1",
            types.ForwardBackwardInput(
                data=[datum],
                loss_fn=loss_fn,
                loss_fn_config=loss_fn_config,
            ),
        ),
    }

    batch = prepare_model_pass_batch(requests)
    assert batch.all_loss_fns == [loss_fn]
    assert batch.all_loss_fn_configs == [loss_fn_config]
    assert batch.all_model_inputs == [datum.model_input]
    assert batch.all_values == [values]
    assert batch.all_returns == [returns]


def test_prepare_sample_batch_session_ids():
    """all_session_ids holds the derived routing key per expanded sample, and None when the request has no session."""
    prompt = types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[1, 2, 3])])
    sampling_params = types.SamplingParams(temperature=0.0, max_tokens=4, seed=0)

    def sample_input(**kwargs):
        return types.SampleInput(
            base_model=BASE_MODEL,
            prompt=prompt,
            sampling_params=sampling_params,
            num_samples=2,
            checkpoint_id="",
            prompt_logprobs=False,
            **kwargs,
        )

    requests = {
        "req_with_session": ("", sample_input(sampling_session_id="sampling_abcd", seq_id=3)),
        "req_no_session": ("", sample_input()),
    }

    batch = prepare_sample_batch(requests)

    assert len(batch.all_session_ids) == len(batch.all_model_inputs)
    # num_samples=2 expands each request into two entries that share its key.
    assert batch.all_session_ids == ["sampling_abcd:3", "sampling_abcd:3", None, None]


@pytest.fixture()
def scheduling_engine():
    """Create a TinkerEngine with only the DB initialized (no backend) for scheduling tests."""
    from sqlalchemy import create_engine

    from skyrl.tinker.db_models import enable_sqlite_wal

    engine = object.__new__(TinkerEngine)
    engine.db_engine = create_engine("sqlite:///:memory:", echo=False)
    enable_sqlite_wal(engine.db_engine)
    SQLModel.metadata.create_all(engine.db_engine)
    return engine


def test_find_single_requests_respects_forward_backward_barriers(scheduling_engine):
    """Regression: optim_step must not run before a preceding forward_backward for the same model.

    Given pending requests [fwdbwd1, optim1, fwdbwd2, optim2] for the same model,
    find_single_requests should only return optim1 (not optim2), because fwdbwd2
    acts as a barrier — optim2 depends on fwdbwd2's gradients.
    """
    engine = scheduling_engine
    model_id = "test_model"

    with Session(engine.db_engine) as session:
        # Insert requests in order: fwdbwd1, optim1, fwdbwd2, optim2
        for req_type in [
            types.RequestType.FORWARD_BACKWARD,
            types.RequestType.OPTIM_STEP,
            types.RequestType.FORWARD_BACKWARD,
            types.RequestType.OPTIM_STEP,
        ]:
            session.add(
                FutureDB(
                    request_type=req_type,
                    model_id=model_id,
                    request_data={},
                    status=RequestStatus.PENDING,
                )
            )
        session.commit()

    with Session(engine.db_engine) as session:
        # find_single_requests should return only optim1 (request_id=2), NOT optim2 (request_id=4)
        singles = engine.find_single_requests(session)
        assert list(singles.keys()) == ["2"]


def test_find_single_requests_no_barrier_when_no_pending_passes(scheduling_engine):
    """When there are no pending forward/forward_backward requests, all single requests are returned."""
    engine = scheduling_engine

    with Session(engine.db_engine) as session:
        for model_id in ["model_a", "model_b"]:
            session.add(
                FutureDB(
                    request_type=types.RequestType.OPTIM_STEP,
                    model_id=model_id,
                    request_data={},
                    status=RequestStatus.PENDING,
                )
            )
        session.commit()

    with Session(engine.db_engine) as session:
        singles = engine.find_single_requests(session)
        assert len(singles) == 2


def test_find_single_requests_barrier_is_per_model(scheduling_engine):
    """A blocked forward_backward on model_a should not block an optim_step on model_b."""
    engine = scheduling_engine

    with Session(engine.db_engine) as session:
        # model_a: fwdbwd(1), optim(2), fwdbwd(3), optim(4)
        # model_b: optim(5)
        for req_type in [
            types.RequestType.FORWARD_BACKWARD,
            types.RequestType.OPTIM_STEP,
            types.RequestType.FORWARD_BACKWARD,
            types.RequestType.OPTIM_STEP,
        ]:
            session.add(
                FutureDB(
                    request_type=req_type,
                    model_id="model_a",
                    request_data={},
                    status=RequestStatus.PENDING,
                )
            )
        session.add(
            FutureDB(
                request_type=types.RequestType.OPTIM_STEP,
                model_id="model_b",
                request_data={},
                status=RequestStatus.PENDING,
            )
        )
        session.commit()

    with Session(engine.db_engine) as session:
        singles = engine.find_single_requests(session)
        # model_a: optim(2) returned, optim(4) blocked by fwdbwd(3)
        # model_b: optim(5) returned (not affected by model_a's barrier)
        assert list(singles.keys()) == ["2", "5"]
        assert singles["2"][0] == "model_a"
        assert singles["5"][0] == "model_b"


def test_process_batch_requests_per_model_completes_incrementally(scheduling_engine):
    """per_model=True must complete each model's futures before processing the next model's group."""
    from types import SimpleNamespace

    from sqlmodel import select

    engine = scheduling_engine
    engine.backend = SimpleNamespace(has_model=lambda model_id: True)

    with Session(engine.db_engine) as session:
        for model_id in ["model_a", "model_a", "model_b"]:
            session.add(
                FutureDB(
                    request_type=types.RequestType.FORWARD_BACKWARD,
                    model_id=model_id,
                    request_data={},
                    status=RequestStatus.PENDING,
                )
            )
        session.commit()
        futures = session.exec(select(FutureDB).order_by(FutureDB.request_id)).all()
        adam = types.AdamParams(learning_rate=1e-4, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0)
        requests = {str(f.request_id): (f.model_id, types.OptimStepInput(adam_params=adam)) for f in futures}

    processed_models = []

    def processor(group):
        (model_id,) = {mid for mid, _ in group.values()}
        if model_id == "model_b":
            with Session(engine.db_engine) as session:
                statuses = [
                    f.status for f in session.exec(select(FutureDB).where(FutureDB.model_id == "model_a")).all()
                ]
                assert statuses == [RequestStatus.COMPLETED] * 2
        processed_models.append(model_id)
        return {request_id: types.OptimStepOutput(metrics={}) for request_id in group}

    engine.process_batch_requests(requests, processor, "forward_backward", per_model=True)

    assert processed_models == ["model_a", "model_b"]
    with Session(engine.db_engine) as session:
        assert all(f.status == RequestStatus.COMPLETED for f in session.exec(select(FutureDB)).all())


def forward_backward_payload() -> dict:
    return types.ForwardBackwardInput(
        data=[
            types.Datum(
                model_input=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[1, 2, 3])]),
                loss_fn_inputs=types.LossFnInputs(
                    target_tokens=types.TensorData(data=[1, 2, 3]),
                    weights=types.TensorData(data=[1.0, 1.0, 1.0]),
                    advantages=types.TensorData(data=[0.0, 0.0, 0.0]),
                    logprobs=types.TensorData(data=[0.0, 0.0, 0.0]),
                ),
            )
        ],
        loss_fn="cross_entropy",
    ).model_dump(mode="json")


def add_futures(engine, specs) -> list[int]:
    with Session(engine.db_engine) as session:
        rows = [
            FutureDB(request_type=request_type, model_id=model_id, request_data=data, status=RequestStatus.PENDING)
            for request_type, model_id, data in specs
        ]
        for row in rows:
            session.add(row)
        session.commit()
        return [row.request_id for row in rows]


def test_find_batchable_model_passes_stops_at_barrier(scheduling_engine):
    """Passes behind an optim_step must not batch with earlier ones, and payloads still parse."""
    engine = scheduling_engine
    payload = forward_backward_payload()
    request_ids = add_futures(
        engine,
        [
            (types.RequestType.FORWARD_BACKWARD, "model_a", payload),
            (types.RequestType.OPTIM_STEP, "model_a", {}),
            (types.RequestType.FORWARD_BACKWARD, "model_a", payload),
            (types.RequestType.FORWARD_BACKWARD, "model_b", payload),
        ],
    )

    with Session(engine.db_engine) as session:
        batchable = engine.find_batchable_model_passes(session, types.RequestType.FORWARD_BACKWARD)

    assert set(batchable) == {str(request_ids[0]), str(request_ids[3])}
    model_id, request_data = batchable[str(request_ids[0])]
    assert model_id == "model_a"
    assert request_data.data[0].loss_fn_inputs.target_tokens.data == [1, 2, 3]


def sample_payload(checkpoint_id: str) -> dict:
    return types.SampleInput(
        prompt=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=[1, 2])]),
        sampling_params=types.SamplingParams(temperature=1.0, max_tokens=4, seed=0),
        num_samples=1,
        checkpoint_id=checkpoint_id,
        prompt_logprobs=False,
    ).model_dump(mode="json")


def test_find_batchable_sample_keeps_one_checkpoint_per_model(scheduling_engine):
    """checkpoint_id is read out of the payload in the database rather than in Python."""
    engine = scheduling_engine
    engine.config = EngineConfig(base_model=BASE_MODEL, backend="fsdp")
    request_ids = add_futures(
        engine,
        [
            (types.RequestType.SAMPLE, "model_a", sample_payload("ckpt_1")),
            (types.RequestType.SAMPLE, "model_a", sample_payload("ckpt_2")),
            (types.RequestType.SAMPLE, "model_a", sample_payload("ckpt_1")),
            (types.RequestType.SAMPLE, "", sample_payload("")),
        ],
    )

    with Session(engine.db_engine) as session:
        batchable = engine.find_batchable_sample(session)

    assert set(batchable) == {str(request_ids[0]), str(request_ids[2]), str(request_ids[3])}
    assert batchable[str(request_ids[0])][1].checkpoint_id == "ckpt_1"


def test_payload_lookup_is_chunked(scheduling_engine):
    """A backlog larger than the per-statement id chunk must still resolve fully."""
    from skyrl.tinker.engine import _MAX_IDS_PER_QUERY

    engine = scheduling_engine
    count = _MAX_IDS_PER_QUERY + 25
    add_futures(engine, [(types.RequestType.FORWARD_BACKWARD, "model_a", forward_backward_payload())] * count)

    with Session(engine.db_engine) as session:
        batchable = engine.find_batchable_model_passes(session, types.RequestType.FORWARD_BACKWARD)

    assert len(batchable) == count


# ---------------------------------------------------------------------------
# Continuous sampling (per-request future completion; drain before blocking ops)
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402
import time  # noqa: E402

from sqlmodel import select  # noqa: E402


class _FakeContinuousBackend:
    """Fake skyrl-train backend exposing the continuous-sampling surface.

    ``order`` records completion order across sample generations and
    blocking ops so tests can assert drain-before-op sequencing.
    """

    def __init__(self):
        self.delays: dict[str, float] = {}
        self.prepare_calls = 0
        self.sample_calls: list[str] = []
        self.order: list[str] = []

    def has_model(self, model_id: str) -> bool:
        return True

    def prepare_for_sampling(self) -> None:
        self.prepare_calls += 1

    async def sample_batch_async(self, prepared):
        request_id = str(prepared.request_batch_slices[0][0])
        self.sample_calls.append(request_id)
        await asyncio.sleep(self.delays.get(request_id, 0.05))
        self.order.append(f"sample:{request_id}")
        out = types.SampleOutput(
            sequences=[types.GeneratedSequence(stop_reason="stop", tokens=[1], logprobs=[0.0])]
        )
        return {request_id: out}

    def optim_step(self, model_id, request_data):
        self.order.append("optim")
        return types.OptimStepOutput(metrics={})


@pytest.fixture()
def continuous_engine(tmp_path):
    """Engine with a file-backed DB (shared across threads) + fake async backend.

    The continuous sampler completes futures from its own thread; a
    ``:memory:`` SQLite would give that thread a separate empty database
    (SingletonThreadPool), so these tests use a real file like production.
    """
    from sqlalchemy import create_engine as _sa_create_engine

    from skyrl.tinker.db_models import enable_sqlite_wal

    engine = object.__new__(TinkerEngine)
    engine.db_engine = _sa_create_engine(f"sqlite:///{tmp_path}/engine.db", echo=False)
    enable_sqlite_wal(engine.db_engine)
    SQLModel.metadata.create_all(engine.db_engine)
    engine.config = EngineConfig(
        base_model=BASE_MODEL,
        checkpoints_base=AnyPath(""),
        backend="megatron",
        session_cleanup_interval_sec=-1,  # cleanup disabled for these tests
    )
    engine.backend = _FakeContinuousBackend()
    engine._continuous_sampling = True
    engine._sampler = None
    engine._last_cleanup_time = time.time()
    return engine


def _wait_for(predicate, timeout: float = 5.0, interval: float = 0.02) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _future_status(engine, request_id: int):
    with Session(engine.db_engine) as session:
        return session.exec(select(FutureDB.status).where(FutureDB.request_id == request_id)).one()


def test_continuous_sampling_completes_futures_independently(continuous_engine):
    """A short generation's future must complete while a long one is still
    in flight — the whole point of continuous sampling (previously every
    future in a wave waited for the wave's slowest generation)."""
    engine = continuous_engine
    rid_slow, rid_fast = add_futures(
        engine,
        [
            (types.RequestType.SAMPLE, "model_a", sample_payload("")),
            (types.RequestType.SAMPLE, "model_a", sample_payload("")),
        ],
    )
    engine.backend.delays = {str(rid_slow): 0.8, str(rid_fast): 0.05}

    engine.process_pending_requests_once()

    assert _wait_for(lambda: _future_status(engine, rid_fast) == RequestStatus.COMPLETED, timeout=3.0)
    assert _future_status(engine, rid_slow) == RequestStatus.PENDING
    assert engine._sampler.inflight_count() == 1

    # Re-running the scheduler must not double-admit the in-flight request.
    engine.process_pending_requests_once()
    assert engine.backend.sample_calls.count(str(rid_slow)) == 1

    assert _wait_for(lambda: _future_status(engine, rid_slow) == RequestStatus.COMPLETED, timeout=3.0)
    assert engine.backend.prepare_calls == 1


def test_blocking_single_request_drains_inflight_samples(continuous_engine):
    """A pending blocking op (optim_step) must wait for in-flight samples to
    finish (drain) before it runs — engines may sleep during the op."""
    engine = continuous_engine
    (rid_sample,) = add_futures(engine, [(types.RequestType.SAMPLE, "model_a", sample_payload(""))])
    engine.backend.delays = {str(rid_sample): 0.4}

    engine.process_pending_requests_once()  # admits the sample
    assert engine._sampler.inflight_count() == 1

    adam = types.AdamParams(learning_rate=1e-4, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0)
    (rid_optim,) = add_futures(
        engine,
        [(types.RequestType.OPTIM_STEP, "model_a", types.OptimStepInput(adam_params=adam).model_dump(mode="json"))],
    )
    engine.process_pending_requests_once()  # must drain, then run the optim step

    assert engine.backend.order == [f"sample:{rid_sample}", "optim"]
    assert _future_status(engine, rid_sample) == RequestStatus.COMPLETED
    assert _future_status(engine, rid_optim) == RequestStatus.COMPLETED


def test_blocking_op_defers_new_sample_admission(continuous_engine):
    """Samples arriving alongside a blocking op are deferred (stay PENDING),
    not processed as a serial convoy batch, and get admitted next iteration."""
    engine = continuous_engine
    adam = types.AdamParams(learning_rate=1e-4, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0)
    rid_optim, rid_sample = add_futures(
        engine,
        [
            (types.RequestType.OPTIM_STEP, "model_a", types.OptimStepInput(adam_params=adam).model_dump(mode="json")),
            (types.RequestType.SAMPLE, "model_a", sample_payload("")),
        ],
    )

    engine.process_pending_requests_once()
    assert _future_status(engine, rid_optim) == RequestStatus.COMPLETED
    assert _future_status(engine, rid_sample) == RequestStatus.PENDING
    assert engine.backend.sample_calls == []

    engine.process_pending_requests_once()
    assert _wait_for(lambda: _future_status(engine, rid_sample) == RequestStatus.COMPLETED, timeout=3.0)


def test_admission_fails_unknown_models_without_touching_backend(continuous_engine):
    """Stale requests for unloaded models (e.g. from a previous server) must
    be failed by the admission filter without triggering an engine build."""
    engine = continuous_engine
    engine.backend.has_model = lambda model_id: False

    (rid,) = add_futures(engine, [(types.RequestType.SAMPLE, "ghost_model", sample_payload(""))])
    engine.process_pending_requests_once()

    assert _future_status(engine, rid) == RequestStatus.FAILED
    assert engine.backend.prepare_calls == 0
    assert engine.backend.sample_calls == []


def test_serial_fallback_when_continuous_disabled(continuous_engine):
    """SKYRL_TINKER_CONTINUOUS_SAMPLING=0 semantics: the serial batch path."""
    engine = continuous_engine
    engine._continuous_sampling = False
    serial_batches = []

    def fake_sample(prepared):
        rids = [str(s[0]) for s in prepared.request_batch_slices]
        serial_batches.append(rids)
        out = types.SampleOutput(
            sequences=[types.GeneratedSequence(stop_reason="stop", tokens=[1], logprobs=[0.0])]
        )
        return {rid: out for rid in rids}

    engine.backend.sample = fake_sample
    rid1, rid2 = add_futures(
        engine,
        [
            (types.RequestType.SAMPLE, "model_a", sample_payload("")),
            (types.RequestType.SAMPLE, "model_a", sample_payload("")),
        ],
    )

    engine.process_pending_requests_once()

    assert serial_batches == [[str(rid1), str(rid2)]]
    assert _future_status(engine, rid1) == RequestStatus.COMPLETED
    assert _future_status(engine, rid2) == RequestStatus.COMPLETED
