"""CPU tests for the Tinker torch-profiler control plane.

Covers the engine reconciling to the control row, and the API's compare-and-swap
that admits only one profiling session at a time.
"""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel

from skyrl.tinker.config import EngineConfig, TinkerTorchProfilerConfig
from skyrl.tinker.db_models import ProfilerControlDB, ProfilerState, enable_sqlite_wal
from skyrl.tinker.engine import TinkerEngine

PROFILER_CFG = {"export_dir": "/tmp/traces", "ranks": [0], "max_session_duration_sec": 7200}
WORKER_CFG = {"enable": True, "ranks": [0], "save_path": "/tmp/traces/120", "active": 5}


@pytest.fixture()
def engine(tmp_path):
    """A TinkerEngine with a real DB and a mock backend -- no GPU, no Ray."""
    eng = object.__new__(TinkerEngine)
    eng.db_engine = create_engine(f"sqlite:///{tmp_path}/t.db", echo=False)
    enable_sqlite_wal(eng.db_engine)
    SQLModel.metadata.create_all(eng.db_engine)
    eng.config = EngineConfig(base_model="m", backend="fsdp", torch_profiler=PROFILER_CFG)
    eng.backend = MagicMock()
    eng.backend.profile_step.return_value = None
    eng._profiling_model_id = None
    eng._profiling_steps = 0
    eng._profiling_error = None
    eng._profiler_cfg = TinkerTorchProfilerConfig(**PROFILER_CFG)
    eng._reset_profiler_row()
    return eng


def _row(engine) -> ProfilerControlDB:
    with Session(engine.db_engine) as s:
        return s.get(ProfilerControlDB, 1)


def _claim(engine, model_id="model-a", version=1, started_at=None, config_json=...):
    """Stand in for the API's start CAS."""
    with Session(engine.db_engine) as s:
        row = s.get(ProfilerControlDB, 1)
        row.desired_state = ProfilerState.RUNNING
        row.owner_model_id = model_id
        row.config_json = json.dumps(WORKER_CFG) if config_json is ... else config_json
        row.version = version
        row.started_at = started_at or datetime.now(timezone.utc)
        s.add(row)
        s.commit()


def _release(engine, version):
    """Stand in for the API's stop CAS."""
    with Session(engine.db_engine) as s:
        row = s.get(ProfilerControlDB, 1)
        row.desired_state = ProfilerState.STOPPED
        row.version = version
        s.add(row)
        s.commit()


class TestEngineReconcile:
    def test_start_is_applied_and_acked(self, engine):
        _claim(engine)
        engine.reconcile_profiler()

        engine.backend.start_profile.assert_called_once_with(WORKER_CFG)
        assert engine._profiling_model_id == "model-a"
        row = _row(engine)
        # The API blocks on this ack, so it must advance even in failure paths.
        assert row.applied_version == 1 and row.error is None

    def test_stop_is_applied_and_acked(self, engine):
        _claim(engine)
        engine.reconcile_profiler()
        with Session(engine.db_engine) as s:
            row = s.get(ProfilerControlDB, 1)
            row.desired_state, row.version = ProfilerState.STOPPED, 2
            s.add(row)
            s.commit()

        engine.reconcile_profiler()

        engine.backend.stop_profile.assert_called_once()
        assert engine._profiling_model_id is None
        assert _row(engine).applied_version == 2

    def test_failed_start_releases_slot_but_still_acks(self, engine):
        engine.backend.start_profile.side_effect = RuntimeError("kineto boom")
        _claim(engine)

        engine.reconcile_profiler()

        row = _row(engine)
        # Without the ack the API would hang to its timeout; without the release
        # a failed start would lock the slot for every other client.
        assert row.applied_version == 1
        assert row.desired_state == ProfilerState.STOPPED and row.owner_model_id is None
        assert "kineto boom" in row.error
        assert engine._profiling_model_id is None

    def test_no_op_when_version_already_applied(self, engine):
        _claim(engine)
        engine.reconcile_profiler()
        engine.reconcile_profiler()
        engine.backend.start_profile.assert_called_once()

    def test_disabled_engine_ignores_the_row(self, engine):
        engine._profiler_cfg = None
        _claim(engine)
        engine.reconcile_profiler()
        engine.backend.start_profile.assert_not_called()

    def test_reset_clears_a_session_left_by_a_crashed_engine(self, engine):
        _claim(engine)
        engine._reset_profiler_row()
        row = _row(engine)
        assert row.desired_state == ProfilerState.STOPPED and row.owner_model_id is None


class TestStepScoping:
    def _optim_step(self, engine, model_id):
        engine.backend.has_model.return_value = True
        engine.process_optim_step(model_id, MagicMock())

    def test_only_the_owning_model_advances_the_profiler(self, engine):
        _claim(engine, model_id="model-a")
        engine.reconcile_profiler()

        self._optim_step(engine, "model-a")
        self._optim_step(engine, "model-b")
        self._optim_step(engine, "model-a")

        assert engine.backend.profile_step.call_count == 2
        assert engine._profiling_steps == 2

    def test_no_stepping_without_a_session(self, engine):
        self._optim_step(engine, "model-a")
        engine.backend.profile_step.assert_not_called()

    def test_step_counters_flush_on_reconcile_not_on_optim_step(self, engine):
        _claim(engine)
        engine.reconcile_profiler()
        self._optim_step(engine, "model-a")
        assert _row(engine).step == 0, "optim_step must not write to the DB"
        engine.reconcile_profiler()
        assert _row(engine).step == 1

    def test_profiler_failure_does_not_break_optim_step(self, engine):
        _claim(engine)
        engine.reconcile_profiler()
        engine.backend.profile_step.side_effect = RuntimeError("step boom")

        self._optim_step(engine, "model-a")  # must not raise

        engine.reconcile_profiler()
        assert "step boom" in _row(engine).error

    def test_upload_error_surfaces_in_status(self, engine):
        _claim(engine)
        engine.reconcile_profiler()
        engine.backend.profile_step.return_value = "trace upload failed: s3 boom"
        self._optim_step(engine, "model-a")
        engine.reconcile_profiler()
        assert "s3 boom" in _row(engine).error


class TestSessionTTL:
    def test_expired_session_is_finalized_and_slot_released(self, engine):
        past = datetime.now(timezone.utc) - timedelta(seconds=PROFILER_CFG["max_session_duration_sec"] + 60)
        _claim(engine, started_at=past)
        engine.reconcile_profiler()  # applies the claim
        engine.reconcile_profiler()  # notices the age

        engine.backend.stop_profile.assert_called_once()
        row = _row(engine)
        assert row.desired_state == ProfilerState.STOPPED and row.owner_model_id is None
        assert "max_session_duration_sec" in row.error
        assert engine._profiling_model_id is None

    def test_fresh_session_is_left_running(self, engine):
        _claim(engine)
        engine.reconcile_profiler()
        engine.reconcile_profiler()
        engine.backend.stop_profile.assert_not_called()
        assert _row(engine).desired_state == ProfilerState.RUNNING


class TestStartCAS:
    """The claim must be atomic: Kineto is process-global, so two live sessions
    either raise or silently corrupt each other's traces."""

    @staticmethod
    def _claim_stmt():
        from sqlmodel import update

        return (
            update(ProfilerControlDB)
            .where(
                ProfilerControlDB.singleton_id == 1,
                ProfilerControlDB.desired_state == ProfilerState.STOPPED,
            )
            .values(desired_state=ProfilerState.RUNNING, version=ProfilerControlDB.version + 1)
        )

    def test_second_claim_matches_no_rows(self, engine):
        with Session(engine.db_engine) as s:
            assert s.exec(self._claim_stmt()).rowcount == 1
            s.commit()
        with Session(engine.db_engine) as s:
            # The 409: the slot is taken, so the predicate no longer matches.
            assert s.exec(self._claim_stmt()).rowcount == 0
            s.commit()
        assert _row(engine).version == 1, "a losing claim must not bump the version"

    def test_release_requires_the_owning_model(self, engine):
        from sqlmodel import update

        _claim(engine, model_id="model-a")

        def release(model_id):
            with Session(engine.db_engine) as s:
                r = s.exec(
                    update(ProfilerControlDB)
                    .where(
                        ProfilerControlDB.singleton_id == 1,
                        ProfilerControlDB.desired_state == ProfilerState.RUNNING,
                        ProfilerControlDB.owner_model_id == model_id,
                    )
                    .values(desired_state=ProfilerState.STOPPED)
                ).rowcount
                s.commit()
                return r

        assert release("model-b") == 0, "another client must not end this session"
        assert release("model-a") == 1


class TestEndpointHelpers:
    def _request(self, backend="fsdp", profiler_cfg=PROFILER_CFG):
        cfg = TinkerTorchProfilerConfig(**profiler_cfg) if profiler_cfg else None
        state = SimpleNamespace(
            profiler_cfg=cfg,
            engine_config=EngineConfig(base_model="m", backend=backend, torch_profiler=profiler_cfg or {}),
        )
        return SimpleNamespace(app=SimpleNamespace(state=state))

    def test_404_when_profiling_not_configured(self):
        from fastapi import HTTPException

        from skyrl.tinker.api import _require_profiling_enabled

        with pytest.raises(HTTPException) as e:
            _require_profiling_enabled(self._request(profiler_cfg=None))
        assert e.value.status_code == 404

    def test_400_for_jax_backend(self):
        from fastapi import HTTPException

        from skyrl.tinker.api import _require_profiling_enabled

        with pytest.raises(HTTPException) as e:
            _require_profiling_enabled(self._request(backend="jax"))
        assert e.value.status_code == 400

    @pytest.mark.parametrize(
        "extra,expected",
        [
            (None, "/tmp/traces/120"),
            ("before-fix", "/tmp/traces/120_before-fix"),
        ],
    )
    def test_export_path_resolution(self, extra, expected):
        from skyrl.tinker.api import StartProfilingRequest, _resolve_export_path

        cfg = TinkerTorchProfilerConfig(**PROFILER_CFG)
        req = StartProfilingRequest(model_id="m", global_step=120, export_path_extra=extra)
        assert _resolve_export_path(cfg, req) == expected

    @pytest.mark.parametrize("extra", ["../escape", "a/b", "..", "."])
    def test_export_path_extra_cannot_escape_export_dir(self, extra):
        from fastapi import HTTPException

        from skyrl.tinker.api import StartProfilingRequest, _resolve_export_path

        cfg = TinkerTorchProfilerConfig(**PROFILER_CFG)
        req = StartProfilingRequest(model_id="m", global_step=1, export_path_extra=extra)
        with pytest.raises(HTTPException) as e:
            _resolve_export_path(cfg, req)
        assert e.value.status_code == 400


class TestRepeatedSessions:
    """Starting, stopping and starting again is the point of the endpoints: the
    server is long-lived and a client may profile several times, possibly with
    identical settings."""

    def test_identical_settings_can_be_reused_across_sessions(self, engine):
        _claim(engine, version=1)
        engine.reconcile_profiler()
        _release(engine, version=2)
        engine.reconcile_profiler()

        _claim(engine, version=3)
        engine.reconcile_profiler()

        assert engine.backend.start_profile.call_count == 2
        assert engine.backend.start_profile.call_args_list[0] == engine.backend.start_profile.call_args_list[1]
        assert engine._profiling_model_id == "model-a"
        row = _row(engine)
        assert row.desired_state == ProfilerState.RUNNING
        assert row.applied_version == 3

    def test_step_counter_resets_between_sessions(self, engine):
        engine.backend.has_model.return_value = True

        _claim(engine, version=1)
        engine.reconcile_profiler()
        engine.process_optim_step("model-a", MagicMock())
        engine.process_optim_step("model-a", MagicMock())
        engine.reconcile_profiler()
        assert _row(engine).step == 2

        _release(engine, version=2)
        engine.reconcile_profiler()
        _claim(engine, version=3)
        engine.reconcile_profiler()

        assert _row(engine).step == 0
        engine.process_optim_step("model-a", MagicMock())
        engine.reconcile_profiler()
        assert _row(engine).step == 1

    def test_a_different_model_may_claim_the_slot_after_release(self, engine):
        _claim(engine, model_id="model-a", version=1)
        engine.reconcile_profiler()
        _release(engine, version=2)
        engine.reconcile_profiler()

        _claim(engine, model_id="model-b", version=3)
        engine.reconcile_profiler()

        assert engine._profiling_model_id == "model-b"
        assert _row(engine).owner_model_id == "model-b"

    def test_error_from_a_previous_session_is_cleared_on_restart(self, engine):
        _claim(engine, version=1)
        engine.reconcile_profiler()
        engine.backend.profile_step.return_value = "trace upload failed: boom"
        engine.backend.has_model.return_value = True
        engine.process_optim_step("model-a", MagicMock())
        engine.reconcile_profiler()
        assert _row(engine).error is not None

        _release(engine, version=2)
        engine.reconcile_profiler()
        engine.backend.profile_step.return_value = None
        _claim(engine, version=3)
        engine.reconcile_profiler()

        assert _row(engine).error is None


class TestBadControlRowState:
    """The control row is shared mutable state; a corrupt row must not wedge the
    slot or crash the engine loop."""

    @pytest.mark.parametrize("config_json", [None, "{not json", ""])
    def test_unreadable_config_releases_the_slot_and_reports(self, engine, config_json):
        _claim(engine, config_json=config_json)

        engine.reconcile_profiler()

        engine.backend.start_profile.assert_not_called()
        row = _row(engine)
        # Acked so the API stops waiting, released so the next client can claim.
        assert row.applied_version == 1
        assert row.desired_state == ProfilerState.STOPPED and row.owner_model_id is None
        assert "unreadable profiler config" in row.error
        assert engine._profiling_model_id is None

    def test_slot_is_reclaimable_after_an_unreadable_config(self, engine):
        _claim(engine, config_json="{not json")
        engine.reconcile_profiler()

        _claim(engine, version=2)
        engine.reconcile_profiler()

        engine.backend.start_profile.assert_called_once()
        assert _row(engine).desired_state == ProfilerState.RUNNING

    def test_missing_row_is_a_no_op_for_the_engine(self, engine):
        with Session(engine.db_engine) as s:
            s.delete(s.get(ProfilerControlDB, 1))
            s.commit()

        engine.reconcile_profiler()  # must not raise

        engine.backend.start_profile.assert_not_called()


class TestStopUploadFailure:
    def test_failure_flushing_the_final_window_is_recorded(self, engine):
        _claim(engine, version=1)
        engine.reconcile_profiler()
        engine.backend.stop_profile.side_effect = RuntimeError("trace upload failed: s3 boom")

        _release(engine, version=2)
        engine.reconcile_profiler()

        row = _row(engine)
        # /stop_profiling turns a non-null error into a 500 rather than reporting
        # success for traces that never landed.
        assert "s3 boom" in row.error
        assert row.applied_version == 2
        assert engine._profiling_model_id is None


class TestRequestValidation:
    def test_missing_required_field_is_rejected(self):
        from pydantic import ValidationError

        from skyrl.tinker.api import StartProfilingRequest

        with pytest.raises(ValidationError):
            StartProfilingRequest(global_step=1)


@pytest.mark.asyncio
async def test_async_claim_is_atomic_on_the_api_path():
    """The endpoints run on AsyncSession, so pin rowcount semantics there too.

    The sync tests above cover the same predicate, but a CAS that silently lost
    its rowcount on the async driver would let two clients hold the slot and only
    fail in production.
    """
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlmodel import update
    from sqlmodel.ext.asyncio.session import AsyncSession

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    async with AsyncSession(engine) as s:
        s.add(ProfilerControlDB(singleton_id=1))
        await s.commit()

    claim = (
        update(ProfilerControlDB)
        .where(
            ProfilerControlDB.singleton_id == 1,
            ProfilerControlDB.desired_state == ProfilerState.STOPPED,
        )
        .values(desired_state=ProfilerState.RUNNING, version=ProfilerControlDB.version + 1)
    )

    async with AsyncSession(engine) as s:
        assert (await s.exec(claim)).rowcount == 1
        await s.commit()
    async with AsyncSession(engine) as s:
        assert (await s.exec(claim)).rowcount == 0, "the slot is taken; this claim must lose"
        await s.commit()
        row = await s.get(ProfilerControlDB, 1)
    assert row.version == 1, "a losing claim must not bump the version"


@pytest.mark.asyncio
async def test_status_endpoint_reports_a_missing_control_row():
    """The row is inserted at startup; if it is gone the server is misconfigured,
    which is a 500 rather than a report of "no session running"."""
    from fastapi import HTTPException
    from sqlalchemy.ext.asyncio import create_async_engine

    from skyrl.tinker.api import profiling_status

    db_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with db_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                db_engine=db_engine,
                profiler_cfg=TinkerTorchProfilerConfig(**PROFILER_CFG),
                engine_config=EngineConfig(base_model="m", backend="fsdp", torch_profiler=PROFILER_CFG),
            )
        )
    )

    with pytest.raises(HTTPException) as excinfo:
        await profiling_status(request)
    assert excinfo.value.status_code == 500
    assert "missing" in excinfo.value.detail
