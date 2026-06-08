"""CPU unit tests for the config-driven torch.profiler wrapper.

These run without a GPU or torch.distributed: ``torch.profiler`` works on CPU,
and ``Profiler`` falls back to rank 0 when distributed is not initialized.

uv run --isolated --extra dev pytest tests/backends/skyrl_train/utils/test_profiler.py -v
"""

import glob
import os
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import patch

from skyrl.backends.skyrl_train.utils.profiler import Profiler


@dataclass
class _ProfCfg:
    """Minimal stand-in for TorchProfilerConfig (avoids importing skyrl_gym)."""

    enable: bool = True
    ranks: List[int] = field(default_factory=lambda: [0])
    save_path: Optional[str] = None
    skip_first: int = 0
    wait: int = 0
    warmup: int = 0
    active: int = 1
    repeat: int = 1
    activities: List[str] = field(default_factory=lambda: ["cpu"])
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = False
    with_flops: bool = False
    with_modules: bool = False
    export_type: str = "chrome_trace"


def _run_loop(prof: Profiler, n_steps: int) -> None:
    prof.start()
    for _ in range(n_steps):
        prof.step()
    prof.stop()


def test_disabled_is_noop(tmp_path):
    prof = Profiler(_ProfCfg(enable=False), default_save_path=str(tmp_path))
    assert prof.prof is None
    assert prof.check() is False
    _run_loop(prof, 5)  # must not raise
    assert glob.glob(os.path.join(str(tmp_path), "*")) == []


def test_rank_not_selected_is_noop(tmp_path):
    # is_initialized() is False in this process -> rank resolves to 0, which is
    # not in ranks=[1], so the profiler must not arm.
    prof = Profiler(_ProfCfg(ranks=[1]), default_save_path=str(tmp_path))
    assert prof.prof is None
    _run_loop(prof, 3)
    assert glob.glob(os.path.join(str(tmp_path), "*")) == []


def test_single_window_writes_one_trace(tmp_path):
    prof = Profiler(
        _ProfCfg(skip_first=0, wait=0, warmup=0, active=1, repeat=1),
        default_save_path=str(tmp_path),
    )
    assert prof.check() is True
    _run_loop(prof, 3)
    traces = glob.glob(os.path.join(str(tmp_path), "*.pt.trace.json*"))
    assert len(traces) == 1, f"expected exactly one trace, got {traces}"
    assert "rank0" in os.path.basename(traces[0])


def test_repeat_writes_multiple_windows(tmp_path):
    # Two cycles of (warmup=1 + active=1); needs >= 2*(1+1) steps to fire twice.
    prof = Profiler(
        _ProfCfg(skip_first=0, wait=0, warmup=1, active=1, repeat=2),
        default_save_path=str(tmp_path),
    )
    _run_loop(prof, 8)
    traces = glob.glob(os.path.join(str(tmp_path), "*.pt.trace.json*"))
    assert len(traces) == 2, f"expected two traces for repeat=2, got {traces}"


def test_skip_first_defers_recording(tmp_path):
    # With skip_first larger than the loop, no window should ever open.
    prof = Profiler(
        _ProfCfg(skip_first=100, wait=0, warmup=0, active=1, repeat=1),
        default_save_path=str(tmp_path),
    )
    _run_loop(prof, 5)
    assert glob.glob(os.path.join(str(tmp_path), "*.pt.trace.json*")) == []


def test_save_path_overrides_default(tmp_path):
    explicit = tmp_path / "explicit"
    prof = Profiler(
        _ProfCfg(save_path=str(explicit)),
        default_save_path=str(tmp_path / "default"),
    )
    assert prof.save_path == str(explicit)


def test_default_save_path_used_when_none(tmp_path):
    default = str(tmp_path / "default")
    prof = Profiler(_ProfCfg(save_path=None), default_save_path=default)
    assert prof.save_path == default


def test_cloud_save_path_falls_back_to_local(tmp_path):
    # ckpt_path (and thus the derived default save_path) can be a cloud URI, which
    # torch.profiler can't write to. The wrapper must fall back to a local dir so
    # the trace isn't silently lost.
    prof = Profiler(_ProfCfg(save_path="s3://my-bucket/run/profiler_traces"), default_save_path=str(tmp_path))
    assert prof.save_path == "./profiler_traces"


def test_kernel_summary_none_when_disabled(tmp_path):
    prof = Profiler(_ProfCfg(enable=False), default_save_path=str(tmp_path))
    assert prof.get_kernel_summary() is None


def test_kernel_summary_empty_before_first_window(tmp_path):
    prof = Profiler(_ProfCfg(skip_first=0, wait=0, warmup=0, active=1), default_save_path=str(tmp_path))
    summary = prof.get_kernel_summary()
    assert summary == {"window_count": 0, "pairs": []}


def test_kernel_summary_populated_after_window(tmp_path):
    prof = Profiler(
        _ProfCfg(skip_first=0, wait=0, warmup=0, active=1, repeat=1, activities=["cpu"]),
        default_save_path=str(tmp_path),
    )
    prof.start()
    for _ in range(3):
        # Do a little CPU work so the profiler records some ops.
        import torch

        _ = torch.randn(64, 64) @ torch.randn(64, 64)
        prof.step()
    prof.stop()

    summary = prof.get_kernel_summary()
    assert summary is not None
    assert summary["window_count"] == 1
    assert isinstance(summary["pairs"], list)
    # Each pair is (name:str, self_us:float); summary is pickle-safe.
    import pickle

    pickle.dumps(summary)
    for name, self_us in summary["pairs"]:
        assert isinstance(name, str)
        assert isinstance(self_us, float)


def test_activities_threaded_to_torch(tmp_path):
    import torch

    prof = Profiler(_ProfCfg(activities=["cpu"]), default_save_path=str(tmp_path))
    # The underlying torch profiler was constructed with the CPU activity only.
    assert torch.profiler.ProfilerActivity.CPU in prof.prof.activities
    assert torch.profiler.ProfilerActivity.CUDA not in prof.prof.activities


def test_step_failure_disables_without_raising(tmp_path):
    prof = Profiler(_ProfCfg(), default_save_path=str(tmp_path))
    prof.start()

    class _Boom:
        def step(self):
            raise RuntimeError("boom")

    # Simulate an internal profiler fault mid-loop; the wrapper must swallow it.
    prof.prof = _Boom()
    prof.step()
    assert prof.enable is False
    assert prof.prof is None
    # Subsequent calls remain safe no-ops.
    prof.step()
    prof.stop()


class TestWorkerProfilerRPCs:
    """The Worker base exposes the profiler-control RPCs and they no-op when
    ``self.profiler`` is None (so the trainer can dispatch them unconditionally,
    and so they're on the snapshotted Ray actor method table without a subclass)."""

    def test_methods_exist_on_worker_base(self):
        from skyrl.backends.skyrl_train.workers.worker import Worker

        for name in ("start_profile", "profile_step", "stop_profile", "dump_profiler_summary"):
            assert callable(getattr(Worker, name)), f"Worker.{name} missing"

    def test_dump_profiler_summary_none_when_profiler_none(self):
        from types import SimpleNamespace

        from skyrl.backends.skyrl_train.workers.worker import Worker

        stub = SimpleNamespace(profiler=None)
        assert Worker.dump_profiler_summary(stub) is None

    def test_dump_profiler_summary_delegates_to_profiler(self):
        from types import SimpleNamespace

        from skyrl.backends.skyrl_train.workers.worker import Worker

        expected = {"window_count": 2, "pairs": [("gemm", 1.5)]}
        stub = SimpleNamespace(profiler=SimpleNamespace(get_kernel_summary=lambda: expected))
        assert Worker.dump_profiler_summary(stub) == expected

    def test_rpcs_noop_when_profiler_none(self):
        from types import SimpleNamespace

        from skyrl.backends.skyrl_train.workers.worker import Worker

        # Call the unbound methods against a stub whose profiler is None; they
        # must early-return without touching anything.
        stub = SimpleNamespace(profiler=None)
        Worker.start_profile(stub)
        Worker.profile_step(stub)
        Worker.stop_profile(stub)

    def test_rpcs_drive_profiler_when_present(self):
        from types import SimpleNamespace

        from skyrl.backends.skyrl_train.workers.worker import Worker

        calls = []
        fake_profiler = SimpleNamespace(
            start=lambda: calls.append("start"),
            step=lambda: calls.append("step"),
            stop=lambda: calls.append("stop"),
        )
        stub = SimpleNamespace(profiler=fake_profiler)
        Worker.start_profile(stub)
        Worker.profile_step(stub)
        Worker.stop_profile(stub)
        assert calls == ["start", "step", "stop"]


class TestBuildProfilerFromPolicyCfg:
    """``build_profiler_from_policy_cfg`` is the production entry both worker
    backends call in ``init_model``. It gates on ``enable`` and composes the
    default ``{ckpt_path}/profiler_traces`` save path when none is set."""

    @staticmethod
    def _trainer_cfg(prof_cfg, ckpt_path):
        from types import SimpleNamespace

        return SimpleNamespace(ckpt_path=ckpt_path, policy=SimpleNamespace(torch_profiler_config=prof_cfg))

    def test_returns_none_when_disabled(self, tmp_path):
        from skyrl.backends.skyrl_train.utils.profiler import (
            build_profiler_from_policy_cfg,
        )

        cfg = self._trainer_cfg(_ProfCfg(enable=False), str(tmp_path))
        assert build_profiler_from_policy_cfg(cfg) is None

    def test_builds_profiler_with_ckpt_default_save_path(self, tmp_path):
        from skyrl.backends.skyrl_train.utils.profiler import (
            Profiler,
            build_profiler_from_policy_cfg,
        )

        cfg = self._trainer_cfg(_ProfCfg(enable=True, save_path=None), str(tmp_path))
        prof = build_profiler_from_policy_cfg(cfg)
        assert isinstance(prof, Profiler)
        assert prof.save_path == os.path.join(str(tmp_path), "profiler_traces")

    def test_explicit_save_path_wins_over_ckpt_default(self, tmp_path):
        from skyrl.backends.skyrl_train.utils.profiler import (
            build_profiler_from_policy_cfg,
        )

        explicit = str(tmp_path / "explicit")
        cfg = self._trainer_cfg(_ProfCfg(enable=True, save_path=explicit), str(tmp_path))
        prof = build_profiler_from_policy_cfg(cfg)
        assert prof.save_path == explicit


class TestWorkerDispatchProfilerRPCs:
    """The WorkerDispatch profiler wrappers dispatch via ``pass_through`` to the
    named model's actor group, no-op for an unknown model, and swallow dispatch
    faults so a profiler error can never abort the training loop. Tested against
    the unbound methods with a stub ``self`` (no live Ray actors required)."""

    @staticmethod
    def _stub(actor_groups):
        from types import SimpleNamespace

        return SimpleNamespace(_actor_groups=actor_groups)

    def _fake_group(self, calls, raises=False):
        from types import SimpleNamespace

        def async_run_ray_method(mode, method, *args, **kwargs):
            calls.append((mode, method))
            if raises:
                raise RuntimeError("dispatch boom")
            return ["sentinel"]

        return SimpleNamespace(async_run_ray_method=async_run_ray_method)

    def test_unknown_model_is_noop(self):
        from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch

        stub = self._stub({})  # no "policy" group
        # None of these should raise or dispatch.
        WorkerDispatch.start_profile(stub, "policy")
        WorkerDispatch.profile_step(stub, "policy")
        WorkerDispatch.stop_profile(stub, "policy")
        assert WorkerDispatch.dump_profiler_summary(stub, "policy") is None

    def test_control_rpcs_dispatch_pass_through(self):
        from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch

        calls = []
        stub = self._stub({"policy": self._fake_group(calls)})
        with patch("skyrl.backends.skyrl_train.workers.worker_dispatch.ray.get", side_effect=lambda x: x):
            WorkerDispatch.start_profile(stub, "policy")
            WorkerDispatch.profile_step(stub, "policy")
            WorkerDispatch.stop_profile(stub, "policy")
        assert calls == [
            ("pass_through", "start_profile"),
            ("pass_through", "profile_step"),
            ("pass_through", "stop_profile"),
        ]

    def test_dump_profiler_summary_returns_per_rank_payload(self):
        from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch

        calls = []
        payload = [{"window_count": 1, "pairs": [("gemm", 2.0)]}, None]
        stub = self._stub({"policy": self._fake_group(calls)})
        with patch("skyrl.backends.skyrl_train.workers.worker_dispatch.ray.get", side_effect=lambda x: payload):
            out = WorkerDispatch.dump_profiler_summary(stub, "policy")
        assert out == payload
        assert calls == [("pass_through", "dump_profiler_summary")]

    def test_dispatch_fault_is_swallowed(self):
        from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch

        calls = []
        stub = self._stub({"policy": self._fake_group(calls, raises=True)})

        def boom(_):
            raise RuntimeError("ray.get boom")

        with patch("skyrl.backends.skyrl_train.workers.worker_dispatch.ray.get", side_effect=boom):
            # Must not propagate -- a profiler fault cannot abort training.
            WorkerDispatch.start_profile(stub, "policy")
            WorkerDispatch.profile_step(stub, "policy")
            WorkerDispatch.stop_profile(stub, "policy")
            assert WorkerDispatch.dump_profiler_summary(stub, "policy") is None


class TestTrainerProfilerHelpers:
    """``RayPPOTrainer`` exposes gated ``_profiler_{start,step,stop}`` helpers so
    every trainer flavor (sync / fully-async / one-step async) drives the
    profiler through one code path. The helpers dispatch only when
    ``_torch_profiler_enabled`` is True, and always target the policy model."""

    @staticmethod
    def _trainer(enable):
        from types import SimpleNamespace

        from skyrl.train.trainer import RayPPOTrainer

        # Build a bare instance without running __init__ (which needs Ray/cfg).
        trainer = object.__new__(RayPPOTrainer)
        calls = []
        trainer.dispatch = SimpleNamespace(
            start_profile=lambda m: calls.append(("start", m)),
            profile_step=lambda m: calls.append(("step", m)),
            stop_profile=lambda m: calls.append(("stop", m)),
        )
        trainer.cfg = SimpleNamespace(
            trainer=SimpleNamespace(policy=SimpleNamespace(torch_profiler_config=SimpleNamespace(enable=enable)))
        )
        return trainer, calls

    def test_helpers_exist(self):
        from skyrl.train.trainer import RayPPOTrainer

        for name in ("_profiler_start", "_profiler_step", "_profiler_stop"):
            assert callable(getattr(RayPPOTrainer, name)), f"RayPPOTrainer.{name} missing"

    def test_noop_when_disabled(self):
        trainer, calls = self._trainer(enable=False)
        trainer._profiler_start()
        trainer._profiler_step()
        trainer._profiler_stop()
        assert calls == []

    def test_dispatch_to_policy_when_enabled(self):
        trainer, calls = self._trainer(enable=True)
        trainer._profiler_start()
        trainer._profiler_step()
        trainer._profiler_stop()
        assert calls == [("start", "policy"), ("step", "policy"), ("stop", "policy")]
