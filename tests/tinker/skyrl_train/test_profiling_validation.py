"""Validation for /start_profiling that reaches into SkyRL-Train config.

Lives here rather than in ``tests/tinker/`` because these exercise
``TorchProfilerConfig`` and the SkyRL-Train backend, which need the fsdp/megatron
extras. The top-level tinker suite runs jax-only. CPU-only: no Ray, no GPU. Run:
  uv run --isolated --extra tinker --extra fsdp --with pytest \\
    pytest tests/tinker/skyrl_train/test_profiling_validation.py
"""

from __future__ import annotations

import pytest

from skyrl.tinker.config import EngineConfig, TinkerTorchProfilerConfig

PROFILER_CFG = {"export_dir": "/tmp/traces", "ranks": [0], "max_session_duration_sec": 7200}
WORKER_CFG = {"enable": True, "ranks": [0], "save_path": "/tmp/traces/120", "active": 5}
# FSDP profiling requires FSDP2-native offload; see test_fsdp_without_cpu_offload_is_rejected.
OFFLOAD_OK = {"trainer.policy.fsdp_config.cpu_offload": True}


class TestWorkerConfigValidation:

    def test_cloud_export_dir_is_accepted(self):
        cfg = TinkerTorchProfilerConfig(export_dir="s3://bucket/traces")
        cfg.validate_startup()

    def test_relative_export_dir_is_rejected(self):
        with pytest.raises(ValueError):
            TinkerTorchProfilerConfig(export_dir="traces/").validate_startup()

    def test_bad_schedule_is_rejected(self):
        from skyrl.tinker.api import _validate_worker_profiler_config

        cfg = EngineConfig(base_model="m", backend="fsdp", backend_config=OFFLOAD_OK)
        with pytest.raises(ValueError):
            _validate_worker_profiler_config({**WORKER_CFG, "active": 0}, cfg)

    def test_stacks_without_stack_is_rejected(self):
        from skyrl.tinker.api import _validate_worker_profiler_config

        cfg = EngineConfig(base_model="m", backend="fsdp", backend_config=OFFLOAD_OK)
        with pytest.raises(ValueError):
            _validate_worker_profiler_config({**WORKER_CFG, "export_type": "stacks", "with_stack": False}, cfg)

    def test_fsdp_without_cpu_offload_is_rejected(self):
        """
        ``colocate_all=False`` alone does not take the policy out of offload
        management, because ``colocate_policy_ref`` defaults to True. The manual
        offload path then swaps parameters the profiler holds references to.
        """
        from skyrl.tinker.api import _validate_worker_profiler_config

        cfg = EngineConfig(base_model="m", backend="fsdp", backend_config={"trainer.placement.colocate_all": False})
        with pytest.raises(ValueError, match="cpu_offload=true"):
            _validate_worker_profiler_config(WORKER_CFG, cfg)

    def test_fsdp_fully_uncolocated_does_not_require_cpu_offload(self):
        """With both colocation flags off nothing offloads, so swap_tensors never
        runs and the manual path is safe to profile."""
        from skyrl.tinker.api import _validate_worker_profiler_config

        cfg = EngineConfig(
            base_model="m",
            backend="fsdp",
            backend_config={
                "trainer.placement.colocate_all": False,
                "trainer.placement.colocate_policy_ref": False,
            },
        )
        _validate_worker_profiler_config(WORKER_CFG, cfg)

    def test_fsdp_with_cpu_offload_is_accepted(self):
        from skyrl.tinker.api import _validate_worker_profiler_config

        cfg = EngineConfig(
            base_model="m",
            backend="fsdp",
            backend_config={"trainer.policy.fsdp_config.cpu_offload": True},
        )
        _validate_worker_profiler_config(WORKER_CFG, cfg)

    def test_megatron_does_not_require_cpu_offload(self):
        from skyrl.tinker.api import _validate_worker_profiler_config

        cfg = EngineConfig(base_model="m", backend="megatron")
        _validate_worker_profiler_config(WORKER_CFG, cfg)

    def test_unknown_option_is_rejected(self):
        from skyrl.tinker.api import _validate_worker_profiler_config

        cfg = EngineConfig(base_model="m", backend="fsdp", backend_config=OFFLOAD_OK)
        with pytest.raises(TypeError):
            _validate_worker_profiler_config({**WORKER_CFG, "not_a_field": 1}, cfg)

    def test_non_integer_schedule_value_is_rejected(self):
        from skyrl.tinker.api import _validate_worker_profiler_config

        cfg = EngineConfig(base_model="m", backend="fsdp", backend_config=OFFLOAD_OK)
        with pytest.raises(TypeError):
            _validate_worker_profiler_config({**WORKER_CFG, "active": "five"}, cfg)


def test_backend_config_cannot_set_trainer_profiler_config():
    """Static config would fight /start_profiling over the single worker.profiler slot."""
    from skyrl.backends.skyrl_train_backend import (
        FSDPBackendOverrides,
        _build_skyrl_train_config,
    )

    overrides = FSDPBackendOverrides(**{"trainer.policy.torch_profiler_config.enable": True})
    with pytest.raises(ValueError, match="--torch-profiler"):
        _build_skyrl_train_config("m", overrides)
