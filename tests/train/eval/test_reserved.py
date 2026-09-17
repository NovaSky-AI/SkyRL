"""
uv run --isolated --extra dev pytest tests/train/eval/test_reserved.py

CPU tests for the reserved eval backend: its config derivation and ``sync`` / ``close`` against a
mocked client. Launching the engine group (``create``) and the load itself need GPUs; see
``tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_load_weights_from_path.py``.
"""

import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from skyrl.train.eval import EvalRequest, EvalSkip, reserved
from skyrl.train.eval.reserved import (
    ReservedEvalBackend,
    _default_export_cache_dir,
    reserved_train_cfg,
)
from tests.train.util import example_dummy_config


def _backend(tmp_path, *, versions=("5",)):
    client = MagicMock()
    client.load_weights_from_path = AsyncMock(return_value=list(versions))
    client.reset_prefix_cache = AsyncMock()
    client.teardown = AsyncMock()
    client.paths_exist = AsyncMock(return_value=[True, True])
    client.aclose = AsyncMock()
    generator = MagicMock(name="generator")
    setup = MagicMock()
    setup.server_groups = [MagicMock(), MagicMock()]
    backend = ReservedEvalBackend(setup=setup, client=client, generator=generator, cache_root=str(tmp_path / "cache"))
    return backend, client, generator, setup


def _export(tmp_path, step, *, complete=True):
    export = tmp_path / f"global_step_{step}" / "policy"
    export.mkdir(parents=True)
    if complete:
        (export / "config.json").write_text("{}")
    return str(export)


@pytest.mark.asyncio
async def test_sync_loads_the_export_invalidates_both_caches_and_leases_the_generator(tmp_path):
    backend, client, generator, _ = _backend(tmp_path)
    export_dir = _export(tmp_path, 5)

    lease = await backend.sync(EvalRequest(global_step=5, export_dir=export_dir))

    client.load_weights_from_path.assert_awaited_once_with(
        export_dir, weight_version="5", cache_dir=os.path.join(str(tmp_path / "cache"), "global_step_5")
    )
    client.reset_prefix_cache.assert_awaited_once()  # the engines' prefix cache
    client.increment_weight_version.assert_called_once()  # the generator's cache salt
    assert lease.generator is generator


@pytest.mark.asyncio
async def test_sync_skips_a_step_whose_export_is_missing_or_incomplete(tmp_path):
    backend, client, _, _ = _backend(tmp_path)

    with pytest.raises(EvalSkip) as missing:
        await backend.sync(EvalRequest(global_step=5, export_dir=str(tmp_path / "global_step_5" / "policy")))
    with pytest.raises(EvalSkip) as incomplete:
        await backend.sync(EvalRequest(global_step=6, export_dir=_export(tmp_path, 6, complete=False)))

    assert (missing.value.reason, incomplete.value.reason) == ("ckpt_missing", "ckpt_missing")
    client.load_weights_from_path.assert_not_awaited()


@pytest.mark.asyncio
async def test_sync_at_step_zero_loads_the_launch_weights_without_an_export(tmp_path):
    backend, client, _, _ = _backend(tmp_path, versions=("0",))

    await backend.sync(EvalRequest(global_step=0, export_dir="org/launch-model"))  # a hub id, not a directory

    assert client.load_weights_from_path.await_args.args == ("org/launch-model",)
    assert client.load_weights_from_path.await_args.kwargs["weight_version"] == "0"


@pytest.mark.asyncio
@pytest.mark.parametrize("versions", [("5", "4"), ()], ids=["one-stale-worker", "no-worker-answered"])
async def test_sync_requires_every_worker_to_attest_the_version(tmp_path, versions):
    backend, _, _, _ = _backend(tmp_path, versions=versions)

    with pytest.raises(EvalSkip) as skip:
        await backend.sync(EvalRequest(global_step=5, export_dir=_export(tmp_path, 5)))

    assert skip.value.reason == "sync_mismatch"


@pytest.mark.asyncio
async def test_sync_reports_a_failed_load(tmp_path):
    backend, client, _, _ = _backend(tmp_path)
    client.load_weights_from_path.side_effect = RuntimeError("server died")

    with pytest.raises(EvalSkip) as skip:
        await backend.sync(EvalRequest(global_step=5, export_dir=_export(tmp_path, 5)))

    assert skip.value.reason == "sync_failed"
    client.increment_weight_version.assert_not_called()


@pytest.mark.asyncio
async def test_close_attempts_every_teardown_step(tmp_path):
    backend, client, _, setup = _backend(tmp_path)
    setup.router.shutdown.side_effect = RuntimeError("router already gone")

    await backend.close()  # logs the failed step and keeps going

    for group in setup.server_groups:
        group.shutdown.assert_called_once()
    client.teardown.assert_awaited_once()


def test_default_export_cache_dir_is_derived_from_the_export_path():
    a = _default_export_cache_dir("s3://bucket/run-a/exports")
    b = _default_export_cache_dir("s3://bucket/run-b/exports")

    assert a != b
    assert a == _default_export_cache_dir("s3://bucket/run-a/exports")
    assert "skyrl_eval_exports" in a


def test_reserved_train_cfg_describes_the_group_and_leaves_the_run_config_alone():
    cfg = example_dummy_config()
    cfg.trainer.placement.colocate_all = True
    cfg.generator.inference_engine.num_engines = 4
    cfg.generator.inference_engine.offload_kv_for_weight_sync = True
    cfg.trainer.eval_dispatch.num_engines = 2
    cfg.trainer.eval_dispatch.engine_overrides = {"tensor_parallel_size": 1, "gpu_memory_utilization": 0.9}

    cfg_r = reserved_train_cfg(cfg)
    ie = cfg_r.generator.inference_engine

    assert (ie.num_engines, ie.tensor_parallel_size, ie.gpu_memory_utilization) == (2, 1, 0.9)
    assert (ie.enable_pd, ie.num_prefill, ie.offload_kv_for_weight_sync, ie.speculative_config) == (
        False,
        0,
        False,
        None,
    )
    assert cfg_r.trainer.placement.colocate_all is False  # never colocated, whatever training does
    assert cfg.trainer.placement.colocate_all is True and cfg.generator.inference_engine.num_engines == 4


def _sentinels(root):
    return [p.name for p in root.iterdir() if p.name.startswith(".reserved_eval_probe_")]


@pytest.mark.asyncio
async def test_probe_passes_when_every_worker_sees_the_export_root(tmp_path):
    backend, client, _, _ = _backend(tmp_path)

    await backend._probe_export_root(str(tmp_path))

    assert client.paths_exist.await_args.args[0].startswith(str(tmp_path))
    assert _sentinels(tmp_path) == []  # cleaned up
    client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_probe_fails_fast_when_a_worker_cannot_see_the_export_root(tmp_path):
    backend, client, _, _ = _backend(tmp_path)
    client.paths_exist.return_value = [True, False]

    with pytest.raises(RuntimeError, match="export_path"):
        await backend._probe_export_root(str(tmp_path))

    assert _sentinels(tmp_path) == []
    client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_probe_closes_the_session_even_when_the_sentinel_cannot_be_removed(tmp_path, monkeypatch):
    backend, client, _, _ = _backend(tmp_path)

    def failing_remove(path):
        raise OSError("transient storage error")

    monkeypatch.setattr(reserved.io, "remove", failing_remove)

    await backend._probe_export_root(str(tmp_path))  # the probe itself passed; the delete is best effort

    client.aclose.assert_awaited_once()
