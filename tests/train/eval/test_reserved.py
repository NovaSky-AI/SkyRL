"""
uv run --isolated --extra dev pytest tests/train/eval/test_reserved.py

CPU tests for the reserved eval backend: its config derivation and ``sync`` / ``close`` against a
mocked client. Launching the engine group (``create``) and the load itself need GPUs; see
``tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_load_weights_from_path.py``.
"""

import os
import sys
import types
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


# ---------------------------------------------------------------------------
# create(): the engines it launched are given back when the rest of setup fails
# ---------------------------------------------------------------------------

_IS = "skyrl.backends.skyrl_train.inference_servers"


def _stub_engine_launch(monkeypatch, *, seen):
    """Stand-ins for what ``create`` imports lazily. ``setup.py`` needs ``vllm_router`` and
    ``build_vllm_cli_args`` needs vLLM, neither of which is installed on a CPU box."""
    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        RemoteInferenceClient,
    )

    setup = MagicMock()
    setup.proxy_url = "http://127.0.0.1:9000"
    setup.server_urls = ["http://127.0.0.1:9001"]
    setup.server_groups = [MagicMock(), MagicMock()]
    fake_setup_module = types.ModuleType(f"{_IS}.setup")
    fake_setup_module.VLLM_START_PORT = 8000
    fake_setup_module.create_inference_servers = MagicMock(return_value=setup)
    monkeypatch.setitem(sys.modules, f"{_IS}.setup", fake_setup_module)
    monkeypatch.setattr(f"{_IS}.utils.build_vllm_cli_args", MagicMock())
    monkeypatch.setattr(RemoteInferenceClient, "paths_exist", AsyncMock(return_value=seen))
    monkeypatch.setattr(RemoteInferenceClient, "aclose", AsyncMock())
    return setup, fake_setup_module.create_inference_servers


def _create(tmp_path, *, make_generator=None):
    cfg = example_dummy_config()
    cfg.trainer.export_path = str(tmp_path)
    make_generator = make_generator or (lambda client: MagicMock(name="generator"))
    return cfg, lambda: ReservedEvalBackend.create(
        cfg, MagicMock(name="tokenizer"), log_path=str(tmp_path / "logs"), make_generator=make_generator
    )


def _assert_shut_down(setup):
    setup.router.shutdown.assert_called_once()
    for group in setup.server_groups:
        group.shutdown.assert_called_once()


def test_create_leaves_the_engines_up_when_setup_succeeds(tmp_path, monkeypatch):
    setup, launch = _stub_engine_launch(monkeypatch, seen=[True])
    cfg, create = _create(tmp_path)

    backend = create()

    assert isinstance(backend, ReservedEvalBackend)
    training = cfg.generator.inference_engine
    assert launch.call_args.kwargs["placement_group"] is None  # its own placement group
    assert launch.call_args.kwargs["start_port"] == 8000 + training.num_engines * training.data_parallel_size * 100
    setup.router.shutdown.assert_not_called()
    for group in setup.server_groups:
        group.shutdown.assert_not_called()


def test_create_shuts_the_engines_down_when_the_probe_fails(tmp_path, monkeypatch):
    setup, _ = _stub_engine_launch(monkeypatch, seen=[False])  # an engine node cannot read export_path
    _, create = _create(tmp_path)

    with pytest.raises(RuntimeError, match="export_path"):
        create()

    _assert_shut_down(setup)


def test_create_shuts_the_engines_down_when_the_generator_factory_raises(tmp_path, monkeypatch):
    setup, _ = _stub_engine_launch(monkeypatch, seen=[True])

    def broken_factory(client):
        raise ValueError("custom generator needs a config this run does not have")

    _, create = _create(tmp_path, make_generator=broken_factory)

    with pytest.raises(ValueError, match="custom generator"):
        create()

    _assert_shut_down(setup)
