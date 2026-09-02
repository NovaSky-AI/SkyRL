"""Tests for ``weight_sync/trainer_engines.py``.

What is worth pinning are the choices it encodes that are silent when wrong:

* ``world_size = inference_world_size + 1`` (the trainer sender is a rank in the
  NCCL group). Off by one and the rendezvous never completes.
* ``packed=True`` on IPC, which overrides vLLM's default. Unpacked IPC holds a
  strong ref to a contiguous copy of every parameter until past
  ``finish_weight_update``, i.e. the whole model resident on the trainer.
* which backends need a per-server init rewrite.
* the capability probes defaulting correctly for an engine that declares nothing.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm", reason="trainer_engines builds vLLM trainer init infos")

pytestmark = pytest.mark.vllm

import torch  # noqa: E402
from vllm.distributed.weight_transfer.base import ParamMeta  # noqa: E402

from skyrl.backends.skyrl_train.weight_sync.control_plane import (  # noqa: E402
    nccl_init_payloads,
    rdt_init_payloads,
)
from skyrl.backends.skyrl_train.weight_sync.trainer_engines import (  # noqa: E402
    _build_init_info,
    _packed_buffer_size_bytes,
    engine_capability,
    maybe_set_reset_prefix_cache,
    teardown_engine,
)

_1GiB = 1024**3


class _Source:
    """Only ``metadata()`` is read here; the engines are not constructed."""

    def __init__(self, shapes=((4, 4),), dtype=None):
        self.metadata_calls = 0
        self._meta = [ParamMeta(f"w{i}", dtype or torch.bfloat16, s) for i, s in enumerate(shapes)]

    def metadata(self):
        self.metadata_calls += 1
        return self._meta

    def __iter__(self):
        return iter(())


def _init_info(backend, *, inference_world_size=4, ie_cfg=None, base_model_path=None, rank=0, source=None):
    return _build_init_info(
        backend=backend,
        ie_cfg=ie_cfg if ie_cfg is not None else SimpleNamespace(),
        rank=rank,
        inference_world_size=inference_world_size,
        source=source if source is not None else _Source(),
        server_urls=["http://a"],
        data_parallel_size=1,
        base_model_path=base_model_path,
    )


class TestPackedBufferSize:
    """A parameter too large for the buffer raises on the IPC path, and vLLM's
    1 GiB default is smaller than a large-vocab embedding matrix."""

    def test_small_models_keep_vllms_default(self):
        assert _packed_buffer_size_bytes(_Source(shapes=((4, 4),))) == _1GiB

    def test_grows_to_fit_the_largest_single_parameter(self):
        # 151936 x 4096 bf16 = Qwen3-235B's embedding: 1.24 GiB, over the default.
        source = _Source(shapes=((151936, 4096), (4096, 4096)))
        assert _packed_buffer_size_bytes(source) == 151936 * 4096 * 2

    def test_sizes_from_the_largest_not_the_total(self):
        # Four 0.5 GiB parameters total 2 GiB but each fits the default; the
        # buffer bounds one chunk, not the model.
        half_gib_rows = (1024**3) // 2 // 2 // 1024
        source = _Source(shapes=((half_gib_rows, 1024),) * 4)
        assert _packed_buffer_size_bytes(source) == _1GiB

    def test_an_empty_source_falls_back_to_the_default(self):
        assert _packed_buffer_size_bytes(_Source(shapes=())) == _1GiB


class TestNcclInitInfo:
    def test_world_size_counts_the_trainer_sender(self):
        info, _ = _init_info("nccl", inference_world_size=4)
        assert info.world_size == 5

    def test_packed_is_on(self):
        info, _ = _init_info("nccl")
        assert info.packed is True

    def test_buffer_is_sized_from_the_source(self):
        source = _Source(shapes=((151936, 4096),))
        info, _ = _init_info("nccl", source=source)
        assert info.packed_buffer_size_bytes == 151936 * 4096 * 2
        # metadata() is a collective on a Megatron source, so a rank that
        # skipped it would hang its peers.
        assert source.metadata_calls == 1

    def test_backend_is_vllms_own_key(self):
        """Separate registries, so only the receive side takes a new name."""
        info, _ = _init_info("nccl")
        assert info.backend == "nccl"

    def test_rank_decides_the_sender(self):
        assert _init_info("nccl", rank=0)[0].is_sender is True
        assert _init_info("nccl", rank=3)[0].is_sender is False

    def test_uses_the_per_server_rank_offset_rewrite(self):
        _, payload_fn = _init_info("nccl")
        assert payload_fn is nccl_init_payloads

    def test_picks_a_free_port(self):
        first, _ = _init_info("nccl")
        assert first.master_port > 0
        assert first.master_address


class TestIpcInitInfo:
    def test_packed_is_forced_on(self):
        info, payload_fn = _init_info("ipc")
        assert info.packed is True
        assert payload_fn is None

    def test_buffer_is_sized_from_the_source(self):
        source = _Source(shapes=((151936, 4096),))
        info, _ = _init_info("ipc", source=source)
        assert info.packed_buffer_size_bytes == 151936 * 4096 * 2

    def test_backend_is_vllms_own_key(self):
        assert _init_info("ipc")[0].backend == "ipc"


class TestShardedRdtInitInfo:
    def test_uses_the_replica_rank_rewrite(self):
        _, payload_fn = _init_info("sharded_rdt")
        assert payload_fn is rdt_init_payloads

    def test_carries_the_consumer_count(self):
        info, _ = _init_info("sharded_rdt", inference_world_size=8)
        assert info.backend == "sharded_rdt"
        assert info.num_consumers == 8

    def test_a_missing_consumer_count_is_rejected(self):
        """The ownership arithmetic is sized from it, and a wrong value silently
        mis-maps consumers onto slices, so there is no safe default."""
        with pytest.raises(ValueError, match="inference world size"):
            _init_info("sharded_rdt", inference_world_size=0)


def _delta_cfg(**overrides):
    delta = SimpleNamespace(
        sync_dir="/tmp/sync",
        local_checkpoint_dir="/tmp/local",
        publish_staging_dir="/tmp/staging",
        max_file_size_in_gb=1.0,
        cloud_download_workers=4,
        publish_num_workers=None,
        checkpoint_load_format="vllm_multi_thread_safetensors",
        multi_thread_safetensors_max_workers=8,
    )
    for key, value in overrides.items():
        setattr(delta, key, value)
    return SimpleNamespace(delta_weight_sync=delta)


class TestDeltaInitInfo:
    def test_does_not_touch_the_source(self):
        """No wire buffer to size, and a Megatron ``metadata()`` is a whole-model
        dry export."""
        source = _Source()
        _init_info("delta", ie_cfg=_delta_cfg(), base_model_path="/m", source=source)
        assert source.metadata_calls == 0

    def test_carries_the_publisher_and_worker_settings(self):
        info, payload_fn = _init_info("delta", ie_cfg=_delta_cfg(), base_model_path="/models/base")
        assert info.backend == "delta"
        assert info.base_model_path == "/models/base"
        assert info.sync_dir == "/tmp/sync"
        # Identical for every server, so no rewrite.
        assert payload_fn is None

    def test_requires_a_base_model_path(self):
        with pytest.raises(ValueError, match="base_model_path"):
            _init_info("delta", ie_cfg=_delta_cfg())

    def test_requires_a_sync_dir(self):
        with pytest.raises(ValueError, match="sync_dir"):
            _init_info("delta", ie_cfg=_delta_cfg(sync_dir=""), base_model_path="/models/base")

    def test_rejects_an_unsupported_load_format(self):
        with pytest.raises(ValueError, match="checkpoint_load_format"):
            _init_info("delta", ie_cfg=_delta_cfg(checkpoint_load_format="nope"), base_model_path="/m")


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="Unknown weight sync backend"):
        _init_info("telepathy")


def test_skyrl_trainer_engines_are_registered():
    """``delta`` and ``sharded_rdt`` are ours; vLLM registers the rest."""
    from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory

    _init_info("ipc")  # any call performs the registration
    for name in ("nccl", "ipc", "delta", "sharded_rdt"):
        assert name in WeightTransferTrainerFactory._registry


class TestBuildTrainerEngineResolvesTheBackend:
    """``build_trainer_engine`` resolves the backend from the same two config
    values the driver uses to configure the servers, and only then builds the
    source. A mismatch here means the trainer and receive engines disagree,
    which the driver has no way to catch."""

    def _build(self, monkeypatch, weight_sync_backend, colocate_all):
        from vllm.distributed.weight_transfer.factory import (
            WeightTransferTrainerFactory,
        )

        from skyrl.backends.skyrl_train.weight_sync.trainer_engines import (
            build_trainer_engine,
        )

        seen = {}

        def _fake_trainer_init(init_info, *, client, source=None):
            seen["init_info"] = init_info
            seen["source"] = source
            return object()

        monkeypatch.setattr(WeightTransferTrainerFactory, "trainer_init", _fake_trainer_init)

        def source_factory(dtype, backend):
            seen["factory_args"] = (dtype, backend)
            return _Source()

        build_trainer_engine(
            ie_cfg=SimpleNamespace(weight_sync_backend=weight_sync_backend, model_dtype="bfloat16"),
            colocate_all=colocate_all,
            rank=0,
            inference_world_size=4,
            source_factory=source_factory,
            server_urls=["http://a"],
            data_parallel_size=1,
            base_model_path=None,
        )
        return seen

    @pytest.mark.parametrize(
        "weight_sync_backend,colocate_all,expected",
        [
            ("nccl", False, "nccl"),
            # The one resolution with no config field of its own.
            ("nccl", True, "ipc"),
            ("sharded_rdt", False, "sharded_rdt"),
            ("rdt", False, "sharded_rdt"),
        ],
    )
    def test_resolution_reaches_both_the_factory_and_the_source(
        self, monkeypatch, weight_sync_backend, colocate_all, expected
    ):
        seen = self._build(monkeypatch, weight_sync_backend, colocate_all)
        assert seen["init_info"].backend == expected
        # The source factory is told the SAME backend, so sharded RDT gets its
        # ownership-aware subclass and nothing else does.
        assert seen["factory_args"][1] == expected

    def test_source_factory_gets_the_inference_dtype(self, monkeypatch):
        seen = self._build(monkeypatch, "nccl", False)
        assert seen["factory_args"][0] is torch.bfloat16

    def test_the_built_source_is_handed_to_the_engine(self, monkeypatch):
        seen = self._build(monkeypatch, "nccl", False)
        assert seen["source"] is not None


class _Bare:
    """An engine that declares nothing — the shape of vLLM's own engines."""


class TestCapabilityProbes:
    def test_defaults_for_an_engine_that_declares_nothing(self):
        engine = _Bare()
        assert engine_capability(engine, "handles_prefix_cache_reset", False) is False
        assert engine_capability(engine, "force_disable_expandable_segments", False) is False
        assert engine_capability(engine, "empty_cache_after_send", True) is True

    def test_reads_a_declared_flag(self):
        engine = _Bare()
        engine.skyrl_empty_cache_after_send = False
        assert engine_capability(engine, "empty_cache_after_send", True) is False

    def test_delta_declares_it_resets_the_prefix_cache(self):
        from skyrl.backends.skyrl_train.weight_sync.delta_trainer import (
            DeltaTrainerWeightTransferEngine,
        )

        assert engine_capability(DeltaTrainerWeightTransferEngine, "handles_prefix_cache_reset", False) is True

    def test_rdt_declares_its_two_memory_flags(self):
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.sharded_rdt_trainer import (
            ShardedRDTTrainerWeightTransferEngine as E,
        )

        assert engine_capability(E, "force_disable_expandable_segments", False) is True
        assert engine_capability(E, "empty_cache_after_send", True) is False

    def test_set_reset_prefix_cache_is_optional(self):
        # No setter must be a no-op, not an AttributeError.
        maybe_set_reset_prefix_cache(_Bare(), True)

        class _WithSetter:
            told = None

            def skyrl_set_reset_prefix_cache(self, reset):
                self.told = reset

        engine = _WithSetter()
        maybe_set_reset_prefix_cache(engine, True)
        assert engine.told is True


class TestTeardown:
    def test_shuts_down_the_engine_and_closes_its_client(self):
        calls = []

        class _Client:
            def close(self):
                calls.append("close")

        class _Engine:
            client = _Client()

            def shutdown(self):
                calls.append("shutdown")

        teardown_engine(_Engine())
        assert calls == ["shutdown", "close"]

    def test_closes_the_client_even_if_shutdown_raises(self):
        """A half-torn-down engine must not leak the session and fan-out pool."""
        calls = []

        class _Client:
            def close(self):
                calls.append("close")

        class _Engine:
            client = _Client()

            def shutdown(self):
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            teardown_engine(_Engine())
        assert calls == ["close"]

    def test_none_is_a_no_op(self):
        teardown_engine(None)
