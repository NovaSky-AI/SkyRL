"""Tests for the spec-decode drafter's weight-sync session.

vLLM's MTP drafter is a separate model that the main session never loads. Each
send therefore ends with a second session, opened on the drafter through
``/start_draft_weight_update``. The failures here are silent: a skipped or
misrouted draft session leaves the drafter on stale weights, and acceptance
decays without any error.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm", reason="the trainer engines subclass vLLM's")

pytestmark = pytest.mark.vllm

import torch  # noqa: E402

from skyrl.backends.skyrl_train.weight_sync.control_plane import (  # noqa: E402
    START_DRAFT_UPDATE_ENDPOINT,
    START_UPDATE_ENDPOINT,
)
from skyrl.backends.skyrl_train.weight_sync.sources import (  # noqa: E402
    MegatronWeightSource,
    is_megatron_draft_param,
    is_megatron_mtp_param,
)
from skyrl.backends.skyrl_train.weight_sync.weight_senders import (  # noqa: E402
    SkyrlDraftSessionMixin,
    SkyrlTrainerCapabilities,
    get_skyrl_ipc_trainer,
    get_skyrl_nccl_trainer,
)


class _RecordingClient:
    """The four-method protocol plus ``draft_session``, recording each call's route."""

    def __init__(self):
        self.calls = []
        self._start = START_UPDATE_ENDPOINT

    def draft_session(self):
        from contextlib import contextmanager

        @contextmanager
        def session():
            self._start = START_DRAFT_UPDATE_ENDPOINT
            try:
                yield
            finally:
                self._start = START_UPDATE_ENDPOINT

        return session()

    def start_weight_update(self):
        self.calls.append(self._start)

    def update_weights(self, update_info):
        self.calls.append(("update", update_info))

    def finish_weight_update(self, weight_version=None):
        self.calls.append("finish")

    def fetch_weights(self, **kwargs):
        self.calls.append("fetch")

    def pause_generation(self):
        self.calls.append("pause")

    def resume_generation(self):
        self.calls.append("resume")

    def reset_prefix_cache(self, reset_running_requests=True):
        self.calls.append("reset")


class _VllmLikeEngine:
    """Stands in for vLLM's trainer engine: one session over ``self.source``."""

    def __init__(self, client, source, fail_on=None):
        self.client = client
        self.source = source
        self._fail_on = fail_on

    def send_weights(self):
        self.client.start_weight_update()
        if self.source is self._fail_on:
            raise RuntimeError("broadcast failed")
        self.client.update_weights([name for name, _ in self.source])
        self.client.finish_weight_update()


class _Engine(SkyrlDraftSessionMixin, SkyrlTrainerCapabilities, _VllmLikeEngine):
    pass


def _source(*names):
    return [(name, torch.zeros(1)) for name in names]


class TestDraftSessionMixin:
    def test_draft_session_follows_the_main_session(self):
        client = _RecordingClient()
        main, draft = _source("model.w", "mtp.w"), _source("mtp.w")
        engine = _Engine(client, main)
        engine.skyrl_draft_source = draft
        engine.send_weights()
        assert client.calls == [
            START_UPDATE_ENDPOINT,
            ("update", ["model.w", "mtp.w"]),
            "finish",
            START_DRAFT_UPDATE_ENDPOINT,
            ("update", ["mtp.w"]),
            "finish",
        ]
        # The main source is back in place for the next round.
        assert engine.source is main

    def test_no_draft_source_sends_one_session(self):
        client = _RecordingClient()
        _Engine(client, _source("model.w")).send_weights()
        assert client.calls == [START_UPDATE_ENDPOINT, ("update", ["model.w"]), "finish"]

    def test_a_failed_draft_send_restores_the_main_source(self):
        client = _RecordingClient()
        main, draft = _source("model.w"), _source("mtp.w")
        engine = _Engine(client, main, fail_on=draft)
        engine.skyrl_draft_source = draft
        with pytest.raises(RuntimeError, match="broadcast failed"):
            engine.send_weights()
        assert engine.source is main
        client.start_weight_update()
        assert client.calls[-1] == START_UPDATE_ENDPOINT

    @pytest.mark.parametrize("get_trainer", [get_skyrl_nccl_trainer, get_skyrl_ipc_trainer])
    def test_push_trainers_wrap_vllms_send(self, get_trainer):
        from vllm.distributed.weight_transfer.base import TrainerWeightTransferEngine

        _, engine_cls = get_trainer()
        vllm_engine = next(c for c in engine_cls.__mro__ if c.__module__.startswith("vllm."))
        assert issubclass(vllm_engine, TrainerWeightTransferEngine)
        # The mixin must precede vLLM's engine so its send_weights wraps vLLM's.
        mro = engine_cls.__mro__
        assert mro.index(SkyrlDraftSessionMixin) < mro.index(vllm_engine)
        assert engine_cls.skyrl_draft_source is None


class TestDeltaDraftSession:
    def _engine(self, draft_source):
        from skyrl.backends.skyrl_train.weight_sync.delta.trainer import (
            DeltaTrainerWeightTransferEngine,
        )

        engine = DeltaTrainerWeightTransferEngine.__new__(DeltaTrainerWeightTransferEngine)
        engine.client = _RecordingClient()
        engine._init_info = SimpleNamespace(sync_dir="/tmp/sync")
        engine._reset_prefix_cache = False
        engine.skyrl_draft_source = draft_source
        return engine

    def test_drafter_reloads_the_same_version_inside_the_pause(self):
        engine = self._engine(_source("mtp.w"))
        update_info = {"target_version": 3}
        engine._apply_receiver_update(update_info)
        assert engine.client.calls == [
            "fetch",
            "pause",
            START_UPDATE_ENDPOINT,
            ("update", update_info),
            "finish",
            START_DRAFT_UPDATE_ENDPOINT,
            ("update", update_info),
            "finish",
            "resume",
        ]

    def test_no_draft_source_reloads_the_main_model_only(self):
        engine = self._engine(None)
        engine._apply_receiver_update({"target_version": 3})
        assert START_DRAFT_UPDATE_ENDPOINT not in engine.client.calls


class TestBuildTrainerEngineDraftSource:
    def _build(self, monkeypatch, speculative_config, weight_sync_backend="nccl"):
        from vllm.distributed.weight_transfer.factory import (
            WeightTransferTrainerFactory,
        )

        from skyrl.backends.skyrl_train.weight_sync.weight_senders import (
            build_trainer_engine,
        )

        seen = {"draft_factory_args": []}
        monkeypatch.setattr(
            WeightTransferTrainerFactory,
            "trainer_init",
            lambda init_info, *, client, source=None: SimpleNamespace(),
        )

        def draft_source_factory(dtype):
            seen["draft_factory_args"].append(dtype)
            return _source("mtp.w")

        seen["engine"] = build_trainer_engine(
            ie_cfg=SimpleNamespace(
                weight_sync_backend=weight_sync_backend,
                model_dtype="bfloat16",
                weight_transfer_threshold_cuda_ipc_GB=1.0,
                fp8_weight_sync_mode=None,
                speculative_config=speculative_config,
            ),
            colocate_all=False,
            rank=0,
            inference_world_size=1,
            source_factory=lambda dtype, backend: _source("model.w"),
            draft_source_factory=draft_source_factory,
            server_urls=["http://a"],
            data_parallel_size=1,
        )
        return seen

    def test_speculative_decoding_attaches_the_draft_source(self, monkeypatch):
        seen = self._build(monkeypatch, {"method": "mtp", "num_speculative_tokens": 1})
        assert seen["draft_factory_args"] == [torch.bfloat16]
        assert [name for name, _ in seen["engine"].skyrl_draft_source] == ["mtp.w"]

    def test_no_speculative_decoding_builds_no_draft_source(self, monkeypatch):
        seen = self._build(monkeypatch, None)
        assert seen["draft_factory_args"] == []
        assert seen["engine"].skyrl_draft_source is None

    def test_sharded_rdt_refuses_before_rendezvous(self, monkeypatch):
        with pytest.raises(ValueError, match="sharded_rdt cannot sync the spec-decode drafter"):
            self._build(monkeypatch, {"method": "mtp"}, weight_sync_backend="sharded_rdt")


class TestMegatronDraftParams:
    @pytest.mark.parametrize(
        "name",
        [
            "mtp.layers.0.transformer_layer.self_attention.linear_qkv.weight",
            "mtp.layers.0.eh_proj.weight",
            "language_model.mtp.layers.0.mlp.experts.linear_fc1.weight0",
        ],
    )
    def test_mtp_block(self, name):
        assert is_megatron_mtp_param(name)
        assert is_megatron_draft_param(name)

    @pytest.mark.parametrize(
        "name",
        [
            "embedding.word_embeddings.weight",
            "output_layer.weight",
            "language_model.embedding.word_embeddings.weight",
            "language_model.output_layer.weight",
        ],
    )
    def test_shared_embedding_and_output_layer_ride_along(self, name):
        assert not is_megatron_mtp_param(name)
        assert is_megatron_draft_param(name)

    @pytest.mark.parametrize(
        "name",
        [
            "decoder.layers.0.self_attention.linear_qkv.weight",
            "decoder.final_layernorm.weight",
            "decoder.layers.0.mlp.router.weight",
            "vision_model.blocks.0.attn.qkv.weight",
            "decoder.layers.0.mlp.smtp.weight",
        ],
    )
    def test_trunk_is_excluded(self, name):
        assert not is_megatron_draft_param(name)


class TestFilteredMegatronWeightSource:
    def test_exports_only_the_selected_tasks_in_one_call(self):
        names = ["embedding.word_embeddings.weight", "decoder.layers.0.w", "mtp.layers.0.a", "mtp.layers.0.b"]

        class _Bridge:
            def __init__(self):
                self.tasks_args = []

            def get_conversion_tasks(self, module):
                return [SimpleNamespace(global_param_name=name) for name in names]

            def export_hf_weights(self, module, show_progress=False, conversion_tasks=None):
                self.tasks_args.append(conversion_tasks)
                return iter([(task.global_param_name, torch.zeros(2)) for task in conversion_tasks])

        bridge = _Bridge()
        source = MegatronWeightSource(bridge, object(), torch.bfloat16, param_filter=is_megatron_draft_param)
        assert [m.name for m in source.metadata()] == [n for n, _ in source]
        assert [n for n, _ in source] == ["embedding.word_embeddings.weight", "mtp.layers.0.a", "mtp.layers.0.b"]
        # One export call per pass keeps every group_key's tasks together.
        assert all(len(args) == 3 for args in bridge.tasks_args)
