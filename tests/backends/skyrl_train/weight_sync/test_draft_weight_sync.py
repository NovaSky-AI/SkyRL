"""MTP weight selection, session ordering, and native draft endpoints."""

import asyncio
from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.inference_servers.layerwise_reload import (
    LayerwiseReloadWorkerMixin,
)
from skyrl.backends.skyrl_train.weight_sync import (
    WEIGHT_UPDATE_TARGET_DRAFT,
    WEIGHT_UPDATE_TARGET_MODEL,
    BroadcastInitInfo,
    BroadcastWeightTransferSender,
    CudaIpcInitInfo,
    CudaIpcWeightTransferSender,
    DeltaInitInfo,
    DeltaWeightTransferSender,
    needs_draft_weight_sync,
)
from skyrl.backends.skyrl_train.weight_sync.base import WeightChunk
from skyrl.backends.skyrl_train.weight_sync.draft_weights import (
    is_megatron_draft_param,
    is_megatron_mtp_param,
    validate_weight_update_target,
)
from skyrl.backends.skyrl_train.weight_sync.transfer_strategy import (
    WeightTransferSender,
)
from skyrl.backends.skyrl_train.weight_sync.weight_extractor import WeightExtractor


class TestNeedsDraftWeightSync:
    @pytest.mark.parametrize(
        "spec, expected",
        [
            (None, False),
            ({}, False),
            ({"method": "mtp", "num_speculative_tokens": 1}, True),
            ({"method": "deepseek_mtp"}, True),
            ({"method": "eagle3", "model": "some/eagle-head"}, False),
            ({"method": "dflash", "model": "some/dflash-head"}, False),
            ({"model": "some/draft-checkpoint"}, False),
            ({"method": "ngram", "prompt_lookup_max": 4}, False),
            ({"method": "suffix"}, False),
        ],
    )
    def test_spec_methods(self, spec, expected):
        assert needs_draft_weight_sync(spec) is expected


class TestMegatronDraftParamSelection:
    @pytest.mark.parametrize(
        "name",
        [
            "mtp.layers.0.transformer_layer.self_attention.linear_qkv.weight",
            "mtp.layers.0.eh_proj.weight",
            "mtp.layers.0.enorm.weight",
            "language_model.mtp.layers.0.mlp.experts.linear_fc1.weight0",
        ],
    )
    def test_mtp_block_params(self, name):
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
    def test_embedding_and_output_layer_ride_along(self, name):
        assert not is_megatron_mtp_param(name)
        assert is_megatron_draft_param(name)

    @pytest.mark.parametrize(
        "name",
        [
            "decoder.layers.0.self_attention.linear_qkv.weight",
            "decoder.layers.61.mlp.linear_fc1.weight",
            "decoder.final_layernorm.weight",
            "decoder.layers.0.mlp.router.weight",
            "vision_model.blocks.0.attn.qkv.weight",
        ],
    )
    def test_trunk_params_excluded(self, name):
        assert not is_megatron_draft_param(name)


def test_validate_weight_update_target():
    assert validate_weight_update_target("model") == WEIGHT_UPDATE_TARGET_MODEL
    assert validate_weight_update_target("draft") == WEIGHT_UPDATE_TARGET_DRAFT
    with pytest.raises(ValueError, match="Unknown weight update target"):
        validate_weight_update_target("drafter")


def _chunk(*names: str) -> WeightChunk:
    return WeightChunk(
        names=list(names),
        dtypes=["bfloat16"] * len(names),
        shapes=[[2]] * len(names),
        tensors=[torch.zeros(2, dtype=torch.bfloat16) for _ in names],
    )


class _FakeExtractor(WeightExtractor):
    def __init__(self, names, draft_names=None):
        self._names = list(names)
        self._draft_names = draft_names

    def extract_weights(self, dtype):
        yield _chunk(*self._names)

    def get_weight_metadata(self, dtype):
        return {
            "names": list(self._names),
            "dtype_names": ["bfloat16"] * len(self._names),
            "shapes": [[2]] * len(self._names),
        }

    def draft_extractor(self):
        if self._draft_names is None:
            return super().draft_extractor()
        return _FakeExtractor(self._draft_names)


class _RecordingSender(WeightTransferSender):
    def __init__(self):
        self.sessions = []

    async def send_chunks(self, chunks, weight_metadata=None, derive_metadata_from_chunks=False, target="model", **kw):
        names = [name for chunk in chunks for name in chunk.names]
        self.sessions.append((target, names, weight_metadata["names"] if weight_metadata else None, kw))

    def teardown(self):
        pass


class TestSenderSessions:
    def test_main_session_only_by_default(self):
        sender = _RecordingSender()
        asyncio.run(sender.send(_FakeExtractor(["a", "b"], draft_names=["mtp.x"]), torch.bfloat16))
        assert [s[0] for s in sender.sessions] == [WEIGHT_UPDATE_TARGET_MODEL]

    def test_draft_session_follows_main_session(self):
        sender = _RecordingSender()
        extractor = _FakeExtractor(["model.embed_tokens.weight", "model.layers.0.w"], draft_names=["mtp.fc.weight"])
        asyncio.run(sender.send(extractor, torch.bfloat16, sync_draft_weights=True, reset_prefix_cache=True))
        assert sender.sessions == [
            (
                WEIGHT_UPDATE_TARGET_MODEL,
                ["model.embed_tokens.weight", "model.layers.0.w"],
                ["model.embed_tokens.weight", "model.layers.0.w"],
                {"reset_prefix_cache": True},
            ),
            (WEIGHT_UPDATE_TARGET_DRAFT, ["mtp.fc.weight"], ["mtp.fc.weight"], {"reset_prefix_cache": True}),
        ]

    def test_extractor_without_draft_weights_fails_loud(self):
        sender = _RecordingSender()
        with pytest.raises(NotImplementedError, match="cannot extract spec-decode draft weights"):
            asyncio.run(sender.send(_FakeExtractor(["a"]), torch.bfloat16, sync_draft_weights=True))
        assert sender.sessions == []


class _FakeClient:
    def __init__(self):
        self.events = []

    async def start_weight_update(self, is_checkpoint_format=True, target="model"):
        self.events.append(("start", target))

    async def update_weights_ipc(self, update_info):
        self.events.append(("ipc", list(update_info["names"])))

    async def update_weights_nccl(self, update_info):
        self.events.append(("nccl", list(update_info["names"])))

    async def finish_weight_update(self, target="model"):
        self.events.append(("finish", target))


def _single_rank(monkeypatch, module):
    monkeypatch.setattr(module.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(module.torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(module.torch.distributed, "barrier", lambda: None)


def test_broadcast_sender_starts_session_on_target(monkeypatch):
    import skyrl.backends.skyrl_train.weight_sync.broadcast_strategy as broadcast_module

    _single_rank(monkeypatch, broadcast_module)
    monkeypatch.setattr(broadcast_module, "nccl_trainer_send_weights", lambda it, group, *, packed: list(it))
    client = _FakeClient()
    sender = BroadcastWeightTransferSender(
        init_info=BroadcastInitInfo(
            master_addr="127.0.0.1", master_port=1, rank_offset=1, world_size=2, override_existing_receiver=False
        ),
        model_update_group=object(),
        inference_client=client,
    )
    metadata = {"names": ["mtp.fc.weight"], "dtype_names": ["bfloat16"], "shapes": [[2]]}

    asyncio.run(sender.send_chunks(iter([_chunk("mtp.fc.weight")]), weight_metadata=metadata, target="draft"))

    assert client.events == [("start", "draft"), ("nccl", ["mtp.fc.weight"]), ("finish", "draft")]


def test_broadcast_fp8_sender_starts_session_on_target(monkeypatch):
    import skyrl.backends.skyrl_train.weight_sync.broadcast_strategy as broadcast_module

    _single_rank(monkeypatch, broadcast_module)
    monkeypatch.setattr(broadcast_module, "nccl_trainer_send_weights", lambda it, group, *, packed: list(it))
    client = _FakeClient()
    sender = BroadcastWeightTransferSender(
        init_info=BroadcastInitInfo(
            master_addr="127.0.0.1", master_port=1, rank_offset=1, world_size=2, override_existing_receiver=False
        ),
        model_update_group=object(),
        inference_client=client,
    )

    asyncio.run(sender.send_chunks(iter([_chunk("mtp.fc.weight")]), derive_metadata_from_chunks=True, target="draft"))

    assert client.events == [("start", "draft"), ("nccl", ["mtp.fc.weight"]), ("finish", "draft")]


def test_cuda_ipc_sender_forwards_target(monkeypatch):
    sender = CudaIpcWeightTransferSender(
        init_info=CudaIpcInitInfo(override_existing_receiver=False, model_dtype_str="bfloat16"),
        inference_client=_FakeClient(),
    )
    calls = []

    async def record(chunks, weight_metadata=None, target="model"):
        calls.append((list(chunks), target))

    monkeypatch.setattr(sender, "_send_chunks_vllm_native", record)
    asyncio.run(sender.send_chunks(iter([]), target="draft"))
    assert calls == [([], "draft")]


def test_delta_sender_rejects_draft_target():
    sender = DeltaWeightTransferSender(
        init_info=DeltaInitInfo(
            override_existing_receiver=False,
            base_model_path="base",
            sync_dir="/tmp/sync",
            local_checkpoint_dir="/tmp/local",
            publish_staging_dir="/tmp/staging",
        ),
        inference_client=object(),
    )
    with pytest.raises(ValueError, match="Delta weight sync cannot sync spec-decode draft weights"):
        asyncio.run(sender.send_chunks(iter([]), target="draft"))


def test_sharded_rdt_sender_rejects_draft_weights():
    from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.sharded_rdt_strategy import (
        ShardedRdtWeightTransferSender,
    )

    sender = ShardedRdtWeightTransferSender.__new__(ShardedRdtWeightTransferSender)
    with pytest.raises(ValueError, match="sharded_rdt cannot sync spec-decode draft weights"):
        asyncio.run(sender.send(_FakeExtractor(["a"]), torch.bfloat16, sync_draft_weights=True))


def test_native_draft_session_target():
    model, config = object(), object()
    worker = LayerwiseReloadWorkerMixin()
    worker._weight_update_active = True
    worker._weight_update_is_draft = True
    worker.weight_transfer_engine = SimpleNamespace(model=model, model_config=config)
    assert worker.skyrl_weight_update_target() == (model, config)
    worker._weight_update_active = False
    with pytest.raises(RuntimeError, match="start_weight_update"):
        worker.skyrl_weight_update_target()


def test_model_session_after_native_draft_session(monkeypatch):
    from skyrl.backends.skyrl_train.inference_servers import layerwise_reload

    monkeypatch.setattr(layerwise_reload, "_PATCHED_LAYERWISE_NUMEL_LOADED", True)
    worker = LayerwiseReloadWorkerMixin()
    worker._weight_update_is_draft = True
    worker.model_runner = SimpleNamespace(model=object())
    worker.model_config = object()
    worker.skyrl_start_weight_update(is_checkpoint_format=False)
    assert worker.skyrl_weight_update_target() == (worker.model_runner.model, worker.model_config)
    worker.skyrl_finish_weight_update()


@pytest.mark.parametrize("target", ["model", "draft"])
def test_client_weight_update_endpoints(monkeypatch, target):
    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        RemoteInferenceClient,
    )

    client = RemoteInferenceClient(
        server_urls=["http://localhost:8000"], proxy_url="http://localhost:8001", data_parallel_size=1
    )
    calls = []

    async def record(endpoint, payload):
        calls.append((endpoint, payload))
        return {}

    monkeypatch.setattr(client, "_call_all_servers", record)

    async def sync():
        await client.start_weight_update(target=target)
        await client.finish_weight_update(target=target)

    asyncio.run(sync())
    if target == "draft":
        assert calls == [("/start_draft_weight_update", {}), ("/finish_weight_update", {})]
    else:
        assert calls == [
            ("/collective_rpc", {"method": "skyrl_start_weight_update", "kwargs": {"is_checkpoint_format": True}}),
            ("/collective_rpc", {"method": "skyrl_finish_weight_update"}),
        ]
