"""The sharded-RDT ``WeightTransferStrategy`` adapter: capability flags, init info,
and that ``send`` never materializes metadata or chunks.

The rendezvous itself (Ray actors, NIXL, CUDA IPC) is covered by the producer/plan
suites and the GPU tests.
"""

from dataclasses import dataclass
from typing import Optional

import pytest

from skyrl.backends.skyrl_train.weight_sync import (
    ShardedRdtInitInfo,
    ShardedRdtTransferStrategy,
    ShardedRdtWeightTransferSender,
    get_transfer_strategy_cls,
)
from skyrl.backends.skyrl_train.weight_sync.transfer_strategy import (
    WeightSyncInitInfo,
    WeightTransferSender,
)


@dataclass
class _IeCfg:
    """Just the fields the strategy reads."""

    model_dtype: str = "torch.bfloat16"
    run_engines_locally: bool = False


class _Client:
    server_urls = ["http://a:1", "http://b:2"]
    data_parallel_size = 1


class _Extractor:
    """Records whether the metadata/chunk channels were touched at all."""

    def __init__(self, derives_metadata_from_chunks: bool = False) -> None:
        self.metadata_calls = 0
        self.extract_calls = 0
        self.derives_metadata_from_chunks = derives_metadata_from_chunks

    def get_weight_metadata(self, dtype):
        self.metadata_calls += 1
        return {"names": [], "dtype_names": [], "shapes": []}

    def extract_weights(self, dtype):
        self.extract_calls += 1
        return iter(())


class _FakeRdtSender:
    """Stands in for RdtWeightSyncSender: records the calls it receives."""

    def __init__(self) -> None:
        self.sent = []
        self.torn_down = 0

    async def send(self, weight_extractor):
        self.sent.append(weight_extractor)

    def teardown(self):
        self.torn_down += 1


class TestInitInfo:
    def test_carries_the_config_derived_args(self, monkeypatch):
        monkeypatch.setattr(
            "skyrl.backends.skyrl_train.weight_sync.sharded_rdt.sharded_rdt_strategy._ray_namespace",
            lambda: "ns",
        )
        info = ShardedRdtTransferStrategy.create_init_info(_IeCfg(), inference_world_size=16)
        assert isinstance(info, ShardedRdtInitInfo)
        assert info.model_dtype == "torch.bfloat16"
        assert info.inference_world_size == 16
        assert info.trainer_actor_namespace == "ns"

    def test_a_missing_world_size_is_refused(self):
        """The consumer count sizes every ownership decision; defaulting it would
        mis-map consumers onto slices with nothing downstream to notice."""
        for bad in (None, 0):
            with pytest.raises(ValueError, match="inference world size"):
                ShardedRdtTransferStrategy.create_init_info(_IeCfg(), inference_world_size=bad)

    def test_base_model_path_is_ignored(self):
        """RDT reads the live model, never a checkpoint, so the delta backend's
        argument must not become a hidden requirement here."""
        info = ShardedRdtTransferStrategy.create_init_info(
            _IeCfg(), inference_world_size=2, base_model_path="/does/not/matter"
        )
        assert info.inference_world_size == 2

    def test_override_existing_receiver_follows_run_engines_locally(self):
        assert ShardedRdtTransferStrategy.create_init_info(
            _IeCfg(run_engines_locally=False), inference_world_size=1
        ).override_existing_receiver
        assert not ShardedRdtTransferStrategy.create_init_info(
            _IeCfg(run_engines_locally=True), inference_world_size=1
        ).override_existing_receiver


class TestCapabilityFlags:
    def test_the_sender_owns_the_inference_side_handshake(self):
        """trainer_init opens the inference side itself, so worker rank 0 must not
        also push init_info — doing both would init the engines twice."""
        assert ShardedRdtTransferStrategy.sender_initializes_receivers is True
        assert get_transfer_strategy_cls("nccl", False).sender_initializes_receivers is False

    def test_every_strategy_accepts_the_weight_extractor(self):
        """The worker passes `weight_extractor` to every strategy uniformly, so all
        of them must accept the keyword even though only sharded_rdt uses it."""
        import inspect

        for backend, colocate in (("nccl", False), ("nccl", True), ("sharded_rdt", False)):
            sig = inspect.signature(get_transfer_strategy_cls(backend, colocate).create_sender)
            assert "weight_extractor" in sig.parameters, f"{backend}/{colocate} rejects weight_extractor"

    def test_expandable_segments_are_forced_off(self):
        """The sidecar shares gathered tensors over CUDA IPC on every run, not only
        under colocation, and VMM-backed memory makes that 5-10x slower."""
        assert ShardedRdtWeightTransferSender.force_disable_expandable_segments is True
        assert WeightTransferSender.force_disable_expandable_segments is False

    def test_the_post_send_empty_cache_is_skipped(self):
        """Publish buffers are reused by the next training step, so returning them to
        CUDA costs 0.25-0.53s per rank at 235B and buys nothing."""
        assert ShardedRdtWeightTransferSender.empty_cache_after_send is False
        assert WeightTransferSender.empty_cache_after_send is True

    def test_the_worker_still_resets_the_prefix_cache(self):
        """Unlike delta, this sender does not touch the prefix cache, so the worker
        must keep doing it."""
        assert ShardedRdtWeightTransferSender.handles_prefix_cache_reset is False


class TestSend:
    @pytest.mark.asyncio
    async def test_send_delegates_to_the_rdt_sender(self):
        inner = _FakeRdtSender()
        sender = ShardedRdtWeightTransferSender(inner)
        extractor = _Extractor()

        await sender.send(extractor, "torch.bfloat16")

        assert inner.sent == [extractor]

    @pytest.mark.asyncio
    async def test_send_never_materializes_metadata_or_chunks(self):
        """get_weight_metadata on the Megatron extractor is a whole-model
        export_hf_weights pass -- ~20s at 235B and model-sized memory, which is what
        the RDT weight source avoids. The push backends' default send() calls it, so
        this override must not."""
        extractor = _Extractor()
        await ShardedRdtWeightTransferSender(_FakeRdtSender()).send(extractor, "torch.bfloat16")

        assert extractor.metadata_calls == 0
        assert extractor.extract_calls == 0

    @pytest.mark.asyncio
    async def test_the_push_backends_kwargs_are_accepted_and_ignored(self):
        """The workers pass one kwarg set to every sender; an unexpected keyword
        here would break the shared call site."""
        inner = _FakeRdtSender()
        await ShardedRdtWeightTransferSender(inner).send(_Extractor(), "torch.bfloat16", reset_prefix_cache=True)
        assert len(inner.sent) == 1

    @pytest.mark.asyncio
    async def test_send_refuses_serialized_fp8_extractors(self):
        """The weight sources publish whole bridge tensors, so an FP8 sync would
        land dequantized weights in modules vLLM built for FP8 -- wrong rollout
        weights rather than an error."""
        inner = _FakeRdtSender()
        extractor = _Extractor(derives_metadata_from_chunks=True)

        with pytest.raises(ValueError, match="serialized FP8"):
            await ShardedRdtWeightTransferSender(inner).send(extractor, "torch.bfloat16")

        assert inner.sent == []

    @pytest.mark.asyncio
    async def test_send_chunks_refuses_with_an_explanation(self):
        """There is no chunk stream to push; the error has to say what to call."""
        with pytest.raises(NotImplementedError, match="pull"):
            await ShardedRdtWeightTransferSender(_FakeRdtSender()).send_chunks(iter(()))

    def test_teardown_forwards(self):
        inner = _FakeRdtSender()
        ShardedRdtWeightTransferSender(inner).teardown()
        assert inner.torn_down == 1


class TestCreateSender:
    @staticmethod
    def _info(namespace: Optional[str] = None) -> ShardedRdtInitInfo:
        return ShardedRdtInitInfo(
            override_existing_receiver=True,
            model_dtype="torch.bfloat16",
            inference_world_size=2,
            trainer_actor_namespace=namespace,
        )

    def test_a_missing_extractor_is_a_loud_error(self):
        """It would otherwise surface deep inside the rendezvous, or worse, as a
        deferred init on the first send (which deadlocks)."""
        with pytest.raises(RuntimeError, match="weight_extractor"):
            ShardedRdtTransferStrategy.create_sender(self._info(), _Client(), weight_extractor=None)

    def test_a_foreign_init_info_is_rejected(self):
        @dataclass
        class _Other(WeightSyncInitInfo):
            pass

        with pytest.raises(ValueError, match="ShardedRdtInitInfo"):
            ShardedRdtTransferStrategy.create_sender(
                _Other(override_existing_receiver=True), _Client(), weight_extractor=_Extractor()
            )

    def test_it_rendezvouses_eagerly_and_wraps_the_sender(self, monkeypatch):
        """create_sender must initialize() before returning: every rank is inside
        init_weight_sync_state here, which is the only window where rank 0 can wait
        on the inference side without the others spinning in gather collectives."""
        built = {}

        class _Recorder:
            def __init__(self, client, model_dtype, world_size, namespace):
                built.update(
                    client=client, model_dtype=model_dtype, world_size=world_size, namespace=namespace, inited=None
                )

            def initialize(self, weight_extractor):
                built["inited"] = weight_extractor

        monkeypatch.setattr(
            "skyrl.backends.skyrl_train.weight_sync.sharded_rdt.rdt_send.RdtWeightSyncSender",
            _Recorder,
        )
        extractor = _Extractor()
        sender = ShardedRdtTransferStrategy.create_sender(
            self._info(namespace="ns"), _Client(), weight_extractor=extractor
        )

        assert isinstance(sender, ShardedRdtWeightTransferSender)
        assert built["model_dtype"] == "torch.bfloat16"
        assert built["world_size"] == 2
        assert built["namespace"] == "ns"
        assert built["inited"] is extractor


@pytest.mark.vllm
class TestVllmEngineMapping:
    def test_it_returns_the_registered_consumer_engine(self):
        """The consumers construct this class through vLLM's factory."""
        pytest.importorskip("vllm")
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.sharded_rdt_engine import (
            ShardedRDTWeightTransferEngine,
        )

        assert ShardedRdtTransferStrategy.get_vllm_transfer_engine() is ShardedRDTWeightTransferEngine
