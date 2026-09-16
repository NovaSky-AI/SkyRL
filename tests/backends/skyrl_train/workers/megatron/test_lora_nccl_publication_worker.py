import gc
import weakref
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

pytest.importorskip(
    "megatron.core.parallel_state",
    reason="adapter publication is implemented by the Megatron worker",
)
pytestmark = pytest.mark.megatron

from skyrl.backends.skyrl_train.weight_sync.lora_nccl.plan import (  # noqa: E402
    LoRANcclConsumerRoute,
)
from skyrl.backends.skyrl_train.weight_sync.lora_nccl.transport import (  # noqa: E402
    LoRANcclTransferReceipt,
)
from skyrl.backends.skyrl_train.weight_sync.lora_transport.bridge_sources import (  # noqa: E402
    LoRABridgeSourceLayout,
)
from skyrl.backends.skyrl_train.weight_sync.lora_transport.consumer_plan import (  # noqa: E402
    LoRAConsumerPull,
)
from skyrl.backends.skyrl_train.weight_sync.lora_transport.contracts import (  # noqa: E402
    LoRASourceSlice,
)
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (  # noqa: E402
    MegatronPolicyWorkerBase,
)


def _record():
    return SimpleNamespace(
        global_param_name="decoder.layers.0.mlp.linear_fc2.adapter.linear_out.weight",
        weight=torch.ones((2, 1), dtype=torch.float32),
        hf_param_names=("down_proj.lora_B.weight",),
        component="linear_out",
        transform="identity",
        alpha=32,
        effective_rank=32,
        tensor_parallel_axis=0,
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
        expert_parallel_axis=None,
        expert_parallel_rank=0,
        expert_parallel_size=1,
        transform_config=(),
    )


@pytest.mark.asyncio
async def test_native_broadcast_bypasses_sender_and_file_sync(monkeypatch):
    worker = object.__new__(MegatronPolicyWorkerBase)
    worker._is_lora = True
    worker.cfg = SimpleNamespace(
        fully_async=SimpleNamespace(enabled=False, clear_kv_cache_on_weight_sync=False),
        placement=SimpleNamespace(colocate_all=False),
        policy=SimpleNamespace(
            megatron_config=SimpleNamespace(
                lora_config=SimpleNamespace(merge_lora=False),
            ),
        ),
    )
    worker._resolve_lora_sync_target = MagicMock(
        return_value=("adapter", "/must/not/be/used"),
    )
    worker._publish_lora_nccl_adapter = AsyncMock()
    worker._save_lora_adapters_and_sync = AsyncMock()
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    client = object()
    inference_config = SimpleNamespace(
        enable_prefix_caching=False,
        model_dtype="bfloat16",
        weight_sync_backend="lora_nccl",
    )

    await worker.broadcast_to_inference_engines(client, inference_config)

    worker._publish_lora_nccl_adapter.assert_awaited_once_with(client, "adapter")
    worker._save_lora_adapters_and_sync.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_native_broadcast_preserves_file_sync(monkeypatch):
    worker = object.__new__(MegatronPolicyWorkerBase)
    worker._is_lora = True
    worker.cfg = SimpleNamespace(
        fully_async=SimpleNamespace(enabled=False, clear_kv_cache_on_weight_sync=False),
        placement=SimpleNamespace(colocate_all=False),
        policy=SimpleNamespace(
            megatron_config=SimpleNamespace(
                lora_config=SimpleNamespace(merge_lora=False),
            ),
        ),
    )
    worker._weight_transfer_sender = SimpleNamespace(
        handles_prefix_cache_reset=False,
        empty_cache_after_send=False,
    )
    worker._resolve_lora_sync_target = MagicMock(return_value=("adapter", "/adapter"))
    worker._publish_lora_nccl_adapter = AsyncMock()
    worker._save_lora_adapters_and_sync = AsyncMock()
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    client = object()
    inference_config = SimpleNamespace(
        enable_prefix_caching=False,
        model_dtype="bfloat16",
        weight_sync_backend="nccl",
    )

    await worker.broadcast_to_inference_engines(client, inference_config)

    worker._save_lora_adapters_and_sync.assert_awaited_once_with(
        "/adapter", client, lora_name="adapter"
    )
    worker._publish_lora_nccl_adapter.assert_not_awaited()


class _Session:
    def __init__(self, fail_generation=None):
        self.fail_generation = fail_generation
        self.generations = []
        self.values = []
        self.close_count = 0

    def send(self, request, tensors):
        self.generations.append(request.generation)
        assert set(tensors) == {
            "decoder.layers.0.mlp.linear_fc2.adapter.linear_out.weight"
        }
        self.values.append(next(iter(tensors.values())).flatten()[0].item())
        if request.generation == self.fail_generation:
            raise RuntimeError("injected send failure")
        return LoRANcclTransferReceipt(
            request.generation,
            "a" * 64,
            "send",
            0,
            1,
            8,
            0.1,
        )

    def close(self):
        self.close_count += 1


def _route(layout):
    source = layout.sources[0]
    return LoRANcclConsumerRoute(
        inference_rank=0,
        source_layout_digest=layout.layout_digest,
        pulls=(
            LoRAConsumerPull(
                source_rank=0,
                source_slice=LoRASourceSlice(
                    source.key,
                    starts=(0, 0),
                    stops=source.shape,
                ),
                value_scale=(1, 1),
            ),
        ),
    )


@pytest.fixture
def publication_environment(monkeypatch):
    from megatron.bridge.models.conversion import peft_bridge

    import skyrl.backends.skyrl_train.weight_sync.lora_nccl.rendezvous as lora_nccl_rendezvous
    from skyrl.backends.skyrl_train.inference_servers import remote_inference_client
    from skyrl.backends.skyrl_train.workers.megatron import megatron_worker

    worker = object.__new__(MegatronPolicyWorkerBase)
    worker.actor_module = SimpleNamespace(
        adapter_weight=torch.ones((2, 1), dtype=torch.float32)
    )
    exported_weight_refs = []

    def export_local_adapter_weights(actor_module):
        record = _record()
        record.weight = actor_module.adapter_weight.detach().clone()
        exported_weight_refs.append(weakref.ref(record.weight))
        return [record]

    worker.bridge = SimpleNamespace(
        export_local_adapter_weights=export_local_adapter_weights
    )
    worker.lora_cls = SimpleNamespace(dim=32)
    worker._logical_model_path = "model"

    class RemoteClient:
        server_urls = ("server",)

        def __init__(self):
            self.inspections = 0
            self.initializations = []
            self.loads = []
            self.resets = []

        async def inspect_lora_transport_routes(self, layout, adapter_config):
            self.inspections += 1
            source_layout = LoRABridgeSourceLayout.from_json_dict(layout)
            return [_route(source_layout).to_json_dict()]

        async def init_lora_nccl_transport(self, rendezvous, layout, adapter_config):
            self.initializations.append((rendezvous, layout, adapter_config))

        async def load_lora_nccl_adapter(self, lora_name, request, producer_ready):
            producer_error = await producer_ready
            if producer_error is not None:
                raise RuntimeError(producer_error)
            self.loads.append((lora_name, request["generation"]))

        async def reset_lora_nccl_transport(self, lora_name):
            self.resets.append(lora_name)

    monkeypatch.setattr(remote_inference_client, "RemoteInferenceClient", RemoteClient)
    monkeypatch.setattr(
        peft_bridge,
        "infer_target_modules_from_adapter_weights",
        lambda weights: ["down_proj"],
    )
    monkeypatch.setattr(
        peft_bridge,
        "build_adapter_config_dict",
        lambda lora_cls, target_modules, base_model_name_or_path: {
            "target_modules": target_modules
        },
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda results, value: results.__setitem__(slice(None), [value]),
    )
    monkeypatch.setattr(
        torch.distributed,
        "broadcast_object_list",
        lambda values, src: None,
    )
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(
        megatron_worker.ray.util,
        "get_node_ip_address",
        lambda: "10.0.0.1",
    )
    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.distributed.utils.get_free_port",
        lambda: 41000,
    )
    sessions = []

    def open_session(source_group, rendezvous, device):
        session = _Session()
        sessions.append(session)
        return session

    monkeypatch.setattr(
        lora_nccl_rendezvous,
        "open_lora_nccl_source_session",
        open_session,
    )
    return worker, RemoteClient(), sessions, exported_weight_refs


@pytest.mark.asyncio
async def test_lora_nccl_reuses_bounded_plan_and_transport_for_ten_publications(
    publication_environment,
):
    worker, client, sessions, exported_weight_refs = publication_environment

    await worker._publish_lora_nccl_adapter(client, "adapter")
    state = worker._lora_nccl_publication_states["adapter"]
    planner = state.planner
    plan = state.plan
    adapter_config = state.adapter_config
    rendezvous = state.rendezvous
    session = state.session
    for value in range(2, 11):
        worker.actor_module.adapter_weight.fill_(value)
        await worker._publish_lora_nccl_adapter(client, "adapter")

    assert client.inspections == 1
    assert len(client.initializations) == 1
    rendezvous_dict = client.initializations[0][0]
    assert rendezvous_dict["source_ranks"] == (0,)
    assert rendezvous_dict["inference_ranks"] == (0,)
    assert rendezvous_dict["master_address"] == "10.0.0.1"
    assert rendezvous_dict["master_port"] == 41000
    assert "groups" not in rendezvous_dict
    assert client.loads == [("adapter", generation) for generation in range(10)]
    assert client.resets == []
    assert len(sessions) == 1
    assert sessions[0].generations == list(range(10))
    assert sessions[0].values == list(range(1, 11))
    assert sessions[0].close_count == 0
    assert list(worker._lora_nccl_publication_states) == ["adapter"]
    final_state = worker._lora_nccl_publication_states["adapter"]
    assert final_state.planner is planner
    assert final_state.plan is plan
    assert final_state.adapter_config is adapter_config
    assert final_state.rendezvous is rendezvous
    assert final_state.session is session
    gc.collect()
    assert all(weight_ref() is None for weight_ref in exported_weight_refs)


@pytest.mark.asyncio
async def test_lora_nccl_publishes_ten_optimizer_backed_replacements(
    publication_environment,
):
    worker, client, sessions, exported_weight_refs = publication_environment
    parameter = torch.nn.Parameter(torch.ones((2, 1), dtype=torch.float32))
    worker.actor_module.adapter_weight = parameter
    optimizer = torch.optim.SGD([parameter], lr=1.0)

    for _ in range(10):
        optimizer.zero_grad()
        parameter.sum().backward()
        optimizer.step()
        await worker._publish_lora_nccl_adapter(client, "adapter")

    assert client.loads == [("adapter", generation) for generation in range(10)]
    assert len(client.initializations) == 1
    assert len(sessions) == 1
    assert sessions[0].generations == list(range(10))
    assert sessions[0].values == [float(-generation) for generation in range(10)]
    assert sessions[0].close_count == 0
    gc.collect()
    assert all(weight_ref() is None for weight_ref in exported_weight_refs)


@pytest.mark.asyncio
async def test_lora_nccl_failure_resets_transport_and_retry_advances_generation(
    publication_environment,
):
    worker, client, sessions, exported_weight_refs = publication_environment

    await worker._publish_lora_nccl_adapter(client, "adapter")
    state = worker._lora_nccl_publication_states["adapter"]
    planner = state.planner
    plan = state.plan
    sessions[0].fail_generation = 1
    worker.actor_module.adapter_weight.fill_(2)
    with pytest.raises(RuntimeError, match="injected send failure"):
        await worker._publish_lora_nccl_adapter(client, "adapter")
    worker.actor_module.adapter_weight.fill_(3)
    await worker._publish_lora_nccl_adapter(client, "adapter")

    assert client.inspections == 1
    assert len(client.initializations) == 2
    assert client.loads == [("adapter", 0), ("adapter", 2)]
    assert client.resets == ["adapter"]
    assert len(sessions) == 2
    assert sessions[0].generations == [0, 1]
    assert sessions[0].values == [1, 2]
    assert sessions[0].close_count == 1
    assert sessions[1].generations == [2]
    assert sessions[1].values == [3]
    final_state = worker._lora_nccl_publication_states["adapter"]
    assert final_state is state
    assert final_state.planner is planner
    assert final_state.plan is plan
    assert final_state.session is sessions[1]
    gc.collect()
    assert all(weight_ref() is None for weight_ref in exported_weight_refs)


@pytest.mark.asyncio
async def test_checkpoint_restore_publishes_restored_values_without_reinitializing_transport(
    publication_environment,
):
    worker, client, sessions, exported_weight_refs = publication_environment
    worker.model = MagicMock()
    worker.optimizer = MagicMock()
    worker.scheduler = MagicMock()
    worker.strategy = MagicMock()

    def load_checkpoint(**kwargs):
        worker.actor_module.adapter_weight.fill_(17)
        return "/checkpoint", {"global_step": 4}

    worker.strategy.load_checkpoint.side_effect = load_checkpoint

    await worker._publish_lora_nccl_adapter(client, "adapter")
    state = worker._lora_nccl_publication_states["adapter"]
    restored_state = worker.load_checkpoint("/checkpoint")
    await worker._publish_lora_nccl_adapter(client, "adapter")

    assert restored_state == {"global_step": 4}
    assert client.inspections == 1
    assert len(client.initializations) == 1
    assert len(sessions) == 1
    assert sessions[0].generations == [0, 1]
    assert sessions[0].values == [1, 17]
    assert worker._lora_nccl_publication_states["adapter"] is state
    gc.collect()
    assert all(weight_ref() is None for weight_ref in exported_weight_refs)


@pytest.mark.asyncio
async def test_delete_adapter_closes_and_removes_native_publication_state(
    publication_environment,
):
    worker, client, sessions, _ = publication_environment
    worker.adapter_store = MagicMock()
    worker._resolve_lora_sync_target = MagicMock(
        return_value=("adapter", "/must/not/be/used")
    )
    worker._is_lora_sync_writer_rank = MagicMock(return_value=False)

    await worker._publish_lora_nccl_adapter(client, "adapter")
    worker.delete_adapter("model")

    assert worker._lora_nccl_publication_states == {}
    assert len(sessions) == 1
    assert sessions[0].close_count == 1
    worker.adapter_store.delete.assert_called_once_with("model")
    worker._is_lora_sync_writer_rank.assert_called_once_with()
