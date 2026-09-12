from types import SimpleNamespace

import pytest
import torch

pytest.importorskip(
    "megatron.core.parallel_state",
    reason="adapter publication is implemented by the Megatron worker",
)
pytestmark = pytest.mark.megatron

from skyrl.backends.skyrl_train.weight_sync.lora_nccl import (  # noqa: E402
    LoRANcclConsumerRoute,
    LoRANcclTransferReceipt,
)
from skyrl.backends.skyrl_train.weight_sync.lora_transport import (  # noqa: E402
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
        tensor_parallel_axis=0,
        tensor_parallel_rank=0,
        tensor_parallel_size=1,
        expert_parallel_axis=None,
        expert_parallel_rank=0,
        expert_parallel_size=1,
        transform_config=(),
    )


class _Session:
    def __init__(self, fail_generation=None):
        self.fail_generation = fail_generation
        self.generations = []
        self.close_count = 0

    def send(self, request, tensors):
        self.generations.append(request.generation)
        assert set(tensors) == {
            "decoder.layers.0.mlp.linear_fc2.adapter.linear_out.weight"
        }
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
            ),
        ),
    )


@pytest.fixture
def publication_environment(monkeypatch):
    from megatron.bridge.models.conversion import peft_bridge

    from skyrl.backends.skyrl_train.inference_servers import remote_inference_client
    from skyrl.backends.skyrl_train.weight_sync import lora_nccl
    from skyrl.backends.skyrl_train.workers.megatron import megatron_worker

    worker = object.__new__(MegatronPolicyWorkerBase)
    worker.actor_module = object()
    worker.bridge = SimpleNamespace(
        export_local_adapter_weights=lambda actor_module: [_record()]
    )
    worker.lora_cls = object()
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
        lora_nccl,
        "open_lora_nccl_source_session",
        open_session,
    )
    return worker, RemoteClient(), sessions


@pytest.mark.asyncio
async def test_lora_nccl_reuses_plan_and_transport(publication_environment):
    worker, client, sessions = publication_environment

    await worker._publish_lora_nccl_adapter(client, "adapter")
    await worker._publish_lora_nccl_adapter(client, "adapter")

    assert client.inspections == 1
    assert len(client.initializations) == 1
    rendezvous = client.initializations[0][0]
    assert rendezvous["source_ranks"] == (0,)
    assert rendezvous["inference_ranks"] == (0,)
    assert rendezvous["master_address"] == "10.0.0.1"
    assert rendezvous["master_port"] == 41000
    assert "groups" not in rendezvous
    assert client.loads == [("adapter", 0), ("adapter", 1)]
    assert client.resets == []
    assert len(sessions) == 1
    assert sessions[0].generations == [0, 1]


@pytest.mark.asyncio
async def test_lora_nccl_failure_resets_transport_and_retry_advances_generation(
    publication_environment,
):
    worker, client, sessions = publication_environment

    await worker._publish_lora_nccl_adapter(client, "adapter")
    sessions[0].fail_generation = 1
    with pytest.raises(RuntimeError, match="injected send failure"):
        await worker._publish_lora_nccl_adapter(client, "adapter")
    await worker._publish_lora_nccl_adapter(client, "adapter")

    assert client.inspections == 1
    assert len(client.initializations) == 2
    assert client.loads == [("adapter", 0), ("adapter", 2)]
    assert client.resets == ["adapter"]
    assert len(sessions) == 2
    assert sessions[0].generations == [0, 1]
    assert sessions[0].close_count == 1
    assert sessions[1].generations == [2]
