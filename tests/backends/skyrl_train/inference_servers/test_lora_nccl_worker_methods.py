import json
from types import SimpleNamespace

import pytest

import skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap as worker_wrap
from skyrl.backends.skyrl_train.weight_sync.lora_nccl import (
    LoRANcclRendezvous,
    LoRANcclTransferReceipt,
    build_lora_nccl_plan,
)
from skyrl.backends.skyrl_train.weight_sync.lora_transport import (
    LoRABridgeSource,
    LoRABridgeSourceLayout,
    LoRAUpdateRequest,
)
from skyrl.backends.skyrl_train.weight_sync.lora_transport.consumer_plan import (
    LoRAConsumerCopy,
    LoRAConsumerPlan,
    LoRAConsumerPull,
)
from skyrl.backends.skyrl_train.weight_sync.lora_transport.contracts import (
    LoRASourceSlice,
)


class _Session:
    def __init__(self, plan_digest):
        self.plan_digest = plan_digest
        self.close_count = 0
        self.requests = []

    def receive(self, request):
        self.requests.append(request)
        return {"model.proj": (["a"], ["b"])}, LoRANcclTransferReceipt(
            request.generation,
            self.plan_digest,
            "receive",
            0,
            1,
            8,
            0.25,
        )

    def close(self):
        self.close_count += 1


def _layout():
    return LoRABridgeSourceLayout(
        "adapter",
        (
            LoRABridgeSource(
                key="adapter.weight",
                source_rank=0,
                hf_param_names=("model.proj.lora_B.weight",),
                component="linear_out",
                transform="identity",
                shape=(1, 2),
                tensor_parallel_axis=0,
                tensor_parallel_rank=0,
                tensor_parallel_size=1,
                expert_parallel_axis=None,
                expert_parallel_rank=0,
                expert_parallel_size=1,
                transform_config=(),
            ),
        ),
    )


def _consumer_plan(layout):
    pull = LoRAConsumerPull(
        0,
        LoRASourceSlice("adapter.weight", (0, 0), (1, 2)),
    )
    module = SimpleNamespace(
        module_name="model.proj",
        factor_shapes=(((1, 2), (1, 2)),),
    )
    return LoRAConsumerPlan(
        layout.layout_digest,
        SimpleNamespace(modules=(module,)),
        (pull,),
        (LoRAConsumerCopy(0, module.module_name, 0, 1, (0, 0), (1, 2)),),
    )


def _rendezvous(layout, consumer_plan):
    plan = build_lora_nccl_plan({0: consumer_plan}, 32)
    return LoRANcclRendezvous.from_plan(
        plan,
        layout.adapter_name,
        "10.0.0.4",
        41000,
    )


def _worker():
    worker = object.__new__(worker_wrap.NewInferenceWorkerWrap)
    worker.model_runner = object()
    worker.device = "cuda:0"
    worker.rank = 0
    return worker


@pytest.fixture(autouse=True)
def _mock_cuda(monkeypatch):
    monkeypatch.setattr(worker_wrap.torch.cuda, "synchronize", lambda device: None)


def test_worker_caches_route_initializes_once_and_stages_received_factors(monkeypatch):
    layout = _layout()
    consumer_plan = _consumer_plan(layout)
    rendezvous = _rendezvous(layout, consumer_plan)
    config = {"r": 2}
    builds = []
    sessions = []
    staged = []

    def build_plan(source_layout, adapter_config, runner):
        builds.append((source_layout, adapter_config, runner))
        return consumer_plan

    def open_session(plan, rank, info, device):
        assert plan is consumer_plan
        assert rank == 0
        assert info == rendezvous
        assert device == "cuda:0"
        session = _Session(info.plan_digest)
        sessions.append(session)
        return session

    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.weight_sync.lora_transport.build_vllm_lora_consumer_plan",
        build_plan,
    )
    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.weight_sync.lora_nccl.open_lora_nccl_receiver_session",
        open_session,
    )
    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.weight_sync.lora_transport.vllm_adapter.stage_vllm_local_lora_factors",
        lambda runner, adapter_id, receiver_plan, factors: staged.append(
            (runner, adapter_id, receiver_plan, factors)
        ),
    )
    worker = _worker()

    first_route = worker.inspect_lora_transport_route(layout.to_json_dict(), config)
    second_route = worker.inspect_lora_transport_route(layout.to_json_dict(), config)
    first_init = worker.init_lora_nccl_transport(
        rendezvous.to_json_dict(),
        layout.to_json_dict(),
        config,
    )
    second_init = worker.init_lora_nccl_transport(
        rendezvous.to_json_dict(),
        layout.to_json_dict(),
        config,
    )
    request = LoRAUpdateRequest.from_layout(layout, 2)
    receipt = worker.stage_lora_nccl_adapter(request.to_json_dict(), 9)

    assert first_route == second_route
    assert len(builds) == 1
    assert len(sessions) == 1
    assert first_init["reused"] is False
    assert second_init["reused"] is True
    assert receipt == {
        "adapter_id": 9,
        "generation": 2,
        "plan_digest": rendezvous.plan_digest,
        "direction": "receive",
        "rank": 0,
        "bucket_count": 1,
        "fp32_bytes": 8,
        "envelope_seconds": 0.25,
    }
    assert sessions[0].requests == [request]
    assert staged == [
        (
            worker.model_runner,
            9,
            consumer_plan.receiver_plan,
            {"model.proj": (["a"], ["b"])},
        )
    ]
    assert worker._skyrl_lora_transport_staged[
        "adapter"
    ].adapter_config_json == json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
    )


def test_worker_rejects_stale_generation_before_receive(monkeypatch):
    layout = _layout()
    consumer_plan = _consumer_plan(layout)
    rendezvous = _rendezvous(layout, consumer_plan)
    session = _Session(rendezvous.plan_digest)
    worker = _worker()
    worker._skyrl_lora_nccl_sessions = {
        "adapter": worker_wrap._LoRANcclWorkerSession(
            rendezvous,
            json.dumps({"r": 2}, sort_keys=True, separators=(",", ":")),
            consumer_plan,
            session,
        )
    }
    active_request = LoRAUpdateRequest.from_layout(layout, 3)
    worker._skyrl_lora_transport_active = {
        "adapter": SimpleNamespace(
            request=active_request,
            adapter_id=8,
            adapter_config_json=json.dumps(
                {"r": 2},
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    }

    with pytest.raises(ValueError, match="stale"):
        worker.stage_lora_nccl_adapter(
            LoRAUpdateRequest.from_layout(layout, 2).to_json_dict(),
            9,
        )

    assert session.requests == []


def test_worker_requires_unload_before_idempotent_transport_close():
    layout = _layout()
    consumer_plan = _consumer_plan(layout)
    rendezvous = _rendezvous(layout, consumer_plan)
    session = _Session(rendezvous.plan_digest)
    worker = _worker()
    worker._skyrl_lora_nccl_sessions = {
        "adapter": worker_wrap._LoRANcclWorkerSession(
            rendezvous,
            "{}",
            consumer_plan,
            session,
        )
    }
    worker._skyrl_lora_transport_active = {"adapter": SimpleNamespace(adapter_id=9)}

    with pytest.raises(ValueError, match="must be unloaded"):
        worker.close_lora_nccl_transport("adapter")
    worker._skyrl_lora_transport_active = {}
    worker.close_lora_nccl_transport("adapter")
    worker.close_lora_nccl_transport("adapter")

    assert session.close_count == 1
    assert worker._skyrl_lora_nccl_sessions == {}


def test_worker_reset_closes_transport_but_preserves_active_generation():
    layout = _layout()
    consumer_plan = _consumer_plan(layout)
    rendezvous = _rendezvous(layout, consumer_plan)
    session = _Session(rendezvous.plan_digest)
    worker = _worker()
    worker._skyrl_lora_nccl_sessions = {
        "adapter": worker_wrap._LoRANcclWorkerSession(
            rendezvous,
            "{}",
            consumer_plan,
            session,
        )
    }
    active = SimpleNamespace(adapter_id=9)
    worker._skyrl_lora_transport_active = {"adapter": active}
    worker._skyrl_lora_nccl_consumer_plans = {"adapter": ("{}", consumer_plan)}

    worker.reset_lora_nccl_transport("adapter")
    worker.reset_lora_nccl_transport("adapter")

    assert session.close_count == 1
    assert worker._skyrl_lora_nccl_sessions == {}
    assert worker._skyrl_lora_transport_active == {"adapter": active}
    assert worker._skyrl_lora_nccl_consumer_plans == {"adapter": ("{}", consumer_plan)}
