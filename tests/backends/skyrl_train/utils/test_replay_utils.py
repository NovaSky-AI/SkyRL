import inspect
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.utils import replay_utils
from skyrl.utils.routed_experts import make_replay_padding_indices
from skyrl.utils.token_metadata import build_token_metadata_layout


@pytest.fixture
def parallel_state(monkeypatch):
    try:
        import megatron.core.parallel_state as mpu
    except ModuleNotFoundError:
        megatron = types.ModuleType("megatron")
        core = types.ModuleType("megatron.core")
        mpu = types.ModuleType("megatron.core.parallel_state")
        megatron.core = core
        core.parallel_state = mpu
        monkeypatch.setitem(sys.modules, "megatron", megatron)
        monkeypatch.setitem(sys.modules, "megatron.core", core)
        monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", mpu)

    monkeypatch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: 0, raising=False)
    return mpu


def test_patch_topk_router_expert_bias_excludes_padding(monkeypatch):
    router_module = types.ModuleType("megatron.core.transformer.moe.router")

    class TopKRouter:
        def __init__(self):
            self.local_tokens_per_expert = torch.zeros(3, dtype=torch.int64)

        def _apply_expert_bias(self, routing_map, padding_mask=None):
            if padding_mask is not None:
                routing_map = routing_map & (~padding_mask)
            self.local_tokens_per_expert += routing_map.sum(dim=0)

    router_module.TopKRouter = TopKRouter
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.moe.router", router_module)

    replay_utils.patch_topk_router_expert_bias_padding_mask()
    router = TopKRouter()
    router._apply_expert_bias(
        torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.bool),
        torch.tensor([False, True]),
    )

    assert torch.equal(router.local_tokens_per_expert, torch.tensor([1, 0, 1]))


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int16, torch.int32])
def test_replay_padding_indices_are_unique(dtype):
    padding = make_replay_padding_indices((2, 3, 4, 3), dtype=dtype)

    assert padding.shape == (2, 3, 4, 3)
    assert torch.equal(padding, torch.tensor([0, 1, 2], dtype=dtype).expand_as(padding))


def test_replay_has_no_dispatcher_specific_patch():
    assert "TokenDispatcher" not in inspect.getsource(replay_utils)


@pytest.mark.parametrize("route_dtype", [torch.uint8, torch.int16, torch.int32])
def test_setup_replay_installs_indices_and_returns_model_mask(monkeypatch, parallel_state, route_dtype):
    router_replay_module = types.ModuleType("megatron.core.transformer.moe.router_replay")

    class RouterReplay:
        global_router_replay_instances = [object()]
        replay_data = None
        action = None

        @classmethod
        def set_replay_data(cls, replay_data):
            cls.replay_data = replay_data

        @classmethod
        def set_global_router_replay_action(cls, action):
            cls.action = action

    class RouterReplayAction:
        REPLAY_FORWARD = "replay_forward"

    router_replay_module.RouterReplay = RouterReplay
    router_replay_module.RouterReplayAction = RouterReplayAction
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.moe.router_replay", router_replay_module)
    monkeypatch.setattr(replay_utils, "_get_current_pp_stage_layer_range", lambda model_config: (1, 1))
    monkeypatch.setattr(
        replay_utils,
        "scatter_router_padding_mask_for_model",
        lambda mask, model, model_config: mask,
    )
    apply_layout = replay_utils.align_token_metadata
    routed_layer_counts = []

    def record_routed_layer_count(metadata, layout, padding_value):
        if metadata.ndim == 4:
            routed_layer_counts.append(metadata.shape[2])
        return apply_layout(metadata, layout, padding_value)

    monkeypatch.setattr(replay_utils, "align_token_metadata", record_routed_layer_count)

    routes = torch.tensor(
        [
            [
                [[0, 1], [0, 1], [0, 1]],
                [[10, 11], [1, 2], [20, 21]],
                [[12, 13], [3, 4], [22, 23]],
                [[14, 15], [5, 6], [24, 25]],
            ]
        ],
        dtype=route_dtype,
    )
    attention_mask = torch.tensor([[0, 1, 1, 1]])
    router_padding_mask = torch.tensor([[1, 0, 0, 1]], dtype=torch.bool)
    metadata_layout = build_token_metadata_layout(
        attention_mask,
        routes.device,
        packed=False,
        fp8_enabled=False,
    )

    model_kwargs = replay_utils.setup_per_microbatch_replay_forward(
        routes,
        router_padding_mask,
        attention_mask,
        model=object(),
        model_config=SimpleNamespace(fp8=None),
        metadata_layout=metadata_layout,
    )

    assert RouterReplay.replay_data[0].tolist() == [[1, 2], [3, 4], [5, 6]]
    assert RouterReplay.replay_data[0].dtype == torch.int32
    assert RouterReplay.action == RouterReplayAction.REPLAY_FORWARD
    assert model_kwargs["padding_mask"].tolist() == [[False, False, True]]
    assert routed_layer_counts == [1]


@pytest.mark.parametrize(
    ("model_kind", "pre_process", "expected"),
    [
        ("gpt", True, [[False, False, True, True]]),
        ("gpt", False, [[True, True]]),
        ("hybrid", True, [[True, True]]),
    ],
)
def test_sequence_parallel_mask_layout(monkeypatch, model_kind, pre_process, expected):
    hybrid_model = types.ModuleType("megatron.core.models.hybrid.hybrid_model")
    tensor_parallel = types.ModuleType("megatron.core.tensor_parallel")
    utils = types.ModuleType("megatron.core.utils")

    class HybridModel:
        def __init__(self):
            self.pre_process = pre_process

    class GPTModel:
        def __init__(self):
            self.pre_process = pre_process

    hybrid_model.HybridModel = HybridModel
    tensor_parallel.scatter_to_sequence_parallel_region = lambda value: value.chunk(2, dim=0)[1]
    utils.unwrap_model = lambda model: model
    monkeypatch.setitem(sys.modules, "megatron.core.models.hybrid.hybrid_model", hybrid_model)
    monkeypatch.setitem(sys.modules, "megatron.core.tensor_parallel", tensor_parallel)
    monkeypatch.setitem(sys.modules, "megatron.core.utils", utils)

    mask = torch.tensor([[0, 0, 1, 1]], dtype=torch.bool)
    model = HybridModel() if model_kind == "hybrid" else GPTModel()
    scattered = replay_utils.scatter_router_padding_mask_for_model(
        mask,
        model,
        SimpleNamespace(sequence_parallel=True),
    )

    assert scattered.tolist() == expected


@pytest.fixture
def router_replay_module(monkeypatch):
    module = types.ModuleType("megatron.core.transformer.moe.router_replay")
    router = SimpleNamespace(replay_backward_list=[], action=None)

    class RouterReplay:
        global_router_replay_instances = [router]

        @classmethod
        def clear_global_indices(cls):
            for instance in cls.global_router_replay_instances:
                instance.replay_backward_list = []

        @classmethod
        def clear_global_router_replay_action(cls):
            for instance in cls.global_router_replay_instances:
                instance.action = None

    module.RouterReplay = RouterReplay
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.moe.router_replay", module)
    return router


def test_router_replay_schedule_clears_stale_forward_only_fifo(router_replay_module):
    router_replay_module.replay_backward_list = ["stale-forward-only"]

    with replay_utils.router_replay_schedule(enabled=True):
        assert router_replay_module.replay_backward_list == []
        router_replay_module.replay_backward_list.extend(["microbatch-0", "microbatch-1"])
        assert router_replay_module.replay_backward_list.pop(0) == "microbatch-0"
        assert router_replay_module.replay_backward_list.pop(0) == "microbatch-1"

    assert router_replay_module.replay_backward_list == []
    assert router_replay_module.action is None


def test_router_replay_schedule_clears_after_exception(router_replay_module):
    with pytest.raises(RuntimeError, match="schedule failed"):
        with replay_utils.router_replay_schedule(enabled=True):
            router_replay_module.replay_backward_list.append("partially-consumed-schedule")
            router_replay_module.action = "replay-backward"
            raise RuntimeError("schedule failed")

    assert router_replay_module.replay_backward_list == []
    assert router_replay_module.action is None
