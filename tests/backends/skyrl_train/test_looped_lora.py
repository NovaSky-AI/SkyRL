from types import MethodType

import pytest
import torch
from torch import nn

from skyrl.backends.skyrl_train.patches.megatron import looped_lora
from skyrl.backends.skyrl_train.patches.megatron.looped_lora import (
    _adapter_only,
    _looped_block_forward,
    _looped_linear_forward,
    _LoopedModuleList,
    _set_context,
)
from skyrl.train.looped_lora import (
    LayerExecution,
    build_looped_lora_schedule,
    get_lora_only_executions_by_physical_layer,
)


class _RecordingLayer(nn.Module):
    def __init__(self, layer_number: int, calls: list[tuple[int, bool]]) -> None:
        super().__init__()
        self.layer_number = layer_number
        self.calls = calls

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.calls.append((self.layer_number - 1, _adapter_only.get()))
        return hidden_states + self.layer_number


class _RecordingBlock(nn.Module):
    def __init__(self, num_layers: int, schedule) -> None:
        super().__init__()
        self.calls: list[tuple[int, bool]] = []
        layers = [_RecordingLayer(index + 1, self.calls) for index in range(num_layers)]
        self.layers = _LoopedModuleList(layers, schedule)
        self.num_layers_per_pipeline_rank = num_layers
        self._looped_lora_original_forward = self.forward
        self.forward = MethodType(_looped_block_forward, self)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class _FakeRMSNormLinear(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.layer_norm_weight = nn.Parameter(torch.ones(hidden_size))
        self.weight = nn.Parameter(torch.eye(hidden_size))
        self.normalization = "RMSNorm"
        self.eps = 1e-6
        self.zero_centered_gamma = False


class _FakeLoraLinear(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.to_wrap = _FakeRMSNormLinear(hidden_size)
        self.adapter = nn.Linear(hidden_size, hidden_size, bias=False)
        self._adapter_enabled = True
        self._base_returns_tuple = True
        self.base_calls = 0

    def _looped_lora_original_forward(self, inputs: torch.Tensor):
        self.base_calls += 1
        return self.to_wrap.weight @ inputs, None

    def adapter_forward(self, adapter: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        return adapter(inputs)


def test_megatron_extra_pass_skips_frozen_linear_and_backpropagates_through_lora() -> None:
    linear = _FakeLoraLinear(hidden_size=3)
    inputs = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    with _set_context(_adapter_only, True):
        output, bias = _looped_linear_forward(linear, inputs)
    output.sum().backward()

    assert bias is None
    assert linear.base_calls == 0
    assert linear.adapter.weight.grad is not None
    assert linear.to_wrap.weight.grad is None
    assert inputs.grad is not None


def test_megatron_extra_pass_repeats_fused_layernorm_sequence_gather(monkeypatch: pytest.MonkeyPatch) -> None:
    linear = _FakeLoraLinear(hidden_size=3)
    linear.to_wrap.return_layernorm_output_gathered = True
    linear.adapter.tp_group = object()
    gathered_group = None

    def gather(inputs: torch.Tensor, group: object) -> torch.Tensor:
        nonlocal gathered_group
        gathered_group = group
        return torch.cat((inputs, inputs), dim=0)

    monkeypatch.setattr(looped_lora, "_gather_sequence_parallel_input", gather)

    with _set_context(_adapter_only, True):
        output, _ = _looped_linear_forward(linear, torch.ones(2, 3))

    assert output.shape == (4, 3)
    assert gathered_group is linear.adapter.tp_group


@pytest.mark.parametrize("repeat_count", [1, 2, 4])
def test_megatron_block_uses_vllm_section_order(repeat_count: int) -> None:
    schedule = build_looped_lora_schedule(
        6,
        [{"start_layer": 1, "end_layer": 4, "repeat_count": repeat_count}],
    )
    block = _RecordingBlock(6, schedule)

    output = block(torch.zeros(1))

    physical_calls = [0, 1, 2, 3] + [1, 2, 3] * (repeat_count - 1) + [4, 5]
    expected_calls = [
        (physical_layer, execution.lora_only) for physical_layer, execution in zip(physical_calls, schedule)
    ]
    assert block.calls == expected_calls
    assert output.item() == sum(index + 1 for index in physical_calls)
    assert [layer.layer_number for layer in block.layers] == list(range(1, 7))


@pytest.mark.parametrize(
    ("repeat_count", "expected_length", "expected_lora_only"),
    [(1, 36, 0), (2, 44, 8), (4, 60, 24)],
)
def test_middle_section_repeats_only_lora_after_first_pass(
    repeat_count: int,
    expected_length: int,
    expected_lora_only: int,
) -> None:
    schedule = build_looped_lora_schedule(
        36,
        [{"start_layer": 14, "end_layer": 22, "repeat_count": repeat_count}],
    )

    assert len(schedule) == expected_length
    assert sum(execution.lora_only for execution in schedule) == expected_lora_only
    assert [execution.physical_layer for execution in schedule if not execution.lora_only] == list(range(36))


def test_sections_can_have_independent_repeat_counts() -> None:
    schedule = build_looped_lora_schedule(
        36,
        [
            {"start_layer": 0, "end_layer": 4, "repeat_count": 2},
            {"start_layer": 4, "end_layer": 20, "repeat_count": 3},
            {"start_layer": 20, "end_layer": 36, "repeat_count": 1},
        ],
    )

    assert len(schedule) == 72
    assert schedule[:4] == tuple(LayerExecution(layer, False) for layer in range(4))
    assert schedule[4:8] == tuple(LayerExecution(layer, True) for layer in range(4))
    assert schedule[8:24] == tuple(LayerExecution(layer, False) for layer in range(4, 20))
    assert schedule[24:40] == tuple(LayerExecution(layer, True) for layer in range(4, 20))
    assert schedule[40:56] == tuple(LayerExecution(layer, True) for layer in range(4, 20))
    assert schedule[56:] == tuple(LayerExecution(layer, False) for layer in range(20, 36))


def test_unconfigured_gaps_execute_once() -> None:
    schedule = build_looped_lora_schedule(
        8,
        [{"start_layer": 2, "end_layer": 4, "repeat_count": 2}],
    )

    assert schedule == (
        LayerExecution(0, False),
        LayerExecution(1, False),
        LayerExecution(2, False),
        LayerExecution(3, False),
        LayerExecution(2, True),
        LayerExecution(3, True),
        LayerExecution(4, False),
        LayerExecution(5, False),
        LayerExecution(6, False),
        LayerExecution(7, False),
    )


def test_lora_only_executions_get_distinct_logical_indices() -> None:
    schedule = build_looped_lora_schedule(
        6,
        [{"start_layer": 2, "end_layer": 4, "repeat_count": 3}],
    )

    assert get_lora_only_executions_by_physical_layer(6, schedule) == (
        (),
        (),
        (4, 6),
        (5, 7),
        (),
        (),
    )


@pytest.mark.parametrize(
    "sections",
    [
        [{"start_layer": 4, "end_layer": 4, "repeat_count": 2}],
        [{"start_layer": -1, "end_layer": 4, "repeat_count": 2}],
        [{"start_layer": 2, "end_layer": 4, "repeat_count": 0}],
        [{"start_layer": 2, "end_layer": 7, "repeat_count": 2}],
        [
            {"start_layer": 2, "end_layer": 5, "repeat_count": 2},
            {"start_layer": 4, "end_layer": 6, "repeat_count": 2},
        ],
    ],
)
def test_invalid_sections_fail_before_model_start(
    sections: list[dict[str, int]],
) -> None:
    with pytest.raises(ValueError):
        build_looped_lora_schedule(6, sections)
