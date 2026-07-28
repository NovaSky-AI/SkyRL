from __future__ import annotations

import asyncio
import copy
import importlib.util
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from skyrl.backends.skyrl_train.workers.fsdp.multi_lora import (
    MultiLoRAExperts,
    MultiLoRALinear,
    MultiLoRAManager,
    inject_multi_lora,
    validate_concurrent_lora_model_support,
)


def _fsdp_policy_worker_class():
    if importlib.util.find_spec("flash_attn") is not None:
        from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import (
            FSDPPolicyWorkerBase,
        )

        return FSDPPolicyWorkerBase

    flash_attn = ModuleType("flash_attn")
    flash_attn.__spec__ = ModuleSpec("flash_attn", loader=None)
    bert_padding = ModuleType("flash_attn.bert_padding")
    bert_padding.__spec__ = ModuleSpec("flash_attn.bert_padding", loader=None)
    bert_padding.pad_input = MagicMock()
    bert_padding.unpad_input = MagicMock()
    find_spec = importlib.util.find_spec

    def find_spec_without_flash_attn(name, package=None):
        if name == "flash_attn":
            return None
        return find_spec(name, package)

    fake_modules = {
        "flash_attn": flash_attn,
        "flash_attn.bert_padding": bert_padding,
    }
    missing = object()
    previous_modules = {name: sys.modules.get(name, missing) for name in fake_modules}
    sys.modules.update(fake_modules)
    try:
        with patch("importlib.util.find_spec", side_effect=find_spec_without_flash_attn):
            from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import (
                FSDPPolicyWorkerBase,
            )
    finally:
        for name, previous in previous_modules.items():
            if previous is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    return FSDPPolicyWorkerBase


def _set_slot(layer: MultiLoRALinear, slot: int, a: torch.Tensor, b: torch.Tensor) -> None:
    with torch.no_grad():
        layer.adapters[slot].lora_A.weight.copy_(a)
        layer.adapters[slot].lora_B.weight.copy_(b)


@pytest.fixture
def fake_grouped_mm(monkeypatch):
    """Exercise routing and autograd on CPU while the CUDA kernel is unavailable."""
    calls = []

    def grouped_mm(inputs, weights, *, offs):
        calls.append((inputs, weights, offs))
        outputs = []
        start = 0
        for group_index, end in enumerate(offs.tolist()):
            outputs.append(inputs[start:end] @ weights[group_index])
            start = end
        return torch.cat(outputs)

    monkeypatch.setattr(F, "grouped_mm", grouped_mm)
    monkeypatch.setattr(MultiLoRALinear, "_validate_grouped_mm_inputs", lambda self, inputs: None)
    monkeypatch.setattr(
        MultiLoRAExperts,
        "_validate_grouped_mm_inputs",
        lambda self, inputs, output_features: None,
    )
    return calls


def _reference_lora_delta(
    layer: MultiLoRALinear,
    inputs: torch.Tensor,
    adapter_indices: torch.Tensor,
) -> torch.Tensor:
    delta = torch.zeros(*inputs.shape[:-1], layer.base_layer.out_features, device=inputs.device, dtype=inputs.dtype)
    for slot_index in adapter_indices.unique(sorted=True).tolist():
        rows = torch.nonzero(adapter_indices == slot_index, as_tuple=False).flatten()
        adapter = layer.adapters[slot_index]
        slot_delta = F.linear(F.linear(inputs.index_select(0, rows), adapter.lora_A.weight), adapter.lora_B.weight)
        delta.index_copy_(0, rows, slot_delta * layer.scaling)
    return delta


def test_lora_sampler_sync_barriers_before_inference_load(monkeypatch, tmp_path):
    FSDPPolicyWorkerBase = _fsdp_policy_worker_class()
    worker = object.__new__(FSDPPolicyWorkerBase)
    worker._is_concurrent_lora = True
    manager = SimpleNamespace(
        alpha=64,
        rank=32,
        slot_for=lambda name: 0,
        adapter_state_dict=lambda slot: {
            "base_model.model.proj.lora_A.weight": torch.ones(1, 1),
            "base_model.model.proj.lora_B.weight": torch.ones(1, 1),
        },
    )
    worker.model = SimpleNamespace(multi_lora_manager=manager)
    worker.cfg = SimpleNamespace(
        policy=SimpleNamespace(
            model=SimpleNamespace(
                path="test-model",
                lora=SimpleNamespace(dropout=0.0, target_modules="all-linear"),
            )
        )
    )
    events = []
    inference_client = SimpleNamespace(
        update_named_weights=AsyncMock(side_effect=lambda request: events.append("load"))
    )
    barrier = MagicMock(side_effect=lambda: events.append("barrier"))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "barrier", barrier)

    asyncio.run(
        worker._save_lora_adapters_and_sync(
            peft_model=None,
            lora_sync_path=str(tmp_path),
            inference_engine_client=inference_client,
            lora_name="adapter",
        )
    )

    inference_client.update_named_weights.assert_awaited_once()
    barrier.assert_called_once_with()
    assert events == ["barrier", "load"]


def test_lora_broadcast_uses_dispatch_fan_in_instead_of_worker_barrier(monkeypatch):
    FSDPPolicyWorkerBase = _fsdp_policy_worker_class()
    worker = object.__new__(FSDPPolicyWorkerBase)
    worker._is_lora = True
    worker._is_concurrent_lora = True
    worker.model = SimpleNamespace(model=object())
    worker.cfg = SimpleNamespace(fully_async=SimpleNamespace(enabled=False, clear_kv_cache_on_weight_sync=False))
    worker._resolve_lora_sync_target = MagicMock(return_value=("adapter", "/tmp/adapter"))
    worker._save_lora_adapters_and_sync = AsyncMock()
    barrier = MagicMock()
    monkeypatch.setattr(torch.cuda, "empty_cache", MagicMock())
    monkeypatch.setattr(torch.distributed, "barrier", barrier)

    asyncio.run(
        worker.broadcast_to_inference_engines(
            inference_engine_client=object(),
            inference_engine_cfg=SimpleNamespace(
                enable_prefix_caching=False,
                model_dtype="bfloat16",
            ),
            model_id="adapter",
        )
    )

    worker._save_lora_adapters_and_sync.assert_awaited_once()
    barrier.assert_not_called()


def test_multimodal_config_is_supported_when_only_the_language_model_is_loaded():
    validate_concurrent_lora_model_support(
        is_multimodal=True,
        language_model_only=True,
        sequence_parallel_size=1,
        remove_microbatch_padding=False,
    )

    with pytest.raises(NotImplementedError, match="language_model_only=True"):
        validate_concurrent_lora_model_support(
            is_multimodal=True,
            language_model_only=False,
            sequence_parallel_size=1,
            remove_microbatch_padding=False,
        )


def test_mixed_adapter_forward_matches_independent_lora_forwards(fake_grouped_mm):
    torch.manual_seed(0)
    base = nn.Linear(3, 2, bias=False)
    layer = MultiLoRALinear(copy.deepcopy(base), max_adapters=2, rank=1, alpha=1, dropout=0)
    _set_slot(layer, 0, torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[2.0], [3.0]]))
    _set_slot(layer, 1, torch.tensor([[0.0, 1.0, 0.0]]), torch.tensor([[5.0], [7.0]]))

    inputs = torch.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
    layer.set_adapter_indices(torch.tensor([0, 1]))
    actual = layer(inputs)

    expected = base(inputs)
    expected[0] += torch.tensor([[2.0, 3.0]])
    expected[1] += torch.tensor([[25.0, 35.0]])
    torch.testing.assert_close(actual, expected)


def test_flattened_batch_major_tokens_preserve_adapter_routing(fake_grouped_mm):
    torch.manual_seed(7)
    batched = MultiLoRALinear(nn.Linear(8, 8, bias=False), max_adapters=2, rank=4, alpha=4, dropout=0)
    _randomize_lora_weights(batched)
    flattened = copy.deepcopy(batched)
    adapter_indices = torch.tensor([1, 0])
    batched.set_adapter_indices(adapter_indices)
    flattened.set_adapter_indices(adapter_indices)
    inputs = torch.randn(2, 3, 8)

    expected = batched(inputs)
    actual = flattened(inputs.flatten(0, 1)).reshape_as(expected)

    torch.testing.assert_close(actual, expected)


def test_backward_only_produces_nonzero_gradients_for_routed_slots(fake_grouped_mm):
    layer = MultiLoRALinear(nn.Linear(3, 2, bias=False), max_adapters=3, rank=2, alpha=2, dropout=0)
    layer.set_adapter_indices(torch.tensor([0, 2]))
    layer(torch.randn(2, 4, 3)).sum().backward()

    assert layer.adapters[0].lora_A.weight.grad is not None
    assert layer.adapters[0].lora_B.weight.grad is not None
    assert layer.adapters[1].lora_A.weight.grad is None or not layer.adapters[1].lora_A.weight.grad.count_nonzero()
    assert layer.adapters[1].lora_B.weight.grad is None or not layer.adapters[1].lora_B.weight.grad.count_nonzero()
    assert layer.adapters[2].lora_A.weight.grad is not None
    assert layer.adapters[2].lora_B.weight.grad is not None


def test_sparse_routing_materializes_only_active_adapter_banks(monkeypatch, fake_grouped_mm):
    layer = MultiLoRALinear(nn.Linear(3, 2, bias=False), max_adapters=8, rank=2, alpha=2, dropout=0)
    recorded_slots = []
    original_weight_banks = layer._weight_banks

    def record_weight_banks(active_slots):
        recorded_slots.append(tuple(active_slots))
        return original_weight_banks(active_slots)

    monkeypatch.setattr(layer, "_weight_banks", record_weight_banks)
    layer.set_adapter_indices(torch.tensor([7, 2, 7]))
    layer(torch.randn(3, 4, 3)).sum().backward()

    assert recorded_slots == [(2, 7)]
    assert layer.adapters[0].lora_A.weight.grad is None
    assert layer.adapters[2].lora_A.weight.grad is not None
    assert layer.adapters[7].lora_A.weight.grad is not None


def _randomize_lora_weights(layer: MultiLoRALinear) -> None:
    with torch.no_grad():
        for adapter in layer.adapters:
            adapter.lora_A.weight.normal_()
            adapter.lora_B.weight.normal_()


def _grad_or_zeros(parameter: nn.Parameter) -> torch.Tensor:
    return torch.zeros_like(parameter) if parameter.grad is None else parameter.grad


class Qwen3MoeExperts(nn.Module):
    """Minimal Transformers-compatible fused expert bank for unit tests."""

    def __init__(self, num_experts=3, hidden_dim=4, intermediate_dim=4):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * intermediate_dim, hidden_dim))
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden_dim, intermediate_dim))
        self.act_fn = F.silu


def _reference_expert_output(
    layer: MultiLoRAExperts,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    adapter_indices: torch.Tensor,
) -> torch.Tensor:
    tokens_per_row = hidden_states.shape[0] // adapter_indices.shape[0]
    token_adapters = adapter_indices.repeat_interleave(tokens_per_row)
    output = torch.zeros_like(hidden_states)
    for token_index in range(hidden_states.shape[0]):
        slot = layer.adapters[int(token_adapters[token_index])]
        for top_k_position in range(top_k_index.shape[1]):
            expert_index = int(top_k_index[token_index, top_k_position])
            token = hidden_states[token_index]
            gate, up = F.linear(token, layer.gate_up_proj[expert_index]).chunk(2)
            for projection, value in (("gate_proj", gate), ("up_proj", up)):
                if projection in layer.projections:
                    value = (
                        value
                        + F.linear(
                            F.linear(token, slot.lora_A[projection][expert_index]),
                            slot.lora_B[projection][expert_index],
                        )
                        * layer.scaling
                    )
                if projection == "gate_proj":
                    gate = value
                else:
                    up = value
            intermediate = layer.act_fn(gate) * up
            down = F.linear(intermediate, layer.down_proj[expert_index])
            if "down_proj" in layer.projections:
                down = (
                    down
                    + F.linear(
                        F.linear(intermediate, slot.lora_A["down_proj"][expert_index]),
                        slot.lora_B["down_proj"][expert_index],
                    )
                    * layer.scaling
                )
            output[token_index] = output[token_index] + down * top_k_weights[token_index, top_k_position]
    return output


def test_grouped_gemm_routing_matches_reference_forward_and_backward(fake_grouped_mm):
    torch.manual_seed(13)
    grouped = MultiLoRALinear(nn.Linear(8, 8, bias=False), max_adapters=3, rank=4, alpha=4, dropout=0)
    _randomize_lora_weights(grouped)
    reference = copy.deepcopy(grouped)
    adapter_indices = torch.tensor([2, 0, 2, 1])
    grouped.set_adapter_indices(adapter_indices)

    grouped_inputs = torch.randn(4, 6, 8, requires_grad=True)
    reference_inputs = grouped_inputs.detach().clone().requires_grad_(True)
    grouped_output = grouped(grouped_inputs)
    reference_output = reference.base_layer(reference_inputs) + _reference_lora_delta(
        reference, reference_inputs, adapter_indices
    )
    torch.testing.assert_close(grouped_output, reference_output)
    assert len(fake_grouped_mm) == 2

    grouped_output.square().sum().backward()
    reference_output.square().sum().backward()
    torch.testing.assert_close(grouped_inputs.grad, reference_inputs.grad)
    for grouped_adapter, reference_adapter in zip(grouped.adapters, reference.adapters):
        for grouped_parameter, reference_parameter in zip(grouped_adapter.parameters(), reference_adapter.parameters()):
            torch.testing.assert_close(_grad_or_zeros(grouped_parameter), _grad_or_zeros(reference_parameter))


def test_grouped_gemm_pads_unaligned_output_width(fake_grouped_mm):
    torch.manual_seed(17)
    grouped = MultiLoRALinear(nn.Linear(8, 1, bias=False), max_adapters=2, rank=4, alpha=4, dropout=0)
    _randomize_lora_weights(grouped)
    reference = copy.deepcopy(grouped)
    adapter_indices = torch.tensor([1, 0, 1])
    grouped.set_adapter_indices(adapter_indices)

    grouped_inputs = torch.randn(3, 2, 8, requires_grad=True)
    reference_inputs = grouped_inputs.detach().clone().requires_grad_(True)
    grouped_output = grouped(grouped_inputs)
    reference_output = reference.base_layer(reference_inputs) + _reference_lora_delta(
        reference, reference_inputs, adapter_indices
    )
    torch.testing.assert_close(grouped_output, reference_output)
    assert fake_grouped_mm[1][1].shape == (2, 4, 4)

    grouped_output.square().sum().backward()
    reference_output.square().sum().backward()
    torch.testing.assert_close(grouped_inputs.grad, reference_inputs.grad)
    for grouped_adapter, reference_adapter in zip(grouped.adapters, reference.adapters):
        for grouped_parameter, reference_parameter in zip(grouped_adapter.parameters(), reference_adapter.parameters()):
            torch.testing.assert_close(_grad_or_zeros(grouped_parameter), _grad_or_zeros(reference_parameter))


def test_moe_grouped_gemm_routes_by_expert_and_adapter(fake_grouped_mm):
    torch.manual_seed(29)
    grouped = MultiLoRAExperts(
        Qwen3MoeExperts(),
        max_adapters=2,
        rank=2,
        alpha=4,
        dropout=0,
        projections=("gate_proj", "up_proj", "down_proj"),
    )
    for adapter in grouped.adapters:
        with torch.no_grad():
            for parameter in adapter.parameters():
                parameter.normal_()
    reference = copy.deepcopy(grouped)
    adapter_indices = torch.tensor([1, 0])
    grouped.set_adapter_indices(adapter_indices)

    top_k_index = torch.tensor([[2, 0], [1, 2], [0, 1], [2, 1]])
    top_k_weights = torch.softmax(torch.randn(4, 2), dim=-1)
    grouped_inputs = torch.randn(4, 4, requires_grad=True)
    reference_inputs = grouped_inputs.detach().clone().requires_grad_(True)

    grouped_output = grouped(grouped_inputs, top_k_index, top_k_weights)
    reference_output = _reference_expert_output(
        reference,
        reference_inputs,
        top_k_index,
        top_k_weights,
        adapter_indices,
    )
    torch.testing.assert_close(grouped_output, reference_output)

    output_grad = torch.randn_like(grouped_output)
    grouped_output.backward(output_grad)
    reference_output.backward(output_grad)
    torch.testing.assert_close(grouped_inputs.grad, reference_inputs.grad)
    for grouped_adapter, reference_adapter in zip(grouped.adapters, reference.adapters):
        for grouped_parameter, reference_parameter in zip(grouped_adapter.parameters(), reference_adapter.parameters()):
            torch.testing.assert_close(
                _grad_or_zeros(grouped_parameter),
                _grad_or_zeros(reference_parameter),
            )


def test_injection_wraps_fused_moe_experts_and_exports_peft_tensors():
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = Qwen3MoeExperts(num_experts=2)

        def get_output_embeddings(self):
            return None

    model = ToyModel()
    manager = inject_multi_lora(
        model,
        max_adapters=2,
        rank=2,
        alpha=4,
        dropout=0,
        target_modules=["gate_proj", "down_proj"],
    )
    slot = manager.register("adapter")

    assert isinstance(model.experts, MultiLoRAExperts)
    assert model.experts.projections == ("gate_proj", "down_proj")
    state = manager.adapter_state_dict(slot)
    assert set(state) == {
        f"base_model.model.experts.{expert_index}.{projection}.{matrix}.weight"
        for expert_index in range(2)
        for projection in ("gate_proj", "down_proj")
        for matrix in ("lora_A", "lora_B")
    }
    assert state["base_model.model.experts.0.gate_proj.lora_A.weight"].shape == (2, 4)
    assert state["base_model.model.experts.0.down_proj.lora_B.weight"].shape == (4, 2)


def test_transformers_qwen3_moe_expert_bank_matches_eager_forward(fake_grouped_mm):
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeExperts as HFQwen3MoeExperts,
    )

    config = Qwen3MoeConfig(
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=8,
        num_experts=3,
        num_experts_per_tok=2,
    )
    eager = HFQwen3MoeExperts(config)
    with torch.no_grad():
        eager.gate_up_proj.normal_(std=0.02)
        eager.down_proj.normal_(std=0.02)
    grouped = MultiLoRAExperts(
        copy.deepcopy(eager),
        max_adapters=2,
        rank=4,
        alpha=8,
        dropout=0,
        projections=("gate_proj", "up_proj", "down_proj"),
    )
    hidden_states = torch.randn(5, 8, requires_grad=True)
    reference_inputs = hidden_states.detach().clone().requires_grad_(True)
    top_k_index = torch.tensor([[2, 0], [1, 2], [0, 1], [2, 1], [0, 2]])
    top_k_weights = torch.softmax(torch.randn(5, 2), dim=-1)

    grouped_output = grouped(hidden_states, top_k_index, top_k_weights)
    eager_output = eager(reference_inputs, top_k_index, top_k_weights)
    torch.testing.assert_close(grouped_output, eager_output)

    output_grad = torch.randn_like(grouped_output)
    grouped_output.backward(output_grad)
    eager_output.backward(output_grad)
    torch.testing.assert_close(hidden_states.grad, reference_inputs.grad)


def test_grouped_gemm_requires_cuda_bfloat16_inputs():
    layer = MultiLoRALinear(nn.Linear(8, 8, bias=False), max_adapters=1, rank=4, alpha=4, dropout=0)

    with pytest.raises(RuntimeError, match="requires CUDA"):
        layer._apply_lora_grouped(torch.empty(1, 2, 8), torch.tensor([0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA grouped GEMM requires a GPU")
def test_cuda_grouped_gemm_matches_reference_forward_and_backward():
    torch.manual_seed(17)
    grouped = MultiLoRALinear(
        nn.Linear(32, 48, bias=False, device="cuda", dtype=torch.bfloat16),
        max_adapters=4,
        rank=8,
        alpha=16,
        dropout=0,
    )
    _randomize_lora_weights(grouped)
    reference = copy.deepcopy(grouped)
    adapter_indices = torch.tensor([3, 0, 3, 3, 0, 0], device="cuda")
    grouped_inputs = torch.randn(6, 16, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    reference_inputs = grouped_inputs.detach().clone().requires_grad_(True)

    grouped_output = grouped.base_layer(grouped_inputs) + grouped._apply_lora_grouped(grouped_inputs, adapter_indices)
    reference_output = reference.base_layer(reference_inputs) + _reference_lora_delta(
        reference, reference_inputs, adapter_indices
    )
    torch.testing.assert_close(grouped_output, reference_output, atol=2e-2, rtol=2e-2)

    output_grad = torch.randn_like(grouped_output)
    grouped_output.backward(output_grad)
    reference_output.backward(output_grad)
    torch.testing.assert_close(grouped_inputs.grad, reference_inputs.grad, atol=2e-2, rtol=2e-2)
    for grouped_adapter, reference_adapter in zip(grouped.adapters, reference.adapters):
        for grouped_parameter, reference_parameter in zip(grouped_adapter.parameters(), reference_adapter.parameters()):
            torch.testing.assert_close(
                _grad_or_zeros(grouped_parameter),
                _grad_or_zeros(reference_parameter),
                atol=2e-2,
                rtol=2e-2,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA grouped GEMM requires a GPU")
def test_cuda_moe_grouped_gemm_matches_reference_forward_and_backward():
    torch.manual_seed(31)
    experts = Qwen3MoeExperts(num_experts=4, hidden_dim=32, intermediate_dim=48).to(
        device="cuda",
        dtype=torch.bfloat16,
    )
    grouped = MultiLoRAExperts(
        experts,
        max_adapters=3,
        rank=8,
        alpha=16,
        dropout=0,
        projections=("gate_proj", "up_proj", "down_proj"),
    )
    with torch.no_grad():
        grouped.gate_up_proj.normal_(std=0.02)
        grouped.down_proj.normal_(std=0.02)
        for adapter in grouped.adapters:
            for parameter in adapter.parameters():
                parameter.normal_(std=0.02)
    reference = copy.deepcopy(grouped)
    adapter_indices = torch.tensor([2, 0, 2], device="cuda")
    grouped.set_adapter_indices(adapter_indices)

    top_k_index = torch.tensor(
        [[3, 0], [1, 3], [2, 1], [0, 2], [3, 1], [2, 0]],
        device="cuda",
    )
    top_k_weights = torch.softmax(
        torch.randn(6, 2, device="cuda", dtype=torch.bfloat16),
        dim=-1,
    )
    grouped_inputs = torch.randn(
        6,
        32,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    reference_inputs = grouped_inputs.detach().clone().requires_grad_(True)

    grouped_output = grouped(grouped_inputs, top_k_index, top_k_weights)
    reference_output = _reference_expert_output(
        reference,
        reference_inputs,
        top_k_index,
        top_k_weights,
        adapter_indices,
    )
    torch.testing.assert_close(grouped_output, reference_output, atol=5e-2, rtol=5e-2)

    output_grad = torch.randn_like(grouped_output)
    grouped_output.backward(output_grad)
    reference_output.backward(output_grad)
    torch.testing.assert_close(grouped_inputs.grad, reference_inputs.grad, atol=5e-2, rtol=5e-2)
    for grouped_adapter, reference_adapter in zip(grouped.adapters, reference.adapters):
        for grouped_parameter, reference_parameter in zip(grouped_adapter.parameters(), reference_adapter.parameters()):
            torch.testing.assert_close(
                _grad_or_zeros(grouped_parameter),
                _grad_or_zeros(reference_parameter),
                atol=5e-2,
                rtol=5e-2,
            )


def test_slot_scoped_adamw_preserves_other_weights_gradients_and_state(fake_grouped_mm):
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(3, 2, bias=False)

        def get_output_embeddings(self):
            return None

    manager = inject_multi_lora(
        ToyModel(),
        max_adapters=2,
        rank=2,
        alpha=2,
        dropout=0,
        target_modules=["proj"],
    )
    slot_a = manager.register("a")
    slot_b = manager.register("b")
    manager.set_adapter_indices(torch.tensor([slot_a, slot_b]))
    layer = manager.layers[0][1]
    layer(torch.randn(2, 3, 3)).square().sum().backward()

    params = list(layer.parameters())
    optimizer = torch.optim.AdamW(params, lr=0.1, weight_decay=0.2)
    b_weights_before = [p.detach().clone() for p in manager.parameters_for_slot(slot_b)]
    b_grads_before = [p.grad.detach().clone() for p in manager.parameters_for_slot(slot_b)]

    with manager.isolate_slot_gradients(slot_a):
        optimizer.step()
        optimizer.zero_grad()

    for before, parameter in zip(b_weights_before, manager.parameters_for_slot(slot_b)):
        torch.testing.assert_close(parameter, before)
        assert parameter not in optimizer.state
    for before, parameter in zip(b_grads_before, manager.parameters_for_slot(slot_b)):
        torch.testing.assert_close(parameter.grad, before)


def test_batched_adapter_optimizer_uses_slot_specific_parameter_groups(fake_grouped_mm):
    layer = MultiLoRALinear(nn.Linear(3, 2, bias=False), max_adapters=3, rank=2, alpha=2, dropout=0)
    manager = MultiLoRAManager([("proj", layer)], max_adapters=3, rank=2, alpha=2, target_modules=["proj"])
    slot_a = manager.register("a")
    slot_b = manager.register("b")
    manager.register("inactive")
    manager.set_optimizer_hparams(
        "a",
        {"learning_rate": 0.1, "beta1": 0.8, "beta2": 0.9, "eps": 1e-6, "weight_decay": 0.0},
    )
    manager.set_optimizer_hparams(
        "b",
        {"learning_rate": 0.01, "beta1": 0.7, "beta2": 0.95, "eps": 1e-5, "weight_decay": 0.2},
    )
    manager.set_adapter_indices(torch.tensor([slot_a, slot_b]))
    layer(torch.randn(2, 4, 3)).square().sum().backward()

    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-4)
    original_groups = optimizer.param_groups
    with manager.select_optimizer_slots(optimizer, ["a", "b"]):
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["lr"] == 0.1
        assert optimizer.param_groups[0]["betas"] == (0.8, 0.9)
        assert optimizer.param_groups[1]["lr"] == 0.01
        assert optimizer.param_groups[1]["weight_decay"] == 0.2
        optimizer.step()

    assert optimizer.param_groups is original_groups
    assert all(parameter in optimizer.state for parameter in manager.parameters_for_slot(slot_a))
    assert all(parameter in optimizer.state for parameter in manager.parameters_for_slot(slot_b))
    assert all(parameter not in optimizer.state for parameter in manager.parameters_for_slot(2))


def test_injection_freezes_base_and_exports_one_slot_in_peft_shape():
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.ModuleDict(
                {
                    "q_proj": nn.Linear(4, 4, bias=False),
                    "router": nn.Linear(4, 2, bias=False),
                }
            )
            self.lm_head = nn.Linear(4, 8, bias=False)

        def get_output_embeddings(self):
            return self.lm_head

    model = ToyModel()
    manager = inject_multi_lora(
        model,
        max_adapters=2,
        rank=2,
        alpha=4,
        dropout=0,
        target_modules="all-linear",
    )

    assert isinstance(model.block["q_proj"], MultiLoRALinear)
    assert isinstance(model.block["router"], nn.Linear)
    assert isinstance(model.lm_head, nn.Linear)
    assert not model.block["q_proj"].base_layer.weight.requires_grad
    assert model.block["q_proj"].adapters[0].lora_A.weight.requires_grad

    slot = manager.register("adapter")
    state = manager.adapter_state_dict(slot)
    assert set(state) == {
        "base_model.model.block.q_proj.lora_A.weight",
        "base_model.model.block.q_proj.lora_B.weight",
    }
    assert state["base_model.model.block.q_proj.lora_A.weight"].shape == (2, 4)
    assert state["base_model.model.block.q_proj.lora_B.weight"].shape == (4, 2)


def test_manager_rejects_invalid_routes_and_capacity_overflow():
    layer = MultiLoRALinear(nn.Linear(2, 2), max_adapters=1, rank=1, alpha=1, dropout=0)

    manager = MultiLoRAManager([("proj", layer)], max_adapters=1, rank=1, alpha=1, target_modules=["proj"])
    assert manager.register("a") == 0
    with pytest.raises(ValueError, match="Maximum number"):
        manager.register("b")
    with pytest.raises(ValueError, match="adapter index"):
        manager.set_adapter_indices(torch.tensor([1]))


def test_deleted_slot_is_reset_and_can_be_reused():
    layer = MultiLoRALinear(nn.Linear(2, 2), max_adapters=1, rank=1, alpha=1, dropout=0)

    manager = MultiLoRAManager([("proj", layer)], max_adapters=1, rank=1, alpha=1, target_modules=["proj"])
    slot = manager.register("a", seed=7)
    initial_a = layer.adapters[slot].lora_A.weight.detach().clone()
    with torch.no_grad():
        layer.adapters[slot].lora_A.weight.fill_(4)
        layer.adapters[slot].lora_B.weight.fill_(5)

    assert manager.delete("a") == slot
    torch.testing.assert_close(
        layer.adapters[slot].lora_A.weight,
        torch.zeros_like(layer.adapters[slot].lora_A.weight),
    )
    torch.testing.assert_close(
        layer.adapters[slot].lora_B.weight,
        torch.zeros_like(layer.adapters[slot].lora_B.weight),
    )
    assert manager.register("b", seed=7) == slot
    torch.testing.assert_close(layer.adapters[slot].lora_A.weight, initial_a)


def test_mixed_adapter_gradients_match_separate_forwards(fake_grouped_mm):
    layer = MultiLoRALinear(nn.Linear(3, 2, bias=False), max_adapters=2, rank=2, alpha=2, dropout=0)
    inputs = torch.randn(2, 4, 3)
    weights = torch.randn(2, 4, 2)

    layer.set_adapter_indices(torch.tensor([0, 1]))
    (layer(inputs) * weights).sum().backward()
    mixed_grads = {
        (slot, name): parameter.grad.detach().clone()
        for slot, adapter in enumerate(layer.adapters)
        for name, parameter in adapter.named_parameters()
    }
    layer.zero_grad(set_to_none=True)

    for row, slot in enumerate((0, 1)):
        layer.set_adapter_indices(torch.tensor([slot]))
        (layer(inputs[row : row + 1]) * weights[row : row + 1]).sum().backward()

    for slot, adapter in enumerate(layer.adapters):
        for name, parameter in adapter.named_parameters():
            torch.testing.assert_close(parameter.grad, mixed_grads[(slot, name)])


def test_training_checkpoint_is_slot_independent_and_restores_adam_state(fake_grouped_mm):
    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(3, 2, bias=False)

        def get_output_embeddings(self):
            return None

        def forward(self, inputs):
            return self.proj(inputs)

    torch.manual_seed(4)
    pristine_model = ToyModel()
    source_model = copy.deepcopy(pristine_model)
    source_manager = inject_multi_lora(
        source_model,
        max_adapters=2,
        rank=2,
        alpha=2,
        dropout=0,
        target_modules=["proj"],
    )
    source_manager.register("filler", seed=3)
    source_slot = source_manager.register("source", seed=7)
    assert source_slot == 1
    source_hparams = {
        "learning_rate": 0.03,
        "beta1": 0.8,
        "beta2": 0.9,
        "eps": 1e-6,
        "weight_decay": 0.1,
    }
    source_manager.set_optimizer_hparams("source", source_hparams)
    source_optimizer = torch.optim.AdamW(
        source_model.parameters(),
        lr=source_hparams["learning_rate"],
        betas=(source_hparams["beta1"], source_hparams["beta2"]),
        eps=source_hparams["eps"],
        weight_decay=source_hparams["weight_decay"],
    )

    inputs = torch.randn(1, 5, 3)
    source_manager.set_adapter_indices(torch.tensor([source_slot]))
    source_model(inputs).square().sum().backward()
    with source_manager.isolate_slot_gradients(source_slot):
        source_optimizer.step()
        source_optimizer.zero_grad()
    checkpoint = source_manager.training_state("source", source_optimizer)

    target_model = copy.deepcopy(pristine_model)
    target_manager = inject_multi_lora(
        target_model,
        max_adapters=2,
        rank=2,
        alpha=2,
        dropout=0,
        target_modules=["proj"],
    )
    target_slot = target_manager.register("target", seed=99)
    other_slot = target_manager.register("other", seed=101)
    assert target_slot == 0
    other_weights_before = [parameter.detach().clone() for parameter in target_manager.parameters_for_slot(other_slot)]
    for parameter in target_manager.parameters_for_slot(target_slot):
        parameter.grad = torch.ones_like(parameter)
    for parameter in target_manager.parameters_for_slot(other_slot):
        parameter.grad = torch.ones_like(parameter)
    target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=1e-4)

    target_manager.load_training_state("target", checkpoint, target_optimizer)

    assert target_manager.optimizer_hparams_for("target") == source_hparams
    assert all(parameter.grad is None for parameter in target_manager.parameters_for_slot(target_slot))
    assert all(parameter.grad is not None for parameter in target_manager.parameters_for_slot(other_slot))
    for source_parameter, target_parameter in zip(
        source_manager.parameters_for_slot(source_slot),
        target_manager.parameters_for_slot(target_slot),
    ):
        torch.testing.assert_close(target_parameter, source_parameter)
        source_state = source_optimizer.state[source_parameter]
        target_state = target_optimizer.state[target_parameter]
        assert source_state.keys() == target_state.keys()
        for key in source_state:
            if isinstance(source_state[key], torch.Tensor):
                torch.testing.assert_close(target_state[key], source_state[key])
            else:
                assert target_state[key] == source_state[key]
    for before, parameter in zip(other_weights_before, target_manager.parameters_for_slot(other_slot)):
        torch.testing.assert_close(parameter, before)

    for optimizer in (source_optimizer, target_optimizer):
        for group in optimizer.param_groups:
            group["lr"] = source_hparams["learning_rate"]
            group["betas"] = (source_hparams["beta1"], source_hparams["beta2"])
            group["eps"] = source_hparams["eps"]
            group["weight_decay"] = source_hparams["weight_decay"]
    source_manager.set_adapter_indices(torch.tensor([source_slot]))
    target_manager.set_adapter_indices(torch.tensor([target_slot]))
    source_model(inputs).square().sum().backward()
    target_model(inputs).square().sum().backward()
    with source_manager.isolate_slot_gradients(source_slot):
        source_optimizer.step()
    with target_manager.isolate_slot_gradients(target_slot):
        target_optimizer.step()

    for source_parameter, target_parameter in zip(
        source_manager.parameters_for_slot(source_slot),
        target_manager.parameters_for_slot(target_slot),
    ):
        torch.testing.assert_close(target_parameter, source_parameter)


def test_training_checkpoint_rejects_signature_mismatch():
    layer = MultiLoRALinear(nn.Linear(2, 2), max_adapters=1, rank=1, alpha=1, dropout=0)
    manager = MultiLoRAManager([("proj", layer)], max_adapters=1, rank=1, alpha=1, target_modules=["proj"])
    manager.register("adapter")
    optimizer = torch.optim.AdamW(layer.parameters())
    state = manager.training_state("adapter", optimizer)
    state["signature"]["rank"] = 2

    with pytest.raises(ValueError, match="signature mismatch"):
        manager.load_training_state("adapter", state, optimizer)
