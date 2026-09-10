from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from skyrl.backends.skyrl_train.workers.fsdp.adapter_store import (
    FSDPAdapterStore,
    FSDPLoraSignature,
)


class ToyLoRAModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(3, 2, bias=False)
        self.base.requires_grad_(False)
        self.lora_A = nn.Linear(3, 2, bias=False)
        self.lora_B = nn.Linear(2, 2, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + self.lora_B(self.lora_A(inputs))


def _signature(rank: int = 2) -> FSDPLoraSignature:
    return FSDPLoraSignature(rank=rank, alpha=4, target_modules=("all-linear",), world_size=1)


def _build_store():
    torch.manual_seed(7)
    model = ToyLoRAModel()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.05,
        betas=(0.8, 0.9),
        weight_decay=0.1,
    )
    store = FSDPAdapterStore()
    pristine = [parameter.detach().clone() for parameter in model.parameters() if parameter.requires_grad]
    store.prime_optimizer_state(model, optimizer)
    for expected, parameter in zip(pristine, (p for p in model.parameters() if p.requires_grad)):
        torch.testing.assert_close(parameter, expected)
    store.register_pristine(model, optimizer, _signature())
    store.create("adapter-a", model, optimizer, _signature())
    store.create("adapter-b", model, optimizer, _signature())
    return model, optimizer, store


def _step(model: ToyLoRAModel, optimizer: torch.optim.Optimizer, scale: float) -> None:
    inputs = torch.tensor([[1.0, -2.0, 0.5]]) * scale
    model(inputs).square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _optimizer_steps(model: nn.Module, optimizer: torch.optim.Optimizer) -> list[int]:
    return [int(optimizer.state[parameter]["step"].item()) for parameter in _trainable_parameters(model)]


def test_adapter_weights_and_adam_state_are_isolated():
    model, optimizer, store = _build_store()

    _step(model, optimizer, scale=1.0)
    adapter_a_weights = [parameter.detach().clone() for parameter in _trainable_parameters(model)]
    assert _optimizer_steps(model, optimizer) == [1, 1]

    store.swap_to("adapter-b", model, optimizer)
    assert _optimizer_steps(model, optimizer) == [0, 0]
    _step(model, optimizer, scale=2.0)
    _step(model, optimizer, scale=2.0)
    adapter_b_weights = [parameter.detach().clone() for parameter in _trainable_parameters(model)]
    assert _optimizer_steps(model, optimizer) == [2, 2]

    store.swap_to("adapter-a", model, optimizer)
    assert _optimizer_steps(model, optimizer) == [1, 1]
    for expected, parameter in zip(adapter_a_weights, _trainable_parameters(model)):
        torch.testing.assert_close(parameter, expected)

    store.swap_to("adapter-b", model, optimizer)
    assert _optimizer_steps(model, optimizer) == [2, 2]
    for expected, parameter in zip(adapter_b_weights, _trainable_parameters(model)):
        torch.testing.assert_close(parameter, expected)


def test_unconsumed_gradients_follow_their_adapter():
    model, optimizer, store = _build_store()
    parameters = _trainable_parameters(model)

    model(torch.ones(1, 3)).sum().backward()
    adapter_a_gradients = [parameter.grad.detach().clone() for parameter in parameters]

    store.swap_to("adapter-b", model, optimizer)
    assert all(parameter.grad is None for parameter in parameters)

    model(torch.full((1, 3), 2.0)).sum().backward()
    adapter_b_gradients = [parameter.grad.detach().clone() for parameter in parameters]

    store.swap_to("adapter-a", model, optimizer)
    for expected, parameter in zip(adapter_a_gradients, parameters):
        torch.testing.assert_close(parameter.grad, expected)

    store.swap_to("adapter-b", model, optimizer)
    for expected, parameter in zip(adapter_b_gradients, parameters):
        torch.testing.assert_close(parameter.grad, expected)


def test_registration_validates_signature_and_parameter_layout():
    model, optimizer, store = _build_store()

    with pytest.raises(ValueError, match="signature mismatch"):
        store.create("wrong-rank", model, optimizer, _signature(rank=4))

    model.base.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="non-adapter"):
        store.swap_to("adapter-b", model, optimizer)


def test_delete_releases_slot_and_clears_current_adapter():
    model, optimizer, store = _build_store()

    store.delete("adapter-a")
    assert store.current_id is None
    assert store.registered_ids() == ["adapter-b"]

    store.swap_to("adapter-b", model, optimizer)
    assert store.current_id == "adapter-b"
    with pytest.raises(KeyError, match="unknown adapter"):
        store.swap_to("adapter-a", model, optimizer)
