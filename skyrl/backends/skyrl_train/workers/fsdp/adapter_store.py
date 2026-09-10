"""CPU-backed adapter storage for the FSDP LoRA worker.

The FSDP policy keeps one PEFT adapter live on GPU. Each registered Tinker
model owns a pinned-CPU snapshot of that adapter's local parameter shards,
gradients, and Adam state. Switching adapters copies only local shards; it
does not gather parameters or issue FSDP collectives.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.distributed as dist

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor


@dataclass(frozen=True)
class FSDPLoraSignature:
    """LoRA and sharding properties that must match across adapter slots."""

    rank: int
    alpha: int
    target_modules: tuple[str, ...]
    world_size: int

    @classmethod
    def from_lora_config(cls, lora_config) -> "FSDPLoraSignature":
        targets = lora_config.target_modules
        target_modules = (targets,) if isinstance(targets, str) else tuple(targets)
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        return cls(
            rank=int(lora_config.rank),
            alpha=int(lora_config.alpha),
            target_modules=target_modules,
            world_size=world_size,
        )


@dataclass
class AdapterSlot:
    """Pinned-CPU mirrors of one worker's local LoRA and optimizer shards."""

    parameter_data: list[torch.Tensor] = field(default_factory=list)
    gradient_data: list[Optional[torch.Tensor]] = field(default_factory=list)
    optimizer_state: list[dict[str, Any]] = field(default_factory=list)


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _cpu_copy(tensor: torch.Tensor) -> torch.Tensor:
    local = _local_tensor(tensor).detach()
    pin_memory = local.device.type == "cuda" or local.is_pinned()
    result = torch.empty_like(local, device="cpu", pin_memory=pin_memory)
    result.copy_(local, non_blocking=pin_memory)
    return result


def _copy_to_live(target: torch.Tensor, source: torch.Tensor) -> None:
    local = _local_tensor(target)
    local.copy_(source, non_blocking=local.device.type == "cuda")


def _synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.current_stream().synchronize()


class FSDPAdapterStore:
    """Registry of CPU adapter slots with exactly one live GPU adapter."""

    def __init__(self) -> None:
        self._slots: dict[str, AdapterSlot] = {}
        self._pristine: Optional[AdapterSlot] = None
        self._current_id: Optional[str] = None
        self._signature: Optional[FSDPLoraSignature] = None
        self._parameter_names: tuple[str, ...] = ()

    @property
    def current_id(self) -> Optional[str]:
        return self._current_id

    @property
    def signature(self) -> Optional[FSDPLoraSignature]:
        return self._signature

    def registered_ids(self) -> list[str]:
        return list(self._slots)

    def num_adapters(self) -> int:
        return len(self._slots)

    @staticmethod
    def _trainable_parameters(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
        parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("FSDPAdapterStore found no trainable LoRA parameters")
        unexpected = [name for name, _ in parameters if "lora_" not in name and "adapter" not in name]
        if unexpected:
            raise RuntimeError(
                "FSDPAdapterStore only supports LoRA parameters, but found trainable non-adapter parameters: "
                + ", ".join(unexpected)
            )
        return parameters

    def _checked_parameters(self, model: torch.nn.Module) -> list[torch.nn.Parameter]:
        named_parameters = self._trainable_parameters(model)
        names = tuple(name for name, _ in named_parameters)
        if self._parameter_names and names != self._parameter_names:
            raise RuntimeError(
                "FSDPAdapterStore trainable parameter layout changed after registration: "
                f"expected {self._parameter_names}, got {names}"
            )
        return [parameter for _, parameter in named_parameters]

    @staticmethod
    def _snapshot_value(value: Any) -> Any:
        return _cpu_copy(value) if isinstance(value, torch.Tensor) else copy.deepcopy(value)

    @staticmethod
    def _clone_slot(slot: AdapterSlot) -> AdapterSlot:
        return AdapterSlot(
            parameter_data=[_cpu_copy(value) for value in slot.parameter_data],
            gradient_data=[None if value is None else _cpu_copy(value) for value in slot.gradient_data],
            optimizer_state=[
                {
                    key: _cpu_copy(value) if isinstance(value, torch.Tensor) else copy.deepcopy(value)
                    for key, value in state.items()
                }
                for state in slot.optimizer_state
            ],
        )

    def _new_snapshot(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> AdapterSlot:
        parameters = self._checked_parameters(model)
        slot = AdapterSlot(
            parameter_data=[_cpu_copy(parameter) for parameter in parameters],
            gradient_data=[None if parameter.grad is None else _cpu_copy(parameter.grad) for parameter in parameters],
            optimizer_state=[
                {key: self._snapshot_value(value) for key, value in optimizer.state.get(parameter, {}).items()}
                for parameter in parameters
            ],
        )
        _synchronize_cuda()
        return slot

    def _snapshot_into(
        self,
        slot: AdapterSlot,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        parameters = self._checked_parameters(model)
        for index, parameter in enumerate(parameters):
            slot.parameter_data[index].copy_(_local_tensor(parameter), non_blocking=parameter.device.type == "cuda")

            gradient = parameter.grad
            if gradient is None:
                slot.gradient_data[index] = None
            elif slot.gradient_data[index] is None:
                slot.gradient_data[index] = _cpu_copy(gradient)
            else:
                local_gradient = _local_tensor(gradient)
                slot.gradient_data[index].copy_(
                    local_gradient,
                    non_blocking=local_gradient.device.type == "cuda",
                )

            live_state = optimizer.state.get(parameter, {})
            saved_state = slot.optimizer_state[index]
            for stale_key in saved_state.keys() - live_state.keys():
                del saved_state[stale_key]
            for key, value in live_state.items():
                saved_value = saved_state.get(key)
                if isinstance(value, torch.Tensor):
                    local_value = _local_tensor(value)
                    if not isinstance(saved_value, torch.Tensor) or saved_value.shape != local_value.shape:
                        saved_state[key] = _cpu_copy(value)
                    else:
                        saved_value.copy_(local_value, non_blocking=local_value.device.type == "cuda")
                else:
                    saved_state[key] = copy.deepcopy(value)
        _synchronize_cuda()

    def _restore(
        self,
        slot: AdapterSlot,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        parameters = self._checked_parameters(model)
        for index, parameter in enumerate(parameters):
            _copy_to_live(parameter, slot.parameter_data[index])

            saved_gradient = slot.gradient_data[index]
            if saved_gradient is None:
                parameter.grad = None
            else:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                _copy_to_live(parameter.grad, saved_gradient)

            live_state = optimizer.state[parameter]
            saved_state = slot.optimizer_state[index]
            for stale_key in live_state.keys() - saved_state.keys():
                del live_state[stale_key]
            for key, saved_value in saved_state.items():
                live_value = live_state.get(key)
                if isinstance(saved_value, torch.Tensor):
                    if not isinstance(live_value, torch.Tensor):
                        raise RuntimeError(f"Optimizer state '{key}' was not materialized before adapter registration")
                    _copy_to_live(live_value, saved_value)
                else:
                    live_state[key] = copy.deepcopy(saved_value)
        _synchronize_cuda()

    @torch.no_grad()
    def prime_optimizer_state(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        """Materialize Adam state without changing the pristine adapter."""

        parameters = self._checked_parameters(model)
        saved_parameters = [_cpu_copy(parameter) for parameter in parameters]
        saved_gradients = [None if parameter.grad is None else _cpu_copy(parameter.grad) for parameter in parameters]
        _synchronize_cuda()

        for parameter in parameters:
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()

        for parameter, saved_parameter in zip(parameters, saved_parameters):
            _copy_to_live(parameter, saved_parameter)
        optimizer.zero_grad(set_to_none=True)

        for parameter in parameters:
            for key, value in optimizer.state[parameter].items():
                if isinstance(value, torch.Tensor):
                    value.zero_()
                elif isinstance(value, (int, float)):
                    optimizer.state[parameter][key] = type(value)(0)

        for parameter, saved_gradient in zip(parameters, saved_gradients):
            if saved_gradient is None:
                parameter.grad = None
            else:
                parameter.grad = torch.zeros_like(parameter)
                _copy_to_live(parameter.grad, saved_gradient)
        _synchronize_cuda()

    def register_pristine(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        signature: FSDPLoraSignature,
    ) -> None:
        if self._pristine is not None:
            raise RuntimeError("FSDPAdapterStore.register_pristine called twice")
        named_parameters = self._trainable_parameters(model)
        self._parameter_names = tuple(name for name, _ in named_parameters)
        self._signature = signature
        self._pristine = self._new_snapshot(model, optimizer)

    def create(
        self,
        model_id: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        signature: FSDPLoraSignature,
    ) -> None:
        if self._pristine is None or self._signature is None:
            raise RuntimeError("FSDPAdapterStore.create called before register_pristine")
        if signature != self._signature:
            raise ValueError(
                f"FSDPAdapterStore LoRA signature mismatch for '{model_id}': "
                f"pristine={self._signature}, requested={signature}"
            )
        if model_id in self._slots:
            raise ValueError(f"FSDPAdapterStore adapter '{model_id}' is already registered")

        self._checked_parameters(model)
        self._slots[model_id] = self._clone_slot(self._pristine)
        if self._current_id is None:
            self._current_id = model_id

    def delete(self, model_id: str) -> None:
        if model_id not in self._slots:
            raise KeyError(f"FSDPAdapterStore unknown adapter '{model_id}'")
        del self._slots[model_id]
        if self._current_id == model_id:
            self._current_id = None

    @torch.no_grad()
    def swap_to(
        self,
        model_id: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if model_id not in self._slots:
            raise KeyError(f"FSDPAdapterStore unknown adapter '{model_id}'")
        if self._current_id == model_id:
            return

        distributed = dist.is_available() and dist.is_initialized()
        if distributed:
            dist.barrier()

        if self._current_id is not None:
            self._snapshot_into(self._slots[self._current_id], model, optimizer)
        self._restore(self._slots[model_id], model, optimizer)
        self._current_id = model_id

        if distributed:
            dist.barrier()
