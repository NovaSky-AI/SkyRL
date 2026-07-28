"""Resident multi-adapter LoRA layers for the FSDP2 policy path.

The model owns a fixed number of adapter slots before FSDP wrapping. A routing
tensor selects a slot per batch row, allowing one forward/backward pass to
accumulate gradients for multiple independent adapters.
"""

from __future__ import annotations

import math
import re
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def validate_concurrent_lora_model_support(
    *,
    is_multimodal: bool,
    language_model_only: bool,
    sequence_parallel_size: int,
    remove_microbatch_padding: bool,
) -> None:
    if is_multimodal and not language_model_only:
        raise NotImplementedError(
            "Concurrent FSDP LoRA currently supports multimodal models only with " "policy.language_model_only=True"
        )
    if sequence_parallel_size != 1:
        raise NotImplementedError("Concurrent FSDP LoRA currently requires sequence_parallel_size=1")
    if remove_microbatch_padding:
        raise NotImplementedError("Concurrent FSDP LoRA currently requires remove_microbatch_padding=False")


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def _shard_seed_offset(tensor: torch.Tensor) -> int:
    """Return an offset shared by replicas but distinct for each shard."""
    if not hasattr(tensor, "device_mesh") or not hasattr(tensor, "placements"):
        return 0

    from torch.distributed.tensor import Shard

    coordinate = tensor.device_mesh.get_coordinate()
    if coordinate is None:
        return 0
    offset = 0
    for mesh_coordinate, placement in zip(coordinate, tensor.placements):
        if isinstance(placement, Shard):
            offset = offset * 1_000_003 + mesh_coordinate + 1
    return offset


def _full_tensor_cpu(tensor: torch.Tensor) -> torch.Tensor:
    full_tensor = tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor
    return full_tensor.detach().cpu()


@torch.no_grad()
def _copy_from_full_tensor(target: torch.Tensor, source: torch.Tensor) -> None:
    local_target = _local_tensor(target)
    source = source.to(device=local_target.device, dtype=local_target.dtype)
    if hasattr(target, "device_mesh") and hasattr(target, "placements"):
        from torch.distributed.tensor import distribute_tensor

        distributed = distribute_tensor(source, target.device_mesh, target.placements)
        local_target.copy_(distributed.to_local())
    else:
        local_target.copy_(source)


class MultiLoRAAdapterSlot(nn.Module):
    """One statically allocated LoRA adapter."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dropout: float,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.lora_B = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, seed: Optional[int] = None) -> None:
        if self.lora_A.weight.is_meta:
            return
        local_a = _local_tensor(self.lora_A.weight)
        local_b = _local_tensor(self.lora_B.weight)
        if seed is None:
            nn.init.kaiming_uniform_(local_a, a=math.sqrt(5))
        else:
            devices = [local_a.device] if local_a.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed + _shard_seed_offset(self.lora_A.weight))
                nn.init.kaiming_uniform_(local_a, a=math.sqrt(5))
        nn.init.zeros_(local_b)

    @torch.no_grad()
    def clear(self) -> None:
        if self.lora_A.weight.is_meta:
            return
        _local_tensor(self.lora_A.weight).zero_()
        _local_tensor(self.lora_B.weight).zero_()
        self.lora_A.weight.grad = None
        self.lora_B.weight.grad = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(self.dropout(inputs))) * self.scaling


class MultiLoRALinear(nn.Module):
    """Wrap a frozen linear layer with statically allocated adapter slots."""

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        max_adapters: int,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if max_adapters <= 0:
            raise ValueError("max_adapters must be positive")
        if rank <= 0:
            raise ValueError("rank must be positive")

        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)
        self.rank = rank
        self.scaling = float(alpha) / rank
        self.dropout_p = float(dropout)
        self.adapters = nn.ModuleList(
            [
                MultiLoRAAdapterSlot(
                    base_layer.in_features,
                    base_layer.out_features,
                    rank,
                    alpha,
                    dropout,
                    device=base_layer.weight.device,
                    dtype=base_layer.weight.dtype,
                )
                for _ in range(max_adapters)
            ]
        )
        self.adapter_indices: Optional[torch.Tensor] = None
        self.active_adapter_slots: Optional[tuple[int, ...]] = None
        self.compact_adapter_indices: Optional[torch.Tensor] = None

    @staticmethod
    def compact_routing(adapter_indices: torch.Tensor) -> tuple[tuple[int, ...], torch.Tensor]:
        active_slots, compact_indices = torch.unique(adapter_indices, sorted=True, return_inverse=True)
        return tuple(int(slot) for slot in active_slots.tolist()), compact_indices

    def set_adapter_indices(
        self,
        adapter_indices: Optional[torch.Tensor],
        *,
        active_slots: Optional[tuple[int, ...]] = None,
        compact_indices: Optional[torch.Tensor] = None,
    ) -> None:
        if (active_slots is None) != (compact_indices is None):
            raise ValueError("active_slots and compact_indices must be provided together")
        self.adapter_indices = adapter_indices
        self.active_adapter_slots = active_slots
        self.compact_adapter_indices = compact_indices

    def _weight_banks(self, active_slots: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack only active slot parameters without changing optimizer ownership."""
        lora_a = torch.stack([self.adapters[slot].lora_A.weight for slot in active_slots])
        lora_b = torch.stack([self.adapters[slot].lora_B.weight for slot in active_slots])
        return lora_a, lora_b

    def _resolve_compact_routing(
        self,
        adapter_indices: torch.Tensor,
        active_slots: Optional[tuple[int, ...]],
    ) -> tuple[tuple[int, ...], torch.Tensor]:
        if active_slots is None:
            return self.compact_routing(adapter_indices)
        return active_slots, adapter_indices

    def _expanded_adapter_indices(self, inputs: torch.Tensor, adapter_indices: torch.Tensor) -> torch.Tensor:
        tokens_per_row = inputs.numel() // (inputs.shape[0] * inputs.shape[-1])
        return adapter_indices.repeat_interleave(tokens_per_row)

    def _validate_grouped_mm_inputs(self, inputs: torch.Tensor) -> None:
        if not hasattr(F, "grouped_mm"):
            raise RuntimeError("Concurrent FSDP LoRA requires torch.nn.functional.grouped_mm")
        if not inputs.is_cuda:
            raise RuntimeError("Concurrent FSDP LoRA grouped GEMM requires CUDA inputs")
        if inputs.dtype != torch.bfloat16:
            raise RuntimeError("Concurrent FSDP LoRA grouped GEMM requires bfloat16 inputs")
        if torch.cuda.get_device_capability(inputs.device)[0] < 8:
            raise RuntimeError("Concurrent FSDP LoRA grouped GEMM requires CUDA compute capability 8.0 or newer")

        # The CUDA grouped-GEMM kernels require row strides aligned to 16 bytes.
        alignment = 16 // inputs.element_size()
        dimensions = (inputs.shape[-1], self.rank)
        if any(dimension % alignment for dimension in dimensions):
            raise ValueError(
                "Concurrent FSDP LoRA grouped GEMM requires input and rank "
                f"dimensions divisible by {alignment}; got {dimensions}"
            )

    def _apply_lora_grouped(
        self,
        inputs: torch.Tensor,
        adapter_indices: torch.Tensor,
        active_slots: Optional[tuple[int, ...]] = None,
    ) -> torch.Tensor:
        """Sort tokens by adapter and execute two ragged grouped GEMMs."""
        self._validate_grouped_mm_inputs(inputs)
        active_slots, compact_indices = self._resolve_compact_routing(adapter_indices, active_slots)
        lora_a, lora_b = self._weight_banks(active_slots)
        flat_inputs = inputs.reshape(-1, inputs.shape[-1])
        expanded_indices = self._expanded_adapter_indices(inputs, compact_indices)
        sorted_indices, sort_order = torch.sort(expanded_indices, stable=True)
        sorted_inputs = flat_inputs.index_select(0, sort_order)
        group_sizes = torch.bincount(sorted_indices, minlength=len(active_slots))
        group_offsets = torch.cumsum(group_sizes, dim=0, dtype=torch.int32)

        intermediate = F.grouped_mm(sorted_inputs, lora_a.transpose(1, 2), offs=group_offsets)
        output_features = self.base_layer.out_features
        alignment = 16 // inputs.element_size()
        output_padding = (-output_features) % alignment
        if output_padding:
            lora_b = F.pad(lora_b, (0, 0, 0, output_padding))
        sorted_delta = F.grouped_mm(intermediate, lora_b.transpose(1, 2), offs=group_offsets)
        if output_padding:
            sorted_delta = sorted_delta[:, :output_features]
        flat_delta = torch.zeros_like(sorted_delta).index_copy(0, sort_order, sorted_delta)
        return flat_delta.reshape(*inputs.shape[:-1], output_features) * self.scaling

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.base_layer(inputs)
        adapter_indices = self.adapter_indices
        if adapter_indices is None:
            return output
        if inputs.ndim < 2:
            raise ValueError(f"MultiLoRALinear expects at least 2-D inputs, got shape={tuple(inputs.shape)}")
        if adapter_indices.numel() == 0:
            return output

        active_slots = self.active_adapter_slots
        compact_indices = self.compact_adapter_indices
        if active_slots is None or compact_indices is None:
            active_slots, compact_indices = self.compact_routing(adapter_indices)
        if inputs.shape[0] != adapter_indices.shape[0]:
            if inputs.ndim != 2 or inputs.shape[0] % adapter_indices.shape[0] != 0:
                raise ValueError(
                    "MultiLoRA routing expects a batch-first tensor or a flattened "
                    "batch-major token tensor: "
                    f"input shape={tuple(inputs.shape)}, adapter_indices={adapter_indices.shape[0]}"
                )
            compact_indices = compact_indices.repeat_interleave(inputs.shape[0] // adapter_indices.shape[0])

        lora_inputs = F.dropout(inputs, p=self.dropout_p, training=self.training)
        delta = self._apply_lora_grouped(lora_inputs, compact_indices, active_slots)
        return output + delta

    def named_adapter_parameters(self, slot_index: int) -> list[tuple[str, nn.Parameter]]:
        slot = self.adapters[slot_index]
        return [
            ("lora_A.weight", slot.lora_A.weight),
            ("lora_B.weight", slot.lora_B.weight),
        ]

    def export_adapter_state(self, slot_index: int) -> dict[str, torch.Tensor]:
        return {name: _full_tensor_cpu(parameter) for name, parameter in self.named_adapter_parameters(slot_index)}


_EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


class MultiLoRAExpertAdapterSlot(nn.Module):
    """One adapter's LoRA weights for a fused bank of routed experts."""

    def __init__(
        self,
        *,
        num_experts: int,
        hidden_features: int,
        intermediate_features: int,
        rank: int,
        projections: Sequence[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.input_features = {
            "gate_proj": hidden_features,
            "up_proj": hidden_features,
            "down_proj": intermediate_features,
        }
        self.output_features = {
            "gate_proj": intermediate_features,
            "up_proj": intermediate_features,
            "down_proj": hidden_features,
        }
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        for projection in projections:
            self.lora_A[projection] = nn.Parameter(
                torch.empty(
                    num_experts,
                    rank,
                    self.input_features[projection],
                    device=device,
                    dtype=dtype,
                )
            )
            self.lora_B[projection] = nn.Parameter(
                torch.empty(
                    num_experts,
                    self.output_features[projection],
                    rank,
                    device=device,
                    dtype=dtype,
                )
            )
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, seed: Optional[int] = None) -> None:
        for projection_offset, projection in enumerate(self.lora_A):
            lora_a = self.lora_A[projection]
            lora_b = self.lora_B[projection]
            if lora_a.is_meta:
                continue
            local_a = _local_tensor(lora_a)
            local_b = _local_tensor(lora_b)
            bound = 1 / math.sqrt(self.input_features[projection])
            if seed is None:
                nn.init.uniform_(local_a, -bound, bound)
            else:
                devices = [local_a.device] if local_a.is_cuda else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(seed + projection_offset * 1_000_003 + _shard_seed_offset(lora_a))
                    nn.init.uniform_(local_a, -bound, bound)
            nn.init.zeros_(local_b)

    @torch.no_grad()
    def clear(self) -> None:
        for parameter in self.parameters():
            if not parameter.is_meta:
                _local_tensor(parameter).zero_()
            parameter.grad = None

    def named_lora_parameters(self) -> list[tuple[str, nn.Parameter]]:
        parameters = []
        for projection in self.lora_A:
            parameters.extend(
                (
                    (f"{projection}.lora_A.weight", self.lora_A[projection]),
                    (f"{projection}.lora_B.weight", self.lora_B[projection]),
                )
            )
        return parameters


class MultiLoRAExperts(nn.Module):
    """Fused routed experts with resident multi-adapter LoRA banks."""

    def __init__(
        self,
        base_layer: nn.Module,
        *,
        max_adapters: int,
        rank: int,
        alpha: float,
        dropout: float,
        projections: Sequence[str],
    ) -> None:
        super().__init__()
        if not projections:
            raise ValueError("At least one expert projection must be selected")

        self.num_experts = int(base_layer.num_experts)
        self.hidden_dim = int(base_layer.hidden_dim)
        self.intermediate_dim = int(base_layer.intermediate_dim)
        self.gate_up_proj = base_layer.gate_up_proj
        self.down_proj = base_layer.down_proj
        self.gate_up_proj.requires_grad_(False)
        self.down_proj.requires_grad_(False)
        self.act_fn = base_layer.act_fn
        self.rank = rank
        self.scaling = float(alpha) / rank
        self.dropout_p = float(dropout)
        self.projections = tuple(projections)
        self.adapters = nn.ModuleList(
            [
                MultiLoRAExpertAdapterSlot(
                    num_experts=self.num_experts,
                    hidden_features=self.hidden_dim,
                    intermediate_features=self.intermediate_dim,
                    rank=rank,
                    projections=self.projections,
                    device=self.gate_up_proj.device,
                    dtype=self.gate_up_proj.dtype,
                )
                for _ in range(max_adapters)
            ]
        )
        self.adapter_indices: Optional[torch.Tensor] = None
        self.active_adapter_slots: Optional[tuple[int, ...]] = None
        self.compact_adapter_indices: Optional[torch.Tensor] = None

    def set_adapter_indices(
        self,
        adapter_indices: Optional[torch.Tensor],
        *,
        active_slots: Optional[tuple[int, ...]] = None,
        compact_indices: Optional[torch.Tensor] = None,
    ) -> None:
        if (active_slots is None) != (compact_indices is None):
            raise ValueError("active_slots and compact_indices must be provided together")
        self.adapter_indices = adapter_indices
        self.active_adapter_slots = active_slots
        self.compact_adapter_indices = compact_indices

    def _validate_grouped_mm_inputs(self, inputs: torch.Tensor, output_features: int) -> None:
        if not hasattr(F, "grouped_mm"):
            raise RuntimeError("Concurrent FSDP LoRA requires torch.nn.functional.grouped_mm")
        if not inputs.is_cuda:
            raise RuntimeError("Concurrent FSDP LoRA grouped GEMM requires CUDA inputs")
        if inputs.dtype != torch.bfloat16:
            raise RuntimeError("Concurrent FSDP LoRA grouped GEMM requires bfloat16 inputs")
        if torch.cuda.get_device_capability(inputs.device)[0] < 8:
            raise RuntimeError("Concurrent FSDP LoRA grouped GEMM requires CUDA compute capability 8.0 or newer")

        alignment = 16 // inputs.element_size()
        dimensions = (inputs.shape[-1], output_features)
        if any(dimension % alignment for dimension in dimensions):
            raise ValueError(
                "Concurrent FSDP LoRA grouped GEMM requires input and output "
                f"dimensions divisible by {alignment}; got {dimensions}"
            )

    @staticmethod
    def _routing(
        group_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        active_groups, compact_indices = torch.unique(group_indices, sorted=True, return_inverse=True)
        sorted_indices, sort_order = torch.sort(compact_indices, stable=True)
        group_sizes = torch.bincount(sorted_indices, minlength=len(active_groups))
        group_offsets = torch.cumsum(group_sizes, dim=0, dtype=torch.int32)
        return active_groups, sort_order, group_offsets

    def _apply_grouped_projection(
        self,
        inputs: torch.Tensor,
        weights: torch.Tensor,
        active_groups: torch.Tensor,
        sort_order: torch.Tensor,
        group_offsets: torch.Tensor,
    ) -> torch.Tensor:
        output_features = weights.shape[1]
        self._validate_grouped_mm_inputs(inputs, output_features)
        selected_weights = weights.index_select(0, active_groups)
        sorted_inputs = inputs.index_select(0, sort_order)
        sorted_output = F.grouped_mm(
            sorted_inputs,
            selected_weights.transpose(1, 2),
            offs=group_offsets,
        )
        return torch.zeros_like(sorted_output).index_copy(0, sort_order, sorted_output)

    def _active_pair_weight_bank(
        self,
        *,
        projection: str,
        matrix: str,
        pair_experts: torch.Tensor,
        pair_adapters: torch.Tensor,
        active_slots: tuple[int, ...],
    ) -> torch.Tensor:
        values = []
        positions = []
        for compact_adapter, slot_index in enumerate(active_slots):
            adapter_positions = torch.nonzero(pair_adapters == compact_adapter, as_tuple=False).flatten()
            if adapter_positions.numel() == 0:
                continue
            expert_indices = pair_experts.index_select(0, adapter_positions)
            parameter_bank = getattr(self.adapters[slot_index], matrix)[projection]
            values.append(parameter_bank.index_select(0, expert_indices))
            positions.append(adapter_positions)

        concatenated_positions = torch.cat(positions)
        restore_order = torch.argsort(concatenated_positions)
        return torch.cat(values).index_select(0, restore_order)

    def _apply_lora_projection(
        self,
        inputs: torch.Tensor,
        *,
        projection: str,
        pair_experts: torch.Tensor,
        pair_adapters: torch.Tensor,
        active_slots: tuple[int, ...],
        sort_order: torch.Tensor,
        group_offsets: torch.Tensor,
    ) -> torch.Tensor:
        lora_a = self._active_pair_weight_bank(
            projection=projection,
            matrix="lora_A",
            pair_experts=pair_experts,
            pair_adapters=pair_adapters,
            active_slots=active_slots,
        )
        lora_b = self._active_pair_weight_bank(
            projection=projection,
            matrix="lora_B",
            pair_experts=pair_experts,
            pair_adapters=pair_adapters,
            active_slots=active_slots,
        )
        lora_inputs = F.dropout(inputs, p=self.dropout_p, training=self.training)
        self._validate_grouped_mm_inputs(lora_inputs, self.rank)
        sorted_inputs = lora_inputs.index_select(0, sort_order)
        intermediate = F.grouped_mm(sorted_inputs, lora_a.transpose(1, 2), offs=group_offsets)
        self._validate_grouped_mm_inputs(intermediate, lora_b.shape[1])
        sorted_delta = F.grouped_mm(intermediate, lora_b.transpose(1, 2), offs=group_offsets)
        return torch.zeros_like(sorted_delta).index_copy(0, sort_order, sorted_delta) * self.scaling

    def _token_adapter_indices(self, num_tokens: int) -> tuple[tuple[int, ...], torch.Tensor] | None:
        if self.adapter_indices is None:
            return None
        if self.adapter_indices.numel() == 0:
            return None
        if num_tokens % self.adapter_indices.shape[0] != 0:
            raise ValueError(
                "MoE token count must be divisible by the adapter routing batch: "
                f"tokens={num_tokens}, adapter_indices={self.adapter_indices.shape[0]}"
            )

        active_slots = self.active_adapter_slots
        compact_indices = self.compact_adapter_indices
        if active_slots is None or compact_indices is None:
            active_slots, compact_indices = MultiLoRALinear.compact_routing(self.adapter_indices)
        tokens_per_row = num_tokens // compact_indices.shape[0]
        return active_slots, compact_indices.repeat_interleave(tokens_per_row)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        top_k = top_k_index.shape[1]
        token_indices = torch.arange(num_tokens, device=hidden_states.device).unsqueeze(1).expand(-1, top_k).reshape(-1)
        expert_indices = top_k_index.reshape(-1)
        routed_inputs = hidden_states.index_select(0, token_indices)

        active_experts, expert_sort_order, expert_offsets = self._routing(expert_indices)
        gate_up = self._apply_grouped_projection(
            routed_inputs,
            self.gate_up_proj,
            active_experts,
            expert_sort_order,
            expert_offsets,
        )
        gate, up = gate_up.chunk(2, dim=-1)

        adapter_routing = self._token_adapter_indices(num_tokens)
        pair_routing = None
        if adapter_routing is not None:
            active_slots, token_adapters = adapter_routing
            routed_adapters = token_adapters.index_select(0, token_indices)
            combined_indices = expert_indices * len(active_slots) + routed_adapters
            active_pairs, pair_sort_order, pair_offsets = self._routing(combined_indices)
            pair_experts = torch.div(active_pairs, len(active_slots), rounding_mode="floor")
            pair_adapters = active_pairs.remainder(len(active_slots))
            pair_routing = (
                active_slots,
                pair_experts,
                pair_adapters,
                pair_sort_order,
                pair_offsets,
            )

            if "gate_proj" in self.projections:
                gate = gate + self._apply_lora_projection(
                    routed_inputs,
                    projection="gate_proj",
                    pair_experts=pair_experts,
                    pair_adapters=pair_adapters,
                    active_slots=active_slots,
                    sort_order=pair_sort_order,
                    group_offsets=pair_offsets,
                )
            if "up_proj" in self.projections:
                up = up + self._apply_lora_projection(
                    routed_inputs,
                    projection="up_proj",
                    pair_experts=pair_experts,
                    pair_adapters=pair_adapters,
                    active_slots=active_slots,
                    sort_order=pair_sort_order,
                    group_offsets=pair_offsets,
                )

        intermediate = self.act_fn(gate) * up
        down = self._apply_grouped_projection(
            intermediate,
            self.down_proj,
            active_experts,
            expert_sort_order,
            expert_offsets,
        )
        if pair_routing is not None and "down_proj" in self.projections:
            active_slots, pair_experts, pair_adapters, pair_sort_order, pair_offsets = pair_routing
            down = down + self._apply_lora_projection(
                intermediate,
                projection="down_proj",
                pair_experts=pair_experts,
                pair_adapters=pair_adapters,
                active_slots=active_slots,
                sort_order=pair_sort_order,
                group_offsets=pair_offsets,
            )

        weighted_output = (down * top_k_weights.reshape(-1, 1)).to(hidden_states.dtype)
        final_hidden_states = torch.zeros_like(hidden_states)
        return final_hidden_states.index_add(0, token_indices, weighted_output)

    def named_adapter_parameters(self, slot_index: int) -> list[tuple[str, nn.Parameter]]:
        return self.adapters[slot_index].named_lora_parameters()

    def export_adapter_state(self, slot_index: int) -> dict[str, torch.Tensor]:
        state = {}
        for name, parameter in self.named_adapter_parameters(slot_index):
            projection, matrix, _ = name.split(".")
            full_parameter = _full_tensor_cpu(parameter)
            for expert_index in range(self.num_experts):
                state[f"{expert_index}.{projection}.{matrix}.weight"] = full_parameter[expert_index]
        return state


@dataclass(frozen=True)
class AdapterRegistration:
    model_id: str
    slot_index: int
    seed: int


class MultiLoRAManager:
    """Model-wide routing and lifecycle manager for resident adapter slots."""

    def __init__(
        self,
        layers: Sequence[tuple[str, MultiLoRALinear | MultiLoRAExperts]],
        *,
        max_adapters: int,
        rank: int,
        alpha: float,
        target_modules: str | Sequence[str],
    ) -> None:
        if not layers:
            raise ValueError("Concurrent LoRA did not match any linear modules")
        self.layers = list(layers)
        self.max_adapters = max_adapters
        self.rank = rank
        self.alpha = float(alpha)
        self.target_modules = target_modules
        self._registrations: dict[str, AdapterRegistration] = {}
        self._optimizer_hparams: dict[str, dict[str, object]] = {}

    @property
    def registrations(self) -> dict[str, AdapterRegistration]:
        return dict(self._registrations)

    def register(
        self,
        model_id: str,
        seed: int = 0,
        optimizer_hparams: Optional[dict[str, object]] = None,
    ) -> int:
        if model_id in self._registrations:
            raise ValueError(f"Adapter '{model_id}' is already registered")
        used = {registration.slot_index for registration in self._registrations.values()}
        available = sorted(set(range(self.max_adapters)) - used)
        if not available:
            raise ValueError(f"Maximum number of resident LoRA adapters ({self.max_adapters}) reached")
        slot_index = available[0]
        for layer_offset, (_, layer) in enumerate(self.layers):
            layer.adapters[slot_index].reset_parameters(seed + layer_offset)
        self._registrations[model_id] = AdapterRegistration(model_id, slot_index, seed)
        if optimizer_hparams is not None:
            self._optimizer_hparams[model_id] = deepcopy(optimizer_hparams)
        return slot_index

    def delete(self, model_id: str) -> int:
        try:
            registration = self._registrations.pop(model_id)
        except KeyError as exc:
            raise ValueError(f"Adapter '{model_id}' is not registered") from exc
        self._optimizer_hparams.pop(model_id, None)
        for _, layer in self.layers:
            layer.adapters[registration.slot_index].clear()
        return registration.slot_index

    def slot_for(self, model_id: str) -> int:
        try:
            return self._registrations[model_id].slot_index
        except KeyError as exc:
            raise ValueError(f"Adapter '{model_id}' is not registered") from exc

    def set_adapter_indices(self, adapter_indices: Optional[torch.Tensor]) -> None:
        active_slots = None
        compact_indices = None
        if adapter_indices is not None:
            if adapter_indices.ndim != 1:
                raise ValueError("adapter_indices must be a 1-D tensor")
            if adapter_indices.dtype not in (torch.int32, torch.int64):
                raise ValueError("adapter_indices must use an integer dtype")
            if adapter_indices.numel() and (
                adapter_indices.min().item() < 0 or adapter_indices.max().item() >= self.max_adapters
            ):
                raise ValueError(f"adapter index must be in [0, {self.max_adapters})")
            active_slots, compact_indices = MultiLoRALinear.compact_routing(adapter_indices)
        for _, layer in self.layers:
            layer.set_adapter_indices(
                adapter_indices,
                active_slots=active_slots,
                compact_indices=compact_indices,
            )

    def parameters_for_slot(self, slot_index: int) -> list[nn.Parameter]:
        return [parameter for _, parameter in self.named_parameters_for_slot(slot_index)]

    def named_parameters_for_slot(self, slot_index: int) -> list[tuple[str, nn.Parameter]]:
        params: list[tuple[str, nn.Parameter]] = []
        for name, layer in self.layers:
            params.extend(
                (f"{name}.{parameter_name}", parameter)
                for parameter_name, parameter in layer.named_adapter_parameters(slot_index)
            )
        return params

    def set_optimizer_hparams(self, model_id: str, optimizer_hparams: dict[str, object]) -> None:
        self.slot_for(model_id)
        self._optimizer_hparams[model_id] = deepcopy(optimizer_hparams)

    def optimizer_hparams_for(self, model_id: str) -> dict[str, object]:
        self.slot_for(model_id)
        return deepcopy(self._optimizer_hparams.get(model_id, {}))

    @contextmanager
    def select_optimizer_slots(self, optimizer: torch.optim.Optimizer, model_ids: Sequence[str]):
        """Expose only selected adapters to one optimizer call."""
        if not optimizer.param_groups:
            raise RuntimeError("Optimizer has no parameter groups")

        original_groups = optimizer.param_groups
        template = {key: value for key, value in original_groups[0].items() if key != "params"}
        selected_groups = []
        for model_id in model_ids:
            slot_index = self.slot_for(model_id)
            hparams = self.optimizer_hparams_for(model_id)
            group = dict(template)
            group["params"] = self.parameters_for_slot(slot_index)
            if hparams:
                group["lr"] = hparams["learning_rate"]
                group["betas"] = (hparams["beta1"], hparams["beta2"])
                group["eps"] = hparams["eps"]
                group["weight_decay"] = hparams["weight_decay"]
            selected_groups.append(group)

        optimizer.param_groups = selected_groups
        try:
            yield
        finally:
            optimizer.param_groups = original_groups

    def clear_slot_gradients(self, slot_index: int) -> None:
        for parameter in self.parameters_for_slot(slot_index):
            parameter.grad = None

    def training_state(self, model_id: str, optimizer: torch.optim.Optimizer) -> dict[str, object]:
        registration = self._registrations.get(model_id)
        if registration is None:
            raise ValueError(f"Adapter '{model_id}' is not registered")

        parameters: dict[str, torch.Tensor] = {}
        optimizer_state: dict[str, dict[str, object]] = {}
        for name, parameter in self.named_parameters_for_slot(registration.slot_index):
            parameters[name] = _full_tensor_cpu(parameter)
            state = optimizer.state.get(parameter)
            if state:
                optimizer_state[name] = {
                    key: _full_tensor_cpu(value) if isinstance(value, torch.Tensor) else deepcopy(value)
                    for key, value in state.items()
                }

        targets = [self.target_modules] if isinstance(self.target_modules, str) else list(self.target_modules)
        return {
            "format": "skyrl.fsdp.concurrent_lora",
            "version": 1,
            "source_model_id": model_id,
            "seed": registration.seed,
            "signature": {
                "rank": self.rank,
                "alpha": self.alpha,
                "target_modules": targets,
            },
            "parameters": parameters,
            "optimizer_state": optimizer_state,
            "optimizer_hparams": self.optimizer_hparams_for(model_id),
        }

    def load_training_state(
        self,
        model_id: str,
        state: dict[str, object],
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if state.get("format") != "skyrl.fsdp.concurrent_lora" or state.get("version") != 1:
            raise ValueError("Unsupported concurrent LoRA checkpoint format")

        signature = state.get("signature")
        if not isinstance(signature, dict):
            raise ValueError("Concurrent LoRA checkpoint is missing its signature")
        expected_targets = [self.target_modules] if isinstance(self.target_modules, str) else list(self.target_modules)
        actual_targets = list(signature.get("target_modules", []))
        if (
            int(signature.get("rank", -1)) != self.rank
            or float(signature.get("alpha", -1)) != self.alpha
            or actual_targets != expected_targets
        ):
            raise ValueError(
                "Concurrent LoRA checkpoint signature mismatch: "
                f"expected rank={self.rank}, alpha={self.alpha}, targets={expected_targets}; "
                f"got rank={signature.get('rank')}, alpha={signature.get('alpha')}, targets={actual_targets}"
            )

        registration = self._registrations.get(model_id)
        if registration is None:
            raise ValueError(f"Adapter '{model_id}' is not registered")
        named_parameters = dict(self.named_parameters_for_slot(registration.slot_index))
        saved_parameters = state.get("parameters")
        if not isinstance(saved_parameters, dict) or set(saved_parameters) != set(named_parameters):
            raise ValueError("Concurrent LoRA checkpoint parameter names do not match the current model")

        saved_optimizer_state = state.get("optimizer_state", {})
        if not isinstance(saved_optimizer_state, dict):
            raise ValueError("Concurrent LoRA checkpoint optimizer state is invalid")

        for name, parameter in named_parameters.items():
            saved_parameter = saved_parameters[name]
            if not isinstance(saved_parameter, torch.Tensor) or tuple(saved_parameter.shape) != tuple(parameter.shape):
                raise ValueError(f"Concurrent LoRA checkpoint tensor shape mismatch for '{name}'")
            _copy_from_full_tensor(parameter, saved_parameter)
            parameter.grad = None

            optimizer.state.pop(parameter, None)
            saved_param_state = saved_optimizer_state.get(name)
            if saved_param_state is None:
                continue
            if not isinstance(saved_param_state, dict):
                raise ValueError(f"Concurrent LoRA checkpoint optimizer state for '{name}' is invalid")
            restored_state: dict[str, object] = {}
            for key, value in saved_param_state.items():
                if isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(parameter.shape):
                    restored_tensor = torch.zeros_like(parameter, dtype=value.dtype)
                    _copy_from_full_tensor(restored_tensor, value)
                    restored_state[key] = restored_tensor
                elif isinstance(value, torch.Tensor):
                    restored_state[key] = value.detach().clone()
                else:
                    restored_state[key] = deepcopy(value)
            optimizer.state[parameter] = restored_state

        saved_seed = int(state.get("seed", registration.seed))
        self._registrations[model_id] = AdapterRegistration(model_id, registration.slot_index, saved_seed)
        optimizer_hparams = state.get("optimizer_hparams", {})
        if isinstance(optimizer_hparams, dict):
            self._optimizer_hparams[model_id] = deepcopy(optimizer_hparams)

    def parameters_except_slot(self, slot_index: int) -> Iterable[nn.Parameter]:
        for index in range(self.max_adapters):
            if index != slot_index:
                yield from self.parameters_for_slot(index)

    @contextmanager
    def isolate_slot_gradients(self, slot_index: int):
        """Temporarily hide other slots from a conventional optimizer step."""
        preserved_grads: list[tuple[nn.Parameter, torch.Tensor]] = []
        for parameter in self.parameters_except_slot(slot_index):
            if parameter.grad is not None:
                preserved_grads.append((parameter, parameter.grad))
                parameter.grad = None
        try:
            yield self.parameters_for_slot(slot_index)
        finally:
            for parameter, grad in preserved_grads:
                parameter.grad = grad

    def adapter_state_dict(self, slot_index: int) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for name, layer in self.layers:
            for parameter_name, parameter in layer.export_adapter_state(slot_index).items():
                state[f"base_model.model.{name}.{parameter_name}"] = parameter
        return state

    def state(self) -> dict:
        return {
            "enabled": True,
            "capacity": self.max_adapters,
            "registered": {model_id: registration.slot_index for model_id, registration in self._registrations.items()},
            "optimizer_hparams": deepcopy(self._optimizer_hparams),
        }


def _matches_pattern(module_name: str, patterns: str | Sequence[str] | None) -> bool:
    if patterns is None:
        return False
    if isinstance(patterns, str):
        patterns = [patterns]
    return any(
        module_name == pattern or module_name.endswith(f".{pattern}") or re.fullmatch(pattern, module_name)
        for pattern in patterns
    )


def _supported_expert_projections(
    module_name: str,
    module: nn.Module,
    *,
    target_modules: str | Sequence[str],
    exclude_modules: str | Sequence[str] | None,
) -> tuple[str, ...]:
    required_attributes = (
        "num_experts",
        "hidden_dim",
        "intermediate_dim",
        "gate_up_proj",
        "down_proj",
        "act_fn",
    )
    if not all(hasattr(module, attribute) for attribute in required_attributes):
        return ()
    if (
        not isinstance(module.gate_up_proj, nn.Parameter)
        or module.gate_up_proj.ndim != 3
        or not isinstance(module.down_proj, nn.Parameter)
        or module.down_proj.ndim != 3
    ):
        return ()
    expected_gate_up_shape = (
        int(module.num_experts),
        2 * int(module.intermediate_dim),
        int(module.hidden_dim),
    )
    expected_down_shape = (
        int(module.num_experts),
        int(module.hidden_dim),
        int(module.intermediate_dim),
    )
    if (
        tuple(module.gate_up_proj.shape) != expected_gate_up_shape
        or tuple(module.down_proj.shape) != expected_down_shape
    ):
        return ()

    projections = []
    for projection in _EXPERT_PROJECTIONS:
        projection_name = f"{module_name}.{projection}"
        if _matches_pattern(module_name, exclude_modules) or _matches_pattern(projection_name, exclude_modules):
            continue
        if target_modules == "all-linear" or _matches_pattern(projection_name, target_modules):
            projections.append(projection)
    return tuple(projections)


def inject_multi_lora(
    model: nn.Module,
    *,
    max_adapters: int,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: str | Sequence[str],
    exclude_modules: str | Sequence[str] | None = None,
) -> MultiLoRAManager:
    """Replace matching linear modules and return their lifecycle manager."""

    model.requires_grad_(False)
    output_embedding = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    replacements: list[tuple[str, MultiLoRALinear | MultiLoRAExperts]] = []

    for module_name, module in list(model.named_modules()):
        if not module_name:
            continue
        projections = _supported_expert_projections(
            module_name,
            module,
            target_modules=target_modules,
            exclude_modules=exclude_modules,
        )
        if not projections:
            continue

        parent_name, _, child_name = module_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        replacement = MultiLoRAExperts(
            module,
            max_adapters=max_adapters,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            projections=projections,
        )
        setattr(parent, child_name, replacement)
        replacements.append((module_name, replacement))

    for module_name, module in list(model.named_modules()):
        if not module_name or not isinstance(module, nn.Linear):
            continue
        if module is output_embedding or _matches_pattern(module_name, exclude_modules):
            continue
        if target_modules != "all-linear" and not _matches_pattern(module_name, target_modules):
            continue
        # Fused routed expert banks are handled above. Skip any model-specific
        # expert internals that remain represented as individual linear layers.
        path_components = set(module_name.lower().split("."))
        if "experts" in path_components or "router" in path_components:
            continue

        parent_name, _, child_name = module_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        replacement = MultiLoRALinear(
            module,
            max_adapters=max_adapters,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        setattr(parent, child_name, replacement)
        replacements.append((module_name, replacement))

    return MultiLoRAManager(
        replacements,
        max_adapters=max_adapters,
        rank=rank,
        alpha=alpha,
        target_modules=target_modules,
    )
