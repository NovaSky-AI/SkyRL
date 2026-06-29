from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import Param, prepare_routing


class LoRAMixin(nn.Module):
    """A mixin for PyTorch modules to add multi-adapter LoRA support.

    This mixin adds LoRA parameters (lora_A, lora_B) and methods to apply
    the low-rank adaptation to a base module's output.

    Provides:
      - init_lora(...)  -> allocate lora_A, lora_B, lora_scaling, lora_ranks
      - apply_lora(x, base_output, adapter_indices)  -> apply LoRA adaptation

    Stored tensors (when enabled):
      lora_A:      [A, in_features, r_max]   (Param, He-uniform)
      lora_B:      [A, r_max, out_features]  (Param, zeros)
      lora_scaling:[A]                       (Buffer, alpha/rank per adapter)
      lora_ranks:  [A]                       (Buffer, int rank per adapter)
    """

    lora_scaling: torch.Tensor | None
    lora_ranks: torch.Tensor | None
    lora_A: nn.Parameter | None
    lora_B: nn.Parameter | None

    def init_lora(
        self,
        *,
        max_lora_adapters: int,
        max_lora_rank: int,
        shape_A: tuple[int, ...],
        shape_B: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        self.max_lora_adapters = int(max_lora_adapters)
        self.max_lora_rank = int(max_lora_rank)

        if self.max_lora_adapters == 0:
            self.lora_scaling = None
            self.lora_ranks = None
            self.lora_A = None
            self.lora_B = None
            return

        self.register_buffer(
            "lora_scaling",
            torch.full((self.max_lora_adapters,), 1.0, dtype=dtype, device=device),
            persistent=True,
        )
        self.register_buffer(
            "lora_ranks",
            torch.full((self.max_lora_adapters,), self.max_lora_rank, dtype=torch.int32, device=device),
            persistent=True,
        )

        self.lora_A = Param(
            *shape_A,
            dtype=dtype,
            kernel_init=lambda t: nn.init.kaiming_uniform_(t, a=math.sqrt(5)),
            device=device,
        )
        self.lora_B = Param(
            *shape_B,
            dtype=dtype,
            kernel_init=nn.init.zeros_,
            device=device,
        )

    def apply_lora(
        self,
        x: torch.Tensor,
        base_output: torch.Tensor,
        adapter_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply multi-adapter LoRA to base module output.

        Args:
          x: Input tensor [B, T, in_features]
          base_output: Base module output [B, T, out_features]
          adapter_indices: Adapter index per batch element [B], broadcasted over sequence length

        Returns:
          base_output + lora_output with per-adapter routing and scaling
        """
        if self.max_lora_adapters == 0 or adapter_indices is None:
            return base_output

        if x.dim() != 3:
            raise ValueError("x must be [B, T, in_features].")
        B, T, in_features = x.shape
        if adapter_indices.dim() != 1 or adapter_indices.size(0) != B:
            raise ValueError("adapter_indices must be shape [B].")

        # Flatten tokens to [N, in]
        x_flat = x.reshape(-1, in_features)  # [B*T, in]
        # Broadcast adapter ids across sequence length
        adapters_flat = adapter_indices.repeat_interleave(T)  # [B*T]

        # Route by adapter (ragged groups)
        x_sorted, group_sizes, unsort_idx, _ = prepare_routing(
            tokens=x_flat,
            indices=adapters_flat,
            num_groups=self.max_lora_adapters,
            adapter_indices=None,
        )

        # Compute LoRA: (x @ A) @ B per-adapter group
        N = x_sorted.size(0)
        out_features = base_output.size(-1)
        y_sorted = torch.empty(N, out_features, dtype=base_output.dtype, device=base_output.device)

        offset = 0
        for adapter_index, group_size in enumerate(group_sizes.tolist()):
            if group_size == 0:
                continue
            start_idx, end_idx = offset, offset + group_size
            adapter_input = x_sorted[start_idx:end_idx]  # [group_size, in_features]
            adapter_rank = int(self.lora_ranks[adapter_index].item())
            if adapter_rank > 0:
                lora_A_matrix = self.lora_A[adapter_index, :, :adapter_rank]  # [in_features, adapter_rank]
                lora_B_matrix = self.lora_B[adapter_index, :adapter_rank, :]  # [adapter_rank, out_features]
                intermediate_result = adapter_input.matmul(lora_A_matrix)  # [group_size, adapter_rank]
                adapter_output = intermediate_result.matmul(lora_B_matrix)  # [group_size, out_features]
            else:
                adapter_output = torch.zeros(group_size, out_features, dtype=y_sorted.dtype, device=y_sorted.device)
            y_sorted[start_idx:end_idx] = adapter_output
            offset = end_idx

        # Unsort back to original token order -> [B*T, out]
        y_flat = y_sorted[unsort_idx]

        # Reshape and scale: lora_output * self.lora_scaling[adapter_indices, None, None]
        y = y_flat.view(B, T, out_features)
        y = y * self.lora_scaling[adapter_indices].view(B, 1, 1)

        return base_output + y


class LoRALinear(LoRAMixin, nn.Linear):
    """An nn.Linear layer with multi-adapter LoRA support.

    Combines base linear transformation with optional per-adapter low-rank updates.

    Forward pass:
      base_out = F.linear(x, weight, bias)
      return self.apply_lora(x, base_out, adapter_indices)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        max_lora_adapters: int,
        max_lora_rank: int,
        dtype: torch.dtype,
        use_bias: bool,
        device: torch.device | str,
    ):
        nn.Linear.__init__(self, in_features, out_features, bias=use_bias, device=device, dtype=dtype)
        LoRAMixin.init_lora(
            self,
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            shape_A=(max_lora_adapters, in_features, max_lora_rank),
            shape_B=(max_lora_adapters, max_lora_rank, out_features),
            dtype=self.weight.dtype,
            device=self.weight.device,
        )

    def forward(self, x: torch.Tensor, adapter_indices: torch.Tensor | None = None) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        return self.apply_lora(x, base_out, adapter_indices)


def update_adapter_config(model: nn.Module, adapter_index: int, lora_rank: int, lora_alpha: float):
    """Update lora_ranks and lora_scaling for a specific adapter across all LoRA layers.

    Note: This method needs to be called BEFORE any training happens, you should not update
    the config for the same adapter index multiple times throughout training (e.g. it will
    invalidate your current training progress and also violate the assumption that lora_B
    is zero).

    Args:
        model: The model containing LoRA layers
        adapter_index: Index of the adapter to update
        lora_rank: Rank to set for this adapter
        lora_alpha: Alpha value to use for computing scaling (alpha / rank)
    """
    scaling = lora_alpha / lora_rank
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRAMixin) and m.max_lora_adapters > 0:
                m.lora_ranks[adapter_index] = lora_rank
                m.lora_scaling[adapter_index] = scaling
                # Zero out columns beyond the rank for this adapter; lora_B is already zero
                m.lora_A.data[adapter_index, :, lora_rank:] = 0.0
