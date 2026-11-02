"""Copy-free forward for Megatron-Bridge's per-expert grouped LoRA adapter.

``GroupedExpertLinearAdapter.forward`` (``share_expert_adapters=False`` on
``TE*GroupedLinear`` experts) prepares its ``torch.nn.functional.grouped_mm``
call by looping over the local experts in Python, padding each expert's token
slice to the expert-TP multiple, concatenating the slices into a new buffer,
gathering the active experts' adapter weights with ``weight[indices]`` and
building the group offsets through a blocking host-to-device copy. With
``expert_tensor_parallel_size=1`` no padding is ever needed, so the
concatenation is a full copy of the expert input (``[tokens * topk, hidden]``
for ``linear_fc1``), the weight gather is a copy of every adapter weight (and
an index-scatter in backward), and the offsets copy synchronizes the CPU with
the GPU once per adapter call.

This patch runs that case straight on the input: one ``grouped_mm`` per
projection over all local experts (zero-token experts become empty groups,
whose weights receive a zero gradient exactly as the gathered path produced),
with the offsets uploaded asynchronously from pinned memory. Every other case
(expert TP > 1, FP8, inputs ``grouped_mm`` cannot take) falls through to the
original forward.
"""

from __future__ import annotations

import inspect
from typing import List

import torch
import torch.nn as nn

# Set by apply_grouped_expert_lora_fast_path_patch; tests swap between the two forwards.
_ORIGINAL_FORWARD = None
_PATCHED_FORWARD = None


def _grouped_offsets(m_splits: List[int], device: torch.device) -> torch.Tensor:
    """Inclusive group offsets for ``grouped_mm`` without a host-device sync."""
    splits = torch.tensor(m_splits, dtype=torch.int32, pin_memory=True)
    return splits.to(device, non_blocking=True).cumsum(0, dtype=torch.int32)


def _fast_grouped_forward(self, x: torch.Tensor, expert_splits: List[int]) -> torch.Tensor:
    """Per-expert LoRA over all local experts with no padding, concatenation or weight gather."""
    offs = _grouped_offsets(expert_splits, x.device)
    if self.dropout_position == "pre":
        x = self.dropout(x)
    if self.config.cpu_offloading and self.config.cpu_offloading_activations:
        x.activation_offloading = True
    # Calling the weight containers (rather than reading ``.weight``) keeps the DDP
    # forward pre-hooks that drive the overlapped parameter gather.
    linear_in_weight = self.linear_in()
    hidden = nn.functional.grouped_mm(x, linear_in_weight.transpose(1, 2), offs=offs)
    hidden = self.activation(hidden)
    # Scale the rank-sized bottleneck; the output projection is bias-free so this is exact.
    hidden = hidden * (self.alpha / self.dim)
    if self.config.cpu_offloading and self.config.cpu_offloading_activations:
        hidden.activation_offloading = True
    linear_out_weight = self.linear_out()
    expert_output = nn.functional.grouped_mm(hidden, linear_out_weight.transpose(1, 2), offs=offs)
    if self.dropout_position == "post":
        expert_output = self.dropout(expert_output)
    return expert_output


def apply_grouped_expert_lora_fast_path_patch() -> None:
    """Install the copy-free forward on ``GroupedExpertLinearAdapter``."""
    from megatron.bridge.peft import utils

    adapter_cls = utils.GroupedExpertLinearAdapter
    if getattr(adapter_cls, "_skyrl_grouped_expert_lora_fast_path", False):
        return
    required = {"_extract_expert_splits", "_can_use_grouped_mm", "_is_te_fp8_enabled", "_process_group_size"}
    missing = [name for name in required if not hasattr(adapter_cls, name) and not hasattr(utils, name)]
    source = inspect.getsource(adapter_cls.forward)
    if missing or "grouped_inputs" not in source:
        raise RuntimeError(
            "GroupedExpertLinearAdapter.forward does not match the Megatron-Bridge revision this "
            "patch was written for; update or drop patch_grouped_expert_lora_fast_path."
        )

    original_forward = adapter_cls.forward

    def patched_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        expert_tp_size = utils._process_group_size(self.expert_tp_group, self.config.expert_tensor_parallel_size or 1)
        if expert_tp_size != 1 or x.shape[0] == 0 or self._is_te_fp8_enabled() or not self._can_use_grouped_mm(x):
            return original_forward(self, x, *args, **kwargs)
        expert_splits = self._extract_expert_splits(args, kwargs)
        if sum(expert_splits) != x.shape[0]:
            return original_forward(self, x, *args, **kwargs)
        return _fast_grouped_forward(self, x, expert_splits)

    global _ORIGINAL_FORWARD, _PATCHED_FORWARD
    _ORIGINAL_FORWARD = original_forward
    _PATCHED_FORWARD = patched_forward
    adapter_cls.forward = patched_forward
    adapter_cls._skyrl_grouped_expert_lora_fast_path = True
