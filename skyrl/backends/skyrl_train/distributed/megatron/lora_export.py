"""Adapter-state transforms for the on-policy LoRA weight sync to vLLM.

Kept free of megatron imports so the transforms are unit-testable on CPU.
"""

from typing import Dict, Optional, Tuple

import torch

LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"


def lora_rank_from_tensors(lora_a: Optional[torch.Tensor], lora_b: torch.Tensor) -> int:
    """Return the LoRA rank encoded in a ``lora_B`` (and, if given, ``lora_A``) tensor.

    Layouts *before* any vLLM flattening: ``lora_B`` is ``(out, rank)`` for a
    plain linear or ``(E, out, rank)`` for fused grouped experts; ``lora_A`` is
    ``(rank, in)`` / ``(E, rank, in)``.
    """
    if lora_b.ndim not in (2, 3):
        raise ValueError(f"lora_B must be 2D (out, rank) or 3D (E, out, rank); got shape {tuple(lora_b.shape)}")
    rank = int(lora_b.shape[-1])
    if lora_a is not None:
        if lora_a.ndim not in (2, 3):
            raise ValueError(f"lora_A must be 2D (rank, in) or 3D (E, rank, in); got shape {tuple(lora_a.shape)}")
        rank_a = int(lora_a.shape[-2])
        if rank_a != rank:
            raise ValueError(f"lora_A rank {rank_a} (shape {tuple(lora_a.shape)}) != lora_B rank {rank}")
    return rank


def fold_lora_rank_scale_for_vllm(
    adapter_state: Dict[str, torch.Tensor], *, config_rank: int
) -> Tuple[Dict[str, torch.Tensor], Dict[int, int]]:
    """Fold per-module LoRA rank differences into ``lora_B`` so vLLM's uniform
    scaling reproduces the trainer's per-module scaling.

    megatron-bridge scales every adapter's output by ``alpha / dim`` where ``dim``
    is that module's *effective* rank. With ``normalize_moe_lora`` the grouped
    expert adapters run at ``rank // moe_router_topk`` while dense adapters keep
    the full rank, so the trainer applies e.g. ``32 / 4 = 8`` to the experts and
    ``32 / 32 = 1`` to everything else.

    vLLM's PEFT loader (``PEFTHelper``) has no ``rank_pattern`` support: it derives
    a single ``lora_alpha / r`` from ``adapter_config.json`` and applies it to every
    module (``LoRALayerWeights.from_config`` -> ``optimize()`` folds it into
    ``lora_b``). Exporting raw tensors under ``r = config_rank`` therefore
    under-scales each reduced-rank adapter by ``config_rank / rank`` -- the sampled
    policy sees 1/8 of the trainer's expert update at rank 32 / top-8. Multiplying
    those ``lora_B`` tensors by the ratio makes::

        (alpha / config_rank) * (ratio * B) @ A == (alpha / rank) * B @ A

    i.e. the sampler applies exactly what the trainer trained.

    Ranks are read from the tensors themselves (see :func:`lora_rank_from_tensors`),
    so this handles both the per-expert 2D layout (``experts.<i>.<proj>``) and the
    fused 3D layout (``experts.gate_up_proj`` / ``experts.down_proj``); call it
    *before* :func:`_convert_moe_experts_lora_to_vllm` flattens the 3D tensors.
    Tensors already at ``config_rank`` pass through untouched. The scale is applied
    in float32 and cast back to the tensor's dtype (exact in bf16 for power-of-two
    ratios such as 8).

    Returns the transformed state (same keys, same order) and a ``{rank: count}``
    summary of the ``lora_B`` tensors that were rescaled.
    """
    if config_rank <= 0:
        raise ValueError(f"config_rank must be positive, got {config_rank}")

    folded: Dict[str, torch.Tensor] = {}
    rescaled: Dict[int, int] = {}
    for key, tensor in adapter_state.items():
        if key.endswith(LORA_B_SUFFIX):
            lora_a = adapter_state.get(key[: -len(LORA_B_SUFFIX)] + LORA_A_SUFFIX)
            rank = lora_rank_from_tensors(lora_a, tensor)
            if rank != config_rank:
                ratio = config_rank / rank
                tensor = (tensor.to(torch.float32) * ratio).to(tensor.dtype)
                rescaled[rank] = rescaled.get(rank, 0) + 1
        folded[key] = tensor
    return folded, rescaled
