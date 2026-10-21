"""Keep Gated DeltaNet state from leaking between sequences packed into one row.

Attention layers find the sequence starts encoded in ``position_ids``; Gated DeltaNet
(linear-attention) layers only see them if the model call passes them explicitly as kwargs.
Without that, a packed row is processed as one sequence and the conv/recurrent state of an
earlier sequence bleeds into every later one.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

import torch

CU_SEQ_LENS_KWARG = "cu_seq_lens_q"


def packed_sequence_kwargs(position_ids: torch.Tensor) -> dict[str, Any]:
    """Flash-attention varlen kwargs for a packed row whose position_ids restart at 0 per sequence.

    ``position_ids`` is the unpadded ``(1, total_tokens)`` tensor SkyRL forwards when
    ``remove_microbatch_padding`` is set. Packing concatenates whole sequences, so each one
    restarts at 0 and the zeros mark the boundaries.
    """
    flat = position_ids.reshape(-1)
    starts = (flat == 0).nonzero().view(-1).to(torch.int32)
    total = torch.tensor([flat.numel()], dtype=torch.int32, device=flat.device)
    cu_seqlens = torch.cat([starts, total])
    # One device sync per micro-batch here replaces the per-attention-layer sync transformers
    # pays when it derives these from position_ids itself.
    max_length = int(cu_seqlens.diff().max())
    return {
        CU_SEQ_LENS_KWARG: cu_seqlens,
        "cu_seq_lens_k": cu_seqlens,
        "max_length_q": max_length,
        "max_length_k": max_length,
        "seq_idx": (torch.cumsum(flat == 0, 0) - 1).to(torch.int32)[None],
    }


def _is_gated_delta_net(module: torch.nn.Module) -> bool:
    return type(module).__name__.endswith("GatedDeltaNet")


# Kernels transformers resolves ahead of its torch fallbacks, and the kwarg each one takes the
# boundaries as. The fallbacks accept and ignore the kwarg, which is exactly the silent leak.
_KERNELS = {
    ("fla.ops.gated_delta_rule", "chunk_gated_delta_rule"): "cu_seqlens",
    ("causal_conv1d", "causal_conv1d_fn"): "seq_idx",
}


def check_boundaries_consumed(gdn: torch.nn.Module) -> None:
    """Raise if ``gdn`` or the kernels it resolves would ignore packed-sequence boundaries.

    Passing the boundaries is only safe if they are actually read; a transformers bump that
    stops consuming ``cu_seq_lens_q``, or a missing varlen kernel that falls back to the
    boundary-blind torch path, would silently reintroduce the cross-sequence leak.
    """
    source = inspect.getsource(type(gdn).forward)
    if f'kwargs.pop("{CU_SEQ_LENS_KWARG}"' not in source:
        raise RuntimeError(
            f"{type(gdn).__name__}.forward does not read {CU_SEQ_LENS_KWARG}; packed sequences "
            "would leak recurrent state across trajectories"
        )
    for (module, name), parameter in _KERNELS.items():
        try:
            kernel = getattr(importlib.import_module(module), name)
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                f"{module}.{name} is required for packed Gated DeltaNet training; the torch "
                "fallback ignores sequence boundaries"
            ) from error
        if parameter not in inspect.signature(kernel).parameters:
            raise RuntimeError(f"{module}.{name} has no {parameter} support")


def gated_delta_net_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Return the model's Gated DeltaNet modules, verifying they honour packed-sequence boundaries."""
    layers = [module for module in model.modules() if _is_gated_delta_net(module)]
    for gdn in layers:
        check_boundaries_consumed(gdn)
    return layers
