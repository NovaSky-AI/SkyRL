"""MoE Rollout Routing Replay (R3) for HuggingFace models under FSDP.

R3 forces the training forward pass to reuse the expert selections vLLM made
during rollout, removing the train/inference routing mismatch that destabilizes
MoE RL. The Megatron backend drives Megatron-Core's native ``RouterReplay``;
HF models have no such class, so here we intercept each router via a forward
hook instead.

transformers v5 factors routing into a ``*TopKRouter`` submodule (the
``.mlp.gate``) whose ``forward`` returns ``(router_logits, router_scores,
router_indices)``. The hook overrides ``router_indices`` with the rollout's
selections and recomputes ``router_scores`` from the live ``router_logits`` so
gradients still flow through the gate at the replayed experts. We match this
softmax ``*TopKRouter`` shape structurally (OlMoE, Qwen2/3-MoE). DeepSeek-V3 /
Moonlight gate differently (``DeepseekV3TopkRouter``: sigmoid, returns only
logits), so none is matched and install raises; use the Megatron backend there.
"""

__all__ = [
    "HFRouterReplayContext",
    "align_replay_indices",
    "install_router_replay_hooks",
]

import re
from typing import List, Optional

import torch
import torch.nn as nn

from skyrl.backends.skyrl_train.distributed.ulysses.utils import slice_input_tensor

# transformers v5 names every router submodule "<arch>TopKRouter" and installs
# it as the MoE block's ``.mlp.gate``; match on that path rather than class name.
_ROUTER_NAME_RE = re.compile(r"(?:^|\.)mlp\.gate$")
_LAYER_INDEX_RE = re.compile(r"layers\.(\d+)\.")


class HFRouterReplayContext:
    """Per-worker holder of the routing decisions the hooks replay.

    ``per_layer_indices[i]`` is the ``[num_tokens, top_k]`` long tensor for
    global transformer layer ``i``, aligned to the token order the router sees.
    """

    def __init__(self) -> None:
        self.active: bool = False
        self.per_layer_indices: Optional[List[torch.Tensor]] = None

    def set(self, per_layer_indices: List[torch.Tensor]) -> None:
        self.per_layer_indices = per_layer_indices
        self.active = True

    def clear(self) -> None:
        self.active = False
        self.per_layer_indices = None


def _parse_layer_index(name: str) -> int:
    match = _LAYER_INDEX_RE.search(name)
    if match is None:
        raise ValueError(f"Could not parse a transformer layer index from router module {name!r}")
    return int(match.group(1))


def _split_per_layer_indices(rollout_expert_indices: torch.Tensor) -> List[torch.Tensor]:
    """Split ``[batch, seq, layers, topk]`` into one ``[batch*seq, topk]`` long
    tensor per layer, matching the router's row-major token flattening."""
    per_layer = rollout_expert_indices.permute(2, 0, 1, 3).contiguous()
    topk = per_layer.shape[-1]
    return [per_layer[i].reshape(-1, topk).long() for i in range(per_layer.shape[0])]


def align_replay_indices(
    rollout_expert_indices: torch.Tensor,
    *,
    num_layers: int,
    nnz_indices: Optional[torch.Tensor],
    sp_size: int,
) -> List[torch.Tensor]:
    """Align rollout expert indices to the token layout the router sees.

    Mirrors :meth:`HFModelWrapper.forward`'s own token preprocessing so the
    per-layer indices line up with the (possibly packed and SP-sliced) tokens:

    - Sample packing: gather with the same ``nnz_indices`` that ``unpad_input``
      produced for the sequences.
    - Ulysses SP: pad+slice the token dim with the same logic applied to the
      sequences, so each rank's router only sees its own slice.

    Args:
        rollout_expert_indices: ``[batch, seq, layers, topk]`` rollout selections,
            already on the worker's compute device. Workers should move the training
            batch to the worker's compute device before calling this method
        num_layers: model's transformer-layer count; vLLM reports one entry per
            layer, so this must equal the indices' layer dimension.
        nnz_indices: flat non-pad token indices from ``unpad_input`` when sample
            packing is on, else ``None``.
        sp_size: Ulysses sequence-parallel world size.

    Returns:
        One ``[num_tokens, topk]`` long tensor per global transformer layer.
    """
    if rollout_expert_indices.dim() != 4:
        raise ValueError(f"Expected 4D replay indices, got shape {tuple(rollout_expert_indices.shape)}.")
    if rollout_expert_indices.shape[2] != num_layers:
        raise ValueError(
            f"rollout_expert_indices has {rollout_expert_indices.shape[2]} layers "
            f"but the model has {num_layers}; vLLM must report routing for every layer."
        )
    layers, topk = rollout_expert_indices.shape[2], rollout_expert_indices.shape[3]
    if nnz_indices is not None:
        # Packed: keep only non-pad tokens, in the same order as ``unpad_input``.
        aligned = rollout_expert_indices.reshape(-1, layers, topk)[nnz_indices].unsqueeze(0)
    else:
        aligned = rollout_expert_indices
    if sp_size > 1:
        # Ulysses SP gives each rank only its slice of the sequence; slice the replay
        # indices the same way so they line up with the tokens that rank's router sees
        # (slice_input_tensor pads to an sp multiple, then slices):
        # https://github.com/NovaSky-AI/SkyRL/blob/skyrl-v0.2.0/skyrl/backends/skyrl_train/distributed/ulysses/utils.py#L122-L135
        aligned = slice_input_tensor(aligned, dim=1, padding=True)
    return _split_per_layer_indices(aligned)


def _make_router_replay_hook(layer_idx: int, ctx: HFRouterReplayContext):
    def hook(module: nn.Module, _args, output):
        if not ctx.active:
            return None  # natural routing
        router_logits, _, _ = output
        idx = ctx.per_layer_indices[layer_idx].to(router_logits.device)
        # fp32 softmax matches the router's own dtype handling.
        probs = torch.softmax(router_logits, dim=-1, dtype=torch.float)
        # Gather at the replayed experts; new_scores stays a function of router_logits,
        # so the gate weights that produced those logits still receive gradients.
        new_scores = probs.gather(-1, idx)
        # Qwen3.5-MoE hardcodes the renorm and omits `norm_topk_prob`, so default to True:
        # https://github.com/huggingface/transformers/blob/v5.8.0/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L765-L781
        if getattr(module, "norm_topk_prob", True):
            new_scores = new_scores / new_scores.sum(dim=-1, keepdim=True)
        return router_logits, new_scores.to(router_logits.dtype), idx

    return hook


def install_router_replay_hooks(
    model: nn.Module, ctx: HFRouterReplayContext
) -> List[torch.utils.hooks.RemovableHandle]:
    """Register a routing-replay forward hook on every softmax ``*TopKRouter`` gate.

    Gates are matched structurally by ``.mlp.gate`` plus the ``top_k`` and
    ``num_experts`` attributes; the score recompute reads ``norm_topk_prob`` when
    present, else normalizes (Qwen3.5-MoE hardcodes the renorm and omits the attr).

    Args:
        model: HuggingFace model to instrument; its routers are hooked in place.
        ctx: Per-worker context the hooks read the replay indices from.

    Returns:
        The registered forward-hook handles, one per matched router.

    Raises:
        NotImplementedError: If no compatible router is found (e.g. a dense model, or
            a sigmoid/grouped router whose gate has a different shape).
    """
    handles = []
    for name, module in model.named_modules():
        # `*TopKRouter` gates expose these attrs and return (logits, scores, indices):
        # https://github.com/huggingface/transformers/blob/v5.8.0/src/transformers/models/olmoe/modeling_olmoe.py#L341-L359
        is_router = bool(_ROUTER_NAME_RE.search(name)) and all(
            hasattr(module, attr) for attr in ("top_k", "num_experts", "weight")
        )
        if not is_router:
            continue
        handles.append(module.register_forward_hook(_make_router_replay_hook(_parse_layer_index(name), ctx)))
    if not handles:
        raise NotImplementedError(
            "Router replay (R3) found no compatible softmax `*TopKRouter` gate among the "
            "model's named modules (expected for OlMoE, Qwen2/3-MoE). Sigmoid/grouped routers "
            "like DeepSeek-V3 / Moonlight have a different gate shape and aren't supported, so "
            "use the Megatron backend for those."
        )
    return handles
