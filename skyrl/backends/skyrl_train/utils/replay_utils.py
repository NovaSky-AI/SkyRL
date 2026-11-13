"""
Utility functions for MoE Router Replay.
"""

from collections.abc import Sequence
from contextlib import contextmanager
from typing import Optional

import torch

from skyrl.backends.skyrl_train.distributed.megatron.token_metadata import TokenLayout
from skyrl.backends.skyrl_train.utils.packed_tensor import PackedTensor


def replay_padding_row(
    topk: int,
    *,
    dtype: torch.dtype,
    device: torch.device | str | int | None = None,
) -> torch.Tensor:
    """Return one dummy route of ``topk`` distinct experts.

    Synthetic padding tokens still have to satisfy Megatron's dropless
    ``tokens * topk`` dispatcher invariant, so they route to distinct experts
    rather than repeating one. ``router_padding_mask`` excludes these rows from
    expert-bias accounting.
    """
    if topk < 1:
        raise ValueError(f"Replay route padding requires a positive topk dimension, got {topk}")
    return torch.arange(topk, dtype=dtype, device=device)


def make_replay_padding_indices(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str | int | None = None,
) -> torch.Tensor:
    """Return dummy routes with ``topk`` distinct experts in every row."""
    if not shape:
        raise ValueError(f"Replay route padding requires a positive topk dimension, got {shape}")
    padding_row = replay_padding_row(shape[-1], dtype=dtype, device=device)
    return padding_row.expand(shape).clone()


def patch_topk_router_layer_number():
    """Monkey-patch TopKRouter.set_layer_number to propagate the global layer
    number to the RouterReplay instance.

    DeepSeek V3 (and similar) architectures have dense FFN layers before the MoE
    layers.  vLLM reports routing indices for ALL transformer layers (including
    dense), but Megatron only creates RouterReplay instances for MoE layers.
    Storing the global layer_number on each RouterReplay instance lets us map
    vLLM's per-layer data to the correct MoE router even when dense layers are
    present.

    Must be called BEFORE model creation (i.e. before make_megatron_module).
    """
    try:
        from megatron.core.transformer.moe.router import TopKRouter
    except ImportError:
        return

    if getattr(TopKRouter, "_set_layer_number_patched", False):
        return

    original_set_layer_number = TopKRouter.set_layer_number

    def patched_set_layer_number(self, layer_number: int):
        original_set_layer_number(self, layer_number)
        if self.router_replay is not None:
            self.router_replay.layer_number = layer_number

    TopKRouter.set_layer_number = patched_set_layer_number
    TopKRouter._set_layer_number_patched = True


def model_scatters_router_padding_mask(model) -> bool:
    """Whether the model scatters ``padding_mask`` to its sequence-parallel layout itself.

    GPTModel scatters its mask beside the embedding on the first PP stage. HybridModel
    scatters only the embedding, so its mask must always arrive already scattered.
    """
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.utils import unwrap_model

    unwrapped_model = unwrap_model(model)
    return not isinstance(unwrapped_model, HybridModel) and bool(unwrapped_model.pre_process)


def _get_current_pp_stage_layer_range(model_config) -> tuple[int, int]:
    """Return the current PP rank's transformer-layer range as (start_layer,
    num_layers).

    Prefer Megatron's own helpers so replay indexing stays aligned with the
    actual model partition, including embedding/loss pipeline accounting.
    """
    import megatron.core.parallel_state as mpu
    from megatron.core.transformer.transformer_block import get_num_layers_to_build
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    pp_rank = mpu.get_pipeline_model_parallel_rank()
    offset = get_transformer_layer_offset(model_config, pp_rank=pp_rank)
    num_layers = get_num_layers_to_build(model_config, pp_rank=pp_rank)
    return offset, num_layers


def _get_local_router_layer_indices(
    stage_layer_range: tuple[int, int],
    hybrid_layer_pattern: Optional[str],
    global_num_layers: int,
    instances: Sequence,
) -> list[int]:
    local_layer_offset, local_num_layers = stage_layer_range
    if local_num_layers == len(instances):
        layer_indices = list(range(local_layer_offset, local_layer_offset + local_num_layers))
    else:
        # Dense-layer mismatch: map each MoE router to its global layer index.
        # Prefer the patched layer_number; fall back to offset-based mapping
        # (assumes dense layers precede MoE layers).
        layer_indices = []
        for local_router_index, router_instance in enumerate(instances):
            layer_number = getattr(router_instance, "layer_number", None)
            if layer_number is not None:
                layer_index = layer_number - 1
            else:
                layer_index = local_layer_offset + local_router_index + (local_num_layers - len(instances))
            layer_indices.append(layer_index)

    layer_indices = _hybrid_to_decoder_layer_indices(hybrid_layer_pattern, global_num_layers, layer_indices)
    if any(layer_index < 0 or layer_index >= global_num_layers for layer_index in layer_indices):
        raise ValueError(
            f"Router replay layer indices {layer_indices} out of range for data with {global_num_layers} layers"
        )
    return layer_indices


# megatron-core's hybrid_layer_allocation.Symbols: the attention symbols ATTENTION "*", DS_ATTENTION "D", MLA "+" and
# the DeepSeek-V4 CSA "C" / HCA "H" / WINDOW "W"; PIPE "|" and MTP_SEPARATOR "/" are separators.
_HYBRID_ATTENTION_SYMBOLS = frozenset("*D+CHW")


def _hybrid_to_decoder_layer_indices(
    hybrid_layer_pattern: Optional[str], global_num_layers: int, layer_indices: list[int]
) -> list[int]:
    """Map HybridModel layer indices to the inference engine's decoder-layer indices.

    Router ``layer_number`` counts hybrid layers, and vLLM returns routed experts per HF decoder layer. Models whose
    decoder layers are single hybrid layers (e.g. Nemotron-H) need no mapping. Models whose decoder layer is an
    attention layer followed by its MLP/MoE layer (DeepSeek-V4/V4.1: ``HECEWE...``; Beam: ``*-*-*E*E...``) have one
    decoder layer per attention symbol: hybrid layer ``h`` belongs to decoder layer ``(attention symbols in
    pattern[:h + 1]) - 1``. Decided from the pattern, not the indices: under pipeline parallelism an early stage's
    hybrid indices all fit below the decoder-layer count and still need mapping.
    """
    if not hybrid_layer_pattern:
        return layer_indices
    flat = hybrid_layer_pattern.split("/")[0].replace("|", "")
    attention = [symbol in _HYBRID_ATTENTION_SYMBOLS for symbol in flat]
    if global_num_layers == len(flat) or global_num_layers != sum(attention):
        return layer_indices
    return [sum(attention[: h + 1]) - 1 for h in layer_indices]


def select_rank_replay_routes(
    rollout_expert_indices: PackedTensor,
    router_padding_mask: torch.Tensor | None,
    layout: TokenLayout,
    *,
    router_replay_instances: Sequence,
    stage_layer_range: Optional[tuple[int, int]],
    hybrid_layer_pattern: Optional[str],
    tp_size: int,
    tp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the routes this rank's routers replay, and align the router padding mask.

    Runs on the host, before the microbatch is uploaded. Replay indices arrive packed to their
    real tokens (``[sum(seqlen), layers, topk]`` plus ``cu_seqlens``) while the router padding
    mask arrives batch-major; both are laid out against the microbatch's ``layout``, which
    already carries the packing and the CP front/back chunk split.

    Returns ``(routes, mask)``:

    - ``routes`` is ``[local_layers, router_tokens, topk]`` in the compact rollout dtype. It
      holds only the layers of ``router_replay_instances`` (this PP stage), and only the
      contiguous sequence-parallel chunk ``tp_rank`` of the tokens, flattened sequence major
      (Megatron's router token order). Padding tokens route to ``replay_padding_row``.
    - ``mask`` is the router padding mask in the model's token layout, before any
      sequence-parallel slice.

    Handles dense-layer mismatch: DeepSeek V3-style models have dense FFN layers before the
    MoE layers. vLLM reports routing indices for ALL transformer layers, but Megatron only has
    RouterReplay instances for MoE layers. Each instance's global layer_number (set by the
    patched TopKRouter.set_layer_number) indexes the correct slice of the data, also when
    pipeline parallelism splits the layers across stages.
    """
    if router_padding_mask is None:
        raise ValueError("router_padding_mask is required with rollout_expert_indices")
    if len(rollout_expert_indices.row_shape) != 2:
        raise ValueError(f"Expected [tokens, layers, topk] replay indices, got {rollout_expert_indices!r}")
    if tuple(router_padding_mask.shape) != layout.batch_shape:
        raise ValueError(
            f"router_padding_mask shape {tuple(router_padding_mask.shape)} does not match "
            f"the batch {layout.batch_shape}"
        )
    if not torch.equal(rollout_expert_indices.cu_seqlens.to(torch.long), layout.cu_lengths):
        raise ValueError(
            f"Replay indices hold segments {rollout_expert_indices.sequence_lengths.tolist()} for "
            f"trajectories of {layout.sequence_lengths} tokens"
        )
    if stage_layer_range is None:
        raise ValueError("rollout_expert_indices were given, but the model builds no MoE layers to replay them")

    num_captured_layers, topk = rollout_expert_indices.row_shape
    local_layer_indices = _get_local_router_layer_indices(
        stage_layer_range, hybrid_layer_pattern, num_captured_layers, router_replay_instances
    )
    aligned_router_padding_mask = layout.place(router_padding_mask.to(torch.bool), True)

    # A route row is its token's ordinal, so lay out ordinals rather than the routes themselves
    # and gather only the rows this rank replays; NO_TOKEN marks a padding token.
    values = rollout_expert_indices.values
    aligned_rows = layout.ordinals
    # Sequence parallelism: each TP rank's routers see one contiguous chunk of the sequence.
    # Megatron flattens router tokens in [sequence, batch] order.
    chunk_size = aligned_rows.shape[1] // tp_size
    rows = aligned_rows[:, tp_rank * chunk_size : (tp_rank + 1) * chunk_size].transpose(0, 1).reshape(-1)

    layers = torch.tensor(local_layer_indices, dtype=torch.long, device=values.device)
    padding_row = replay_padding_row(topk, dtype=values.dtype, device=values.device)
    if values.shape[0] == 0:
        local = padding_row.expand(rows.shape[0], len(local_layer_indices), topk).clone()
    else:
        local = values[rows.clamp(min=0)[:, None], layers[None, :]]
        local[rows < 0] = padding_row
    return local.transpose(0, 1).contiguous(), aligned_router_padding_mask


def install_replay_forward(routes: torch.Tensor) -> None:
    """Install one microbatch's ``[local_layers, router_tokens, topk]`` routes for replay.

    Widens the compact rollout dtype to the int32 Megatron's routers consume.
    """
    from megatron.core.transformer.moe.router_replay import (
        RouterReplay,
        RouterReplayAction,
    )

    RouterReplay.set_replay_data(list(routes.to(torch.int32).unbind(0)))
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)


def setup_per_microbatch_replay_backward() -> None:
    """Switch RouterReplay to backward mode so that activation-checkpoint
    recomputation during the backward pass consumes indices from
    ``replay_backward_list`` in FIFO order (populated during the forward pass).
    """
    from megatron.core.transformer.moe.router_replay import (
        RouterReplay,
        RouterReplayAction,
    )

    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)


def clear_router_replay():
    """Clear all router replay state."""
    from megatron.core.transformer.moe.router_replay import RouterReplay

    RouterReplay.clear_global_indices()
    RouterReplay.clear_global_router_replay_action()


@contextmanager
def router_replay_schedule(enabled: bool):
    """Isolate global RouterReplay state to one Megatron pipeline schedule.

    The backward FIFO spans all microbatches in a training schedule, so it must
    only be cleared at schedule boundaries. Forward-only schedules leave that
    FIFO unconsumed, and failed schedules may leave it partially consumed.
    """
    if not enabled:
        yield
        return

    clear_router_replay()
    try:
        yield
    finally:
        clear_router_replay()
