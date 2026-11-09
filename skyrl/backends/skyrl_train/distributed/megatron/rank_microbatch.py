"""Turn one host microbatch into exactly what this Megatron rank consumes, then upload it once.

A training worker holds each microbatch on the host as a :class:`HostMicrobatch`. Each rank needs
only part of it, and in its own layout: input ids for its pipeline stage, its context-parallel
shard, its sequence-parallel chunk of the router routes, and loss inputs only on the last stage.
:func:`build_rank_microbatch` derives all of that on the host, from a :class:`RankView` of the rank
and the :class:`MicrobatchFeatures` it trains with, as pure CPU torch; :meth:`RankMicrobatch.to`
then moves every tensor in one pinned, non-blocking copy. No field is uploaded whole and derived on
the device.

Every token-aligned field reaches the rank's layout through one :class:`TokenLayout`.
"""

import copy
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.moe.router_replay import RouterReplay

from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
    get_model_config,
    packed_seq_params_for,
    to_te_attention_mask,
)
from skyrl.backends.skyrl_train.distributed.megatron.token_metadata import TokenLayout
from skyrl.backends.skyrl_train.training_batch import (
    PACKED_BATCH_FIELD_TYPES,
    BatchField,
    TensorList,
    append_packed_field_padding,
    append_tensor_list_padding,
    packed_dummy_row_segments,
)
from skyrl.backends.skyrl_train.utils import replay_utils
from skyrl.backends.skyrl_train.utils.loss_normalization import (
    MinibatchLossNormalization,
)
from skyrl.backends.skyrl_train.utils.packed_ragged_tensor import PackedRaggedTensor
from skyrl.backends.skyrl_train.utils.packed_tensor import PackedTensor
from skyrl.backends.skyrl_train.utils.sample_support import PackedSampleSupport
from skyrl.backends.skyrl_train.utils.sample_support_replay import (
    PreparedSampleSupport,
    missing_sample_support_message,
    prepare_sample_support_inputs,
    reject_unsupported_sample_support_packing,
)

# Each tensor's offset in the upload. PyTorch's CUDA allocator hands out 512-byte-aligned blocks,
# so every view is as aligned as its own allocation would be. The upload is one contiguous copy
# either way, so the offsets do not affect its bandwidth.
_UPLOAD_ALIGNMENT = 512


@dataclass(frozen=True)
class RankView:
    """Every fact about this rank and its model chunk that decides what it takes from a microbatch."""

    tp_size: int
    tp_rank: int
    cp_size: int
    cp_rank: int
    # Pipeline placement ignores virtual chunks, as Megatron's data layout does.
    is_first_stage: bool
    is_last_stage: bool
    is_vlm: bool = False
    # False when the model applies the CP split itself (Qwen3VLModel): every CP rank then gets
    # the full packed stream, and the model's rank-local outputs keep the usual CP layout.
    shard_for_cp: bool = True
    sequence_parallel: bool = False
    # The model scatters ``padding_mask`` to its SP layout itself (GPTModel's first stage).
    model_scatters_padding_mask: bool = False
    # This stage's transformer layers, and the model's hybrid layer pattern, which map its routers to
    # the replayed layers. None when the model has no MoE layers.
    stage_layer_range: Optional[tuple[int, int]] = None
    hybrid_layer_pattern: Optional[str] = None

    @classmethod
    def for_model(
        cls,
        model,
        *,
        is_vlm: bool,
        shard_for_cp: bool = True,
    ) -> "RankView":
        """Describe this process's rank for one model chunk."""
        import megatron.core.parallel_state as mpu

        model_config = get_model_config(model)
        return cls(
            tp_size=mpu.get_tensor_model_parallel_world_size(),
            tp_rank=mpu.get_tensor_model_parallel_rank(),
            cp_size=mpu.get_context_parallel_world_size(),
            cp_rank=mpu.get_context_parallel_rank(),
            is_first_stage=mpu.is_pipeline_first_stage(ignore_virtual=True),
            is_last_stage=mpu.is_pipeline_last_stage(ignore_virtual=True),
            is_vlm=is_vlm,
            shard_for_cp=shard_for_cp,
            sequence_parallel=bool(getattr(model_config, "sequence_parallel", False)),
            model_scatters_padding_mask=replay_utils.model_scatters_router_padding_mask(model),
            # A stage may own no MoE layer (and so no router) and still receive routes.
            stage_layer_range=(
                replay_utils._get_current_pp_stage_layer_range(model_config)
                if getattr(model_config, "num_moe_experts", None)
                else None
            ),
            hybrid_layer_pattern=getattr(model_config, "hybrid_layer_pattern", None),
        )


@dataclass(frozen=True)
class MicrobatchFeatures:
    """The training features that decide which inputs a microbatch carries and how they are laid out."""

    packed: bool
    sample_support: bool = False
    block_sparse: bool = False
    mtp: bool = False
    # FP8 GEMMs need the token dimension padded further.
    fp8_enabled: bool = False
    fp8_recipe: Optional[str] = None
    # Extra unpacked-length alignment (``mlp_chunks_for_training``, DSV4 compression ratios).
    seq_align: int = 1
    # The model builds its causal mask itself and takes no attention mask (DSV4 hybrid attention).
    implicit_causal_mask: bool = False


@dataclass
class ModelInputs:
    """The model call's inputs, in this rank's token layout."""

    input_ids: torch.Tensor
    position_ids: Optional[torch.Tensor]
    # Transformer Engine padding mask (True at padding), or None when packed.
    attention_mask: Optional[torch.Tensor]
    packed_seq_params: Optional[PackedSeqParams]
    # Router padding mask in the model's sequence-parallel layout, when routers mask padding.
    padding_mask: Optional[torch.Tensor]
    vlm: dict[str, torch.Tensor] = field(default_factory=dict)

    def model_kwargs(self) -> dict[str, Any]:
        """Keyword arguments beyond the positional ids, positions and attention mask."""
        kwargs: dict[str, Any] = {"packed_seq_params": self.packed_seq_params, **self.vlm}
        if self.padding_mask is not None:
            kwargs["padding_mask"] = self.padding_mask
        return kwargs


@dataclass(kw_only=True)
class LossTensors:
    """Loss inputs a host microbatch carries to the last stage unchanged."""

    old_action_log_probs: Optional[torch.Tensor] = None
    base_action_log_probs: Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None
    loss_mask: Optional[torch.Tensor] = None
    rollout_action_logprobs: Optional[torch.Tensor] = None
    response_mask: Optional[torch.Tensor] = None
    num_microbatches: Optional[int] = None
    # ``[batch]`` real rows and the optimizer step's denominators (forward_backward only).
    real_sample_mask: Optional[torch.Tensor] = None
    loss_normalization: Optional[MinibatchLossNormalization] = None
    # calculate_per_token_loss (VLM CP) counts, attached by the worker for forward_backward.
    num_tokens_local: Optional[int] = None
    num_tokens_global: Optional[int] = None


@dataclass(kw_only=True)
class HostMicrobatch:
    """One microbatch of a worker's rows, on the host: what every rank derives its part from."""

    sequences: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    num_actions: int
    # Per row, the lengths of the packed segments it holds.
    sub_seq_lengths: Optional[TensorList] = None
    # Router replay routes in the compact rollout dtype.
    rollout_expert_indices: Optional[BatchField] = None
    router_padding_mask: Optional[torch.Tensor] = None
    rollout_sample_support: Optional[PackedSampleSupport] = None
    pixel_values: Optional[TensorList] = None
    image_grid_thw: Optional[TensorList] = None
    # A microbatch of padding rows only, which equalizes microbatch counts across DP ranks.
    is_padding_batch: bool = False
    loss: LossTensors = field(default_factory=LossTensors)

    @property
    def batch_size(self) -> int:
        return self.sequences.shape[0]

    def padded(self, batch_size: int) -> "HostMicrobatch":
        """This microbatch with dummy rows up to ``batch_size``, as Megatron's schedule needs every
        microbatch to have one size.

        A dummy row attends one token, so it is non-degenerate: no fully masked row (NaN in dense
        attention's softmax) and no zero-length segment (rejected by the packed kernel). It trains
        nothing: its loss, response and real-sample masks are zero and its router padding mask is set.
        """
        count = batch_size - self.batch_size
        if count <= 0:
            return self

        def first_token(mask: torch.Tensor, n: int) -> torch.Tensor:
            rows = mask.new_zeros((n, *mask.shape[1:]))
            rows[:, 0] = 1
            return rows

        def positions(position_ids: torch.Tensor, n: int) -> torch.Tensor:
            return torch.arange(position_ids.shape[1], device=position_ids.device).unsqueeze(0).expand(n, -1)

        def padding(mask: torch.Tensor, n: int) -> torch.Tensor:
            return torch.ones((n, *mask.shape[1:]), dtype=torch.bool, device=mask.device)

        loss = self.loss
        return dataclasses.replace(
            self,
            sequences=_pad_rows("sequences", self.sequences, count),
            attention_mask=_pad_rows("attention_mask", self.attention_mask, count, first_token),
            position_ids=_pad_rows("position_ids", self.position_ids, count, positions),
            sub_seq_lengths=_pad_rows("sub_seq_lengths", self.sub_seq_lengths, count),
            rollout_expert_indices=_pad_rows("rollout_expert_indices", self.rollout_expert_indices, count),
            router_padding_mask=_pad_rows("router_padding_mask", self.router_padding_mask, count, padding),
            rollout_sample_support=_pad_rows("rollout_sample_support", self.rollout_sample_support, count),
            pixel_values=_pad_rows("pixel_values", self.pixel_values, count),
            image_grid_thw=_pad_rows("image_grid_thw", self.image_grid_thw, count),
            loss=dataclasses.replace(
                loss,
                old_action_log_probs=_pad_rows("old_action_log_probs", loss.old_action_log_probs, count),
                base_action_log_probs=_pad_rows("base_action_log_probs", loss.base_action_log_probs, count),
                advantages=_pad_rows("advantages", loss.advantages, count),
                loss_mask=_pad_rows("loss_mask", loss.loss_mask, count),
                rollout_action_logprobs=_pad_rows("rollout_action_logprobs", loss.rollout_action_logprobs, count),
                response_mask=_pad_rows("response_mask", loss.response_mask, count),
                real_sample_mask=_pad_rows("real_sample_mask", loss.real_sample_mask, count),
            ),
        )


def _pad_rows(
    key: str,
    value: Any,
    count: int,
    fill: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
) -> Any:
    """``value`` with ``count`` dummy rows appended, ``fill``'s or zeros; ``key`` names its padding rule
    when it is packed or ragged."""
    if value is None:
        return None
    if isinstance(value, PACKED_BATCH_FIELD_TYPES):
        # Per-token fields cover the dummy attended token; response fields do not.
        return append_packed_field_padding(key, value, segment_lengths=packed_dummy_row_segments(key, count))
    if isinstance(value, TensorList):
        return append_tensor_list_padding(key, value, count)
    rows = fill(value, count) if fill is not None else value.new_zeros((count, *value.shape[1:]))
    return torch.cat([value, rows], dim=0)


@dataclass(kw_only=True)
class LossInputs(LossTensors):
    """What the last stage's loss reads, in the layout its logits arrive in."""

    num_actions: int
    # Canonical padded sequence length S.
    seq_len: int
    # Canonical ``[batch, S]`` bool mask.
    attention_mask: torch.Tensor
    # Canonical ``[batch, S]`` token ids (unpacked, without sample-support replay).
    sequences: Optional[torch.Tensor] = None
    # This CP rank's packed next-token targets ``[1, T // CP]``.
    packed_targets: Optional[torch.Tensor] = None
    cu_seqlens_padded: Optional[torch.Tensor] = None
    sub_seq_lengths: Optional[list[list[int]]] = None
    # Unpacked: flat source and destination rows and per-row counts that restore
    # ``[batch, S', ...]`` model outputs to canonical ``[batch, S, ...]``.
    recover: Optional[tuple[torch.Tensor, torch.Tensor, list[int]]] = None
    # Packed MTP: the ``[1, T]`` real-token mask of real rows, which masks cross-segment draft targets.
    mtp_packed_mask: Optional[torch.Tensor] = None
    sample_support: Optional[PreparedSampleSupport] = None
    # Block-sparse fused LM head: the positions whose next token is trained, in the logits' layout.
    active_mask: Optional[torch.Tensor] = None

    @property
    def packed(self) -> bool:
        return self.cu_seqlens_padded is not None


@dataclass
class RankMicrobatch:
    """One microbatch as this rank consumes it."""

    model_inputs: ModelInputs
    # ``[local_layers, router_tokens, topk]`` routes in the compact rollout dtype.
    replay_routes: Optional[torch.Tensor]
    loss_inputs: Optional[LossInputs]

    def to(self, device: torch.device | str | int) -> "RankMicrobatch":
        """Move every tensor to ``device`` in one pinned, non-blocking copy."""
        device = torch.device(device)
        tensors: dict[int, torch.Tensor] = {}
        _map_tensors(self, lambda tensor: tensors.setdefault(id(tensor), tensor))

        offsets: dict[int, int] = {}
        total = 0
        for key, tensor in tensors.items():
            offsets[key] = total
            total += -(-tensor.numel() * tensor.element_size() // _UPLOAD_ALIGNMENT) * _UPLOAD_ALIGNMENT

        staging = torch.empty(total, dtype=torch.uint8, pin_memory=device.type == "cuda")
        for key, tensor in tensors.items():
            nbytes = tensor.numel() * tensor.element_size()
            if nbytes:
                staging[offsets[key] : offsets[key] + nbytes].copy_(
                    tensor.detach().contiguous().reshape(-1).view(torch.uint8)
                )
        # The caching host allocator keeps the pinned buffer until this copy completes.
        uploaded = staging.to(device, non_blocking=True)

        moved: dict[int, torch.Tensor] = {}
        for key, tensor in tensors.items():
            nbytes = tensor.numel() * tensor.element_size()
            if nbytes:
                start = offsets[key]
                moved[key] = uploaded[start : start + nbytes].view(tensor.dtype).view(tensor.shape)
            else:
                moved[key] = torch.empty(tensor.shape, dtype=tensor.dtype, device=device)
        return _map_tensors(self, lambda tensor: moved[id(tensor)])


def _map_tensors(value: Any, fn) -> Any:
    """Rebuild ``value`` with ``fn`` applied to every tensor it holds.

    Containers are rebuilt without running their constructors, so no validation or derived
    field (``PackedSeqParams.seq_idx``) is recomputed on the device.
    """
    if torch.is_tensor(value):
        return fn(value)
    if isinstance(value, dict):
        return {key: _map_tensors(item, fn) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_map_tensors(item, fn) for item in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        attributes = {item.name: getattr(value, item.name) for item in dataclasses.fields(value)}
        rebuilt = copy.copy(value)
    elif isinstance(value, (PackedTensor, PackedRaggedTensor, TensorList)):
        attributes = dict(vars(value))
        # Not copy.copy: it would follow a pickling hook (``__reduce__``), which can serialize every
        # tensor and rerun the constructor.
        rebuilt = object.__new__(type(value))
    else:
        return value
    for name, item in attributes.items():
        object.__setattr__(rebuilt, name, _map_tensors(item, fn))
    return rebuilt


def recover_canonical_layout(values: torch.Tensor, loss_inputs: LossInputs) -> torch.Tensor:
    """Restore unpacked ``[batch, S', ...]`` model outputs to canonical ``[batch, S, ...]``."""
    source_rows, destination_rows, row_counts = loss_inputs.recover
    batch_size, trailing = values.shape[0], values.shape[2:]
    flat = values.reshape(-1, *trailing)
    out = values.new_zeros((batch_size * loss_inputs.seq_len, *trailing))
    # Row by row, so the gathered rows never hold more than one sequence at once.
    start = 0
    for count in row_counts:
        out.index_copy_(
            0, destination_rows[start : start + count], flat.index_select(0, source_rows[start : start + count])
        )
        start += count
    return out.view(batch_size, loss_inputs.seq_len, *trailing)


def _sequence_parallel_chunk(mask: torch.Tensor, view: RankView) -> torch.Tensor:
    """The router mask as this rank's model sees it at the routers."""
    if not view.sequence_parallel or view.model_scatters_padding_mask:
        return mask
    chunk = mask.shape[1] // view.tp_size
    return mask[:, view.tp_rank * chunk : (view.tp_rank + 1) * chunk].contiguous()


def _build_active_mask(loss_mask: torch.Tensor, num_actions: int, sequence_length: int) -> torch.Tensor:
    """Expand the action loss mask into prediction-row coordinates."""
    if loss_mask.dim() != 2 or loss_mask.shape[1] != num_actions:
        raise ValueError(f"Expected loss_mask shape [batch, {num_actions}], got {tuple(loss_mask.shape)}")
    if num_actions > sequence_length - 1:
        raise ValueError(f"num_actions={num_actions} exceeds the {sequence_length - 1} prediction rows")
    active_mask = torch.zeros((loss_mask.shape[0], sequence_length - 1), dtype=torch.bool, device=loss_mask.device)
    if num_actions > 0:
        active_mask[:, -num_actions:] = loss_mask > 0
    return active_mask


def build_rank_microbatch(host: HostMicrobatch, view: RankView, features: MicrobatchFeatures) -> RankMicrobatch:
    """Derive, on the host, everything this rank consumes from one microbatch."""
    sequences = host.sequences
    attention_mask = host.attention_mask.to(torch.bool)
    position_ids = host.position_ids
    # ``list[list[int]]`` is what packing and the packed scatter-back consume.
    sub_seq_lengths = [t.tolist() for t in host.sub_seq_lengths] if host.sub_seq_lengths is not None else None
    rollout_expert_indices = host.rollout_expert_indices

    vlm: dict[str, torch.Tensor] = {}
    if host.pixel_values is not None and view.is_first_stage:
        vlm["pixel_values"] = torch.cat(host.pixel_values.tensors, dim=0)
    if host.image_grid_thw is not None:
        vlm["image_grid_thw"] = torch.cat(host.image_grid_thw.tensors, dim=0)

    sample_support = host.rollout_sample_support if features.sample_support and view.is_last_stage else None
    if features.sample_support and view.is_last_stage:
        if sample_support is None:
            raise ValueError(missing_sample_support_message("Megatron"))
        reject_unsupported_sample_support_packing(sub_seq_lengths)
    if rollout_expert_indices is not None and sub_seq_lengths is not None:
        raise ValueError(
            "router replay does not support controller-packed multi-subsequence rows: the replay "
            "layout assumes one trajectory per row"
        )

    layout = TokenLayout.build(
        attention_mask,
        packed=features.packed,
        sub_seq_lengths=sub_seq_lengths,
        tp_size=view.tp_size,
        cp_size=view.cp_size,
        cp_rank=view.cp_rank,
        fp8_enabled=features.fp8_enabled,
        fp8_recipe=features.fp8_recipe,
        seq_align=features.seq_align,
    )
    # The loss reads CP-local outputs either way; only the model's inputs skip the split.
    input_layout = layout if not features.packed or view.shard_for_cp else layout.full()
    # The first stage embeds the ids; VLMs merge vision features on every stage.
    input_ids = input_layout.place(sequences, 0) if view.is_first_stage or view.is_vlm else sequences
    packed_seq_params = packed_seq_params_for(layout) if features.packed else None
    if features.packed:
        model_attention_mask = None
        # The trunk ignores position_ids under THD packing (rotary comes from packed_seq_params),
        # but the native MTP block rolls and re-embeds them per depth, so it needs them packed.
        model_position_ids = None
        if features.mtp:
            model_position_ids = input_layout.place(position_ids, 0) if view.is_first_stage else position_ids
    else:
        model_attention_mask = ~layout.padding_mask
        # Qwen-style VLMs recompute 3D mRoPE positions internally from
        # image_grid_thw and ignore any position_ids passed in.
        model_position_ids = None if view.is_vlm else layout.place(position_ids, 0)

    replay_routes = None
    padding_mask = None
    if rollout_expert_indices is not None:
        replay_routes, router_padding_mask = replay_utils.select_rank_replay_routes(
            rollout_expert_indices,
            host.router_padding_mask,
            layout,
            router_replay_instances=RouterReplay.global_router_replay_instances,
            stage_layer_range=view.stage_layer_range,
            hybrid_layer_pattern=view.hybrid_layer_pattern,
            tp_size=view.tp_size,
            tp_rank=view.tp_rank,
        )
        padding_mask = _sequence_parallel_chunk(router_padding_mask, view)

    model_inputs = ModelInputs(
        input_ids=input_ids,
        position_ids=model_position_ids,
        attention_mask=None if features.implicit_causal_mask else to_te_attention_mask(model_attention_mask),
        packed_seq_params=packed_seq_params,
        padding_mask=padding_mask,
        vlm=vlm,
    )
    if not view.is_last_stage:
        return RankMicrobatch(model_inputs=model_inputs, replay_routes=replay_routes, loss_inputs=None)

    num_actions = host.num_actions
    loss_inputs = LossInputs(
        num_actions=num_actions,
        seq_len=attention_mask.shape[1],
        attention_mask=attention_mask,
        sub_seq_lengths=sub_seq_lengths,
        # Every ``LossTensors`` field, as the host carries it.
        **vars(host.loss),
    )
    if features.block_sparse:
        if loss_inputs.loss_mask is None:
            raise ValueError("The triton_block_sparse fused LM-head backend requires a loss mask")
        active_mask = _build_active_mask(loss_inputs.loss_mask, num_actions, loss_inputs.seq_len)
        # Packed, position ``t`` predicts token ``t + 1`` of its segment, like the rolled targets.
        loss_inputs.active_mask = (
            layout.place(torch.nn.functional.pad(active_mask, (0, 1), value=False), False)
            if features.packed
            else active_mask
        )
    if sample_support is not None:
        loss_inputs.sample_support = prepare_sample_support_inputs(
            sequences, loss_inputs.loss_mask, sample_support, num_actions, layout
        )
    if features.packed:
        # The same tensor as packed_seq_params', so the upload carries it once.
        loss_inputs.cu_seqlens_padded = packed_seq_params.cu_seqlens_q_padded
        if sample_support is None:
            loss_inputs.packed_targets = layout.place(sequences, 0, next_token=True)
        if features.mtp:
            real_tokens = attention_mask & loss_inputs.real_sample_mask.to(torch.bool)[:, None]
            loss_inputs.mtp_packed_mask = layout.place(real_tokens.to(torch.float32), 0.0)
    else:
        if sample_support is None:
            loss_inputs.sequences = sequences
        loss_inputs.recover = layout.recover()
    return RankMicrobatch(model_inputs=model_inputs, replay_routes=replay_routes, loss_inputs=loss_inputs)
