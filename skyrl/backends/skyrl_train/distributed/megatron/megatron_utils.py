# Utils ported from Verl
# https://github.com/volcengine/verl/blob/e1603dc97f3c20c58feed1f5be34acd5c72a830c/verl/utils/megatron_utils.py#L4
# https://github.com/volcengine/verl/blob/dfa3933ac44b545fca1f6a8519fd07394a2cde1c/verl/models/mcore/util.py
# The original copyright is reproduced below:

# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from loguru import logger
from megatron.core import parallel_state as mpu
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.optimizer import ChainedOptimizer
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe.moe_utils import (
    clear_aux_losses_tracker,
    get_moe_layer_wise_logging_tracker,
    reduce_aux_losses_tracker_across_ranks,
)
from megatron.core.utils import get_attr_wrapped_model, unwrap_model

from skyrl.backends.skyrl_train.distributed.megatron.packing_utils import (
    get_packed_seq_align_size,
    get_unpacked_seq_align_size,
)

ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, Float16Module)


def make_batch_generator(batches, vpp_size):
    """
    Creates a batch generator suitable for Megatron pipeline parallelism,
    handling virtual pipeline parallelism (VPP).

    If VPP is used (vpp_size > 1), it duplicates the batch iterator for each
    virtual pipeline stage. Otherwise, it returns a single iterator.

    Args:
        batches: An iterable (e.g., list) of micro-batches.
        vpp_size (int): The virtual pipeline model parallel size.

    Returns:
        An iterator or a list of iterators over the micro-batches.
    """
    if vpp_size > 1:
        # has vpp
        batch_generator = [batches] * vpp_size  # number of vpp chunks
        batch_generator = [iter(b) for b in batch_generator]
    else:
        # no vpp
        batch_generator = iter(batches)
    return batch_generator


def print_model_size(model: nn.Module, name: str = None):
    n_params, scale = get_model_size(model, scale="auto")
    if name is None:
        name = model.__class__.__name__
    logger.info(f"{name} contains {n_params:.2f}{scale} parameters")


def get_model_size(model: nn.Module, scale="auto"):
    n_params = sum(p.numel() for p in model.parameters())

    if scale == "auto":
        if n_params > 1e9:
            scale = "B"
        elif n_params > 1e6:
            scale = "M"
        elif n_params > 1e3:
            scale = "K"
        else:
            scale = ""

    if scale == "B":
        n_params = n_params / 1e9
    elif scale == "M":
        n_params = n_params / 1e6
    elif scale == "K":
        n_params = n_params / 1e3
    elif scale == "":
        pass
    else:
        raise NotImplementedError(f"Unknown scale {scale}")

    return n_params, scale


def get_moe_metrics(
    loss_scale: float,
    total_loss_dict: Optional[dict] = None,
    per_layer_logging: bool = False,
) -> dict[str, Any]:
    """Returns Mixture of Experts (MoE) auxiliary-loss metrics.

    This function reduces MoE auxiliary losses across ranks, aggregates them, and
    returns a dictionary of metrics.

    Args:
        loss_scale: Scale factor to apply to each auxiliary loss (e.g., 1/num_microbatches).
        total_loss_dict: If provided, accumulate means into this dict (by name).
        per_layer_logging: If True, include per-layer values in the returned dict.

    Returns:
        dict[str, Any]: A flat dict of aggregated metrics. For each aux loss name,
        the mean value is returned under the same key (e.g., "load_balancing_loss").
        If per_layer_logging is True, per-layer values are returned under keys of the
        form "moe/{name}_layer_{i}".
    """
    reduce_aux_losses_tracker_across_ranks()
    tracker = get_moe_layer_wise_logging_tracker()

    metrics: dict[str, Any] = {}
    if len(tracker) > 0:
        aux_losses = {k: v["values"].float() * loss_scale for k, v in tracker.items()}
        for name, loss_list in aux_losses.items():
            # Megatron-Core aggregates aux losses across layers and normalizes by number of MoE layers
            num_layers = int(loss_list.numel()) if loss_list.numel() > 0 else 1
            aggregated_value = loss_list.sum() / num_layers
            metrics[name] = float(aggregated_value.item())
            if total_loss_dict is not None:
                if name not in total_loss_dict:
                    total_loss_dict[name] = aggregated_value
                else:
                    total_loss_dict[name] += aggregated_value

            if per_layer_logging:
                for i, loss in enumerate(loss_list.tolist()):
                    metrics[f"moe/{name}_layer_{i}"] = float(loss)

    clear_aux_losses_tracker()
    return metrics


def freeze_moe_router(model_or_models: Union[nn.Module, List[nn.Module]]):
    models = model_or_models
    if not isinstance(model_or_models, list):
        models = [model_or_models]

    for model in models:
        for layer in model.decoder.layers:
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
                if getattr(layer.mlp.router, "weight", None) is not None:
                    layer.mlp.router.weight.requires_grad = False
                if getattr(layer.mlp.router, "bias", None) is not None:
                    layer.mlp.router.bias.requires_grad = False
    # modified in-place
    return model_or_models


def _convert_moe_experts_lora_to_vllm(
    adapter_state: Dict[str, "torch.Tensor"],
) -> Dict[str, "torch.Tensor"]:
    """Rewrite fused-MoE expert LoRA tensors into the layout vLLM expects.

    Megatron-Bridge exports fused experts as 3D tensors keyed
    ``...mlp.experts.gate_up_proj`` (w13) / ``...mlp.experts.down_proj`` (w2),
    with ``lora_A=(E, rank, in)`` and ``lora_B=(E, out, rank)``. vLLM's 3D-MoE
    loader (``FusedMoE3DWithLoRA`` / ``_stack_moe_lora_weights``) instead expects
    the flat PEFT layout keyed ``...experts.base_layer`` (w13) / ``...experts``
    (w2), with ``lora_A=(rank*E, in)`` and ``lora_B=(out, rank*E)``. This is the
    exact inverse of vLLM's per-expert reshape. Non-expert tensors pass through.
    """
    converted: Dict[str, "torch.Tensor"] = {}
    for key, tensor in adapter_state.items():
        is_gate_up = ".mlp.experts.gate_up_proj." in key
        is_down = ".mlp.experts.down_proj." in key
        if (is_gate_up or is_down) and tensor.ndim == 3:
            if key.endswith(".lora_A.weight"):
                # (E, rank, in) -> (rank*E [expert-major], in)
                tensor = tensor.reshape(-1, tensor.shape[-1]).contiguous()
            elif key.endswith(".lora_B.weight"):
                # (E, out, rank) -> (out, rank*E [expert-minor])
                tensor = tensor.permute(1, 2, 0).contiguous().reshape(tensor.shape[1], -1)
            if is_gate_up:
                key = key.replace(".mlp.experts.gate_up_proj.", ".mlp.experts.base_layer.")
            else:
                key = key.replace(".mlp.experts.down_proj.", ".mlp.experts.")
        converted[key] = tensor
    return converted


@torch.no_grad()
def offload_megatron_grads_to_cpu(models):
    for model_chunk in models:
        if isinstance(model_chunk, DDP):
            # use megatron DDP built in function to offload grads to cpu
            # https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/core/distributed/distributed_data_parallel.py#L575
            model_chunk.offload_grad_buffers(synchronize=False, empty_cache=False)
        else:
            for _, param in model_chunk.named_parameters():
                if param.grad is not None:
                    param.grad = param.grad.to("cpu", non_blocking=True)
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def load_megatron_grads_to_gpu(models):
    for model_chunk in models:
        if isinstance(model_chunk, DDP):
            model_chunk.restore_grad_buffers(synchronize=False)
        else:
            for _, param in model_chunk.named_parameters():
                if param.grad is not None:
                    param.grad = param.grad.to(torch.cuda.current_device(), non_blocking=True)
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def offload_megatron_model_to_cpu(models):
    """
    In megatron, the model and optimizer storage are:
    - bf16 parameter data chunked in model parallel group
    - fp32 grad chunked in model parallel group
    - fp32 main_parameter chunked in model and dp group
    - fp32 optimizer state chunked in model and dp group
    """
    for model_chunk in models:
        if isinstance(model_chunk, DDP):
            for buffer in model_chunk.buffers + model_chunk.expert_parallel_buffers:
                # use megatron buffer built in function to offload to cpu
                # https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/core/distributed/param_and_grad_buffer.py#L964
                buffer.offload_to_cpu(move_params=True, move_grads=False)

            # LoRA-aware offloading: offload non-lora base weights that live
            # outside the fused Megatron buffers (e.g. HF/bridge "to_wrap" weights).
            for name, param in model_chunk.named_parameters():
                if (
                    param.is_cuda
                    and not param.requires_grad
                    and "adapter" not in name
                    and param.data.storage().size() > 0
                ):
                    cpu_tensor = param.data.detach().cpu().pin_memory()
                    param._offload_cpu_data = cpu_tensor
                    param._offload_cuda_numel = param.data.numel()
                    param.data = torch.empty(0, dtype=param.data.dtype, device=param.data.device)
        else:
            for _, param in model_chunk.named_parameters():
                param.data = param.data.to("cpu", non_blocking=True)
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def load_megatron_model_to_gpu(models):
    for model_chunk in models:
        if isinstance(model_chunk, DDP):
            for buffer in model_chunk.buffers + model_chunk.expert_parallel_buffers:
                buffer.reload_from_cpu(move_params=True, move_grads=False)

            # Restore any LoRA-frozen base weights that were offloaded above.
            device_id = torch.cuda.current_device()
            for name, param in model_chunk.named_parameters():
                if hasattr(param, "_offload_cpu_data") and param.data.storage().size() == 0:
                    restored = param._offload_cpu_data.to(device_id, non_blocking=True)
                    param.data = restored
        else:
            device_id = torch.cuda.current_device()
            for _, param in model_chunk.named_parameters():
                param.data = param.data.to(device_id, non_blocking=True)
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def offload_megatron_copy_params(optimizers):
    """
    Offload optimizer parameters to CPU. Supports both Megatron optimizers
    and `ChainedOptimizer`, which wraps a list of underlying optimizers.

    Args:
        optimizers: The optimizer or ChainedOptimizer instance.
    """

    def _iter_opts(opt):
        if isinstance(opt, ChainedOptimizer):
            return opt.chained_optimizers
        return [opt]

    def offload_tensor_to_cpu(tensor):
        if tensor is None:
            return
        tensor.data = tensor.data.to("cpu", non_blocking=True)

    def offload_group_to_cpu(group):
        if group is None:
            return

        if isinstance(group, list):
            for param_group in group:
                if isinstance(param_group, list):
                    for param in param_group:
                        offload_tensor_to_cpu(param)
                else:
                    offload_tensor_to_cpu(param_group)
        else:
            offload_tensor_to_cpu(group)

    # Offload all parameter groups to CPU for each underlying optimizer

    for _opt in _iter_opts(optimizers):
        if hasattr(_opt, "shard_fp32_from_float16_groups"):
            offload_group_to_cpu(_opt.shard_fp32_from_float16_groups)


@torch.no_grad()
def load_megatron_copy_params(optimizers):
    """
    Load optimizer parameters back to GPU. Handles ChainedOptimizer.

    Args:
        optimizers: Optimizer or ChainedOptimizer instance.
    """

    def _iter_opts(opt):
        if isinstance(opt, ChainedOptimizer):
            return opt.chained_optimizers
        return [opt]

    def load_tensor_to_gpu(tensor):
        if tensor is None:
            return
        device_id = torch.cuda.current_device()
        tensor.data = tensor.data.to(device_id, non_blocking=True)

    def load_group_to_gpu(group):
        if group is None:
            return

        if isinstance(group, list):
            for param_group in group:
                if isinstance(param_group, list):
                    for param in param_group:
                        load_tensor_to_gpu(param)
                else:
                    load_tensor_to_gpu(param_group)
        else:
            load_tensor_to_gpu(group)

    # Load all parameter groups to GPU for each underlying optimizer

    for _opt in _iter_opts(optimizers):
        if hasattr(_opt, "shard_fp32_from_float16_groups"):
            load_group_to_gpu(_opt.shard_fp32_from_float16_groups)


@torch.no_grad()
def offload_megatron_optimizer(optimizers):
    def _iter_opts(opt):
        if isinstance(opt, ChainedOptimizer):
            return opt.chained_optimizers
        return [opt]

    for _opt in _iter_opts(optimizers):
        offload_megatron_copy_params(_opt)
        opt_state_dict_values = _opt.optimizer.state.values()
        for v in opt_state_dict_values:
            if "exp_avg" in v:
                v["exp_avg"] = v["exp_avg"].to("cpu", non_blocking=True)
            if "exp_avg_sq" in v:
                v["exp_avg_sq"] = v["exp_avg_sq"].to("cpu", non_blocking=True)
        gc.collect()
        torch.cuda.empty_cache()


@torch.no_grad()
def load_megatron_optimizer(optimizers):
    def _iter_opts(opt):
        if isinstance(opt, ChainedOptimizer):
            return opt.chained_optimizers
        return [opt]

    for _opt in _iter_opts(optimizers):
        load_megatron_copy_params(_opt)
        # if we are using HybridDeviceOptimizer, we need to only move gpu optimizer state to gpu
        if hasattr(_opt.optimizer, "_move_new_state_to_right_device"):
            _opt.optimizer._move_new_state_to_right_device()
        else:
            opt_state_dict_values = _opt.optimizer.state.values()
            for v in opt_state_dict_values:
                if "exp_avg" in v:
                    v["exp_avg"] = v["exp_avg"].to(torch.cuda.current_device(), non_blocking=True)
                if "exp_avg_sq" in v:
                    v["exp_avg_sq"] = v["exp_avg_sq"].to(torch.cuda.current_device(), non_blocking=True)
        gc.collect()
        torch.cuda.empty_cache()


def preprocess_packed_seqs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pre_process: bool = True,
    sub_seq_lengths: Optional[list[list[int]]] = None,
    fp8_enabled: bool = False,
) -> tuple[torch.Tensor, PackedSeqParams]:
    """
    Preprocess packed sequences.

    Two modes:

    - ``sub_seq_lengths is None`` (default): each row is assumed to hold a
      single sub-sequence whose length is recovered from
      ``attention_mask.sum(dim=-1)``. ``cu_seqlens`` enumerates one entry
      per row. This is the historical SkyRL behavior used by the RL path
      and the existing SFT path without mini-batch packing.
    - ``sub_seq_lengths is not None``: each row may contain multiple
      sub-sequences concatenated end-to-end. ``sub_seq_lengths[r]`` lists
      the per-sub-sequence valid token counts for row ``r``. Tokens
      ``input_ids[r, :sum(sub_seq_lengths[r])]`` are assumed to be the
      concatenated sub-sequences in order; any trailing tokens in the row
      are pad. ``cu_seqlens`` enumerates every sub-sequence across every
      row.

    CP splits sequence into CP*2 chunks, and each GPU gets 2 chunks (GPU0
    gets first and last chunks, GPU1 gets second and second last chunks,
    and so on), this is for load balancing with causal masking.
    See https://github.com/NVIDIA/TransformerEngine/issues/1368
    """
    tp_size = mpu.get_tensor_model_parallel_world_size()
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    align_size = get_packed_seq_align_size(tp_size, cp_size, fp8_enabled=fp8_enabled)

    batch_size = input_ids.shape[0]

    if sub_seq_lengths is not None:
        if len(sub_seq_lengths) != batch_size:
            raise ValueError(f"sub_seq_lengths has {len(sub_seq_lengths)} rows but batch size is {batch_size}")

        # Flatten per-sub-seq lengths into a single 1-D tensor; the i-th
        # entry of the flattened list maps to the i-th cu_seqlens segment.
        flat_seqlens: list[int] = []
        # Per-row, per-sub-seq starting column within the original padded row.
        # We need this to gather sub-seq tokens from the padded input_ids.
        # NOTE: the controller-side collator (``PackedDataCollator``)
        # advances ``row_offset += round_up(length, align_size)`` between
        # consecutive sub-sequences in the same row so that flash-attn varlen
        # sees TP/CP-aligned segment boundaries. We MUST mirror that here —
        # otherwise sub-seq i (for i > 0) would be read starting inside the
        # alignment-pad gap of sub-seq i-1, returning pad tokens.
        row_index_of_subseq: list[int] = []
        intra_row_offset_of_subseq: list[int] = []
        for r, lens in enumerate(sub_seq_lengths):
            running = 0
            for length in lens:
                length_int = int(length)
                flat_seqlens.append(length_int)
                row_index_of_subseq.append(r)
                intra_row_offset_of_subseq.append(running)
                # Pad each sub-seq independently to align_size, matching the
                # collator's row layout.
                pad = (align_size - length_int % align_size) % align_size
                running += length_int + pad

        seqlens_in_batch = torch.tensor(flat_seqlens, dtype=torch.int32, device=input_ids.device)
        num_subseqs = len(flat_seqlens)
    else:
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        num_subseqs = batch_size

    pad_size = (align_size - seqlens_in_batch % align_size) % align_size
    seqlens_in_batch_padded = seqlens_in_batch + pad_size

    cu_seqlens = torch.zeros(num_subseqs + 1, dtype=torch.int32, device=input_ids.device)
    cu_seqlens[1:] = torch.cumsum(seqlens_in_batch, dim=0)
    cu_seqlens_padded = torch.zeros(num_subseqs + 1, dtype=torch.int32, device=input_ids.device)
    cu_seqlens_padded[1:] = torch.cumsum(seqlens_in_batch_padded, dim=0)

    # ----------------------------------------------------------------------------
    # Move the index information needed in the subsequent loop to the CPU at once,
    # to avoid frequent .item() calls in the loop that cause D2H synchronization
    # ----------------------------------------------------------------------------
    seqlens_in_batch_cpu: list[int] = seqlens_in_batch.tolist()
    seqlens_in_batch_padded_cpu: list[int] = seqlens_in_batch_padded.tolist()
    cu_seqlens_padded_cpu: list[int] = cu_seqlens_padded.tolist()

    # Pure Python int calculation to avoid further synchronization
    max_seqlen_in_batch = max(seqlens_in_batch_padded_cpu)

    shape = list(input_ids.shape[1:])
    shape[0] = sum(seqlens_in_batch_padded_cpu) // cp_size
    if pre_process:
        input_ids_rmpad = torch.zeros(shape, dtype=input_ids.dtype, device=input_ids.device)
        for i in range(num_subseqs):
            if sub_seq_lengths is not None:
                row_idx = row_index_of_subseq[i]
                offset = intra_row_offset_of_subseq[i]
                seqlen = seqlens_in_batch_cpu[i]
                seq_tokens = input_ids[row_idx, offset : offset + seqlen]
            else:
                seqlen = seqlens_in_batch_cpu[i]
                seq_tokens = input_ids[i, attention_mask[i]]

            if cp_size <= 1:
                start_idx = cu_seqlens_padded_cpu[i]
                input_ids_rmpad[start_idx : start_idx + seqlen] = seq_tokens
                continue

            seqlen_padded_i = seqlens_in_batch_padded_cpu[i]
            seqlen_cp = seqlen_padded_i // cp_size
            half_seqlen = seqlen_cp // 2
            start_idx = cu_seqlens_padded_cpu[i] // cp_size
            d = seq_tokens
            if d.shape[0] < seqlen_padded_i:
                d = torch.nn.functional.pad(d, (0, seqlen_padded_i - d.shape[0]))
            input_ids_rmpad[start_idx : start_idx + half_seqlen] = d[
                half_seqlen * cp_rank : half_seqlen * (cp_rank + 1)
            ]

            remain_start = seqlen_padded_i - half_seqlen * (cp_rank + 1)
            remain_end = seqlen_padded_i - half_seqlen * cp_rank
            remain_end = min(remain_end, d.shape[0])
            remain_len = remain_end - remain_start
            if remain_len > 0:
                input_ids_rmpad[start_idx + half_seqlen : start_idx + half_seqlen + remain_len] = d[
                    remain_start:remain_end
                ]

    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_padded,
        max_seqlen_q=max_seqlen_in_batch,
        cu_seqlens_kv=cu_seqlens_padded,
        max_seqlen_kv=max_seqlen_in_batch,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
    )
    if pre_process:
        return input_ids_rmpad.unsqueeze(0), packed_seq_params
    else:
        return input_ids, packed_seq_params


def model_packs_sequences_internally(model: Union[nn.Module, List[nn.Module]]) -> bool:
    """Whether the model packs sequences inside its own ``forward``.

    True for ``Qwen3VLModel`` (e.g. Qwen3.5 via the VL bridge), which would
    double-pack and corrupt the GDN ``cu_seqlens`` under SkyRL sample packing, so
    :class:`MegatronModelWrapper` refuses packing for it. Returns ``False`` when
    mbridge / Qwen3VL is not importable, so other models are unaffected.
    """
    try:
        from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import (
            Qwen3VLModel,
        )
    except ImportError:
        return False

    chunks = model if isinstance(model, (list, tuple)) else [model]
    for chunk in chunks:
        unwrapped = unwrap_model(chunk)
        unwrapped_list = unwrapped if isinstance(unwrapped, (list, tuple)) else [unwrapped]
        if any(isinstance(m, Qwen3VLModel) for m in unwrapped_list):
            return True
    return False


def remove_left_padding(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    pre_process: bool = True,
    fp8_enabled: bool = False,
):
    """
    Remove left padding from input_ids, attention_mask and position_ids
    return new_input_ids, new_attention_mask, new_position_ids
    """
    assert attention_mask.ndim == 2
    assert position_ids.ndim == 2
    cp_size = mpu.get_context_parallel_world_size()
    assert cp_size == 1, "Context parallel size without seq_pack is not supported"
    batch_size = input_ids.shape[0]
    shape = list(input_ids.shape)  # batch_size, seq_len,...
    seq_lens = attention_mask.sum(dim=1)
    seq_len = seq_lens.max().item()
    align_size = get_unpacked_seq_align_size(mpu.get_tensor_model_parallel_world_size(), fp8_enabled=fp8_enabled)
    pad_size = (align_size - seq_len % align_size) % align_size
    seq_len = seq_len + pad_size
    shape[1] = seq_len
    if pre_process:
        new_input_ids = torch.zeros(dtype=input_ids.dtype, device=input_ids.device, size=shape)
    new_attention_mask = torch.zeros(
        dtype=attention_mask.dtype, device=attention_mask.device, size=(batch_size, seq_len)
    )
    new_position_ids = torch.zeros(dtype=position_ids.dtype, device=position_ids.device, size=(batch_size, seq_len))
    for i in range(batch_size):
        if pre_process:
            new_input_ids[i, : seq_lens[i]] = input_ids[i, attention_mask[i]]
        new_attention_mask[i, : seq_lens[i]] = attention_mask[i, attention_mask[i]]
        new_position_ids[i, : seq_lens[i]] = position_ids[i, attention_mask[i]]
    if pre_process:
        return new_input_ids, new_attention_mask, new_position_ids
    else:
        return input_ids, new_attention_mask, new_position_ids


def recover_left_padding(
    result,
    attention_mask: torch.Tensor,
    original_attention_mask: torch.Tensor,
    origin_seqlen: int,
    post_process: bool = True,
):
    """
    Recover left padding from result
    return result
    """
    if not post_process:
        return result
    shape = list(result.shape)
    batch_size = shape[0]
    shape[1] = origin_seqlen
    new_result = torch.zeros(dtype=result.dtype, device=result.device, size=shape)
    for i in range(batch_size):
        new_result[i, original_attention_mask[i]] = result[i, attention_mask[i]]
    return new_result


def get_model_config(model):
    return get_attr_wrapped_model(model, "config", allow_none=False)


def broadcast_object_across_pp_ranks(obj):
    """Broadcast an object across pipeline parallel ranks.

    From Nemo-RL: https://github.com/NVIDIA-NeMo/RL/blob/0a769cc3553a265dd1ca4648de0a7d0b1ad5ece6/nemo_rl/models/policy/megatron_policy_worker.py#L136

    This utility function handles broadcasting an object from the rank that owns it
    to all other pipeline parallel ranks. If only one rank has the object (non-None),
    it will be broadcast to all other ranks.

    Args:
        obj: The object to broadcast. Can be None on ranks that don't own it.

    Returns:
        The object on all ranks (either the original or the broadcast copy).

    Raises:
        ValueError: If the object doesn't exist on any pipeline parallel rank.
    """
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    pp_group = mpu.get_pipeline_model_parallel_group()

    if pp_size == 1:
        return obj

    # ------------------------------------------------------------------
    # 1. Gather presence flags from all PP ranks to find the source rank
    # ------------------------------------------------------------------
    has_obj = obj is not None
    obj_flags = [None] * pp_size
    torch.distributed.all_gather_object(obj_flags, has_obj, group=pp_group)

    # ------------------------------------------------------------------
    # 2. Identify the owning rank (the only rank with True flag)
    # ------------------------------------------------------------------
    src_rank = None  # Rank *inside* the PP group
    for rank, flag in enumerate(obj_flags):
        if flag:
            src_rank = rank
            break

    if src_rank is None:
        raise ValueError("Object must exist on at least one PP rank")

    # ------------------------------------------------------------------
    # 3. Broadcast the object from the source rank to all ranks
    # ------------------------------------------------------------------
    # Use broadcast_object_list which is more robust than all_gather_object
    obj_list = [obj]
    pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
    global_src = pp_ranks[src_rank]
    torch.distributed.broadcast_object_list(obj_list, src=global_src, group=pp_group)

    return obj_list[0]


def to_te_attention_mask(attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Convert a 2-D keep-mask to a 4-D padding mask for Transformer Engine attention.

    ``remove_left_padding`` returns a 2-D ``[batch, seq]`` keep-mask where 1 marks a real token.
    Transformer Engine's sliding-window ``get_full_mask`` expects a mask broadcastable to
    ``[batch, 1, q_seq, kv_seq]``; a 2-D mask collides the batch dimension with the sequence
    dimension and fails for ``micro_batch_size > 1``. Return a ``[batch, 1, 1, seq]`` padding mask
    where True marks padding. A ``None`` mask (packed sequences) or an already higher-rank mask is
    returned unchanged.
    """
    if attention_mask is None or attention_mask.dim() != 2:
        return attention_mask
    return (~attention_mask.bool())[:, None, None, :]
