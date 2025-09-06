# Utils ported from Verl
# https://github.com/volcengine/verl/blob/e1603dc97f3c20c58feed1f5be34acd5c72a830c/verl/utils/megatron_utils.py#L4
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
import torch
import torch.nn as nn
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.transformer.module import Float16Module
from megatron.core.optimizer import ChainedOptimizer
from megatron.core import parallel_state as mpu
from megatron.core.utils import get_attr_wrapped_model

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
    print(f"{name} contains {n_params:.2f}{scale} parameters")


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


def freeze_moe_router(model):
    for layer in model.decoder.layers:
        if hasattr(layer.mlp, "router"):
            if hasattr(layer.mlp.router, "weight"):
                layer.mlp.router.weight.requires_grad = False
            if hasattr(layer.mlp.router, "bias"):
                layer.mlp.router.bias.requires_grad = False
        if hasattr(layer.mlp, "shared_experts"):
            if hasattr(layer.mlp.shared_experts, "gate_weight"):
                layer.mlp.shared_experts.gate_weight.requires_grad = False
            if hasattr(layer.mlp.shared_experts, "gate_bias"):
                layer.mlp.shared_experts.gate_bias.requires_grad = False


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
            model_chunk_all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
            for buffers in model_chunk_all_buffers:
                for buffer in buffers:
                    # offload parameters
                    if buffer.param_data.storage().size() > 0:
                        buffer.param_data.cpu_data = buffer.param_data.data.cpu().pin_memory()
                        buffer.param_data_size = buffer.param_data.storage().size()
                        buffer.param_data.storage().resize_(0)

                    assert buffer.param_data_size == buffer.param_data.cpu_data.storage().size()

                    if buffer.grad_data.storage().size() > 0:
                        # if the grad_data size is already zero, we assume that it is already offloaded
                        buffer.grad_data_size = buffer.grad_data.storage().size()
                        buffer.grad_data.storage().resize_(0)
        else:
            # we need this for ref module
            for _, param in model_chunk.named_parameters():
                param.data = param.data.to("cpu", non_blocking=True)
                if param.grad is not None:
                    param.grad = param.grad.to("cpu", non_blocking=True)
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def load_megatron_model_to_gpu(models, load_grad=True):
    for model_chunk in models:
        if isinstance(model_chunk, DDP):
            model_chunk_all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
            for buffers in model_chunk_all_buffers:
                for buffer in buffers:
                    # sometimes, we don't want to load grad for pure inference
                    if load_grad:
                        buffer.grad_data.storage().resize_(buffer.grad_data_size)
                        buffer.grad_data.zero_()

                    if buffer.param_data.storage().size() == 0:
                        buffer.param_data.storage().resize_(buffer.param_data_size)
                        # copy data from cpu to cuda
                        buffer.param_data.copy_(buffer.param_data.cpu_data, non_blocking=True)
        else:
            # we need this for ref module
            device_id = torch.cuda.current_device()
            for _, param in model_chunk.named_parameters():
                param.data = param.data.to(device_id, non_blocking=True)
                if param.grad is not None:
                    param.grad = param.grad.to(device_id, non_blocking=True)
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


def remove_left_padding(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    sequence_parallel: bool = False,
    pre_process: bool = True,
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
    if sequence_parallel:
        sp_world_size = mpu.get_tensor_model_parallel_world_size()
        pad_size = (sp_world_size - seq_len % sp_world_size) % sp_world_size
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
