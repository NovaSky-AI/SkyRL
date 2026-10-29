# This code is adapted from VERL
# https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py
# The original copyright is reproduced below:
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from collections import OrderedDict
from typing import Union

import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig
from packaging import version
from peft.utils.save_and_load import get_peft_model_state_dict
from torch.distributed import DeviceMesh
from torch.distributed.device_mesh import init_device_mesh

from skyrl.backends.skyrl_train.distributed import host_offload
from skyrl.backends.skyrl_train.utils.page_cache import drop_checkpoint_page_cache
from skyrl.train.config import FSDPConfig

if version.parse(torch.__version__) >= version.parse("2.6"):
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        FSDPModule,
        MixedPrecisionPolicy,
        fully_shard,
    )
elif version.parse(torch.__version__) >= version.parse("2.4"):
    from torch.distributed._composable.fsdp import (
        CPUOffloadPolicy,
        FSDPModule,
        MixedPrecisionPolicy,
        fully_shard,
    )
else:
    fully_shard, MixedPrecisionPolicy, FSDPModule, CPUOffloadPolicy = None, None, None, None


def init_fn(x: torch.nn.Module):
    if torch.distributed.get_rank() != 0:
        x = x.to_empty(device=torch.cuda.current_device(), recurse=False)
        torch.cuda.empty_cache()
    return x


def should_use_meta_init(use_meta_tensor=True, mesh: DeviceMesh = None) -> bool:
    """Return True when this rank should create an empty model on meta device."""
    if not use_meta_tensor:
        return False
    if mesh is None:
        return torch.distributed.get_rank() != 0
    return mesh.get_coordinate()[-1] != 0


def _materialize_meta_buffers(model):
    # Materialize any leftover meta buffers (e.g. non-persistent inv_freq from
    # RotaryEmbedding created via from_config on meta device).  We must NOT call
    # model.to_empty() because that would wipe already-loaded FSDP parameters.
    for module in model.modules():
        for key in list(module._buffers.keys()):
            buf = module._buffers[key]
            if buf is not None and buf.device.type == "meta":
                module._buffers[key] = torch.empty(buf.shape, dtype=buf.dtype, device="cpu")


@torch.no_grad()
def _module_to_cpu(model, empty_cache: bool = True):
    _materialize_meta_buffers(model)
    model.to("cpu", non_blocking=True)
    if empty_cache:
        torch.cuda.empty_cache()


@torch.no_grad()
def offload_fsdp2_model_to_cpu(model, empty_cache: bool = True, frozen_only: bool = False):
    """Copy the model's GPU storages into one exact-size pinned host buffer and free them.

    Replaces ``model.to("cpu", non_blocking=True)``, whose pinned caching allocator costs up
    to 2x and keeps the memory after the backload (see ``host_offload``).
    """
    _materialize_meta_buffers(model)
    host_offload.offload_module(model, frozen_only=frozen_only)
    if empty_cache:
        torch.cuda.empty_cache()


@torch.no_grad()
def load_fsdp2_model_to_gpu(model):
    device = torch.cuda.current_device()
    host_offload.onload_module(model)
    # Tensors that were on CPU before the offload (e.g. materialized meta buffers) move to the
    # GPU here, as ``model.to(device)`` did.
    if any(t.device.type == "cpu" for t in [*model.parameters(), *model.buffers()]):
        model.to(device, non_blocking=True)


@torch.no_grad()
def offload_fsdp_optimizer(optimizer):
    if not optimizer.state:
        return
    host_offload.offload_optimizer(optimizer)


@torch.no_grad()
def load_fsdp_optimizer(optimizer, device_id):
    if not optimizer.state:
        return
    host_offload.onload_optimizer(optimizer)
    # CPU state (e.g. Adam's step) moves to the device, as ``value.to(device_id)`` did.
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and value.device.type == "cpu":
                state[key] = value.to(device_id, non_blocking=True)


def _sync_non_persistent_buffers(model: torch.nn.Module, loaded_sd: dict):
    """Broadcast non-persistent buffers (e.g. inv_freq) from rank 0 to all ranks.

    Non-persistent buffers are excluded from state_dict so they are never loaded
    by the parameter broadcast loop.  On non-rank-0 meta-init they remain on the
    meta device with no data; rank 0 has the correctly computed values.
    """
    for module in model.modules():
        non_persistent = getattr(module, "_non_persistent_buffers_set", set())
        for key in sorted(non_persistent):
            buf = module._buffers.get(key)
            if buf is None:
                continue
            if dist.get_rank() == 0:
                src = buf.detach().cuda()
            else:
                src = torch.empty(buf.shape, dtype=buf.dtype, device="cuda")
            dist.broadcast(src, src=0)
            module._buffers[key] = src.cpu()


def fsdp2_load_full_state_dict(model: torch.nn.Module, full_sd: dict, cpu_offload=None):
    """
    Loads a full state dict (assumed populated on rank 0) into a sharded FSDP2
    model by broadcasting from rank 0 and distributing per-parameter shards.

    Uses PyTorch's `set_model_state_dict` with `full_state_dict=True` +
    `broadcast_from_rank0=True`, which internally does the per-parameter
    broadcast and DTensor distribution that we used to do manually. The
    utility also releases intermediate staging tensors as it goes, so the
    caching allocator can return memory to the device pool after init.

    Args:
        model (`torch.nn.Module`):
            The model to load the state dict into, expected to be on meta
            device on all ranks except rank 0 (or on meta on all ranks and
            full_sd populated on rank 0).
        full_sd (`dict`): The full state dict to load (only rank 0 needs
            real data; non-rank-0 ranks may pass an empty dict).
    """
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        set_model_state_dict,
    )

    set_model_state_dict(
        model,
        full_sd,
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
    )

    # Broadcast non-persistent buffers (e.g. inv_freq from RotaryEmbedding)
    # that are excluded from state_dict.  On non-rank-0 meta-init these are
    # still on the meta device with no data; rank 0 has the correct values.
    _sync_non_persistent_buffers(model, {})

    if cpu_offload:
        # Caller asked for CPU-resident params; the offload path is still
        # broken for FSDP2 but we leave the request explicit so a future fix
        # has an obvious hook.
        _module_to_cpu(model)
    return model


def resolve_sharded_load_fn(reference: str, model_path: str):
    """Resolve ``ModelConfig.sharded_load_fn`` (``<module>.<function>`` in the model's remote code)."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    return get_class_from_dynamic_module(reference, model_path)


@torch.no_grad()
def fsdp2_load_sharded(model: torch.nn.Module, fsdp_kwargs: dict, config, load_fn, model_path: str, device):
    """Shard a meta-initialized model with FSDP2 and let ``load_fn`` fill each rank's shards.

    Tensors that are already materialized before sharding (e.g. LoRA adapters, see
    ``init_meta_lora_adapters``) keep their values: each rank copies its local slice. Everything
    still on meta must be filled by ``load_fn(model, model_path)``.
    """
    from torch.distributed.tensor import DTensor, distribute_tensor

    # References, not copies: fully_shard copies each rank's chunk into new storage and leaves the
    # original tensor alone, and to_empty only swaps the module's tensors. A clone here doubled the
    # host copy of the full (unsharded) adapters on every rank, ~2 x 50 GB per rank for ML4 LoRA r32.
    initialized = {
        name: t.detach()
        for name, t in [*model.named_parameters(), *model.named_buffers()]
        if t is not None and not t.is_meta
    }
    apply_fsdp2(model, fsdp_kwargs, config)
    model.to_empty(device=device)
    tensors = dict(model.named_parameters()) | dict(model.named_buffers())
    for name in list(initialized):
        full = initialized.pop(name)
        t = tensors[name]
        if isinstance(t, DTensor):
            local = distribute_tensor(full.to(t.dtype), t.device_mesh, t.placements, src_data_rank=None).to_local()
            t.to_local().copy_(local)
        else:
            t.copy_(full)
    load_fn(model, model_path)
    # Every local rank has finished reading before the checkpoint's page cache is dropped.
    if dist.is_initialized():
        dist.barrier()
    drop_checkpoint_page_cache(model_path)
    return model


def fsdp2_get_full_state_dict(model: torch.nn.Module, cpu_offload=True, rank0_only=True):
    """
    Get the full state dict from an FSDP2 model using proper PyTorch FSDP2 APIs.
    This function will gather the complete state dict on rank 0 only by default.

    Args:
        model (`torch.nn.Module`): The FSDP2 model to get state dict from
        cpu_offload (`bool`): Whether to offload to CPU
        rank0_only (`bool`): Whether to gather full state dict only on rank 0

    Returns:
        dict: The full state dict (only on rank 0 if rank0_only=True, empty dict on other ranks)
    """
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )

    # All ranks must participate in the collective operation
    options = StateDictOptions(
        full_state_dict=True, cpu_offload=cpu_offload, broadcast_from_rank0=False  # We want to get, not set
    )

    # This must be called on all ranks for the collective operation to work
    state_dict = get_model_state_dict(model, options=options)

    # If rank0_only is True, clear the state_dict on non-rank-0 processes
    if rank0_only and dist.get_rank() != 0:
        # Clear the state dict on non-rank-0 processes to save memory
        state_dict.clear()

    return state_dict


def apply_fsdp2(model, fsdp_kwargs, config: Union[FSDPConfig, DictConfig]):
    """model: AutoModelForCausalLM"""
    assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
    default_transformer_cls_names_to_wrap = getattr(model, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = (
        config.wrap_policy.get("transformer_layer_cls_to_wrap", None)
        if getattr(config, "wrap_policy", {}).get("transformer_layer_cls_to_wrap", None)
        else default_transformer_cls_names_to_wrap
    )

    if isinstance(fsdp_transformer_layer_cls_to_wrap, str):
        fsdp_transformer_layer_cls_to_wrap = [fsdp_transformer_layer_cls_to_wrap]
    elif isinstance(fsdp_transformer_layer_cls_to_wrap, set):
        fsdp_transformer_layer_cls_to_wrap = list(fsdp_transformer_layer_cls_to_wrap)

    assert len(fsdp_transformer_layer_cls_to_wrap) > 0 and fsdp_transformer_layer_cls_to_wrap[0] is not None

    modules = []
    for name, module in model.named_modules():
        if module.__class__.__name__ in fsdp_transformer_layer_cls_to_wrap or (
            isinstance(module, nn.Embedding) and not model.config.tie_word_embeddings
        ):
            modules.append(module)
    # Untied output embedding as its own unit: a checkpointed lm head (chunked token logprobs)
    # is recomputed in backward, after the root has resharded under reshard_after_forward=True.
    get_out = getattr(model, "get_output_embeddings", None)
    lm_head = get_out() if callable(get_out) and not model.config.tie_word_embeddings else None
    if isinstance(lm_head, nn.Module) and all(lm_head is not m for m in modules):
        modules.append(lm_head)

    for idx, module in enumerate(modules):
        fully_shard(module, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)  # fsdp2 will not reshard_after_forward for root module


def fsdp2_clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False, foreach=None):
    """torch.nn.utils.clip_grad_norm_ can't run on cpu parameter DTensor"""
    from torch.nn.utils.clip_grad import _clip_grads_with_norm_, _get_total_norm

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        # prevent generators from being exhausted
        parameters = list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = _get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
    total_norm = total_norm.to(torch.cuda.current_device(), non_blocking=True)
    _clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(
            "cuda", mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh


"""
Adapted from Cruise.
"""

HALF_LIST = [16, "16", "fp16", "float16", torch.float16]
FLOAT_LIST = [32, "32", "fp32", "float32", torch.float32]
BFLOAT_LIST = ["bf16", "bfloat16", torch.bfloat16]


class PrecisionType:
    """Type of precision used.

    >>> PrecisionType.HALF == 16
    True
    >>> PrecisionType.HALF in (16, "16")
    True
    """

    HALF = "16"
    FLOAT = "32"
    FULL = "64"
    BFLOAT = "bf16"
    MIXED = "mixed"

    @staticmethod
    def supported_type(precision: Union[str, int]) -> bool:
        return any(x == precision for x in PrecisionType)

    @staticmethod
    def supported_types() -> list[str]:
        return [x.value for x in PrecisionType]

    @staticmethod
    def is_fp16(precision):
        return precision in HALF_LIST

    @staticmethod
    def is_fp32(precision):
        return precision in FLOAT_LIST

    @staticmethod
    def is_bf16(precision):
        return precision in BFLOAT_LIST

    @staticmethod
    def to_dtype(precision):
        if precision in HALF_LIST:
            return torch.float16
        elif precision in FLOAT_LIST:
            return torch.float32
        elif precision in BFLOAT_LIST:
            return torch.bfloat16
        else:
            raise RuntimeError(f"unexpected precision: {precision}")

    @staticmethod
    def to_str(precision):
        if precision == torch.float16:
            return "fp16"
        elif precision == torch.float32:
            return "fp32"
        elif precision == torch.bfloat16:
            return "bf16"
        else:
            raise RuntimeError(f"unexpected precision: {precision}")


# Reference: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py
def layered_summon_lora_params(fsdp_module, rank0_only: bool = False) -> OrderedDict:
    """Full LoRA params on CPU, gathered one submodule at a time. Collective.

    With ``rank0_only`` the other ranks join the gathers but keep nothing.
    """
    keep = not rank0_only or not dist.is_initialized() or dist.get_rank() == 0

    def __prefix_submodules(module, prefix):
        for name, submodule in module.named_modules():
            if name.startswith(prefix) and "." not in name[len(prefix) :]:
                yield name, submodule

    lora_params = OrderedDict()
    prefix_list = [
        "base_model.model.",
        "base_model.model.model.",
        "base_model.model.model.layers.",
        "base_model.model.model.language_model.layers.",
    ]
    for prefix in prefix_list:
        for name, submodule in __prefix_submodules(fsdp_module, prefix):
            if name.endswith(".model") or name.endswith(".layers"):
                continue
            sub_state_dict = {f"{name}.{k}": v for k, v in submodule.state_dict().items()}
            sub_lora_params = get_peft_model_state_dict(fsdp_module, state_dict=sub_state_dict)
            for param_name, param in sub_lora_params.items():
                full = param.full_tensor() if hasattr(param, "full_tensor") else param
                if keep:
                    lora_params[param_name] = full.detach().cpu()
                del full
            torch.cuda.empty_cache()
    return lora_params


def lora_adapter_config(peft_model) -> dict:
    """The PEFT ``adapter_config`` (JSON-serializable) of a model's default adapter."""
    from dataclasses import asdict

    peft_config = asdict(peft_model.peft_config["default"])
    peft_config["task_type"] = peft_config["task_type"].value
    peft_config["peft_type"] = peft_config["peft_type"].value
    target_modules = peft_config["target_modules"]
    # A str is a regex PEFT full-matches against module names; keep it whole.
    peft_config["target_modules"] = target_modules if isinstance(target_modules, str) else list(target_modules)
    return peft_config


def collect_lora_params(module) -> OrderedDict:
    """
    collect lora params or full params if base model is not ready in vllm
    requires `module` to be a `PeftModel` (or an FSDP2-wrapped one).
    """
    lora_params = get_peft_model_state_dict(module)
    lora_params = {
        name: param.full_tensor().detach().cpu() if hasattr(param, "full_tensor") else param.detach().cpu()
        for name, param in lora_params.items()
    }
    torch.cuda.empty_cache()
    return lora_params
