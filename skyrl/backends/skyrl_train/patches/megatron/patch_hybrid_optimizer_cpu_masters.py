"""Fully CPU-offloaded optimizer without FP32 master weights on the GPU.

With ``optimizer_cpu_offload=True`` megatron-core wraps the optimizer in ``HybridDeviceOptimizer``
(HDO), but by default the ``DistributedOptimizer`` still builds its FP32 main-param shards on the
GPU first (``_build_model_and_main_param_groups``: ``shard_model_param.clone().float()`` on the
param's device) and hands those to the HDO, which copies them to the host and keeps the GPU shards
as the "GPU params" it copies the updated values back into. So the GPU holds 4 bytes per locally
owned parameter for the whole run, and every step adds a transient FP32 copy of the grads
(``_copy_model_grads_to_main_grads``: ``shard_main_param.grad = shard_model_grad.float()``). At
expert-DP 1 (EP == world size) a rank owns every one of its local experts' parameters: on
Qwen3.8-Flash-Next at EP16 that is ~7.9B parameters, ~31 GB of GPU FP32 masters next to 25 GB of
BF16 weights and 25 GB of BF16 grads, which OOMs at init.

megatron-core already has the right data flow, used when ``use_precision_aware_optimizer`` is set
together with CPU offload (``OptimizerConfig.use_precision_aware_optimizer_no_fp8_or_ds_fp8``):
the DistributedOptimizer creates no main params, gives the HDO the BF16 *shards of the param
buffer* (views, no copy) and the BF16 grad-buffer shards as ``decoupled_grad`` (views, no copy);
the HDO owns the FP32 masters on the host (``param_to_fp32_param``), copies grads D2H inside its
step, and writes the updated values straight back into the param-buffer shards. Grad norm and
clipping run on the GPU over the BF16 grad views. Three pieces make it the default for a fully
offloaded optimizer:

1. :func:`route_main_params_to_hybrid_optimizer` sets that derived flag on the run's
   ``OptimizerConfig`` when the optimizer is distributed and fully offloaded
   (``optimizer_offload_fraction >= 1``). Only the derived flag: the user-facing
   ``use_precision_aware_optimizer`` (TE FusedAdam dtype knobs) is not used by the HDO path.
2. ``HybridDeviceOptimizer._get_sub_optimizer_param_groups`` builds each offloaded master directly
   as one (pinned) host FP32 tensor. Upstream does ``param.detach().clone().cpu().pin_memory()``
   and then ``.float()`` + ``.pin_memory()``: a GPU clone, a pinned BF16 copy that is dropped into
   the pinned-host cache (never returned to the OS), a pageable FP32 copy and the pinned FP32
   copy. Here it is one pinned allocation filled by a chunked D2H copy (the BF16 -> FP32 cast
   needs a GPU temporary of at most one chunk). Maps and grouping are otherwise unchanged.
3. ``get_megatron_optimizer`` first builds an HDO over the *full model params* and the
   DistributedOptimizer then replaces it with one over its shards. The first one's host copies
   (every model param, BF16 pinned + FP32 pinned) are thrown away, but their pinned blocks stay in
   PyTorch's host cache. Its sub-optimizer construction is deferred while the distributed
   optimizer is being built, and completed afterwards for any HDO still in use un-initialized
   (e.g. the non-distributed ``Float16OptimizerWithFloat16Params`` path).

(2) and (3) only change where host copies live; (1) changes the main-param ownership. SkyRL
applies (1) for full fine-tuning only (``megatron_worker.init_model``): the multi-adapter store
(``adapter_store.py``) snapshots ``shard_fp32_from_float16_groups``, which are ``None`` here.

Not covered: megatron-core's legacy (non-dist-ckpt) ``load_state_dict`` path and the HDO's
``dummy_step`` allocate their placeholders on the GPU, as they do upstream.

Upstream: the HDO path is megatron-core's own; no PR needed for (1). (2) and (3) are worth
proposing to NVIDIA/Megatron-LM (``cpu_offloading/hybrid_optimizer.py`` and
``optimizer/__init__.py``).
"""

import functools
import threading
from typing import Any, Dict, List, Tuple

import torch
from loguru import logger

_APPLIED = False
# Elements per D2H copy when building a host master: bounds the GPU temporary of the BF16 -> FP32
# cast (64 Mi elements = 256 MiB of FP32).
_COPY_CHUNK_NUMEL = 64 * 1024 * 1024

_state = threading.local()


def optimizer_fully_offloaded(config: Any) -> bool:
    """Whether ``config`` puts the whole distributed optimizer (state and step) on the host."""
    return (
        bool(getattr(config, "optimizer_cpu_offload", False))
        and float(getattr(config, "optimizer_offload_fraction", 0.0) or 0.0) >= 1.0
        and bool(getattr(config, "use_distributed_optimizer", False))
    )


def route_main_params_to_hybrid_optimizer(config: Any) -> bool:
    """Let the HybridDeviceOptimizer own the FP32 masters on the host; build none on the GPU.

    Returns True when the config was changed. No-op unless the distributed optimizer is fully
    offloaded, or when the flag is already set (``use_precision_aware_optimizer=True``).
    """
    if not optimizer_fully_offloaded(config):
        return False
    if getattr(config, "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False):
        return False
    config.use_precision_aware_optimizer_no_fp8_or_ds_fp8 = True
    return True


def host_copy(src: torch.Tensor, dtype: torch.dtype, pin: bool) -> torch.Tensor:
    """A host tensor of ``dtype`` with ``src``'s values, allocated once (pinned if ``pin``).

    Copies in chunks so a cast on the way (BF16 -> FP32) needs at most one chunk of device
    temporary. Pinning needs an accelerator; without one the copy is pageable.
    """
    dst = torch.empty(tuple(src.shape), dtype=dtype, device="cpu", pin_memory=pin and torch.cuda.is_available())
    flat_dst = dst.view(-1)
    flat_src = src.detach().reshape(-1)
    for start in range(0, flat_src.numel(), _COPY_CHUNK_NUMEL):
        end = min(start + _COPY_CHUNK_NUMEL, flat_src.numel())
        flat_dst[start:end].copy_(flat_src[start:end])
    return dst


def _get_sub_optimizer_param_groups(self, offload_fraction: float) -> Tuple[List, List, Dict, Dict, Dict]:
    """``HybridDeviceOptimizer._get_sub_optimizer_param_groups`` with one host copy per master.

    Same selection (params in order until ``offload_fraction`` of the GPU numel is offloaded),
    same maps and same pinning as upstream: an FP32-converted host master is pinned iff
    ``pin_cpu_params``; an already-FP32 offloaded param is always pinned.
    """
    params = [param for group in self.param_groups for param in group["params"]]
    gpu_params_total_numel = sum(param.numel() for param in params if param.is_cuda)
    offload_threshold = gpu_params_total_numel * offload_fraction
    offload_params_numel = 0
    cpu_param_groups, gpu_param_groups = [], []
    gpu_params_map_cpu_copy: Dict[torch.Tensor, torch.Tensor] = {}
    cpu_copys_map_gpu_param: Dict[torch.Tensor, torch.Tensor] = {}
    param_to_fp32_param: Dict[torch.Tensor, torch.Tensor] = {}
    for group in self.param_groups:
        gpu_group = group.copy()
        cpu_group = group.copy()
        gpu_group["params"] = []
        cpu_group["params"] = []
        for orig_param in group["params"]:
            needs_fp32 = self.param_update_in_fp32 and orig_param.dtype != torch.float32
            if offload_params_numel < offload_threshold and orig_param.is_cuda:
                inner = host_copy(
                    orig_param,
                    torch.float32 if needs_fp32 else orig_param.dtype,
                    pin=self.pin_cpu_params if needs_fp32 else True,
                )
                offload_params_numel += orig_param.numel()
                gpu_params_map_cpu_copy[orig_param] = inner
                cpu_copys_map_gpu_param[inner] = orig_param
                if needs_fp32:
                    param_to_fp32_param[orig_param] = inner
            elif needs_fp32:
                inner = orig_param.detach().clone().float()
                param_to_fp32_param[orig_param] = inner
            else:
                inner = orig_param
            (gpu_group if inner.is_cuda else cpu_group)["params"].append(inner)
        if gpu_group["params"]:
            gpu_param_groups.append(gpu_group)
        if cpu_group["params"]:
            cpu_param_groups.append(cpu_group)
    return cpu_param_groups, gpu_param_groups, gpu_params_map_cpu_copy, cpu_copys_map_gpu_param, param_to_fp32_param


def complete_deferred_init(optimizer: Any) -> int:
    """Build the sub-optimizers of every HDO under ``optimizer`` whose construction was deferred."""
    completed = 0
    seen = set()
    stack = [optimizer]
    previous = getattr(_state, "defer", False)
    _state.defer = False
    try:
        while stack:
            opt = stack.pop()
            if opt is None or id(opt) in seen:
                continue
            seen.add(id(opt))
            if getattr(opt, "_skyrl_init_deferred", False):
                opt._init_sub_optimizers()
                completed += 1
            stack.extend(getattr(opt, "chained_optimizers", None) or [])
            inner = getattr(opt, "optimizer", None)
            if inner is not None and inner is not opt:
                stack.append(inner)
    finally:
        _state.defer = previous
    return completed


def apply_hybrid_optimizer_cpu_masters_patch() -> bool:
    """Patch the HDO's host-master construction and defer the throwaway HDO. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return True
    import megatron.core.optimizer as optimizer_module
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import (
        HybridDeviceOptimizer,
    )
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    init_sub_optimizers = HybridDeviceOptimizer._init_sub_optimizers
    HybridDeviceOptimizer._get_sub_optimizer_param_groups = _get_sub_optimizer_param_groups

    @functools.wraps(init_sub_optimizers)
    def _init_sub_optimizers(self):
        if getattr(_state, "defer", False):
            # Construction inside get_megatron_optimizer, before DistributedOptimizer replaces it.
            self._skyrl_init_deferred = True
            return
        self._skyrl_init_deferred = False
        init_sub_optimizers(self)

    HybridDeviceOptimizer._init_sub_optimizers = _init_sub_optimizers

    build = optimizer_module._get_megatron_optimizer_based_on_param_groups

    @functools.wraps(build)
    def _get_megatron_optimizer_based_on_param_groups(config, *args, **kwargs):
        if not (
            getattr(config, "optimizer_cpu_offload", False) and getattr(config, "use_distributed_optimizer", False)
        ):
            return build(config, *args, **kwargs)
        previous = getattr(_state, "defer", False)
        _state.defer = True
        try:
            result = build(config, *args, **kwargs)
        finally:
            _state.defer = previous
        complete_deferred_init(result[0] if isinstance(result, tuple) else result)
        return result

    optimizer_module._get_megatron_optimizer_based_on_param_groups = _get_megatron_optimizer_based_on_param_groups

    dist_init = DistributedOptimizer.__init__

    @functools.wraps(dist_init)
    def __init__(self, *args, **kwargs):
        # The HDO the distributed optimizer builds over its shards is the one that is kept.
        previous = getattr(_state, "defer", False)
        _state.defer = False
        try:
            dist_init(self, *args, **kwargs)
        finally:
            _state.defer = previous

    DistributedOptimizer.__init__ = __init__
    _APPLIED = True
    logger.info(
        "Megatron optimizer CPU offload: HybridDeviceOptimizer builds one host copy per master "
        "and the pre-DistributedOptimizer HDO is no longer materialized"
    )
    return True
