"""Fused MoE-LoRA under DP attention + expert parallelism: make LoRA decisions from all ranks' tokens.

With the naive DP/EP all-gather (``allgather_reducescatter``), every rank's MoE layer runs its local experts
on the tokens of all DP ranks. vLLM dispatches the per-token LoRA ids alongside the activations
(``lora_ctx.local_token_lora_mapping``). But ``PunicaWrapperGPU`` still takes three things from the
rank's own batch metadata (``token_mapping_meta``):

* ``no_lora_flag``: if this rank's own batch has no LoRA tokens (e.g. an idle DP rank running a dummy
  forward, or a batch of base-model requests), ``add_lora_fused_moe`` returns early. The LoRA delta of
  this rank's experts is then dropped for every other rank's tokens.
* ``active_lora_ids`` / ``num_active_loras``: the kernels only visit the LoRA slots active in the local
  batch, so tokens of another rank whose adapter slot is not active here get no delta.
* ``prepare_tensors`` exits early on an all-no-LoRA batch without writing the mapping buffer, so the stale
  ids of an earlier batch are what gets dispatched to the other ranks for these tokens.

When a dispatched mapping is passed in, this patch gives both MoE-LoRA entry points the global view:
never skip, and visit every slot ``[0, max_loras)`` (``adapter_enabled`` still masks empty slots, and
tokens with id -1 are still skipped by the kernels). The grid is the same ``max_loras + 1`` vLLM uses
when ``specialize_active_lora`` is off. It also always writes the mapping buffer. Without EP nothing
changes.

Opt-in: ``SKYRL_VLLM_MOE_LORA_EP_GLOBAL_META=1`` (applied in every vLLM worker by
``new_inference_worker_wrap``).
"""

import functools
import os

from loguru import logger

ENV = "SKYRL_VLLM_MOE_LORA_EP_GLOBAL_META"
_APPLIED = False


class _GlobalMeta:
    """Proxy of ``LoRAKernelMeta`` whose ``meta_args`` reports every LoRA slot as active."""

    def __init__(self, meta):
        self._meta = meta

    def __getattr__(self, name):
        return getattr(self._meta, name)

    def meta_args(self, token_nums, specialize_active_lora):
        import torch

        args = self._meta.meta_args(token_nums, specialize_active_lora)
        meta = self._meta
        cache = getattr(meta, "_skyrl_ep_global", None)
        if cache is None:
            n = meta.active_lora_ids.numel()  # max_loras + 1
            ids = torch.full_like(meta.active_lora_ids, -1)
            ids[: n - 1] = torch.arange(n - 1, dtype=ids.dtype, device=ids.device)
            cache = (ids, torch.zeros_like(meta.no_lora_flag_cpu), torch.tensor([n], dtype=torch.int32))
            meta._skyrl_ep_global = cache
        ids, no_lora, num_active = cache
        return (args[0], args[1], args[2], args[3], ids, no_lora, num_active)


def _with_global_meta(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        mapping = kwargs.get("token_lora_mapping")
        meta = self.token_mapping_meta
        # Without EP the mapping is None or a view of the local metadata buffer (w13 hands it on to w2).
        if mapping is None or mapping.data_ptr() == meta.token_lora_mapping.data_ptr():
            return fn(self, *args, **kwargs)
        self.token_mapping_meta = _GlobalMeta(meta)
        try:
            return fn(self, *args, **kwargs)
        finally:
            self.token_mapping_meta = meta

    return wrapper


def apply_moe_lora_ep_global_meta_patch() -> bool:
    """Patch PunicaWrapperGPU's MoE-LoRA entry points and LoRAKernelMeta.prepare_tensors. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return True
    if os.environ.get(ENV, "0").lower() not in ("1", "true"):
        return False
    try:
        from vllm.lora.ops.triton_ops.lora_kernel_metadata import LoRAKernelMeta
        from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU
    except ImportError:
        return False
    for name in ("moe_lora_align_block_size", "add_lora_fused_moe"):
        if not hasattr(PunicaWrapperGPU, name):
            logger.warning(f"{ENV}=1 but PunicaWrapperGPU has no {name}; not patching")
            return False
    PunicaWrapperGPU.moe_lora_align_block_size = _with_global_meta(PunicaWrapperGPU.moe_lora_align_block_size)
    PunicaWrapperGPU.add_lora_fused_moe = _with_global_meta(PunicaWrapperGPU.add_lora_fused_moe)

    orig_prepare = LoRAKernelMeta.prepare_tensors

    @functools.wraps(orig_prepare)
    def prepare_tensors(self, token_lora_mapping):
        orig_prepare(self, token_lora_mapping)
        if bool(self.no_lora_flag_cpu[0]):
            # vLLM skips this copy for an all-no-LoRA batch; EP dispatches the buffer to the other ranks.
            self.token_lora_mapping[: token_lora_mapping.size(0)].copy_(token_lora_mapping, non_blocking=True)

    LoRAKernelMeta.prepare_tensors = prepare_tensors
    _APPLIED = True
    logger.info("Fused MoE-LoRA: EP uses the dispatched token->LoRA mapping with every LoRA slot active")
    return True
