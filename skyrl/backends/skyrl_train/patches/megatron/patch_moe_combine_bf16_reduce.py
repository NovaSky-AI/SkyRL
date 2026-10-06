"""Skip the FP32 upcast before the MoE combine's 2-rank expert-TP reduce-scatter.

``MoEAlltoAllTokenDispatcher.combine_preprocess`` reduce-scatters the expert outputs across the
expert-tensor-parallel group as ``reduce_scatter(hidden.to(self.probs.dtype)).to(hidden.dtype)``.
With an FP32 router (``moe_router_dtype="float32"``, as GLM-5.3-Flash uses) that materializes an
FP32 copy of every token-expert row this rank received: ~19 GiB/GPU at 576k tokens on
GLM-5.3-Flash (TP8, EP32 x ETP2), the allocation that ran that length out of memory.

With exactly two ranks the reduction is a single addition per element, and NCCL's Ring and PAT
kernels compute a bf16 sum in FP32 and round once -- equal to the FP32 sum rounded back to bf16.
Checked bit-for-bit (NCCL 2.29.7, B200) on up to 512M elements spanning five orders of magnitude,
within a node and across nodes; for a 2-rank reduce-scatter NCCL's default picks Ring at every
size from 1 KiB to 1 GiB. NVLS is the exception: it reduces in the NVSwitch, and a forced
``NCCL_ALGO=NVLS`` bf16 reduce-scatter differs from the FP32 path in ~10-20% of elements, by at
most 1 bf16 ulp. So the upcast is skipped only for a 2-rank group and only while ``NCCL_ALGO``
doesn't ask for NVLS; larger groups (several roundings in bf16) keep it too.

The change is one dtype decision, made explicit as ``_tp_reduce_dtype``; ``combine_preprocess``
below is megatron-core's (pinned rev) with only the reduce-scatter's ``.to(...)`` changed, so it
can be diffed against upstream mechanically. It is installed only while megatron-core's own
``combine_preprocess`` still matches ``_PINNED_COMBINE_PREPROCESS``; after a pin bump that changes
it, the patch logs a warning and does nothing rather than replace newer code with a stale copy.
"""

import inspect
import os
import textwrap

import torch
from loguru import logger

_APPLIED = False

# megatron-core's MoEAlltoAllTokenDispatcher.combine_preprocess at the pinned rev, verbatim.
_PINNED_COMBINE_PREPROCESS = '''
def combine_preprocess(self, hidden_states):
    """Prepares hidden states for token combination after expert computations.

    This may involve un-sorting tokens and a Reduce-Scatter in the tensor
    parallel dimension.
    """
    # Unpermutation 2: Unsort tokens by local expert.
    if self.num_local_experts > 1:
        if self.drop_and_pad:
            hidden_states = (
                hidden_states.view(
                    self.num_local_experts,
                    self.tp_size * self.ep_size,
                    self.capacity,
                    *hidden_states.size()[1:],
                )
                .transpose(0, 1)
                .contiguous()
                .flatten(start_dim=0, end_dim=2)
            )
        else:
            hidden_states, _ = sort_chunks_by_idxs(
                hidden_states,
                self.num_global_tokens_per_local_expert.T.ravel(),
                self.restore_output_by_local_experts,
                fused=self.config.moe_permute_fusion,
            )

    if self.tp_size > 1:
        if self.output_splits_tp is None:
            input_split_sizes = None
        else:
            input_split_sizes = self.output_splits_tp.tolist()
        hidden_states = reduce_scatter_to_sequence_parallel_region(
            hidden_states.to(self.probs.dtype),
            group=self.tp_group,
            input_split_sizes=input_split_sizes,
        ).to(hidden_states.dtype)

    return hidden_states
'''


def _nccl_algo_may_use_nvls() -> bool:
    """Whether ``NCCL_ALGO`` asks for NVLS (e.g. ``NVLS`` or ``reducescatter:nvls``, not ``^NVLS``)."""
    algo = os.environ.get("NCCL_ALGO", "").lower()
    return "nvls" in algo and not algo.startswith("^")


def _tp_reduce_dtype(self, hidden_states: torch.Tensor) -> torch.dtype:
    """Dtype for the expert-TP reduce-scatter in ``combine_preprocess``.

    Reducing in the router dtype (e.g. FP32) avoids several bf16 roundings when the group has more
    than two ranks. With exactly two, the reduction is one addition, which NCCL's Ring/PAT kernels
    accumulate in FP32 and round once -- identical to the FP32 sum cast back -- so the upcast only
    costs an FP32 copy of every token-expert row. NVLS rounds differently (<= 1 bf16 ulp), so the
    upcast stays when ``NCCL_ALGO`` asks for it.
    """
    if self.tp_size == 2 and not _nccl_algo_may_use_nvls():
        return hidden_states.dtype
    return self.probs.dtype


def _combine_preprocess(self, hidden_states):
    """``_PINNED_COMBINE_PREPROCESS`` with the reduce-scatter in ``self._tp_reduce_dtype(...)``."""
    from megatron.core.transformer.moe.token_dispatcher import (
        reduce_scatter_to_sequence_parallel_region,
        sort_chunks_by_idxs,
    )

    # Unpermutation 2: Unsort tokens by local expert.
    if self.num_local_experts > 1:
        if self.drop_and_pad:
            hidden_states = (
                hidden_states.view(
                    self.num_local_experts,
                    self.tp_size * self.ep_size,
                    self.capacity,
                    *hidden_states.size()[1:],
                )
                .transpose(0, 1)
                .contiguous()
                .flatten(start_dim=0, end_dim=2)
            )
        else:
            hidden_states, _ = sort_chunks_by_idxs(
                hidden_states,
                self.num_global_tokens_per_local_expert.T.ravel(),
                self.restore_output_by_local_experts,
                fused=self.config.moe_permute_fusion,
            )

    if self.tp_size > 1:
        if self.output_splits_tp is None:
            input_split_sizes = None
        else:
            input_split_sizes = self.output_splits_tp.tolist()
        hidden_states = reduce_scatter_to_sequence_parallel_region(
            hidden_states.to(self._tp_reduce_dtype(hidden_states)),
            group=self.tp_group,
            input_split_sizes=input_split_sizes,
        ).to(hidden_states.dtype)

    return hidden_states


def _normalized_source(fn) -> str:
    return textwrap.dedent(inspect.getsource(fn)).strip()


def patch_moe_combine_bf16_reduce() -> bool:
    """Install ``_tp_reduce_dtype`` and the matching ``combine_preprocess``. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return True
    from megatron.core.transformer.moe.token_dispatcher import (
        MoEAlltoAllTokenDispatcher,
    )

    if hasattr(MoEAlltoAllTokenDispatcher, "_tp_reduce_dtype"):
        logger.warning(
            "megatron-core's MoEAlltoAllTokenDispatcher already has _tp_reduce_dtype; "
            "delete patch_moe_combine_bf16_reduce.py (see patches/megatron/README.md)"
        )
        return False
    if _normalized_source(MoEAlltoAllTokenDispatcher.combine_preprocess) != _PINNED_COMBINE_PREPROCESS.strip():
        logger.warning(
            "megatron-core's MoEAlltoAllTokenDispatcher.combine_preprocess changed since this patch was "
            "written; not applying patch_moe_combine_bf16_reduce (re-sync it or delete it)"
        )
        return False

    MoEAlltoAllTokenDispatcher._tp_reduce_dtype = _tp_reduce_dtype
    MoEAlltoAllTokenDispatcher.combine_preprocess = _combine_preprocess
    _APPLIED = True
    logger.info("MoE combine: 2-rank expert-TP reduce-scatter in the activation dtype (no FP32 upcast)")
    return True
