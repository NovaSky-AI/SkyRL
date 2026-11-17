"""MTP roll under context parallelism: exchange the CP boundary tokens with one all-gather instead of P2P.

megatron-core's MTP roll (``multi_token_prediction.roll_tensor``, packed and unpacked) sends each rank's zigzag-chunk
boundary tokens to its CP neighbours through ``_exchange_cp_roll_boundaries``, one ``batch_isend_irecv`` per call. These
are the only point-to-point ops on the CP communicator, so the first call sets up its P2P connections lazily, which can
deadlock when the CP group spans nodes. This replaces only that helper: one ``all_gather_into_tensor`` of both boundary
vectors, then a lookup. megatron-core's roll, its alignment check and its trailing-segment handling stay unchanged.
Installed when MTP is trained with CP > 1; ``SKYRL_MTP_ROLL_ALLGATHER=0`` keeps megatron-core's.
"""

import os
from typing import Any

from loguru import logger

_ENABLE_ENV = "SKYRL_MTP_ROLL_ALLGATHER"

_APPLIED = False
_ORIGINAL: Any = None


def exchange_cp_roll_boundaries_allgather(first_boundary, second_boundary, cp_group):
    """Drop-in for megatron-core's ``_exchange_cp_roll_boundaries``: the next rank's first and the previous rank's second
    boundary. The last rank keeps its own second boundary (and its gradient); rank 0 gets zeros."""
    import torch

    cp, rank = cp_group.size(), torch.distributed.get_rank(group=cp_group)
    wire = torch.stack((first_boundary, second_boundary)).detach()
    is_bool = wire.dtype == torch.bool  # bool cannot be all-gathered on every backend
    wire = wire.to(torch.uint8) if is_bool else wire
    # Concatenated output: gloo rejects the stacked [cp, 2, ...] form that NCCL also accepts.
    out = wire.new_empty((cp * wire.shape[0], *wire.shape[1:]))
    torch.distributed.all_gather_into_tensor(out, wire.contiguous(), group=cp_group)
    out = out.view(cp, *wire.shape)
    out = out.bool() if is_bool else out
    first = out[rank + 1, 0] if rank < cp - 1 else second_boundary
    second = out[rank - 1, 1] if rank > 0 else torch.zeros_like(second_boundary)
    return first, second


def should_apply_mtp_roll_allgather(mtp_enabled: bool, context_parallel_size: int) -> bool:
    """On iff MTP is trained with CP > 1; ``SKYRL_MTP_ROLL_ALLGATHER`` = 1 / 0 forces it on / off."""
    env = os.environ.get(_ENABLE_ENV)
    if env in ("0", "1"):
        return env == "1"
    return bool(mtp_enabled) and int(context_parallel_size or 1) > 1


def apply_mtp_roll_allgather_patch(mtp_enabled: bool = False, context_parallel_size: int = 1) -> bool:
    """Install before the first MTP forward when ``should_apply_mtp_roll_allgather``; returns True if installed now."""
    global _APPLIED, _ORIGINAL
    if _APPLIED or not should_apply_mtp_roll_allgather(mtp_enabled, context_parallel_size):
        return False
    from megatron.core.transformer import multi_token_prediction as mtp

    if not hasattr(mtp, "_exchange_cp_roll_boundaries"):  # megatron-core before NVIDIA/Megatron-LM#8389
        logger.warning("MTP: this megatron-core has no _exchange_cp_roll_boundaries; the CP roll stays on P2P")
        return False
    _ORIGINAL = mtp._exchange_cp_roll_boundaries
    mtp._exchange_cp_roll_boundaries = exchange_cp_roll_boundaries_allgather
    _APPLIED = True
    logger.info("MTP: CP roll boundaries are exchanged with one all-gather per call instead of P2P")
    return True
