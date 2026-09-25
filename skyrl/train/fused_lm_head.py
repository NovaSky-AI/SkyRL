from enum import StrEnum


class FusedLmHeadBackend(StrEnum):
    TORCH = "torch"
    TRITON = "triton"
    TRITON_BLOCK_SPARSE = "triton_block_sparse"


FUSED_LM_HEAD_BACKENDS = tuple(backend.value for backend in FusedLmHeadBackend)
