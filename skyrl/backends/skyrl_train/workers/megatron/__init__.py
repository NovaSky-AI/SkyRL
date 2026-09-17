"""Megatron backend workers.

The package import runs :func:`patch_fa4_cute_import` first: with the pinned flash-attn and
the cutlass DSL that vLLM >= 0.28.1 pulls in, ``import megatron.core`` otherwise aborts on a
broken FlashAttention-4 probe (see the patch module). Every Megatron entry point in SkyRL
lives under this package, so this is the earliest common place to apply it.
"""

from skyrl.backends.skyrl_train.patches.megatron.patch_fa4_cute_import import (
    patch_fa4_cute_import,
)

patch_fa4_cute_import()
