"""Allow full activation recompute for SkyRL's mHC transformer layer (GLM-5.3-Flash).

megatron-core refuses ``enable_mhc_connections`` together with ``recompute_granularity="full"``
in ``TransformerConfig.__post_init__``. That guard is for megatron-core's own
``HyperConnectionTransformerLayer``, which threads ``CheckpointWithoutOutputManager`` recompute
state through every mHC site. SkyRL's layer (``mcore_ext/mhc_transformer_layer.py``) takes no such
manager -- it raises if handed one -- and nothing else in it depends on the recompute mode. Under
full recompute megatron-core's generic ``checkpointed_forward`` (``megatron/core/recompute.py``)
saves each checkpointed chunk's n-stream input ``[s, b, n * hidden_size]`` and replays
``layer(hidden_states=...)`` in backward; ``TransformerBlock`` expands to and contracts from the
n streams outside that loop, so the checkpointed region sees an ordinary tensor.

Keeping only each layer's input is what lets GLM-5.3-Flash train past ~32k tokens per GPU: the
alternative (selective recompute) keeps every mHC and KDA activation, which grows linearly with
sequence length and has no other lever (KDA has no context-parallel path, and fine-grained
activation offloading has no hooks in the KDA / mHC modules).

``finalize_provider`` bypasses only that one check. It runs the provider's ``finalize()`` twice,
which Megatron-Bridge documents as safe: once with mHC hidden, so every full-recompute validation
still runs, then once with full recompute hidden, so every mHC validation runs and derives the
final state. Full recompute is restored afterwards. ``__post_init__`` derives no fields from the
recompute mode, only validates it, so the second pass leaves the same state a single pass would.

Retire when megatron-core accepts mHC with full recompute (see ``README.md`` in this folder).
"""

from typing import Any

from loguru import logger

_FULL_RECOMPUTE_FIELDS = ("recompute_granularity", "recompute_method", "recompute_num_layers")


def uses_skyrl_mhc_layer(provider: Any) -> bool:
    """Whether ``provider`` builds SkyRL's mHC layer rather than megatron-core's."""
    from skyrl.backends.skyrl_train.patches.megatron.glm5_next.provider import (
        Glm5NextModelProvider,
    )

    return isinstance(provider, Glm5NextModelProvider)


def needs_mhc_full_recompute_bypass(provider: Any) -> bool:
    return (
        bool(getattr(provider, "enable_mhc_connections", False))
        and getattr(provider, "recompute_granularity", None) == "full"
        and uses_skyrl_mhc_layer(provider)
    )


def finalize_provider(provider: Any) -> None:
    """``provider.finalize()``, accepting full recompute for SkyRL's mHC layer."""
    if not needs_mhc_full_recompute_bypass(provider):
        provider.finalize()
        return

    recompute = {k: getattr(provider, k) for k in _FULL_RECOMPUTE_FIELDS}

    # Pass 1: full-recompute validations (method, num_layers, CPU offloading, CUDA graphs, ...).
    provider.enable_mhc_connections = False
    try:
        provider.finalize()
    finally:
        provider.enable_mhc_connections = True

    # Pass 2: mHC validations, with the one guard against full recompute out of the way.
    for k in _FULL_RECOMPUTE_FIELDS:
        setattr(provider, k, None)
    try:
        provider.finalize()
    finally:
        for k, v in recompute.items():
            setattr(provider, k, v)

    logger.info(
        "mHC + full activation recompute enabled for SkyRL's HyperConnectionTransformerLayer "
        f"(recompute_method={recompute['recompute_method']}, "
        f"recompute_num_layers={recompute['recompute_num_layers']})"
    )
