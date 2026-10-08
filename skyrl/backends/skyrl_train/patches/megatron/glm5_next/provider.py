"""Model provider for GLM-5.3-Flash (``glm5_next``)."""

from dataclasses import dataclass
from typing import Callable, ClassVar, Optional, Union

from loguru import logger
from megatron.bridge.models.mla_provider import MLAModelProvider
from megatron.core.transformer.spec_utils import ModuleSpec

from skyrl.backends.skyrl_train.patches.megatron.glm5_next.layer_specs import (
    build_glm5_next_layer_spec,
)


@dataclass
class Glm5NextModelProvider(MLAModelProvider):
    """Megatron configuration and provider for GLM-5.3-Flash.

    GLM-5.3-Flash is a hybrid of KDA linear-attention layers and NoPE-MLA DeepSeek-sparse-
    attention (DSA) layers, sigmoid-routed MoE with a shared expert (dense MLP in the first
    layer), clamped SwiGLU, and Manifold-Constrained Hyper-Connections (mHC) on every block.

    The attention pattern uses the generic ``linear_attention_freq`` list (1 = KDA, 0 = DSA) and
    the KDA geometry the generic ``linear_*`` fields, so the model composes from the same
    configuration surface as the other hybrid models. The mHC field names mirror the
    ``TransformerConfig`` fields of Megatron-LM ``main`` so the backported
    ``HyperConnectionModule`` reads them unchanged.
    """

    transformer_layer_spec: Union[ModuleSpec, Callable] = build_glm5_next_layer_spec

    # SkyRL's mHC layer (mcore_ext/mhc_transformer_layer.py) supports full activation recompute;
    # see ``finalize``. MegatronWorker.init_configs downgrades full recompute only for mHC
    # providers without this flag.
    supports_mhc_full_recompute: ClassVar[bool] = True

    # Manifold-Constrained Hyper-Connections.
    enable_mhc_connections: bool = True
    mhc_num_residual_streams: int = 4
    mhc_sinkhorn_iterations: int = 20
    mhc_init_gating_factor: float = 0.01
    use_fused_mhc: bool = False
    mhc_fused_backend: str = "auto"
    # GLM normalizes the flattened streams with a standard RMSNorm (``rsqrt(mean(x^2) + eps)``,
    # eps = ``rms_norm_eps``) before the mHC mapping; megatron-core's ``HyperConnectionModule``
    # uses ``1 / (rms(x) + 1e-6)``. The residual streams of this model are small enough that the
    # placement of the epsilon changes the mixing weights, so both knobs are explicit here;
    # ``mhc_norm_eps_inside_sqrt`` selects mcore_ext's RMSNorm-input subclass.
    mhc_norm_eps: float = 1e-5
    mhc_norm_eps_inside_sqrt: bool = True

    # KDA forget gate: ``lower_bound * sigmoid(exp(A_log) * (f + dt_bias))``; ``None`` selects
    # the unbounded ``-exp(A_log) * softplus(f + dt_bias)`` gate.
    kda_gate_lower_bound: Optional[float] = -5.0

    # DSA indexer k-pool compression: the indexer scores groups of ``dsa_indexer_kpool``
    # consecutive keys and selects ``dsa_indexer_topk // dsa_indexer_kpool`` groups (plus the
    # incomplete tail group when ``dsa_indexer_kpool_always_select_tail``). Recorded from the HF
    # config; see ``glm5_next.dsa`` for what the Megatron path currently supports.
    dsa_indexer_kpool: int = 1
    dsa_indexer_kpool_always_select_tail: bool = True

    def finalize(self) -> None:
        """``MLAModelProvider.finalize()``, accepting full activation recompute with mHC.

        megatron-core's ``TransformerConfig.__post_init__`` raises for ``enable_mhc_connections``
        with ``recompute_granularity="full"``. The guard protects megatron-core's own mHC layer,
        which threads recompute managers through every mHC site; SkyRL's
        ``HyperConnectionTransformerLayer`` uses none, so megatron-core's generic
        ``checkpointed_forward`` checkpoints it like any layer. Only that one check is bypassed:
        finalize runs twice (Megatron-Bridge documents repeated finalize as safe), once with mHC
        hidden so every full-recompute validation runs, then once with full recompute hidden so
        every mHC validation runs and derives the final state. ``__post_init__`` derives nothing
        from the recompute mode, so the result matches a single pass. Delete once the
        megatron-core pin includes NVIDIA/Megatron-LM#7521, which removes the guard.
        """
        if not (self.enable_mhc_connections and self.recompute_granularity == "full"):
            super().finalize()
            return
        recompute_fields = ("recompute_granularity", "recompute_method", "recompute_num_layers")
        recompute = {k: getattr(self, k) for k in recompute_fields}

        self.enable_mhc_connections = False  # pass 1: full-recompute validations
        try:
            super().finalize()
        finally:
            self.enable_mhc_connections = True

        for k in recompute_fields:  # pass 2: mHC validations, the one guard out of the way
            setattr(self, k, None)
        try:
            super().finalize()
        finally:
            for k, v in recompute.items():
                setattr(self, k, v)
        logger.info(
            "mHC + full activation recompute enabled for SkyRL's HyperConnectionTransformerLayer "
            f"(recompute_method={recompute['recompute_method']}, "
            f"recompute_num_layers={recompute['recompute_num_layers']})"
        )
