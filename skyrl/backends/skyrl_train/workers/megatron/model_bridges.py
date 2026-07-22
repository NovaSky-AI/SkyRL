"""Register megatron-bridge implementations for model architectures not yet
supported upstream.

Import this module at the top of ``megatron_worker.py`` so that bridges are
registered before any ``AutoBridge.from_hf_pretrained`` call.

All registrations are guarded by a top-level ``try/except ImportError`` so that
the rest of the codebase still works in CPU-only (no megatron-bridge) environments.
"""

try:
    from megatron.bridge.models.conversion.mapping_registry import (
        MegatronMappingRegistry,
    )
    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
    from megatron.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge
    from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
    from megatron.bridge.models.qwen.qwen35_bridge import Qwen35Bridge, Qwen35MoEBridge
    from megatron.core.models.gpt.gpt_model import GPTModel

    @MegatronModelBridge.register_bridge(
        source="Glm4MoeLiteForCausalLM",
        target=GPTModel,
    )
    class GLM47FlashBridge(DeepSeekV3Bridge):
        """Bridge for GLM-4.7-Flash (Glm4MoeLiteForCausalLM).

        GLM-4.7-Flash is architecturally identical to DeepSeek-V3 (MLA + MoE)
        but its HF config differs in rope_scaling format:
        - DeepSeek: rope_scaling has factor/mscale/mscale_all_dim, top-level rope_theta
        - GLM-4.7-Flash: rope_scaling has rope_theta/rope_type, no mscale fields

        We reuse DeepSeekV3Bridge.provider_bridge() (which sets all critical
        TP/MoE/MLA provider attributes) by temporarily normalizing the HF config
        rope fields so the base CONFIG_MAPPING can handle them.
        """

        def build_conversion_tasks(self, hf_pretrained, megatron_model):
            """Filter out None tasks from the base implementation.

            megatron-bridge 0.3.1 build_conversion_tasks returns None entries
            for params with no mapping, but load_weights_hf_to_megatron
            doesn't guard against them.
            """
            tasks = super().build_conversion_tasks(hf_pretrained, megatron_model)
            return [t for t in tasks if t is not None]

        def provider_bridge(self, hf_pretrained: PreTrainedCausalLM):
            hf_config = hf_pretrained.config

            # GLM-4.7-Flash stores rope_theta inside rope_scaling dict and
            # doesn't have factor/mscale/mscale_all_dim.  Normalize to the
            # format DeepSeekV3Bridge (and its CONFIG_MAPPING) expects.
            orig_rope_scaling = hf_config.rope_scaling
            orig_rope_theta = getattr(hf_config, "rope_theta", None)
            rope_theta = orig_rope_scaling.get("rope_theta", 10000.0) if orig_rope_scaling else 10000.0
            hf_config.rope_scaling = None
            hf_config.rope_theta = rope_theta

            try:
                provider = super().provider_bridge(hf_pretrained)
            finally:
                hf_config.rope_scaling = orig_rope_scaling
                if orig_rope_theta is None and hasattr(hf_config, "rope_theta"):
                    delattr(hf_config, "rope_theta")
                else:
                    hf_config.rope_theta = orig_rope_theta

            provider.moe_router_score_function = "sigmoid"
            # NOTE: MTP is now honored. DeepSeekV3Bridge.provider_bridge already sets
            # provider.mtp_num_layers from hf_config.num_nextn_predict_layers, and
            # megatron-bridge's get_common_mapping_list emits the MTP weight mappings
            # (enorm/hnorm/eh_proj/shared_head.* + nextn transformer layers) for HF
            # round-tripping. SkyRL's MegatronWorker.init_configs can still override
            # provider.mtp_num_layers via policy.megatron_config.mtp_num_layers (e.g.
            # set it to 0 to force-disable, or leave None to use the model default).
            return provider

    # Qwen3.5 (language-model-only) -> GPTModel.
    #
    # Qwen3.5 checkpoints dispatch to the VL bridge -> Qwen3VLModel, which packs
    # sequences inside its own forward and breaks under SkyRL sample packing. When
    # only the LM is wanted (language_model_only=True), route to the native
    # GPTModel + GDN thd path instead. The stock text bridges assume a flat text
    # checkpoint (top-level config, hf_prefix="model."); these subclasses adapt
    # them to the unified VL checkpoint (text_config, model.language_model.*),
    # like Qwen35VLBridge but targeting GPTModel.

    class _TextConfigShim:
        """Present ``text_config`` as ``.config`` for the stock text bridges.

        Other attribute access passes through to the real ``hf_pretrained``.
        """

        def __init__(self, hf_pretrained, text_config):
            object.__setattr__(self, "_orig", hf_pretrained)
            object.__setattr__(self, "config", text_config)

        def __getattr__(self, name):
            return getattr(self._orig, name)

    class _Qwen35LMOnlyBridgeMixin:
        """Adapt a stock Qwen3.5 text->GPTModel bridge to a unified VL checkpoint.

        Feeds ``text_config`` into the inherited provider logic; MTP mappings are
        omitted (disabled for training).
        """

        def provider_bridge(self, hf_pretrained):
            hf_config = hf_pretrained.config
            text_config = hf_config.get_text_config()
            # tie_word_embeddings lives on the top-level VL config, not text_config.
            if not hasattr(text_config, "tie_word_embeddings"):
                text_config.tie_word_embeddings = getattr(hf_config, "tie_word_embeddings", False)
            return super().provider_bridge(_TextConfigShim(hf_pretrained, text_config))

    @MegatronModelBridge.register_bridge(
        source="Qwen3_5MoeTextForCausalLM",
        target=GPTModel,
        model_type="qwen3_5_moe_text",
    )
    class Qwen35MoELMBridge(_Qwen35LMOnlyBridgeMixin, Qwen35MoEBridge):
        """MoE Qwen3.5 language model (``model.language_model.*``) -> GPTModel."""

        def mapping_registry(self) -> MegatronMappingRegistry:
            return MegatronMappingRegistry(
                *self._get_moe_lm_mappings(hf_prefix="model.language_model.", megatron_prefix="")
            )

    @MegatronModelBridge.register_bridge(
        source="Qwen3_5TextForCausalLM",
        target=GPTModel,
        model_type="qwen3_5_text",
    )
    class Qwen35DenseLMBridge(_Qwen35LMOnlyBridgeMixin, Qwen35Bridge):
        """Dense Qwen3.5 language model (``model.language_model.*``) -> GPTModel."""

        def mapping_registry(self) -> MegatronMappingRegistry:
            return MegatronMappingRegistry(
                *self._get_dense_lm_mappings(hf_prefix="model.language_model.", megatron_prefix="")
            )

    # VL arch -> sentinel ...ForCausalLM key registered above. The ForCausalLM
    # suffix passes AutoBridge's filter; not being a real transformers class makes
    # dispatch fall back to the string key -> our bridge.
    _QWEN35_LM_SENTINEL = {
        "Qwen3_5MoeForConditionalGeneration": "Qwen3_5MoeTextForCausalLM",
        "Qwen3_5ForConditionalGeneration": "Qwen3_5TextForCausalLM",
    }

    def maybe_force_qwen35_text_bridge(bridge, hf_config) -> bool:
        """Rewrite a Qwen3.5 bridge's ``architectures`` to the text sentinel so it
        dispatches to the GPTModel LM bridge instead of the VL bridge.

        Returns ``True`` if rewritten (caller gates on ``language_model_only``).
        """
        archs = list(getattr(hf_config, "architectures", []) or [])
        sentinel = next((_QWEN35_LM_SENTINEL[a] for a in archs if a in _QWEN35_LM_SENTINEL), None)
        if sentinel is None:
            return False
        bridge.hf_pretrained.config.architectures = [sentinel]
        return True

except ImportError:

    def maybe_force_qwen35_text_bridge(bridge, hf_config) -> bool:  # noqa: D103
        # megatron-bridge not installed (e.g. CPU-only environment): nothing to force.
        return False
