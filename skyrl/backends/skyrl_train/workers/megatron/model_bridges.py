"""Register megatron-bridge implementations for model architectures not yet
supported upstream.

Import this module at the top of ``megatron_worker.py`` so that bridges are
registered before any ``AutoBridge.from_hf_pretrained`` call.

All registrations are guarded by a top-level ``try/except ImportError`` so that
the rest of the codebase still works in CPU-only (no megatron-bridge) environments.
"""

try:
    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
    from megatron.bridge.models.deepseek.deepseek_v3_bridge import DeepSeekV3Bridge
    from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
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
            # TODO (erictang000): follow up when Megatron-Bridge supports MTP
            # layers for DeepSeek-V3 style models
            provider.mtp_num_layers = None
            return provider

except ImportError:
    pass  # megatron-bridge not installed (e.g. CPU-only environment)
