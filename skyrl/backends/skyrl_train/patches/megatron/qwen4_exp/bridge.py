"""Hugging Face <-> Megatron bridge for Qwen3.8-Flash-Next (``Qwen4ExpForConditionalGeneration``).

Only the language model is bridged (``model.language_model.*`` and ``lm_head``). Not bridged:

- the vision tower (``model.visual.*``): stays frozen in the inference engine;
- the MTP head (``mtp.*``): not built for training (``mtp_num_layers=None``), so the rollout
  engine's drafter keeps the base weights;
- the PLE n-gram table and its hashing buffers (``ple.ple_embedding.*``): ~51B frozen parameters,
  loaded rank-locally by ``Qwen4ExpPLE.load_table`` and never exported.

GDN, gated attention and the shared-expert MoE reuse megatron-bridge's Qwen3.5 mappings with
the fused input-layernorm entries removed (Qwen4Exp has no ``input_layernorm`` /
``post_attention_layernorm`` / final ``norm``).
"""

import copy

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.conversion.utils import moe_experts_stored_packed
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.qwen.qwen35_bridge import (
    Qwen35MoEBridge,
    _apply_qwen35_common_config,
    _apply_qwen35_moe_config,
)
from megatron.core.models.gpt.gpt_model import GPTModel

from skyrl.backends.skyrl_train.patches.megatron.qwen4_exp.provider import (
    Qwen4ExpModelProvider,
)

HF_PREFIX = "model.language_model."
_HC = ("hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight")
_PLE = (
    "key_proj.weight",
    "value_proj.weight",
    "norm_key.weight",
    "norm_query.weight",
    "norm_conv.weight",
    "conv1d.weight",
)
_INDEXER = ("index_qk_proj.weight", "q_layernorm.weight", "k_layernorm.weight")
# Qwen3.5 mappings that do not exist in Qwen4Exp (no pre-norms, no final norm).
_DROPPED_MEGATRON_PARAMS = (
    "decoder.final_layernorm.weight",
    "decoder.layers.*.pre_mlp_layernorm.weight",
    "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
    "decoder.layers.*.self_attention.in_proj.layer_norm_weight",
)


@MegatronModelBridge.register_bridge(
    source="Qwen4ExpForConditionalGeneration",
    target=GPTModel,
    provider=Qwen4ExpModelProvider,
    model_type="qwen4_exp",
)
class Qwen4ExpBridge(MegatronModelBridge):
    """Megatron Bridge for the Qwen3.8-Flash-Next language model.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen3.8-Flash-Next")
        >>> provider = bridge.to_megatron_provider()
    """

    def hf_config_to_provider_kwargs(self, hf_config) -> dict:
        """Map the nested text configuration with the common config mappings."""
        return super().hf_config_to_provider_kwargs(getattr(hf_config, "text_config", hf_config))

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Qwen4ExpModelProvider:
        hf_config = hf_pretrained.config
        text = hf_config.text_config
        provider = super().provider_bridge(hf_pretrained)

        # transformers renames the dense-indexed attention layers to ``qwen_sparse_attention`` on
        # load; megatron-bridge's Qwen3.5 helpers only know ``full_attention``.
        qwen35_text = copy.copy(text)
        qwen35_text.layer_types = [
            "full_attention" if t == "qwen_sparse_attention" else t for t in (text.layer_types or [])
        ]
        _apply_qwen35_common_config(provider, qwen35_text)
        _apply_qwen35_moe_config(provider, qwen35_text)
        provider.position_embedding_type = "rope"
        rope = getattr(text, "rope_parameters", None) or {}
        provider.rotary_base = rope.get("rope_theta", provider.rotary_base)
        provider.share_embeddings_and_output_weights = bool(getattr(hf_config, "tie_word_embeddings", False))
        provider.hetereogenous_dist_checkpoint = True
        provider.mtp_num_layers = None
        provider.layernorm_epsilon = text.rms_norm_eps
        # mcore's fused gated norm hardcodes the SiLU gate; Qwen4Exp's GDN gate is sigmoid.
        provider.gdn_gated_output_norm_fusion = False

        # HF falls back to ``hidden_act`` when ``output_gate_type`` is unset.
        gate = getattr(text, "output_gate_type", None) or text.hidden_act
        if gate != "sigmoid":
            raise NotImplementedError(f"Qwen4Exp GDN output gate {gate!r} is not supported (only sigmoid).")
        provider.qwen4_hc_count = text.hc_count
        provider.qwen4_hc_lowrank = text.hc_lowrank
        provider.qwen4_ple_layer_ids = list(text.ple_layer_ids or [])
        provider.qwen4_ple_embed_dim = text.ple_embed_dim
        provider.qwen4_ple_conv_kernel_size = text.ple_conv_kernel_size
        provider.qwen4_ngram_size = text.ngram_size
        provider.qwen4_heads_per_ngram = text.heads_per_ngram
        eos = text.eos_token_id
        provider.qwen4_eos_token_id = eos[0] if isinstance(eos, list) else eos
        provider.qwen4_indexer_n_heads = text.indexer_n_heads
        provider.qwen4_indexer_kv_heads = text.indexer_kv_heads
        provider.qwen4_indexer_head_dim = text.indexer_head_dim
        provider.qwen4_indexer_budget = text.indexer_budget
        provider.qwen4_indexer_compress_ratio = text.indexer_compress_ratio
        provider.qwen4_hf_path = str(hf_pretrained.model_name_or_path)
        if len(provider.qwen4_ple_layer_ids) > 1:
            raise NotImplementedError("Qwen4Exp with more than one PLE layer is not supported yet.")
        if provider.qwen4_ple_layer_ids:
            provider.qwen4_ple_hf_prefix = f"{HF_PREFIX}layers.{provider.qwen4_ple_layer_ids[0] - 1}.ple."
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        experts_packed = moe_experts_stored_packed(getattr(self, "hf_pretrained", None), f"{HF_PREFIX}layers.")
        mappings = [
            m
            for m in Qwen35MoEBridge._get_moe_lm_mappings(
                hf_prefix=HF_PREFIX, megatron_prefix="", experts_packed=experts_packed
            )
            if m.megatron_param not in _DROPPED_MEGATRON_PARAMS
        ]
        # Parallelism is detected by module class name; Qwen3.5 registers only ``GatedDeltaNet``.
        AutoMapping.register_module_type("Qwen4ExpGatedDeltaNet", "column")
        layer, hf_layer = "decoder.layers.*", f"{HF_PREFIX}layers.*"
        for site in ("attn_hyper_connection", "mlp_hyper_connection"):
            for name in (*_HC, "block_inject_weight.weight"):
                mappings.append(ReplicatedMapping(f"{layer}.{site}.{name}", f"{hf_layer}.{site}.{name}"))
        for name in _HC:
            mappings.append(
                ReplicatedMapping(f"decoder.final_layernorm.{name}", f"{HF_PREFIX}hyper_connection_mixer.{name}")
            )
        for name in _PLE:
            mappings.append(ReplicatedMapping(f"{layer}.ple.{name}", f"{hf_layer}.ple.{name}"))
        for name in _INDEXER:
            mappings.append(
                ReplicatedMapping(f"{layer}.self_attention.indexer.{name}", f"{hf_layer}.self_attn.indexer.{name}")
            )
        return MegatronMappingRegistry(*mappings)
