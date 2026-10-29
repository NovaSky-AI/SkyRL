"""HF <-> Megatron bridge for DeepSeek-V4.1 (text only), built on megatron-bridge's ``DeepSeekV4Bridge``.

The checkpoint keeps DeepSeek's inference naming (``layers.N.attn.wq_a.weight`` ...), like V4. Its
``config.json`` nests the text model under ``text_config``; vLLM's ``DeepseekV41Config`` flattens it and is
registered with transformers here, so ``AutoBridge.from_hf_pretrained`` can read it.

Text only: vision / aligner / image embeddings, the gate's ``bias_vl`` and the DSpark MTP layers are not
built or mapped. The Engram tables are frozen buffers loaded by ``engram.py`` and never mapped (vLLM keeps
its own copy), so weight sync skips them.
"""

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.deepseek.deepseek_v4_bridge import (
    DeepSeekV4Bridge,
    _HCAlphaMapping,
    _HCAlphaSecondaryMapping,
)
from megatron.bridge.models.deepseek.deepseek_v4_hybrid_provider import (
    DeepSeekV4HybridModelProvider,
)
from megatron.core.models.hybrid.hybrid_model import HybridModel

from skyrl.backends.skyrl_train.patches.megatron.deepseek_v41.engram import EngramSpec
from skyrl.backends.skyrl_train.patches.megatron.deepseek_v41.layer_specs import (
    deepseek_v41_hybrid_stack_spec,
)
from skyrl.backends.skyrl_train.patches.megatron.deepseek_v41.model import (
    apply_deepseek_v41_patches,
)

# Node-local group the Engram table rows are sharded over.
ENGRAM_GROUP_SIZE = 8


def register_deepseek_v41_config() -> None:
    from transformers import AutoConfig
    from vllm.transformers_utils.configs.deepseek_v41 import DeepseekV41Config

    try:
        AutoConfig.register("deepseek_v41", DeepseekV41Config)
    except ValueError:
        pass  # already registered


register_deepseek_v41_config()
apply_deepseek_v41_patches()


def _layer_roles(hf_config):
    n = int(hf_config.num_hidden_layers)
    ratios = [int(r) for r in hf_config.compress_ratios[:n]]
    kv_sources = set(hf_config.kv_source_layer_ids)
    index_sources = set(hf_config.index_source_layer_ids)
    return n, ratios, kv_sources, index_sources


@MegatronModelBridge.register_bridge(
    source="DeepseekV41ForCausalLM",
    target=HybridModel,
    provider=DeepSeekV4HybridModelProvider,
    model_type="deepseek_v41",
)
class DeepSeekV41Bridge(DeepSeekV4Bridge):
    """DeepSeek-V4.1-Flash language model -> megatron-core ``HybridModel``."""

    def provider_bridge(self, hf_pretrained):
        hf_config = hf_pretrained.config
        _, ratios, kv_sources, index_sources = _layer_roles(hf_config)
        # The V4 provider logic wants V4's ratio symbols (window / CSA) and no hash or MTP layers; the real
        # V4.1 ratios go to ``csa2_compress_ratios``.
        saved = {
            key: getattr(hf_config, key, None)
            for key in ("compress_ratios", "num_hash_layers", "num_nextn_predict_layers")
        }
        hf_config.compress_ratios = [4 if r else 0 for r in ratios]
        hf_config.num_hash_layers = 0
        hf_config.num_nextn_predict_layers = 0
        try:
            provider = super().provider_bridge(hf_pretrained)
        finally:
            for key, value in saved.items():
                setattr(hf_config, key, value)

        provider.hybrid_stack_spec = deepseek_v41_hybrid_stack_spec
        provider.moe_num_hash_layers = 0
        provider.mtp_num_layers = None
        provider.share_embeddings_and_output_weights = False
        provider.layernorm_epsilon = float(hf_config.rms_norm_eps)  # 1e-20
        provider.attention_latent_norm_epsilon = float(hf_config.rms_norm_eps)

        engram = None
        if getattr(hf_config, "engram_layer_ids", None):
            path = str(hf_pretrained.model_name_or_path)
            engram = EngramSpec(
                layer_ids=tuple(hf_config.engram_layer_ids),
                num_embeddings=tuple(hf_config.engram_num_embeddings),
                max_ngram_size=int(hf_config.engram_max_ngram_size),
                vocab_size=int(hf_config.engram_vocab_size),
                n_heads=int(hf_config.engram_n_heads),
                head_dim=int(hf_config.engram_head_dim),
                pad_token_id=int(hf_config.engram_pad_token_id),
                compressed_vocab_size=int(hf_config.engram_compressed_vocab_size),
                tokenizer_path=path,
                hf_path=path,
            )
        # V4.1 attention roles under upstream's names (NVIDIA/Megatron-LM#8224: ``dsv4_version`` + ``csa2_*``), as
        # logical DeepSeek layer ids. Upstream keeps the V4.1 ratios in ``csa_compress_ratios``, which the pinned
        # TransformerConfig limits to 0 / 4 / 128 (the V4 placeholders above build the layer pattern).
        candidate_source = int(getattr(hf_config, "candidate_source_layer_id", -1))
        provider.dsv4_version = "v4.1"
        provider.csa2_compress_ratios = ratios
        provider.csa2_kv_source_layers = sorted(kv_sources)
        provider.csa2_index_source_layers = sorted(index_sources)
        provider.csa2_candidate_source_layer = candidate_source if candidate_source >= 0 else None
        provider.csa2_candidate_topk_blocks = int(getattr(hf_config, "candidate_topk_blocks", 0))
        provider.csa2_candidate_block_size = int(getattr(hf_config, "candidate_block_size", 0))
        # Engram is a separate upstream PR (NVIDIA/Megatron-LM#8225), so it stays apart from the attention roles.
        provider.dsv41_engram = engram
        provider.dsv41_engram_group_size = ENGRAM_GROUP_SIZE
        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        hf_config = self.hf_config
        n, ratios, kv_sources, index_sources = _layer_roles(hf_config)
        engram_layers = set(getattr(hf_config, "engram_layer_ids", None) or [])

        def hc(prefix, ck, kind):
            mappings = [
                ReplicatedMapping(f"{prefix}.mapping_proj.weight", f"{ck}.hc_{kind}_fn"),
                ReplicatedMapping(f"{prefix}.bias", f"{ck}.hc_{kind}_base"),
                _HCAlphaMapping(
                    megatron_pre=f"{prefix}.alpha_pre",
                    megatron_post=f"{prefix}.alpha_post",
                    megatron_res=f"{prefix}.alpha_res",
                    hf_param=f"{ck}.hc_{kind}_scale",
                ),
                _HCAlphaSecondaryMapping(f"{prefix}.alpha_post", f"{ck}.hc_{kind}_scale", 1),
                _HCAlphaSecondaryMapping(f"{prefix}.alpha_res", f"{ck}.hc_{kind}_scale", 2),
            ]
            return mappings

        mappings = [
            AutoMapping("embedding.word_embeddings.weight", "embed.weight"),
            AutoMapping("output_layer.weight", "head.weight"),
            AutoMapping("decoder.final_norm.weight", "norm.weight"),
        ]
        for i in range(n):
            ck = f"layers.{i}"
            wrapper = f"decoder.layers.{2 * i}"
            inner = f"{wrapper}.inner_layer"
            attn = f"{inner}.self_attention"
            core = f"{attn}.core_attention"
            mappings += [
                AutoMapping(f"{inner}.input_layernorm.weight", f"{ck}.attn_norm.weight"),
                AutoMapping(f"{attn}.linear_q_down_proj.weight", f"{ck}.attn.wq_a.weight"),
                AutoMapping(f"{attn}.q_layernorm.weight", f"{ck}.attn.q_norm.weight"),
                AutoMapping(f"{attn}.linear_q_up_proj.weight", f"{ck}.attn.wq_b.weight"),
                AutoMapping(f"{attn}.linear_kv_proj.weight", f"{ck}.attn.wkv.weight"),
                AutoMapping(f"{attn}.kv_layernorm.weight", f"{ck}.attn.kv_norm.weight"),
                ReplicatedMapping(f"{attn}.linear_o_group_proj", f"{ck}.attn.wo_a.weight"),
                AutoMapping(f"{attn}.linear_proj.weight", f"{ck}.attn.wo_b.weight"),
                ColumnParallelMapping(f"{core}.attn_sink", f"{ck}.attn.attn_sink"),
            ]
            if ratios[i] and i in kv_sources:
                mappings += [
                    ReplicatedMapping(f"{core}.compressor.linear_wkv.weight", f"{ck}.attn.compressor.wkv.weight"),
                    ReplicatedMapping(f"{core}.compressor.norm.weight", f"{ck}.attn.compressor.norm.weight"),
                ]
                if ratios[i] > 1:
                    mappings.append(
                        ReplicatedMapping(
                            f"{core}.compressor.linear_wgate.weight", f"{ck}.attn.compressor.wgate.weight"
                        )
                    )
            if ratios[i] and i in index_sources:
                mappings += [
                    ReplicatedMapping(f"{core}.indexer.linear_wq_b.weight", f"{ck}.attn.indexer.wq_b.weight"),
                    ReplicatedMapping(
                        f"{core}.indexer.linear_weights_proj.weight", f"{ck}.attn.indexer.weights_proj.weight"
                    ),
                ]
                if i in kv_sources:
                    mappings += [
                        ReplicatedMapping(f"{core}.indexer.linear_wk.weight", f"{ck}.attn.indexer.wk.weight"),
                        ReplicatedMapping(f"{core}.indexer.k_norm.weight", f"{ck}.attn.indexer.k_norm.weight"),
                    ]
            if i in engram_layers:
                mappings += [
                    ReplicatedMapping(f"{wrapper}.engram.linear_wkv.weight", f"{ck}.engram.wkv.weight"),
                    ReplicatedMapping(f"{wrapper}.engram.q_weight", f"{ck}.engram.q_weight"),
                    ReplicatedMapping(f"{wrapper}.engram.k_weight", f"{ck}.engram.k_weight"),
                ]
            mappings += hc(f"{wrapper}.hyper_connection", ck, "attn")

            wrapper = f"decoder.layers.{2 * i + 1}"
            inner = f"{wrapper}.inner_layer"
            mappings += [
                AutoMapping(f"{inner}.pre_mlp_layernorm.weight", f"{ck}.ffn_norm.weight"),
                AutoMapping(f"{inner}.mlp.router.weight", f"{ck}.ffn.gate.weight"),
                AutoMapping(f"{inner}.mlp.router.expert_bias", f"{ck}.ffn.gate.bias"),
                GatedMLPMapping(
                    megatron_param=f"{inner}.mlp.experts.linear_fc1.weight*",
                    gate=f"{ck}.ffn.experts.*.w1.weight",
                    up=f"{ck}.ffn.experts.*.w3.weight",
                ),
                AutoMapping(f"{inner}.mlp.experts.linear_fc2.weight*", f"{ck}.ffn.experts.*.w2.weight"),
                GatedMLPMapping(
                    megatron_param=f"{inner}.mlp.experts.local_experts.*.linear_fc1.weight",
                    gate=f"{ck}.ffn.experts.*.w1.weight",
                    up=f"{ck}.ffn.experts.*.w3.weight",
                ),
                AutoMapping(f"{inner}.mlp.experts.local_experts.*.linear_fc2.weight", f"{ck}.ffn.experts.*.w2.weight"),
                GatedMLPMapping(
                    megatron_param=f"{inner}.mlp.shared_experts.linear_fc1.weight",
                    gate=f"{ck}.ffn.shared_experts.w1.weight",
                    up=f"{ck}.ffn.shared_experts.w3.weight",
                ),
                AutoMapping(f"{inner}.mlp.shared_experts.linear_fc2.weight", f"{ck}.ffn.shared_experts.w2.weight"),
            ]
            mappings += hc(f"{wrapper}.hyper_connection", ck, "ffn")
        return MegatronMappingRegistry(*mappings)
