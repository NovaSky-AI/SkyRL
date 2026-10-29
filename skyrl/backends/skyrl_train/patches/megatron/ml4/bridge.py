"""Checkpoint <-> Megatron bridge for Mistral Large 4 (``ML4ForCausalLM``).

The HF side of every mapping is the ML4 trainer's module name, which is vLLM's name for the
Pixtral-wrapped text model (``language_model.model.layers.{i}.self_attn.q_proj.weight``).
LoRA adapters exported through this bridge therefore carry the keys vLLM loads, the same
contract as the FSDP trainer (``ml4_train/lora.py``). The Mistral-format, ModelOpt-quantized
checkpoint is read through :class:`~.state_source.ML4CheckpointStateSource`, which renames and
dequantizes it with the FSDP loader's own name map.

Not bridged: the vision tower and MTP head (text-only RL; the loader skips them).
"""

import os
from typing import Any

import torch
import torch.nn.functional as F
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from megatron.bridge.models.hf_pretrained.state import StateDict
from megatron.core.models.gpt.gpt_model import GPTModel

from skyrl.backends.skyrl_train.patches.megatron.ml4.provider import ML4ModelProvider
from skyrl.backends.skyrl_train.patches.megatron.ml4.state_source import (
    ML4CheckpointStateSource,
)

_INDEXER_PARAMS = (
    "q_input_norm.weight",
    "q_input_norm.bias",
    "wq_b.weight",
    "wk_weights_proj.weight",
    "k_norm.weight",
    "k_norm.bias",
    "ssmax.s",
    "ssmax.b",
)


class _BroadcastMapping(ReplicatedMapping):
    """One checkpoint tensor into a replicated parameter of every layer (``lerope.alpha``).

    The Megatron side has the layer wildcard, the checkpoint side has none, so the wildcard
    count check does not apply. Import-only: the model never exports it (LoRA training).
    """

    def _validate_patterns(self):
        pass


def ml4_provider_kwargs(c) -> dict[str, Any]:
    """ML4Config (the checkpoint's ``config.json``, ``configuration_ml4.ML4Config``) -> provider kwargs."""
    n = c.num_hidden_layers
    if c.q_lora_rank is not None:
        raise NotImplementedError("ML4 q-LoRA (q_lora_rank) is not supported")
    if c.scoring_func != "softmax" or (c.n_group or 1) != 1:
        raise NotImplementedError(f"ML4 routing: scoring_func={c.scoring_func}, n_group={c.n_group}")
    if not c.norm_topk_prob:
        raise NotImplementedError("ML4 routing without top-k renormalization")
    moe = [
        1 if (c.n_routed_experts and i >= c.first_k_dense_replace and i % c.moe_layer_freq == 0) else 0
        for i in range(n)
    ]
    return dict(
        num_layers=n,
        hidden_size=c.hidden_size,
        ffn_hidden_size=c.intermediate_size,
        num_attention_heads=c.num_attention_heads,
        kv_channels=c.v_head_dim,
        vocab_size=c.vocab_size,
        seq_length=getattr(c, "seq_length", None) or 8192,
        layernorm_epsilon=c.rms_norm_eps,
        normalization="RMSNorm",
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        add_qkv_bias=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        share_embeddings_and_output_weights=bool(c.tie_word_embeddings),
        make_vocab_size_divisible_by=128,
        bias_activation_fusion=False,
        apply_rope_fusion=False,
        masked_softmax_fusion=False,
        persist_layer_norm=False,
        gradient_accumulation_fusion=False,
        position_embedding_type="none",
        final_logit_softcapping=c.softmax_tanh_gating,
        # MoE
        num_moe_experts=c.n_routed_experts,
        moe_layer_freq=moe,
        moe_ffn_hidden_size=c.moe_intermediate_size,
        moe_shared_expert_intermediate_size=(
            (c.moe_intermediate_size * c.n_shared_experts) if c.n_shared_experts else None
        ),
        moe_router_topk=c.num_experts_per_tok,
        moe_router_score_function="softmax",
        moe_router_pre_softmax=True,
        moe_router_topk_scaling_factor=c.routed_scaling_factor,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_permute_fusion=False,
        moe_router_fusion=False,
        # ML4 attention
        ml4_kv_lora_rank=c.kv_lora_rank,
        ml4_qk_nope_head_dim=c.qk_nope_head_dim,
        ml4_qk_rope_head_dim=c.qk_rope_head_dim,
        ml4_v_head_dim=c.v_head_dim,
        ml4_rope_parameters=dict(c.rope_parameters),
        ml4_lerope=bool(c.lerope),
        ml4_ssmax_scaling=bool(c.ssmax_scaling),
        ml4_nope_layer_flags=list(c.nope_layer_flags),
        ml4_kv_source_layers=list(c.kv_source_layers),
        ml4_index_topk=c.index_topk,
        ml4_index_n_heads=c.index_n_heads,
        ml4_index_head_dim=c.index_head_dim,
        ml4_indexer_norm_eps=c.indexer_norm_eps,
        ml4_indexer_ssmax_scaling=bool(c.indexer_ssmax_scaling),
        ml4_indexer_nope_layer_flags=list(c.indexer_nope_layer_flags),
        ml4_index_sharing=list(c.index_sharing),
        ml4_indexer_fp8_simulation=bool(getattr(c, "indexer_fp8_simulation", True)),
        ml4_fp8_kv_cache=bool(getattr(c, "fp8_kv_cache", False)),
        ml4_router_logits_bf16=bool(getattr(c, "router_logits_bf16", True)),
        ml4_attn_query_chunk=int(getattr(c, "attn_query_chunk", 1024)),
        **_quant_kwargs(c),
    )


def _quant_kwargs(c) -> dict[str, Any]:
    q = getattr(c, "checkpoint_quantization", None) or {}
    if q.get("quant_algo") != "MIXED_PRECISION":
        return {}
    if q.get("quant_method") != "modelopt" or q.get("ignore"):
        raise NotImplementedError(f"ML4 checkpoint quantization {q.get('quant_method')}, ignore={q.get('ignore')}")
    layers = dict(q["quantized_layers"])
    algos = {v["quant_algo"] for v in layers.values()}
    if not algos <= {"NVFP4", "FP8_PB_WO"}:
        raise NotImplementedError(f"ML4 quant algos {algos}")
    nvfp4 = {k for k, v in layers.items() if v["quant_algo"] == "NVFP4"}
    if any(not k.endswith(".experts") for k in nvfp4):
        raise NotImplementedError("NVFP4 only on routed experts")
    for k in nvfp4:
        if layers[k].get("group_size", q.get("group_size")) != 16:
            raise ValueError(f"{k}: NVFP4 group size must be 16")
    out = dict(
        ml4_nvfp4_experts=bool(nvfp4),
        ml4_quantized_layers=layers,
        ml4_nvfp4_act_qat=bool(getattr(c, "nvfp4_act_qat", True)),
        ml4_fp8_act_qat=bool(getattr(c, "fp8_act_qat", True)),
    )
    if nvfp4 and out["ml4_nvfp4_act_qat"]:
        # same hard requirements as ml4_train.qat (_check_rollout_moe_activation, _rollout_shards)
        act = getattr(c, "rollout_moe_activation", None)
        if act != "nvfp4_per_token":
            raise ValueError(
                "nvfp4_act_qat models vLLM's per-token NVFP4 MoE activation; set "
                f"rollout_moe_activation='nvfp4_per_token' in the model config (got {act!r})"
            )
        tp, mtp = getattr(c, "rollout_tp", None), getattr(c, "rollout_moe_tp", None)
        if not isinstance(tp, int) or not isinstance(mtp, int) or tp < 1:
            raise ValueError(f"nvfp4_act_qat needs integer rollout_tp and rollout_moe_tp, got {tp!r}, {mtp!r}")
        if mtp != tp:
            raise ValueError(f"rollout_moe_tp={mtp} must equal rollout_tp={tp}")
        out["ml4_rollout_moe_tp"] = mtp
    return out


@MegatronModelBridge.register_bridge(
    source="ML4ForCausalLM",
    target=GPTModel,
    provider=ML4ModelProvider,
    model_type="mistral_large_4_train",
)
class ML4Bridge(MegatronModelBridge):
    """Megatron Bridge for the ML4 language model (``ML4ForCausalLM``)."""

    MODEL_CONFIG_CLASS = None
    SUPPORTS_HF_PRETRAINED_EXPORT = False

    def provider_bridge(self, hf_pretrained) -> ML4ModelProvider:
        path = getattr(hf_pretrained, "model_name_or_path", None)
        if path and os.path.isdir(str(path)) and getattr(hf_pretrained, "_state_dict_accessor", None) is None:
            hf_pretrained._state_dict_accessor = StateDict(ML4CheckpointStateSource(str(path)))
        return ML4ModelProvider(**ml4_provider_kwargs(hf_pretrained.config))

    # ---- NVFP4 experts: packed tensors loaded directly from the checkpoint ----------------------

    def _megatron_global_param_names_all_pp_ranks(self, megatron_model):
        names = super()._megatron_global_param_names_all_pp_ranks(megatron_model)
        return [n for n in names if ".nvfp4" not in n]

    def load_weights_hf_to_megatron(self, hf_pretrained, megatron_model, allowed_mismatched_params=None):
        out = super().load_weights_hf_to_megatron(hf_pretrained, megatron_model, allowed_mismatched_params)
        load_nvfp4_experts(megatron_model, hf_pretrained.state.source)
        return out

    def mapping_registry(self) -> MegatronMappingRegistry:
        hf = "language_model.model"
        hl = f"{hf}.layers.*"
        ha = f"{hl}.self_attn"
        ml = "decoder.layers.*"
        ma = f"{ml}.self_attention"
        m = [
            AutoMapping("embedding.word_embeddings.weight", f"{hf}.embed_tokens.weight"),
            AutoMapping("output_layer.weight", "language_model.lm_head.weight"),
            ReplicatedMapping("decoder.final_layernorm.weight", f"{hf}.norm.weight"),
            ReplicatedMapping(f"{ml}.input_layernorm.weight", f"{hl}.input_layernorm.weight"),
            ReplicatedMapping(f"{ml}.pre_mlp_layernorm.weight", f"{hl}.post_attention_layernorm.weight"),
            # attention
            ColumnParallelMapping(f"{ma}.linear_q_proj.weight", f"{ha}.q_proj.weight"),
            ReplicatedMapping(f"{ma}.linear_kv_down_proj.weight", f"{ha}.kv_a_proj_with_mqa.weight"),
            ReplicatedMapping(f"{ma}.kv_layernorm.weight", f"{ha}.kv_a_layernorm.weight"),
            ColumnParallelMapping(f"{ma}.linear_kv_up_proj.weight", f"{ha}.kv_b_proj.weight"),
            RowParallelMapping(f"{ma}.linear_proj.weight", f"{ha}.o_proj.weight"),
            ColumnParallelMapping(f"{ma}.ssmax.s", f"{ha}.ssmax.s"),
            ColumnParallelMapping(f"{ma}.ssmax.b", f"{ha}.ssmax.b"),
            _BroadcastMapping(f"{ma}.lerope_alpha", f"{hf}.rotary_emb.lerope_alpha"),
        ]
        m += [ReplicatedMapping(f"{ma}.indexer.{p}", f"{ha}.indexer.{p}") for p in _INDEXER_PARAMS]
        # MLPs: dense (layer 0), shared expert, routed experts; router
        m += [
            GatedMLPMapping(
                f"{ml}.mlp.linear_fc1.weight", gate=f"{hl}.mlp.gate_proj.weight", up=f"{hl}.mlp.up_proj.weight"
            ),
            AutoMapping(f"{ml}.mlp.linear_fc2.weight", f"{hl}.mlp.down_proj.weight"),
            GatedMLPMapping(
                f"{ml}.mlp.shared_experts.linear_fc1.weight",
                gate=f"{hl}.mlp.shared_experts.gate_proj.weight",
                up=f"{hl}.mlp.shared_experts.up_proj.weight",
            ),
            AutoMapping(f"{ml}.mlp.shared_experts.linear_fc2.weight", f"{hl}.mlp.shared_experts.down_proj.weight"),
            GatedMLPMapping(
                f"{ml}.mlp.experts.linear_fc1.weight*",
                gate=f"{hl}.mlp.experts.*.gate_proj.weight",
                up=f"{hl}.mlp.experts.*.up_proj.weight",
            ),
            AutoMapping(f"{ml}.mlp.experts.linear_fc2.weight*", f"{hl}.mlp.experts.*.down_proj.weight"),
            ReplicatedMapping(f"{ml}.mlp.router.weight", f"{hl}.mlp.gate.weight"),
        ]
        return MegatronMappingRegistry(*m)


@torch.no_grad()
def load_nvfp4_experts(megatron_model, source: ML4CheckpointStateSource) -> int:
    """Fill every local NVFP4 expert (packed codes, block scales, weight_scale_2) of this rank from
    the checkpoint, global expert = EP rank * local experts + local index. gate/up share the gate's
    weight_scale_2 (vLLM keeps only w1's for w13; the loader requires w1 == w3)."""
    from megatron.core import parallel_state
    from megatron.core.utils import unwrap_model

    from skyrl.backends.skyrl_train.patches.megatron.ml4.experts import (
        ML4NVFP4GroupedMLP,
    )

    models = megatron_model if isinstance(megatron_model, list) else [megatron_model]
    ep_rank = parallel_state.get_expert_model_parallel_rank()
    n = 0
    for model in unwrap_model(models):
        for name, mod in model.named_modules():
            if not isinstance(mod, ML4NVFP4GroupedMLP):
                continue
            layer = mod.layer_number - 1 if hasattr(mod, "layer_number") else None
            if layer is None:
                layer = _layer_of(model, name)
            fc1 = getattr(mod.linear_fc1, "to_wrap", mod.linear_fc1)
            fc2 = getattr(mod.linear_fc2, "to_wrap", mod.linear_fc2)
            E = mod.num_local_experts
            for le in range(E):
                ge = ep_rank * E + le
                pre = f"language_model.model.layers.{layer}.mlp.experts.{ge}."
                t = source.load_tensors(
                    [
                        pre + f"{p}.{s}"
                        for p in ("gate_proj", "up_proj", "down_proj")
                        for s in ("weight_packed", "weight_scale", "weight_scale_2")
                    ]
                )
                g2, u2 = t[pre + "gate_proj.weight_scale_2"], t[pre + "up_proj.weight_scale_2"]
                if not torch.equal(g2, u2):
                    raise ValueError(f"{pre}: w1/w3 weight_scale_2 differ")
                fc1.nvfp4.packed[le].copy_(
                    torch.cat([t[pre + "gate_proj.weight_packed"], t[pre + "up_proj.weight_packed"]])
                )
                fc1.nvfp4.scale[le].copy_(
                    torch.cat([t[pre + "gate_proj.weight_scale"], t[pre + "up_proj.weight_scale"]])
                )
                fc1.nvfp4_ws2[le] = g2.float()
                fc2.nvfp4.packed[le].copy_(t[pre + "down_proj.weight_packed"])
                fc2.nvfp4.scale[le].copy_(t[pre + "down_proj.weight_scale"])
                fc2.nvfp4_ws2[le] = t[pre + "down_proj.weight_scale_2"].float()
                n += 1
    return n


def _layer_of(model, module_name: str) -> int:
    import re

    i = int(re.search(r"layers\.(\d+)\.", module_name).group(1))
    return model.decoder.layers[i].layer_number - 1
