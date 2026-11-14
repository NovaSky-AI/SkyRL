"""Hugging Face <-> Megatron bridge for the Inkling2 language model (``Inkling2ForConditionalGeneration``).

Only the language model (``model.llm.*``) is bridged; the vision tower and the MTP draft of the
checkpoint have no Megatron counterpart and are left untouched.

Routed experts are stored packed, all experts of a layer in one tensor (``w13_weight [E, 2I, H]``
with gate / up rows interleaved, ``w2_weight [E, H, I]``), in BF16 or in the release's NVFP4 /
MXFP8 formats (with ``*_scale`` / ``*_amax`` / ``*_outer_scale_denominator`` companions). Import
dequantizes one expert at a time; export writes BF16 packed tensors.
"""

import math
import re
from contextlib import contextmanager
from typing import Dict, Iterator, Mapping, Optional

import torch
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.utils.common_utils import extract_expert_number_from_param
from megatron.core.models.gpt.gpt_model import GPTModel
from torch import nn

from skyrl.backends.skyrl_train.patches.megatron.inkling2 import configuration
from skyrl.backends.skyrl_train.patches.megatron.inkling2.provider import (
    Inkling2ModelProvider,
)
from skyrl.backends.skyrl_train.patches.megatron.inkling2.quant import (
    dequantize_expert_weight,
)

configuration.register()

HF_PREFIX = "model.llm"
_EXPERT_WEIGHT_RE = re.compile(r"\.mlp\.experts\.w(13|2)_weight$")
_QUANT_SUFFIXES = ("_scale", "_amax", "_outer_scale_denominator")


class PackedExperts:
    """One layer's packed expert tensor (and its scale tensors), dequantized per expert.

    The bridge's import cache keeps the object returned for a grouped mapping for the whole
    load, so this holds no tensors itself: they live in ``cache``, which keeps one layer at a
    time.
    """

    def __init__(self, name: str, state: Mapping[str, torch.Tensor], cache: Dict[str, Dict[str, torch.Tensor]]):
        self.name = name
        self._state = state
        self._cache = cache

    def _tensors(self) -> Dict[str, torch.Tensor]:
        if self.name not in self._cache:
            self._cache.clear()
            tensors = {self.name: self._state[self.name]}
            if tensors[self.name].dtype in (torch.uint8, torch.float8_e4m3fn):
                for suffix in _QUANT_SUFFIXES:
                    try:
                        tensors[self.name + suffix] = self._state[self.name + suffix]
                    except KeyError:
                        pass
            self._cache[self.name] = tensors
        return self._cache[self.name]

    def expert(self, e: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        tensors = self._tensors()
        device = torch.cuda.current_device()
        weight = tensors[self.name]
        sliced = {self.name: weight[e : e + 1].to(device)}
        for suffix in ("_scale", "_amax"):
            if self.name + suffix in tensors:
                sliced[self.name + suffix] = tensors[self.name + suffix][e : e + 1].to(device)
        if self.name + "_outer_scale_denominator" in tensors:
            sliced[self.name + "_outer_scale_denominator"] = tensors[self.name + "_outer_scale_denominator"].to(device)
        return dequantize_expert_weight(self.name, sliced, dtype)[0]


class PackedExpertMapping(AutoMapping):
    """Megatron's per-expert ``linear_fc{1,2}.weight<e>`` <-> HF packed ``w13`` / ``w2``.

    ``w13`` rows are gate / up interleaved in HF and ``[gate; up]`` stacked in Megatron.
    """

    is_grouped_export = True

    def __init__(self, megatron_param: str, hf_param: str, gated: bool):
        super().__init__(megatron_param, hf_param)
        self.allow_hf_name_mismatch = True
        self.gated = gated

    def resolve(self, captures):
        megatron_param, hf_param = self._resolve_names(captures)
        return PackedExpertMapping(megatron_param, hf_param, self.gated)

    @property
    def group_key(self) -> str:
        return self.hf_param

    def hf_to_megatron(self, hf_weights, megatron_module: nn.Module) -> torch.Tensor:
        e = extract_expert_number_from_param(self.megatron_param)
        weight = hf_weights.expert(e) if isinstance(hf_weights, PackedExperts) else hf_weights[e]
        if self.gated:
            weight = torch.cat([weight[0::2], weight[1::2]], dim=0)
        return super().hf_to_megatron(weight, megatron_module)

    def megatron_to_hf(self, megatron_weights: Optional[torch.Tensor], megatron_module: Optional[nn.Module]):
        if megatron_weights is not None and self.gated:
            # [..., 2I, H] (EP gathers may add a leading expert dim) -> rows interleaved.
            gate, up = torch.chunk(megatron_weights, 2, dim=-2)
            megatron_weights = torch.stack([gate, up], dim=-2).reshape(megatron_weights.shape)
        return super().megatron_to_hf(megatron_weights, megatron_module)


class TransposedColumnParallelMapping(ColumnParallelMapping):
    """A 2-D parameter stored transposed in HF (``[W, C]`` vs Megatron's ``[C, W]``), split over
    TP ranks along ``C``."""

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        return super().hf_to_megatron(hf_weights.t().contiguous(), megatron_module)

    def megatron_to_hf(self, megatron_weights, megatron_module):
        return {k: v.t().contiguous() for k, v in super().megatron_to_hf(megatron_weights, megatron_module).items()}


class AttnQueryMapping(ReplicatedMapping):
    """``decoder.layers.<i>.attn_query`` <-> ``attn_res.attn_queries.<i - 1>`` (layer 0 has none)."""

    def resolve(self, captures):
        (layer,) = captures
        return ReplicatedMapping(
            self.megatron_param.replace("*", layer, 1),
            self.hf_param.replace("*", str(int(layer) - 1), 1),
        )


class LayerTypeMapping(ReplicatedMapping):
    """A Megatron parameter whose HF name depends on the layer's attention type (local / global)."""

    def __init__(self, megatron_param: str, hf_local: str, hf_global: str, local_layer_ids):
        super().__init__(megatron_param, hf_local)
        self.hf_local, self.hf_global = hf_local, hf_global
        self.local_layer_ids = set(local_layer_ids)

    def resolve(self, captures):
        (layer,) = captures
        hf = self.hf_local if int(layer) in self.local_layer_ids else self.hf_global
        return ReplicatedMapping(self.megatron_param.replace("*", layer, 1), hf.replace("*", layer, 1))


@MegatronModelBridge.register_bridge(
    source="Inkling2ForConditionalGeneration",
    target=GPTModel,
    provider=Inkling2ModelProvider,
    model_type=configuration.MODEL_TYPE,
)
class Inkling2Bridge(MegatronModelBridge):
    """Megatron Bridge for the Inkling2 language model.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("/path/to/inkling2")
        >>> provider = bridge.to_megatron_provider()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._expert_cache: Dict[str, Dict[str, torch.Tensor]] = {}

    @contextmanager
    def weight_sync_export(self) -> Iterator[None]:
        """Export packed expert weights one expert at a time (``...mlp.experts.<e>.w13_weight``,
        what SkyRL's vLLM model loads) instead of as whole-layer tensors. Entered by
        ``MegatronWeightSource`` for weight sync; checkpoint export keeps the packed tensors.

        Megatron-Bridge's grouped export buffers a layer's experts and stacks them: a layer's ``w13``
        is ~15 GiB in BF16, twice over during the stack. Next to a full fine-tuning trainer's
        params and grads there is no room on the GPU, and Megatron-Bridge falls back to stacking
        on the CPU, layer by layer (a full-model sync took over half an hour).
        """
        # Class-level: Megatron-Bridge builds a new bridge instance for every ``AutoBridge._model_bridge``
        # access and every export, so a flag on ``self`` would never reach the exporting instance.
        Inkling2Bridge._stream_experts_depth += 1
        try:
            yield
        finally:
            Inkling2Bridge._stream_experts_depth -= 1

    _stream_experts_depth = 0

    @property
    def _stream_experts(self) -> bool:
        return Inkling2Bridge._stream_experts_depth > 0

    def _accumulate_grouped_export(
        self, task, converted_weights_dict, model_config, grouped_buffers, hf_state_dict, grouped_sources=None
    ):
        if not (self._stream_experts and isinstance(task.mapping, PackedExpertMapping)):
            return super()._accumulate_grouped_export(
                task, converted_weights_dict, model_config, grouped_buffers, hf_state_dict, grouped_sources
            )
        # Same expert numbering as Megatron-Bridge's: an EP-gathered value has a leading [ep_size]
        # dim, rank i's copy of this local expert being global expert local + i * experts_per_rank.
        ep_size = task.mapping.ep_size
        experts_per_rank = model_config.num_moe_experts // ep_size
        local = extract_expert_number_from_param(task.param_name) % experts_per_rank
        out: Dict[str, torch.Tensor] = {}
        for group_key, value in converted_weights_dict.items():
            base, kind = group_key.rsplit(".", 1)
            if ep_size > 1 and value.ndim > 0 and value.shape[0] == ep_size:
                pieces = [(local + i * experts_per_rank, value[i]) for i in range(ep_size)]
            else:
                pieces = [(local, value)]
            for expert, tensor in pieces:
                name = f"{base}.{expert}.{kind}"
                out[name] = tensor
                if grouped_sources is not None:
                    grouped_sources.setdefault(name, []).append(task.param_name)
        return out

    def hf_config_to_provider_kwargs(self, hf_config) -> dict:
        return super().hf_config_to_provider_kwargs(getattr(hf_config, "text_config", hf_config))

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Inkling2ModelProvider:
        text = hf_pretrained.config.text_config
        provider = super().provider_bridge(hf_pretrained)
        if text.num_attention_heads != text.swa_num_attention_heads or (
            text.num_key_value_heads,
            text.head_dim,
        ) != (text.swa_num_key_value_heads, text.swa_head_dim):
            raise NotImplementedError("Inkling2 local and global layers with different head geometry.")
        if text.n_shared_experts or text.pre_norm_pre_affine or text.final_norm_pre_affine:
            raise NotImplementedError("Inkling2 shared experts and pre-affine norms are not supported.")
        provider.num_layers = text.num_hidden_layers
        provider.hidden_size = text.hidden_size
        provider.num_attention_heads = text.num_attention_heads
        provider.num_query_groups = text.num_key_value_heads
        provider.kv_channels = text.head_dim
        provider.ffn_hidden_size = text.intermediate_size
        provider.vocab_size = text.vocab_size
        provider.unpadded_vocab_size = text.unpadded_vocab_size
        provider.should_pad_vocab = False
        provider.share_embeddings_and_output_weights = False
        provider.layernorm_epsilon = text.rms_norm_eps
        provider.seq_length = text.model_max_length
        provider.num_moe_experts = text.n_routed_experts
        provider.moe_router_topk = text.num_experts_per_tok
        provider.moe_ffn_hidden_size = text.intermediate_size
        provider.moe_router_topk_scaling_factor = text.route_scale
        provider.moe_router_enable_expert_bias = bool(text.use_gate_bias)
        provider.moe_shared_expert_intermediate_size = None
        provider.activation_func_tanh_clamp_scale = text.mlp_gate_softcap
        provider.activation_func_tanh_clamp_scale_linear = text.mlp_up_softcap
        provider.local_layer_ids = list(text.local_layer_ids)
        provider.sliding_window_size = text.sliding_window_size
        provider.d_rel = text.d_rel
        provider.rel_extent = text.rel_extent
        provider.qk_l2_clip_threshold = text.qk_l2_clip_threshold
        provider.rel_l2_clip_threshold = text.rel_l2_clip_threshold
        provider.log_scaling_n_floor = text.log_scaling_n_floor
        provider.log_scaling_alpha = text.log_scaling_alpha
        provider.sconv_kernel_size = text.sconv_kernel_size
        provider.n_layers_per_attn_res_block = text.n_layers_per_attn_res_block
        provider.attn_res_normalize_values = text.attn_res_normalize_values
        provider.final_logit_softcapping = text.final_logit_softcapping
        provider.mtp_num_layers = None
        return provider

    def maybe_modify_loaded_hf_weight(self, hf_param, hf_state_dict: Mapping[str, torch.Tensor]):
        if isinstance(hf_param, str) and _EXPERT_WEIGHT_RE.search(hf_param):
            return PackedExperts(hf_param, hf_state_dict, self._expert_cache)
        return super().maybe_modify_loaded_hf_weight(hf_param, hf_state_dict)

    def _merge_grouped_export_adapter_weights(self, task, converted_weights_dict, adapter_weights, num_moe_experts):
        """Megatron-Bridge adds the LoRA delta, which is in Megatron's ``[gate; up]`` row layout, to
        the already-converted HF tensor. For the interleaved ``w13`` the rows are put back in
        Megatron order for the merge."""
        mapping = task.mapping
        if not (isinstance(mapping, PackedExpertMapping) and mapping.gated):
            return super()._merge_grouped_export_adapter_weights(
                task, converted_weights_dict, adapter_weights, num_moe_experts
            )
        stacked = {k: torch.cat([v[..., 0::2, :], v[..., 1::2, :]], dim=-2) for k, v in converted_weights_dict.items()}
        merged = super()._merge_grouped_export_adapter_weights(task, stacked, adapter_weights, num_moe_experts)
        interleaved = {}
        for k, v in merged.items():
            gate, up = torch.chunk(v, 2, dim=-2)
            interleaved[k] = torch.stack([gate, up], dim=-2).reshape(v.shape)
        return interleaved

    def _build_packed_expert_linear_out_by_base(
        self, megatron_model, base_hf_weight_names, per_expert_linear_out, is_expert
    ):
        """Adapter export: Megatron-Bridge stacks each expert's ``lora_B`` in Megatron's ``[gate; up]``
        row layout; for ``w13`` the rows are interleaved like the base weight they apply to."""
        stacked = super()._build_packed_expert_linear_out_by_base(
            megatron_model, base_hf_weight_names, per_expert_linear_out, is_expert
        )
        for name, tensor in stacked.items():
            if name.endswith(".mlp.experts.w13_weight"):
                gate, up = torch.chunk(tensor, 2, dim=-2)
                stacked[name] = torch.stack([gate, up], dim=-2).reshape(tensor.shape)
        return stacked

    @staticmethod
    def split_for_weight_sync(name: str, tensor: torch.Tensor):
        """Weight sync sends packed expert weights one expert at a time
        (``...mlp.experts.<e>.w13_weight``), which the SkyRL vLLM model loads. A whole layer's
        ``w13`` is 15 GiB in BF16; as one tensor it sets the packed transfer buffer's size, and
        the inference engine cannot spare two of those next to the BF16 model."""
        if _EXPERT_WEIGHT_RE.search(name) is None:
            return [(name, tensor)]
        base, kind = name.rsplit(".", 1)
        return [(f"{base}.{e}.{kind}", tensor[e]) for e in range(tensor.shape[0])]

    @staticmethod
    def weight_sync_numel(name: str, shape) -> int:
        """Largest element count checkpoint tensor ``name`` puts on the weight-sync wire (see
        :meth:`split_for_weight_sync`). NVFP4 stores two values per byte, hence the 2x bound."""
        if _EXPERT_WEIGHT_RE.search(name.removesuffix("_scale").removesuffix("_amax")) is not None:
            return 2 * math.prod(shape[1:])
        return math.prod(shape)

    def load_weights_hf_to_megatron(self, *args, **kwargs):
        try:
            return super().load_weights_hf_to_megatron(*args, **kwargs)
        finally:
            self._expert_cache.clear()

    def mapping_registry(self) -> MegatronMappingRegistry:
        hf_layer = f"{HF_PREFIX}.layers.*"
        hf_attn = f"{hf_layer}.attn"
        layer = "decoder.layers.*"
        attn = f"{layer}.self_attention"
        # A pretrained wrapper on import / full export; the bare config on the adapter-export path.
        config = getattr(self.hf_pretrained, "config", self.hf_pretrained)
        local_ids = getattr(config, "text_config", config).local_layer_ids

        mappings = [
            AutoMapping("embedding.word_embeddings.weight", f"{HF_PREFIX}.embed.weight"),
            AutoMapping("output_layer.weight", f"{HF_PREFIX}.unembed.weight"),
            ReplicatedMapping("embedding.embed_norm.weight", f"{HF_PREFIX}.embed_norm.weight"),
            ReplicatedMapping("decoder.final_layernorm.norm.weight", f"{HF_PREFIX}.norm.weight"),
            ReplicatedMapping("decoder.final_layernorm.output_query", f"{HF_PREFIX}.attn_res.output_query"),
            AttnQueryMapping(f"{layer}.attn_query", f"{HF_PREFIX}.attn_res.attn_queries.*"),
            ReplicatedMapping(f"{layer}.mlp_query", f"{HF_PREFIX}.attn_res.mlp_queries.*"),
            LayerTypeMapping(
                f"{layer}.input_layernorm.weight",
                f"{hf_layer}.attn_norm.weight",
                f"{hf_layer}.q_input_norm.weight",
                local_ids,
            ),
            ReplicatedMapping(f"{layer}.k_input_layernorm.weight", f"{hf_layer}.k_input_norm.weight"),
            ReplicatedMapping(f"{layer}.v_input_layernorm.weight", f"{hf_layer}.v_input_norm.weight"),
            ReplicatedMapping(f"{layer}.pre_mlp_layernorm.weight", f"{hf_layer}.mlp_norm.weight"),
            ReplicatedMapping(f"{layer}.attn_sconv.weight", f"{hf_layer}.attn_sconv.weight"),
            ReplicatedMapping(f"{layer}.mlp_sconv.weight", f"{hf_layer}.mlp_sconv.weight"),
            RowParallelMapping(f"{attn}.linear_proj.weight", f"{hf_attn}.wo_ud.weight"),
            AutoMapping(f"{layer}.mlp.router.weight", f"{hf_layer}.mlp.gate.weight"),
            AutoMapping(f"{layer}.mlp.router.expert_bias", f"{hf_layer}.mlp.gate.bias"),
            PackedExpertMapping(
                f"{layer}.mlp.experts.linear_fc1.weight*", f"{hf_layer}.mlp.experts.w13_weight", gated=True
            ),
            PackedExpertMapping(
                f"{layer}.mlp.experts.linear_fc2.weight*", f"{hf_layer}.mlp.experts.w2_weight", gated=False
            ),
        ]
        # Split over TP ranks by (kv) head: rows of the projections, the per-head clip scales and
        # the per-channel k / v convolution weights.
        for megatron_name, hf_name in (
            ("linear_q.weight", "wq_du.weight"),
            ("linear_k.weight", "wk_dv.weight"),
            ("linear_v.weight", "wv_dv.weight"),
            ("linear_r.weight", "wr_du.weight"),
            ("linear_gate.weight", "gate_attn.weight"),
            ("linear_sink.weight", "attn_sink_proj.weight"),
            ("q_l2_clip_scale", "q_l2_clip.scale"),
            ("rel_l2_clip_scale", "rel_l2_clip.scale"),
            ("k_sconv.weight", "k_sconv.weight"),
            ("v_sconv.weight", "v_sconv.weight"),
            ("k_sconv.weight_proj.weight", "k_sconv.weight_proj"),
            ("v_sconv.weight_proj.weight", "v_sconv.weight_proj"),
        ):
            mappings.append(ColumnParallelMapping(f"{attn}.{megatron_name}", f"{hf_attn}.{hf_name}"))
        for conv in ("k_sconv", "v_sconv"):
            mappings.append(TransposedColumnParallelMapping(f"{attn}.{conv}.static_w", f"{hf_attn}.{conv}.static_w"))
        mappings.append(ReplicatedMapping(f"{attn}.rel_proj", f"{hf_attn}.rel_logits_proj.proj"))
        return MegatronMappingRegistry(*mappings)
