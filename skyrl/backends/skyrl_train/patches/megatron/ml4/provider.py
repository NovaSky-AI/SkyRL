"""Model provider for Mistral Large 4 (``ML4ForCausalLM``)."""

from dataclasses import dataclass, field
from typing import Callable, Optional, Union

from megatron.bridge.models.gemma.modules import extend_instance
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.spec_utils import ModuleSpec

from skyrl.backends.skyrl_train.patches.megatron.ml4.experts import install_fp8_act_qat
from skyrl.backends.skyrl_train.patches.megatron.ml4.layer_specs import (
    build_ml4_layer_spec,
)
from skyrl.backends.skyrl_train.patches.megatron.ml4.modules import ML4OutputLayer


@dataclass
class ML4ModelProvider(GPTModelProvider):
    """Megatron configuration and provider for ML4.

    ML4's attention geometry and per-layer patterns live in ``ml4_*`` fields read by
    :class:`~.modules.ML4Attention`; Megatron's own MLA/DSA fields are unused (the model is a
    ``GPTModel`` with a custom attention module and no position embedding: rotary is applied
    inside the attention). ``final_logit_softcapping`` is the tanh cap of the output logits
    (applied by :class:`~.modules.ML4OutputLayer`; SkyRL's fused lm-head logprob path bypasses the
    output layer and has no cap, so it must stay off for ML4 -- checked in the worker).
    """

    transformer_layer_spec: Union[ModuleSpec, Callable] = build_ml4_layer_spec
    position_embedding_type: str = "none"
    final_logit_softcapping: Optional[float] = 30.0

    ml4_kv_lora_rank: int = 512
    ml4_qk_nope_head_dim: int = 64
    ml4_qk_rope_head_dim: int = 64
    ml4_v_head_dim: int = 128
    ml4_rope_parameters: dict = field(default_factory=dict)
    ml4_lerope: bool = True
    ml4_ssmax_scaling: bool = True
    ml4_nope_layer_flags: list = field(default_factory=list)
    ml4_kv_source_layers: list = field(default_factory=list)
    ml4_index_topk: Optional[int] = 2048
    ml4_index_n_heads: int = 32
    ml4_index_head_dim: int = 128
    ml4_indexer_norm_eps: float = 1e-6
    ml4_indexer_ssmax_scaling: bool = True
    ml4_indexer_nope_layer_flags: list = field(default_factory=list)
    ml4_index_sharing: list = field(default_factory=list)
    ml4_indexer_fp8_simulation: bool = True
    ml4_fp8_kv_cache: bool = False
    ml4_router_logits_bf16: bool = True
    ml4_attn_query_chunk: int = 1024
    # ModelOpt MIXED_PRECISION checkpoint: NVFP4 experts kept packed, FP8_PB_WO linears
    ml4_nvfp4_experts: bool = False
    ml4_quantized_layers: dict = field(default_factory=dict)
    # rollout-matching QAT (ml4_train/qat.py); rollout_moe_tp = vLLM TP of the rollout engines
    ml4_nvfp4_act_qat: bool = True
    ml4_fp8_act_qat: bool = True
    ml4_rollout_moe_tp: int = 1
    ml4_moe_grouped_chunk: int = 16

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> GPTModel:
        model = super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        if self.ml4_fp8_act_qat and self.ml4_quantized_layers:
            # before LoRA wraps the linears: the hook stays on the base linear
            install_fp8_act_qat(model, self)
        if model.post_process and self.final_logit_softcapping:
            extend_instance(model.output_layer, ML4OutputLayer)
        return model
