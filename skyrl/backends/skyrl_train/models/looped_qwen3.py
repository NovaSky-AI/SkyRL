from collections.abc import Iterable

import torch
from torch import nn
from transformers import Qwen3Config
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
from vllm.model_executor.layers.attention.encoder_only_attention import Attention
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import SupportsLoRA
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer, Qwen3Model
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    extract_layer_index,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType

from skyrl.train.looped_lora import (
    LayerExecution,
    build_looped_lora_schedule,
    get_lora_only_executions_by_physical_layer,
)

logger = init_logger(__name__)

try:
    from vllm.model_executor.models.interfaces import LocalArgmaxMixin
except ImportError:

    class LocalArgmaxMixin:  # vLLM < 0.30
        pass


def _apply_lora_delta(layer: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    if not isinstance(layer, BaseLinearLayerWithLoRA):
        raise RuntimeError("Fast looped LoRA requires vLLM LoRA wrapping")

    output = inputs.new_zeros((*inputs.shape[:-1], sum(layer.output_slices)))
    return layer._apply_lora_to_output(inputs, output)


class LoopedLoraQwen3DecoderLayer(Qwen3DecoderLayer):
    def __init__(
        self,
        config: Qwen3Config,
        lora_only_execution_indices: tuple[int, ...],
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        per_layer_sliding_window: int | None = None,
    ) -> None:
        super().__init__(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            per_layer_sliding_window=per_layer_sliding_window,
        )
        if not getattr(config, "is_causal", True):
            raise ValueError("Fast looped LoRA only supports causal Qwen3 models")

        model_prefix = prefix.rsplit(".layers.", 1)[0]
        self.lora_only_attn = nn.ModuleDict()
        for execution_index in lora_only_execution_indices:
            self.lora_only_attn[str(execution_index)] = Attention(
                self.self_attn.num_heads,
                self.self_attn.head_dim,
                self.self_attn.scaling,
                num_kv_heads=self.self_attn.num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                per_layer_sliding_window=per_layer_sliding_window,
                prefix=f"{model_prefix}.looped_lora_layers.{execution_index}.self_attn.attn",
                attn_type=AttentionType.DECODER,
            )

    def forward_lora_only(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        execution_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        qkv = _apply_lora_delta(self.self_attn.qkv_proj, hidden_states)
        q, k, v = qkv.split(
            [self.self_attn.q_size, self.self_attn.kv_size, self.self_attn.kv_size],
            dim=-1,
        )
        q = self.self_attn.q_norm(
            q.view(
                *q.shape[:-1],
                q.shape[-1] // self.self_attn.head_dim,
                self.self_attn.head_dim,
            )
        ).view(q.shape)
        k = self.self_attn.k_norm(
            k.view(
                *k.shape[:-1],
                k.shape[-1] // self.self_attn.head_dim,
                self.self_attn.head_dim,
            )
        ).view(k.shape)
        q, k = self.self_attn.rotary_emb(positions, q, k)
        hidden_states = self.lora_only_attn[str(execution_index)](q, k, v)
        hidden_states = _apply_lora_delta(self.self_attn.o_proj, hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = _apply_lora_delta(self.mlp.gate_up_proj, hidden_states)
        hidden_states = self.mlp.act_fn(hidden_states)
        hidden_states = _apply_lora_delta(self.mlp.down_proj, hidden_states)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class LoopedLoraQwen3Model(Qwen2Model):
    if hasattr(Qwen3Model, "hf_to_vllm_mapper"):
        hf_to_vllm_mapper = Qwen3Model.hf_to_vllm_mapper

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("Fast looped LoRA requires pipeline_parallel_size=1")
        if vllm_config.parallel_config.tensor_parallel_size != 1:
            raise ValueError("Fast looped LoRA requires tensor_parallel_size=1")

        config = vllm_config.model_config.hf_config.get_text_config()
        if config.model_type != "qwen3":
            raise ValueError(f"Fast looped LoRA requires a Qwen3 model, got {config.model_type!r}")

        sections = getattr(config, "looped_lora_sections", None)
        if not sections:
            raise ValueError("Fast looped LoRA requires at least one configured section")
        schedule = build_looped_lora_schedule(config.num_hidden_layers, sections)
        lora_only_by_physical_layer = get_lora_only_executions_by_physical_layer(config.num_hidden_layers, schedule)

        def make_decoder_layer(
            config: Qwen3Config,
            cache_config: CacheConfig | None = None,
            quant_config: QuantizationConfig | None = None,
            prefix: str = "",
            per_layer_sliding_window: int | None = None,
        ) -> nn.Module:
            physical_layer = extract_layer_index(prefix)
            return LoopedLoraQwen3DecoderLayer(
                config=config,
                lora_only_execution_indices=lora_only_by_physical_layer[physical_layer],
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
                per_layer_sliding_window=per_layer_sliding_window,
            )

        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=make_decoder_layer,
        )
        self.looped_lora_schedule: tuple[LayerExecution, ...] = schedule
        logger.info(
            "Fast looped LoRA schedule: physical_layers=%d executions=%d lora_only_executions=%d sections=%s",
            config.num_hidden_layers,
            len(schedule),
            sum(execution.lora_only for execution in schedule),
            sections,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            raise ValueError("Fast looped LoRA does not support pipeline intermediate tensors")
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_input_ids(input_ids)

        hidden_states = inputs_embeds
        residual = None
        for execution_index, execution in enumerate(self.looped_lora_schedule):
            layer = self.layers[execution.physical_layer]
            if execution.lora_only:
                hidden_states, residual = layer.forward_lora_only(
                    positions,
                    hidden_states,
                    residual,
                    execution_index,
                )
            else:
                hidden_states, residual = layer(positions, hidden_states, residual)

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class SkyRLLoopedQwen3ForCausalLM(LocalArgmaxMixin, nn.Module, SupportsLoRA):
    if hasattr(LoopedLoraQwen3Model, "hf_to_vllm_mapper"):
        hf_to_vllm_mapper = LoopedLoraQwen3Model.hf_to_vllm_mapper
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        self.quant_config = vllm_config.quant_config
        self.model = LoopedLoraQwen3Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        if not get_pp_group().is_last_rank:
            raise ValueError("Fast looped LoRA requires pipeline_parallel_size=1")

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            tie_weights = getattr(self.lm_head, "tie_weights", None)
            self.lm_head = (
                tie_weights(self.model.embed_tokens)
                if tie_weights is not None
                else self.model.embed_tokens
            )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


def register() -> None:
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(
        "SkyRLLoopedQwen3ForCausalLM",
        "skyrl.backends.skyrl_train.models.looped_qwen3:SkyRLLoopedQwen3ForCausalLM",
    )
