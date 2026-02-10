# Adapted from
# https://github.com/huggingface/transformers/blob/7769f660935b5d48b73bf6711d0a78b6f8f98739/src/transformers/models/olmo3/modeling_olmo3.py#L1

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils.generic import TransformersKwargs
from transformers.processing_utils import Unpack

from tx.models.configs import Olmo3Config
from tx.extra.torch.layers.rotary_embedding import RotaryEmbedding
from tx.extra.torch.models.modeling_outputs import ModelOutput, CausalLMOutput


class Olmo3RMSNorm(nn.Module):
    """Olmo3RMSNorm is equivalent to T5LayerNorm.

    For the original implementation, please refer to:
    https://github.com/huggingface/transformers/blob/7769f660935b5d48b73bf6711d0a78b6f8f98739/src/transformers/models/t5/modeling_t5.py#L46-L68
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class Olmo3Attention(nn.Module):

    def __init__(
        self,
        config: Olmo3Config,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        # Guard against partial configuration with falsy head_dim
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        # self.num_heads = config.num_attention_heads
        # self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True  # Seems redundant

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Olmo3RMSNorm(config.num_attention_heads * self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Olmo3RMSNorm(config.num_key_value_heads * self.head_dim, eps=config.rms_norm_eps)
        # TODO(jwj): Support sliding window attention.
        # assert config.layer_types is not None
        # self.attention_type = config.layer_types[layer_idx]
        # self.sliding_window = config.sliding_window if self.attention_type == "sliding_attention" else None
        # self._rope_theta = _rope_theta(config)

        # TODO(jwj): Support YaRN-style scaling.
        self.rotary_emb = RotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            is_neox_style=True,
            dtype=config.dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # hidden_states: [B, T, C]
        # B is the batch size, T is the sequence length, and C is the embedding dimension.
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Project then norm on full dimension
        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)

        # Reshape to [B, n_heads, T, head_dim]
        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)

        # TODO(jwj): Support KV cache update.

        # TODO(jwj): Support GQA for Olmo 3 32B.
        # Olmo 3 7B uses MHA, so we skip repeating kv.
        attn_mask_bool = attention_mask[:, None, None, :].to(dtype=torch.bool)
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_mask_bool,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()  # Why contiguous?
        attn_output = self.o_proj(attn_output)

        return attn_output


class Olmo3MLP(nn.Module):

    def __init__(self, config: Olmo3Config) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        assert config.hidden_act == "silu", "Only SiLU is supported for Olmo 3 MLP."
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        x = self.down_proj(gate_up)

        return x


class Olmo3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Olmo3Config,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.self_attn = Olmo3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Olmo3MLP(config)
        self.post_attention_layernorm = Olmo3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Olmo3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states, position_ids, attention_mask, **kwargs)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# Class Olmo3RotaryEmbedding(nn.Module):


class Olmo3Model(nn.Module):

    def __init__(self, config: Olmo3Config) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.ModuleList(
            [Olmo3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Olmo3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ModelOutput:
        input_embeds = self.embed_tokens(input_ids)

        hidden_states = input_embeds
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states, position_ids=position_ids, attention_mask=attention_mask, **kwargs
            )

        hidden_states = self.norm(hidden_states)
        return ModelOutput(
            last_hidden_state=hidden_states,
        )


class Olmo3ForCausalLM(nn.Module):

    def __init__(self, config: Olmo3Config) -> None:
        super().__init__()
        self.vocab_size = config.vocab_size

        self.model = Olmo3Model(config)
        assert not config.tie_word_embeddings, "Tie word embeddings is not supported for Olmo3ForCausalLM."
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutput:
        outputs = self.model(input_ids, position_ids=position_ids, attention_mask=attention_mask, **kwargs)

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        return CausalLMOutput(
            logits=logits,
            last_hidden_state=outputs.last_hidden_state,
        )


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig

    config_path = hf_hub_download("allenai/Olmo-3-7B-Instruct", "config.json")
    print("Downloaded config.json to ", config_path)

    config = AutoConfig.from_pretrained(config_path)
    config = Olmo3Config(
        config=config,
        max_lora_adapters=1,
        max_lora_rank=1,
        shard_attention_heads=True,
    )
    model = Olmo3ForCausalLM(config)
    outputs = model.forward(
        input_ids=torch.randint(0, 100, (1, 10)),
        position_ids=torch.arange(10).unsqueeze(0),
        attention_mask=torch.ones(1, 10),
    )
    print(outputs)
