from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from tx.models.configs import Qwen3Config
from tx.torch.layers.lora import LoRALinear
from tx.torch.models.outputs import ModelOutput, CausalLMOutput
from tx.torch.utils.generator import KVCache, compute_positions


def apply_rope(inputs: torch.Tensor, position_ids: torch.Tensor, head_dim: int, theta: int) -> torch.Tensor:
    """Apply rotary position embeddings to input tensor.

    Args:
        inputs: Input tensor of shape [B, n_head, T, head_dim]
        position_ids: Position IDs of shape [B, T] (can include negative positions for left-padding)
        head_dim: Dimension of each attention head
        theta: RoPE theta parameter

    Returns:
        Tensor with rotary embeddings applied
    """
    fraction = 2 * torch.arange(0, head_dim // 2, dtype=torch.float32, device=inputs.device) / head_dim
    timescale = theta**fraction
    x = position_ids[:, None, :, None] / timescale[None, None, None, :]  # [B, 1, T, dim/2]
    sin, cos = x.sin().to(dtype=inputs.dtype), x.cos().to(dtype=inputs.dtype)  # [B, 1, T, dim/2]
    a, b = inputs.chunk(2, dim=-1)
    return torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1)


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
        max_lora_adapters: int,
        max_lora_rank: int,
    ):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.num_heads
        self.gqa_groups = self.num_heads // self.num_kv_heads

        self.q_proj = LoRALinear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            use_bias=False,
            dtype=dtype,
            device=device,
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
        )
        self.k_proj = LoRALinear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            use_bias=False,
            dtype=dtype,
            device=device,
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
        )
        self.v_proj = LoRALinear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            use_bias=False,
            dtype=dtype,
            device=device,
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
        )
        self.o_proj = LoRALinear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            use_bias=False,
            dtype=dtype,
            device=device,
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
        )

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps).to(dtype=dtype)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps).to(dtype=dtype)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        # [B, n_kv, T, H] -> [B, n_head, T, H]
        return x if self.num_kv_heads == self.num_heads else x.repeat_interleave(self.gqa_groups, dim=1)

    def forward(
        self,
        x: torch.Tensor,  # [B, T, D]
        *,
        attention_mask: torch.Tensor,  # [B, T_kv] (1=keep, 0=mask)
        positions: torch.Tensor,  # [B, T]
        adapter_indices: torch.Tensor | None = None,  # [B] or None
        kv_cache: tuple[torch.Tensor, torch.Tensor, int] | None = None,  # (k_cache, v_cache, cache_position)
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape

        # Project and reshape to [B, T, num_heads, head_dim]
        q = self.q_norm(self.q_proj(x, adapter_indices=adapter_indices).view(B, T, self.num_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x, adapter_indices=adapter_indices).view(B, T, self.num_kv_heads, self.head_dim))
        v = self.v_proj(x, adapter_indices=adapter_indices).view(B, T, self.num_kv_heads, self.head_dim)

        # Transpose to [B, n, T, H]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE
        q = apply_rope(q, positions, self.head_dim, self.config.rope_theta)
        k = apply_rope(k, positions, self.head_dim, self.config.rope_theta)

        # Handle KV cache update
        if kv_cache is not None:
            k_cache, v_cache, cache_pos = kv_cache

            T_new = k.size(2)
            with torch.no_grad():
                k_cache[:, :, cache_pos : cache_pos + T_new, :].copy_(k)
                v_cache[:, :, cache_pos : cache_pos + T_new, :].copy_(v)

            k, v = k_cache, v_cache

        updated_cache = (k, v)

        # Attention (causal only during prefill, GQA handled via repeat)
        k_full = self._repeat_kv(k)
        v_full = self._repeat_kv(v)

        # Use SDPA with bool mask
        attn_mask_bool = attention_mask[:, None, None, :].to(dtype=torch.bool)
        attn_output = F.scaled_dot_product_attention(
            q,
            k_full,
            v_full,
            attn_mask=attn_mask_bool,
            dropout_p=0.0,
            is_causal=(kv_cache is None),
        )

        output = attn_output.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.o_proj(output, adapter_indices=adapter_indices), updated_cache


class Qwen3MLP(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
        max_lora_adapters: int,
        max_lora_rank: int,
    ):
        super().__init__()
        self.gate_proj = LoRALinear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=False,
            dtype=dtype,
            device=device,
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
        )
        self.up_proj = LoRALinear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=False,
            dtype=dtype,
            device=device,
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
        )
        self.down_proj = LoRALinear(
            config.intermediate_size,
            config.hidden_size,
            use_bias=False,
            dtype=dtype,
            device=device,
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
        )

    def forward(self, x: torch.Tensor, adapter_indices: torch.Tensor | None = None) -> torch.Tensor:
        gate_out = self.gate_proj(x, adapter_indices)
        up_out = self.up_proj(x, adapter_indices)
        return self.down_proj(F.silu(gate_out) * up_out, adapter_indices)


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
        max_lora_adapters: int,
        max_lora_rank: int,
    ):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(dtype=dtype)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(dtype=dtype)
        self.self_attn = Qwen3Attention(
            config, dtype=dtype, device=device, max_lora_adapters=max_lora_adapters, max_lora_rank=max_lora_rank
        )
        self.mlp = Qwen3MLP(
            config, dtype=dtype, device=device, max_lora_adapters=max_lora_adapters, max_lora_rank=max_lora_rank
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        positions: torch.Tensor,
        adapter_indices: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor, int] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, updated_cache = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            positions=positions,
            adapter_indices=adapter_indices,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, adapter_indices=adapter_indices)
        hidden_states = residual + hidden_states

        return hidden_states, updated_cache


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config, *, dtype: torch.dtype, device: torch.device | str):
        super().__init__()
        self.config = config
        max_lora_adapters = getattr(config, "max_lora_adapters", 0)
        max_lora_rank = getattr(config, "max_lora_rank", 8)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(
                    config, dtype=dtype, device=device, max_lora_adapters=max_lora_adapters, max_lora_rank=max_lora_rank
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(dtype=dtype)

    def forward(
        self,
        input_ids: torch.Tensor,  # [B, T]
        *,
        attention_mask: torch.Tensor,  # [B, T_kv] (1=keep, 0=mask)
        positions: torch.Tensor,  # [B, T]
        output_hidden_states: bool | None = None,
        adapter_indices: torch.Tensor | None = None,  # [B] or None
        kv_cache: KVCache | None = None,
    ) -> ModelOutput:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        hidden_states = self.embed_tokens(input_ids)
        all_hidden_states = []
        updated_keys, updated_values = [], []

        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            hidden_states, (k, v) = layer(
                hidden_states,
                attention_mask=attention_mask,
                positions=positions,
                adapter_indices=adapter_indices,
                kv_cache=kv_cache and (kv_cache.keys[layer_idx], kv_cache.values[layer_idx], kv_cache.cache_position),
            )
            updated_keys.append(k)
            updated_values.append(v)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        # Increment cache_position if cache exists, or use sequence length for new cache
        new_cache_position = (
            kv_cache.cache_position + input_ids.shape[1] if kv_cache is not None else input_ids.shape[1]
        )

        return ModelOutput(
            last_hidden_state=hidden_states,
            kv_cache=KVCache(keys=updated_keys, values=updated_values, cache_position=new_cache_position),
            hidden_states=all_hidden_states if output_hidden_states else None,
        )


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config, *, dtype: torch.dtype, device: torch.device | str):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config, dtype=dtype, device=device)
        if not self.config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=dtype)

    @staticmethod
    def is_lora_param(name: str) -> bool:
        """Return True if a parameter name corresponds to LoRA weights."""
        return "lora_A" in name or "lora_B" in name

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        positions: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        adapter_indices: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> CausalLMOutput:
        if positions is None:
            positions = compute_positions(attention_mask)

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            positions=positions,
            output_hidden_states=output_hidden_states,
            adapter_indices=adapter_indices,
            kv_cache=kv_cache,
        )
        hidden_states = outputs.last_hidden_state
        if self.config.tie_word_embeddings:
            logits = F.linear(hidden_states, self.model.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden_states)

        return CausalLMOutput(
            logits=logits,
            last_hidden_state=outputs.last_hidden_state,
            kv_cache=outputs.kv_cache,
            hidden_states=outputs.hidden_states,
        )
