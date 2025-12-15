from flax import nnx
import jax
from jax import numpy as jnp
from jax.sharding import get_abstract_mesh
from transformers import LlamaConfig

from tx.layers.lora import LoRAEmbed, LoRALinear
from tx.layers.rotary_embedding import apply_rope
from tx.layers.layernorm import RMSNorm
from tx.models.types import CausalLMOutput, ModelOutput
from tx.utils.generator import GeneratorMixin, KVCache, compute_positions


class Llama3Attention(nnx.Module):
    """Multi-head attention with Grouped Query Attention (GQA) support."""

    def __init__(self, config: LlamaConfig, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        tp = get_abstract_mesh().shape.get("tp", 1)
        shard_attention_heads = config.shard_attention_heads
        if shard_attention_heads:
            assert self.num_heads % tp == 0, f"num_heads={self.num_heads} must be divisible by tp={tp}"
            assert self.num_kv_heads % tp == 0, f"num_kv_heads={self.num_kv_heads} must be divisible by tp={tp}"
        tp_shard = "tp" if shard_attention_heads else None
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.num_heads

        self.q_proj = LoRALinear(
            in_features=config.hidden_size,
            out_features=self.num_heads * self.head_dim,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=False,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, tp_shard)),
            rngs=rngs,
        )

        self.k_proj = LoRALinear(
            in_features=config.hidden_size,
            out_features=self.num_kv_heads * self.head_dim,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=False,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, tp_shard)),
            rngs=rngs,
        )

        self.v_proj = LoRALinear(
            in_features=config.hidden_size,
            out_features=self.num_kv_heads * self.head_dim,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=False,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, tp_shard)),
            rngs=rngs,
        )

        self.o_proj = LoRALinear(
            in_features=self.num_heads * self.head_dim,
            out_features=config.hidden_size,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=False,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (tp_shard, None)),
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        *,
        attention_mask: jax.Array,
        positions: jax.Array,
        adapter_indices: jax.Array | None = None,
        kv_cache: tuple[jax.Array, jax.Array, int] | None = None,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        B, T, _ = x.shape

        # Project and reshape to [B, T, num_heads, head_dim]
        q = self.q_proj(x, adapter_indices=adapter_indices).reshape(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x, adapter_indices=adapter_indices).reshape(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x, adapter_indices=adapter_indices).reshape(B, T, self.num_kv_heads, self.head_dim)

        # Apply RoPE
        q = apply_rope(q, positions, self.head_dim, self.config.rope_theta)
        k = apply_rope(k, positions, self.head_dim, self.config.rope_theta)

        # Handle KV cache
        if kv_cache is not None:
            k_cache, v_cache, cache_position = kv_cache
            k = jax.lax.dynamic_update_slice(k_cache, k, (0, cache_position, 0, 0))
            v = jax.lax.dynamic_update_slice(v_cache, v, (0, cache_position, 0, 0))

        updated_cache = (k, v)

        # Attention (causal only during prefill, GQA handled natively by dot_product_attention)
        attn_output = jax.nn.dot_product_attention(
            q,
            k,
            v,
            scale=1.0 / self.head_dim**0.5,
            mask=attention_mask[:, None, None, :].astype(bool),
            is_causal=kv_cache is None,
        )

        output = attn_output.reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(output, adapter_indices=adapter_indices), updated_cache


class Llama3MLP(nnx.Module):

    def __init__(self, config: LlamaConfig, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.gate_proj = LoRALinear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, "tp")),
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            rngs=rngs,
        )
        self.up_proj = LoRALinear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, "tp")),
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            rngs=rngs,
        )
        self.down_proj = LoRALinear(
            config.intermediate_size,
            config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("tp", None)),
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, adapter_indices: jax.Array | None = None) -> jax.Array:
        gate_out = self.gate_proj(x, adapter_indices)
        up_out = self.up_proj(x, adapter_indices)
        return self.down_proj(nnx.silu(gate_out) * up_out, adapter_indices)


class Llama3DecoderLayer(nnx.Module):

    def __init__(self, config: LlamaConfig, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
        self.self_attn = Llama3Attention(config, dtype=dtype, rngs=rngs)
        self.mlp = Llama3MLP(config, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        hidden_states: jax.Array,
        *,
        attention_mask: jax.Array,
        positions: jax.Array,
        adapter_indices: jax.Array | None = None,
        kv_cache: tuple[jax.Array, jax.Array, int] | None = None,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
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
        mlp_output = self.mlp(hidden_states, adapter_indices=adapter_indices)
        hidden_states = residual + mlp_output

        return hidden_states, updated_cache


class Llama3Model(nnx.Module):

    def __init__(self, config: LlamaConfig, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config

        self.embed_tokens = LoRAEmbed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            param_dtype=dtype,
            embedding_init=nnx.with_partitioning(nnx.initializers.normal(), ("tp", None)),
            rngs=rngs,
        )
        self.layers = nnx.List(
            [Llama3DecoderLayer(config, dtype=dtype, rngs=rngs) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        input_ids: jax.Array,
        *,
        attention_mask: jax.Array,
        positions: jax.Array,
        output_hidden_states: bool | None = None,
        adapter_indices: jax.Array | None = None,
        kv_cache: KVCache | None = None,
    ) -> ModelOutput:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        hidden_states = self.embed_tokens(input_ids, adapter_indices=adapter_indices)
        all_hidden_states: list[jax.Array] = []
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
        new_cache_position = kv_cache.cache_position + 1 if kv_cache is not None else input_ids.shape[1]

        return ModelOutput(
            last_hidden_state=hidden_states,
            kv_cache=KVCache(keys=updated_keys, values=updated_values, cache_position=new_cache_position),
            hidden_states=all_hidden_states if output_hidden_states else None,
        )


class Llama3ForCausalLM(nnx.Module, GeneratorMixin):

    def __init__(self, config: LlamaConfig, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.model = Llama3Model(config, dtype=dtype, rngs=rngs)

        if not self.config.tie_word_embeddings:
            self.lm_head = LoRALinear(
                config.hidden_size,
                config.vocab_size,
                use_bias=False,
                dtype=dtype,
                param_dtype=dtype,
                kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, "tp")),
                max_lora_adapters=config.max_lora_adapters,
                max_lora_rank=config.max_lora_rank,
                rngs=rngs,
            )

    @staticmethod
    def is_lora_param(path: tuple, _value) -> bool:
        """Return True if a parameter path corresponds to LoRA weights."""
        return any(name in path for name in ("lora_A", "lora_B"))

    def __call__(
        self,
        input_ids: jax.Array,
        *,
        attention_mask: jax.Array,
        positions: jax.Array | None = None,
        output_hidden_states: bool | None = None,
        adapter_indices: jax.Array | None = None,
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
            logits = hidden_states @ self.model.embed_tokens.embedding.value.T
        else:
            logits = self.lm_head(hidden_states, adapter_indices=adapter_indices)

        return CausalLMOutput(
            logits=logits,
            last_hidden_state=outputs.last_hidden_state,
            kv_cache=outputs.kv_cache,
            hidden_states=outputs.hidden_states,
        )


LlamaForCausalLM = Llama3ForCausalLM
