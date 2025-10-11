from flax import nnx
import jax
from jax import numpy as jnp
from jax.sharding import get_abstract_mesh
from transformers import Qwen3Config

from tx.layers.lora import LoRAExpert, LoRALinear
from tx.layers.util import Param, prepare_routing


class RMSNorm(nnx.Module):
    def __init__(self, size: int, *, eps: float = 1e-6, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.eps = eps
        self.weight = Param(
            size, dtype=dtype, kernel_init=nnx.with_partitioning(nnx.initializers.normal(), jax.P(None)), rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return self.weight * x / rms


def apply_rope(inputs: jax.Array, position_ids: jax.Array, head_dim: int, theta: int) -> jax.Array:
    fraction = 2 * jnp.arange(0, head_dim // 2, dtype=jnp.float32) / head_dim
    timescale = jnp.pow(theta, fraction)
    x = (position_ids[..., None] / timescale[None, None, :])[..., None, :]
    sin, cos = jnp.sin(x), jnp.cos(x)
    a, b = jnp.split(inputs, 2, axis=-1)
    return jnp.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1).astype(inputs.dtype)


class Qwen3Attention(nnx.Module):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        tp = get_abstract_mesh().shape.get("tp", 1)
        assert self.num_heads % tp == 0, f"num_heads={self.num_heads} must be divisible by tp={tp}"
        assert self.num_kv_heads % tp == 0, f"num_kv_heads={self.num_kv_heads} must be divisible by tp={tp}"
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.num_heads
        max_lora_adapters = getattr(config, "max_lora_adapters", 0)
        max_lora_rank = getattr(config, "max_lora_rank", 8)

        self.q_proj = LoRALinear(
            in_features=config.hidden_size,
            out_features=self.num_heads * self.head_dim,
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=False,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), jax.P(None, "tp")),
            rngs=rngs,
        )
        self.k_proj = LoRALinear(
            in_features=config.hidden_size,
            out_features=self.num_kv_heads * self.head_dim,
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=False,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), jax.P(None, "tp")),
            rngs=rngs,
        )
        self.v_proj = LoRALinear(
            in_features=config.hidden_size,
            out_features=self.num_kv_heads * self.head_dim,
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=False,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), jax.P(None, "tp")),
            rngs=rngs,
        )
        self.o_proj = LoRALinear(
            in_features=self.num_heads * self.head_dim,
            out_features=config.hidden_size,
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=False,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), jax.P("tp", None)),
            rngs=rngs,
        )

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        *,
        attention_mask: jax.Array | None = None,
        adapter_indices: jax.Array | None = None,
    ) -> jax.Array:

        # Reshape each: [B,T,H*D] -> [B,T,H,D]
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x, adapter_indices=adapter_indices).reshape(B, T, self.num_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x, adapter_indices=adapter_indices).reshape(B, T, self.num_kv_heads, self.head_dim))
        v = self.v_proj(x, adapter_indices=adapter_indices).reshape(B, T, self.num_kv_heads, self.head_dim)

        position_ids = jnp.arange(x.shape[1])[None, :].repeat(x.shape[0], axis=0)

        q = apply_rope(q, position_ids, self.head_dim, self.config.rope_theta)
        k = apply_rope(k, position_ids, self.head_dim, self.config.rope_theta)

        if self.num_kv_heads != self.num_heads:
            num_groups = self.num_heads // self.num_kv_heads
            k = jnp.repeat(k, num_groups, axis=2)
            v = jnp.repeat(v, num_groups, axis=2)

        attn_output = jax.nn.dot_product_attention(
            q,
            k,
            v,
            scale=1.0 / self.head_dim**0.5,
            mask=attention_mask[:, None, None, :].astype(bool) if attention_mask is not None else None,
            is_causal=True,
        )

        attn_out_flat = attn_output.reshape(B, T, self.num_heads * self.head_dim)  # [B,T,H,D] -> [B,T,H*D]
        return self.o_proj(attn_out_flat, adapter_indices=adapter_indices)


class Qwen3MLP(nnx.Module):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        max_lora_adapters = getattr(config, "max_lora_adapters", 0)
        max_lora_rank = getattr(config, "max_lora_rank", 8)
        self.gate_proj = LoRALinear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), jax.P(None, "tp")),
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            rngs=rngs,
        )
        self.up_proj = LoRALinear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), jax.P(None, "tp")),
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            rngs=rngs,
        )
        self.down_proj = LoRALinear(
            config.intermediate_size,
            config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), jax.P("tp", None)),
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, adapter_indices: jax.Array | None = None) -> jax.Array:
        gate_out = self.gate_proj(x, adapter_indices)
        up_out = self.up_proj(x, adapter_indices)
        return self.down_proj(nnx.silu(gate_out) * up_out, adapter_indices)


class Qwen3Experts(nnx.Module):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        max_lora_adapters = getattr(config, "max_lora_adapters", 0)
        max_lora_rank = getattr(config, "max_lora_rank", 8)

        self.gate_proj = LoRAExpert(
            config.num_experts,
            config.hidden_size,
            config.moe_intermediate_size,
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), jax.P(None, None, "tp")),
            rngs=rngs,
        )
        self.up_proj = LoRAExpert(
            config.num_experts,
            config.hidden_size,
            config.moe_intermediate_size,
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), jax.P(None, None, "tp")),
            rngs=rngs,
        )
        self.down_proj = LoRAExpert(
            config.num_experts,
            config.moe_intermediate_size,
            config.hidden_size,
            max_lora_adapters=max_lora_adapters,
            max_lora_rank=max_lora_rank,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), jax.P(None, "tp", None)),
            rngs=rngs,
        )

    def __call__(
        self, hidden_states: jax.Array, router_logits: jax.Array, adapter_indices: jax.Array | None = None
    ) -> jax.Array:
        # Get top-k experts for each token and compute routing weights
        routing_weights, selected_experts = jax.lax.top_k(router_logits, k=self.config.num_experts_per_tok)
        routing_weights = nnx.softmax(routing_weights, axis=-1)

        # Prepare for ragged_dot by sorting tokens based on their assigned expert
        selected_experts_flat = selected_experts.ravel()
        hidden_states_expanded = jnp.repeat(hidden_states, self.config.num_experts_per_tok, axis=0)
        adapter_indices_expanded = (
            jnp.repeat(adapter_indices, self.config.num_experts_per_tok) if adapter_indices is not None else None
        )
        hidden_states_sorted, group_sizes, unsort_indices, adapter_indices_sorted = prepare_routing(
            hidden_states_expanded,
            selected_experts_flat,
            self.config.num_experts,
            adapter_indices=adapter_indices_expanded,
        )

        # Apply expert layers using LoRAExpert
        gate_out = self.gate_proj(hidden_states_sorted, group_sizes, adapter_indices_sorted)
        up_out = self.up_proj(hidden_states_sorted, group_sizes, adapter_indices_sorted)
        down_out = self.down_proj(nnx.silu(gate_out) * up_out, group_sizes, adapter_indices_sorted)

        # Unsort and combine the expert outputs
        unsorted_out = down_out[unsort_indices]
        reshaped_out = unsorted_out.reshape(-1, self.config.num_experts_per_tok, self.config.hidden_size)
        return jnp.sum(reshaped_out * routing_weights[..., None], axis=1)


class Qwen3MoeSparseMoeBlock(nnx.Module):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.gate = nnx.Linear(
            config.hidden_size,
            config.num_experts,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), jax.P(None, None)),
            rngs=rngs,
        )
        self.experts = Qwen3Experts(config, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        hidden_states: jax.Array,
        *,
        adapter_indices: jax.Array | None = None,
        return_router_logits: bool = False,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        (batch_size, seq_len, hidden_size) = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_size)
        # Expand adapter_indices to match flattened hidden_states
        if adapter_indices is not None:
            adapter_indices = jnp.repeat(adapter_indices, seq_len)
        router_logits = self.gate(hidden_states)

        hidden_states = self.experts(hidden_states, router_logits, adapter_indices)
        hidden_states = hidden_states.reshape(batch_size, seq_len, hidden_size)

        if return_router_logits:
            return hidden_states, router_logits
        return hidden_states


class Qwen3DecoderLayer(nnx.Module):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
        self.self_attn = Qwen3Attention(config, dtype=dtype, rngs=rngs)
        if getattr(config, "num_experts", None):
            self.mlp = Qwen3MoeSparseMoeBlock(config, dtype=dtype, rngs=rngs)
        else:
            self.mlp = Qwen3MLP(config, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        hidden_states: jax.Array,
        *,
        attention_mask: jax.Array | None = None,
        adapter_indices: jax.Array | None = None,
    ) -> jax.Array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            adapter_indices=adapter_indices,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, adapter_indices=adapter_indices)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3Model(nnx.Module):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.embed_tokens = nnx.Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            embedding_init=nnx.with_partitioning(nnx.initializers.normal(), jax.P("tp", None)),
            rngs=rngs,
        )
        self.layers = nnx.List(
            [Qwen3DecoderLayer(config, dtype=dtype, rngs=rngs) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        input_ids: jax.Array,
        *,
        attention_mask: jax.Array | None = None,
        output_hidden_states: bool | None = None,
        adapter_indices: jax.Array | None = None,
    ) -> dict[str, jax.Array | list[jax.Array]]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        hidden_states = self.embed_tokens(input_ids)

        all_hidden_states: list[jax.Array] = []

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                adapter_indices=adapter_indices,
            )

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states,
        }


class Qwen3ForCausalLM(nnx.Module):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.model = Qwen3Model(config, dtype=dtype, rngs=rngs)
        if not self.config.tie_word_embeddings:
            self.lm_head = nnx.Linear(
                config.hidden_size,
                config.vocab_size,
                use_bias=False,
                dtype=dtype,
                param_dtype=dtype,
                kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), jax.P(None, "tp")),
                rngs=rngs,
            )

    def __call__(
        self,
        input_ids: jax.Array,
        *,
        attention_mask: jax.Array | None = None,
        output_hidden_states: bool | None = None,
        adapter_indices: jax.Array | None = None,
    ) -> dict[str, jax.Array | list[jax.Array]]:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            adapter_indices=adapter_indices,
        )
        hidden_states = outputs["last_hidden_state"]
        if self.config.tie_word_embeddings:
            logits = hidden_states @ self.model.embed_tokens.embedding.value.T
        else:
            logits = self.lm_head(hidden_states)

        return {"logits": logits, **outputs}
