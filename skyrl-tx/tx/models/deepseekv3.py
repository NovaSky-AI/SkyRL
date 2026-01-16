import math
from flax import nnx
import jax
from jax import numpy as jnp
from jax.sharding import get_abstract_mesh
from transformers import DeepseekV3Config

from tx.layers.lora import LoRAEmbed, LoRAExpert, LoRALinear
from tx.layers.rotary_embedding import apply_rope
from tx.layers.util import prepare_routing
from tx.layers.layernorm import RMSNorm
from tx.models.types import CausalLMOutput, ModelOutput
from tx.utils.generator import GeneratorMixin, KVCache, compute_positions


class DeepseekV3MLA(nnx.Module):
    """Multi-Head Latent Attention (MLA) Layer."""

    def __init__(self, config: DeepseekV3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.num_heads = config.num_attention_heads

        tp = get_abstract_mesh().shape.get("tp", 1)
        shard_attention_heads = config.shard_attention_heads
        if shard_attention_heads:
            assert self.num_heads % tp == 0, f"num_heads={self.num_heads} must be divisible by tp={tp}"
        tp_shard = "tp" if shard_attention_heads else None

        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim

        if self.q_lora_rank == 0:
            self.wq = LoRALinear(
                in_features=config.hidden_size,
                out_features=self.num_heads * self.qk_head_dim,
                max_lora_adapters=config.max_lora_adapters,
                max_lora_rank=config.max_lora_rank,
                dtype=dtype,
                param_dtype=dtype,
                use_bias=False,
                kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("fsdp", tp_shard)),
                rngs=rngs,
            )
            self.wq_a = None
            self.q_norm = None
            self.wq_b = None
        else:
            self.wq = None
            self.wq_a = LoRALinear(
                in_features=config.hidden_size,
                out_features=self.q_lora_rank,
                max_lora_adapters=config.max_lora_adapters,
                max_lora_rank=config.max_lora_rank,
                dtype=dtype,
                param_dtype=dtype,
                use_bias=False,
                kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("fsdp", None)),
                rngs=rngs,
            )
            self.q_norm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
            self.wq_b = LoRALinear(
                in_features=self.q_lora_rank,
                out_features=self.num_heads * self.qk_head_dim,
                max_lora_adapters=config.max_lora_adapters,
                max_lora_rank=config.max_lora_rank,
                dtype=dtype,
                param_dtype=dtype,
                use_bias=False,
                kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, tp_shard)),
                rngs=rngs,
            )

        self.wkv_a = LoRALinear(
            in_features=config.hidden_size,
            out_features=self.kv_lora_rank + self.qk_rope_head_dim,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=False,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("fsdp", None)),
            rngs=rngs,
        )
        self.kv_norm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)

        self.wkv_b = LoRALinear(
            in_features=self.kv_lora_rank,
            out_features=self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=False,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, tp_shard)),
            rngs=rngs,
        )

        self.wo = LoRALinear(
            in_features=self.num_heads * self.v_head_dim,
            out_features=config.hidden_size,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=False,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (tp_shard, "fsdp")),
            rngs=rngs,
        )

        # NOTE: Rope config has underwent significant changes upstream, revisit on version bump
        self.softmax_scale = self.qk_head_dim**-0.5
        max_seq_len = config.max_position_embeddings
        original_seq_len = config.rope_scaling.get("original_max_position_embeddings", max_seq_len)
        rope_factor = config.rope_scaling.get("factor", 1.0)
        mscale = config.rope_scaling.get("mscale", 1.0)
        if config.max_position_embeddings > original_seq_len:
            mscale = 0.1 * mscale * math.log(rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

    def __call__(
        self,
        x: jax.Array,
        *,
        attention_mask: jax.Array,
        positions: jax.Array,
        freqs_cis: jax.Array,
        adapter_indices: jax.Array | None = None,
        kv_cache: tuple[jax.Array, jax.Array, int] | None = None,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        B, T, _ = x.shape

        if self.q_lora_rank == 0:
            q = self.wq(x, adapter_indices=adapter_indices)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x, adapter_indices=adapter_indices)), adapter_indices=adapter_indices)

        q = q.reshape(B, T, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = jnp.split(q, [self.qk_nope_head_dim], axis=-1)
        q_pe = apply_rope(q_pe, positions, self.qk_rope_head_dim, self.config.rope_theta)

        kv_out = self.wkv_a(x, adapter_indices=adapter_indices)
        kv, k_pe = jnp.split(kv_out, [self.kv_lora_rank], axis=-1)

        k_pe = apply_rope(k_pe, positions, self.qk_rope_head_dim, self.config.rope_theta)

        kv_normalized = self.kv_norm(kv)
        kv_expanded = self.wkv_b(kv_normalized, adapter_indices=adapter_indices)
        kv_expanded = kv_expanded.reshape(B, T, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = jnp.split(kv_expanded, [self.qk_nope_head_dim], axis=-1)

        q = jnp.concatenate([q_nope, q_pe], axis=-1)

        k = jnp.concatenate([k_nope, jnp.broadcast_to(k_pe, (B, T, self.num_heads, self.qk_rope_head_dim))], axis=-1)

        if kv_cache is not None:
            k_cache, v_cache, cache_position = kv_cache
            k = jax.lax.dynamic_update_slice(k_cache, k, (0, cache_position, 0, 0))
            v = jax.lax.dynamic_update_slice(v_cache, v, (0, cache_position, 0, 0))

        updated_cache = (k, v)

        attn_output = jax.nn.dot_product_attention(
            q,
            k,
            v,
            scale=self.softmax_scale,
            mask=attention_mask[:, None, None, :].astype(bool),
            is_causal=kv_cache is None,
        )

        output = attn_output.reshape(B, T, self.num_heads * self.v_head_dim)
        return self.wo(output, adapter_indices=adapter_indices), updated_cache


class DeepseekV3MLP(nnx.Module):

    def __init__(self, config: DeepseekV3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.gate_proj = LoRALinear(
            config.hidden_size,
            config.intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("fsdp", "tp")),
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
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("fsdp", "tp")),
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
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("tp", "fsdp")),
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, adapter_indices: jax.Array | None = None) -> jax.Array:
        gate_out = self.gate_proj(x, adapter_indices)
        up_out = self.up_proj(x, adapter_indices)
        return self.down_proj(nnx.silu(gate_out) * up_out, adapter_indices)


class DeepseekV3Gate(nnx.Module):

    def __init__(self, config: DeepseekV3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.num_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_expert_groups = getattr(config, "n_expert_groups", 1)
        self.n_limited_groups = getattr(config, "n_limited_groups", 1)
        self.score_func = getattr(config, "score_func", "softmax")
        self.route_scale = getattr(config, "route_scale", 1.0)

        self.gate = nnx.Linear(
            config.hidden_size,
            self.num_experts,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, None)),
            rngs=rngs,
        )

        # Bias only for specific model sizes (7168 hidden_size in original)
        self.use_bias = config.hidden_size == 7168
        if self.use_bias:
            from tx.layers.util import Param

            self.bias = Param(
                self.num_experts,
                dtype=jnp.float32,
                kernel_init=nnx.with_partitioning(nnx.initializers.zeros_init(), (None,)),
                rngs=rngs,
            )

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Compute routing weights and selected expert indices.

        Args:
            x: Input tensor of shape [num_tokens, hidden_size].

        Returns:
            Tuple of (routing_weights, selected_expert_indices).
        """
        scores = self.gate(x)

        if self.score_func == "softmax":
            scores = nnx.softmax(scores, axis=-1)
        else:
            scores = nnx.sigmoid(scores)

        original_scores = scores

        if self.use_bias:
            scores = scores + self.bias.value

        # Group-based expert selection
        if self.n_expert_groups > 1:
            num_tokens = x.shape[0]
            experts_per_group = self.num_experts // self.n_expert_groups
            scores = scores.reshape(num_tokens, self.n_expert_groups, experts_per_group)

            if not self.use_bias:
                group_scores = jnp.max(scores, axis=-1)
            else:
                top2, _ = jax.lax.top_k(scores, 2)
                group_scores = jnp.sum(top2, axis=-1)

            _, top_group_indices = jax.lax.top_k(group_scores, self.n_limited_groups)

            # Create mask for non-selected groups
            mask = jnp.ones((num_tokens, self.n_expert_groups), dtype=bool)
            batch_indices = jnp.arange(num_tokens)[:, None]
            mask = mask.at[batch_indices, top_group_indices].set(False)
            mask = jnp.broadcast_to(mask[:, :, None], scores.shape)

            scores = jnp.where(mask, -jnp.inf, scores)
            scores = scores.reshape(num_tokens, self.num_experts)

        # Select top-k experts
        _, selected_experts = jax.lax.top_k(scores, self.num_experts_per_tok)
        weights = jnp.take_along_axis(original_scores, selected_experts, axis=-1)

        if self.score_func == "sigmoid":
            weights = weights / jnp.sum(weights, axis=-1, keepdims=True)

        weights = weights * self.route_scale
        return weights.astype(x.dtype), selected_experts


class DeepseekV3Experts(nnx.Module):

    def __init__(self, config: DeepseekV3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.num_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        moe_inter_dim = config.moe_intermediate_size

        self.gate_proj = LoRAExpert(
            self.num_experts,
            config.hidden_size,
            moe_inter_dim,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, "fsdp", "tp")),
            rngs=rngs,
        )
        self.up_proj = LoRAExpert(
            self.num_experts,
            config.hidden_size,
            moe_inter_dim,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, "fsdp", "tp")),
            rngs=rngs,
        )
        self.down_proj = LoRAExpert(
            self.num_experts,
            moe_inter_dim,
            config.hidden_size,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, "tp", "fsdp")),
            rngs=rngs,
        )

    def __call__(
        self, hidden_states: jax.Array, router_weights: jax.Array, selected_experts: jax.Array,
        adapter_indices: jax.Array | None = None
    ) -> jax.Array:
        # Prepare for ragged_dot by sorting tokens based on their assigned expert
        selected_experts_flat = selected_experts.ravel()
        hidden_states_expanded = jnp.repeat(hidden_states, self.num_experts_per_tok, axis=0)
        adapter_indices_expanded = (
            jnp.repeat(adapter_indices, self.num_experts_per_tok) if adapter_indices is not None else None
        )
        hidden_states_sorted, group_sizes, unsort_indices, adapter_indices_sorted = prepare_routing(
            hidden_states_expanded,
            selected_experts_flat,
            self.num_experts,
            adapter_indices=adapter_indices_expanded,
        )

        gate_out = self.gate_proj(hidden_states_sorted, group_sizes, adapter_indices_sorted)
        up_out = self.up_proj(hidden_states_sorted, group_sizes, adapter_indices_sorted)
        down_out = self.down_proj(nnx.silu(gate_out) * up_out, group_sizes, adapter_indices_sorted)

        # Unsort and combine the expert outputs
        unsorted_out = down_out[unsort_indices]
        reshaped_out = unsorted_out.reshape(-1, self.num_experts_per_tok, self.config.hidden_size)
        return jnp.sum(reshaped_out * router_weights[..., None], axis=1)


class DeepseekV3MoE(nnx.Module):

    def __init__(self, config: DeepseekV3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.gate = DeepseekV3Gate(config, dtype=dtype, rngs=rngs)
        self.experts = DeepseekV3Experts(config, dtype=dtype, rngs=rngs)

        n_shared_experts = getattr(config, "n_shared_experts", 0)
        if n_shared_experts > 0:
            self.shared_experts = DeepseekV3SharedMLP(config, dtype=dtype, rngs=rngs)
        else:
            self.shared_experts = None

    def __call__(
        self,
        hidden_states: jax.Array,
        *,
        adapter_indices: jax.Array | None = None,
    ) -> jax.Array:
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.reshape(-1, hidden_size)

        if adapter_indices is not None:
            adapter_indices_flat = jnp.repeat(adapter_indices, seq_len)
        else:
            adapter_indices_flat = None

        router_weights, selected_experts = self.gate(hidden_states_flat)
        expert_output = self.experts(hidden_states_flat, router_weights, selected_experts, adapter_indices_flat)

        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states_flat, adapter_indices_flat)
            expert_output = expert_output + shared_output

        return expert_output.reshape(batch_size, seq_len, hidden_size)


class DeepseekV3SharedMLP(nnx.Module):
    """Always active shared experts."""

    def __init__(self, config: DeepseekV3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        n_shared_experts = getattr(config, "n_shared_experts", 2)
        moe_inter_dim = config.moe_intermediate_size
        shared_inter_dim = n_shared_experts * moe_inter_dim

        self.gate_proj = LoRALinear(
            config.hidden_size,
            shared_inter_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("fsdp", "tp")),
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            rngs=rngs,
        )
        self.up_proj = LoRALinear(
            config.hidden_size,
            shared_inter_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("fsdp", "tp")),
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            rngs=rngs,
        )
        self.down_proj = LoRALinear(
            shared_inter_dim,
            config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("tp", "fsdp")),
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, adapter_indices: jax.Array | None = None) -> jax.Array:
        gate_out = self.gate_proj(x, adapter_indices)
        up_out = self.up_proj(x, adapter_indices)
        return self.down_proj(nnx.silu(gate_out) * up_out, adapter_indices)


class DeepseekV3DecoderLayer(nnx.Module):

    def __init__(
        self, config: DeepseekV3Config, layer_idx: int, *, dtype: jnp.dtype, rngs: nnx.Rngs
    ) -> None:
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
        self.self_attn = DeepseekV3MLA(config, dtype=dtype, rngs=rngs)

        # Use dense MLP for initial layers, MoE for the rest
        n_dense_layers = getattr(config, "n_dense_layers", 1)
        if layer_idx < n_dense_layers:
            self.mlp = DeepseekV3MLP(config, dtype=dtype, rngs=rngs)
        else:
            self.mlp = DeepseekV3MoE(config, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        hidden_states: jax.Array,
        *,
        attention_mask: jax.Array,
        positions: jax.Array,
        freqs_cis: jax.Array,
        adapter_indices: jax.Array | None = None,
        kv_cache: tuple[jax.Array, jax.Array, int] | None = None,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, updated_cache = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            positions=positions,
            freqs_cis=freqs_cis,
            adapter_indices=adapter_indices,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states, adapter_indices=adapter_indices)
        hidden_states = residual + mlp_output

        return hidden_states, updated_cache


class DeepseekV3Model(nnx.Module):

    def __init__(self, config: DeepseekV3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
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
            [DeepseekV3DecoderLayer(config, layer_idx=i, dtype=dtype, rngs=rngs) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)

        # Precompute RoPE frequencies
        # qk_rope_head_dim = config.qk_rope_head_dim
        # original_seq_len = getattr(config, "original_seq_len", config.max_position_embeddings)
        # rope_factor = getattr(config, "rope_factor", 1.0)
        # beta_fast = getattr(config, "beta_fast", 32)
        # beta_slow = getattr(config, "beta_slow", 1)

        # TODO: Swap out like llama's rope?
        # self.freqs_cis = precompute_freqs_cis(
        #     dim=qk_rope_head_dim,
        #     max_seq_len=config.max_position_embeddings,
        #     rope_theta=config.rope_theta,
        #     original_seq_len=original_seq_len,
        #     rope_factor=rope_factor,
        #     beta_fast=beta_fast,
        #     beta_slow=beta_slow,
        # )

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
                freqs_cis=self.freqs_cis,
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


class DeepseekV3ForCausalLM(nnx.Module, GeneratorMixin):

    def __init__(self, config: DeepseekV3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.model = DeepseekV3Model(config, dtype=dtype, rngs=rngs)

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
