import math
from flax import nnx
import jax
from jax import numpy as jnp
from jax.sharding import get_abstract_mesh
from transformers import DeepseekV3Config

from tx.layers.lora import LoRAEmbed, LoRAExpert, LoRALinear
from tx.layers.rotary_embedding import apply_rope_interleave
from tx.layers.util import Param, prepare_routing
from tx.layers.layernorm import RMSNorm
from tx.models.types import CausalLMOutput, ModelOutput
from tx.utils.generator import GeneratorMixin, KVCache, compute_positions


class DeepseekV3Attention(nnx.Module):
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

        if self.q_lora_rank is None:
            self.q_proj = LoRALinear(
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
            self.q_a_proj = None
            self.q_a_layernorm = None
            self.q_b_proj = None
        else:
            self.q_proj = None
            self.q_a_proj = LoRALinear(
                in_features=config.hidden_size,
                out_features=self.q_lora_rank,
                max_lora_adapters=config.max_lora_adapters,
                max_lora_rank=config.max_lora_rank,
                dtype=dtype,
                param_dtype=dtype,
                use_bias=config.attention_bias,
                kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("fsdp", None)),
                rngs=rngs,
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
            self.q_b_proj = LoRALinear(
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

        self.kv_a_proj_with_mqa = LoRALinear(
            in_features=config.hidden_size,
            out_features=self.kv_lora_rank + self.qk_rope_head_dim,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=config.attention_bias,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("fsdp", None)),
            rngs=rngs,
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)

        self.kv_b_proj = LoRALinear(
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

        self.o_proj = LoRALinear(
            in_features=self.num_heads * self.v_head_dim,
            out_features=config.hidden_size,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=config.attention_bias,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (tp_shard, "fsdp")),
            rngs=rngs,
        )

        self.scaling = self.qk_head_dim ** (-0.5)

        rope_params = config.rope_scaling
        # HF has both fields, which are set to same on initialization.
        rope_type = rope_params.get("rope_type") or rope_params.get("type", "default")
        if rope_type != "default":
            mscale_all_dim = rope_params.get("mscale_all_dim", 0)
            scaling_factor = rope_params.get("factor", 1.0)
            if mscale_all_dim:
                mscale = 1.0
                if scaling_factor > 1.0:
                    mscale = 0.1 * mscale_all_dim * math.log(scaling_factor) + 1.0
                self.scaling = self.scaling * mscale * mscale

    def __call__(
        self,
        x: jax.Array,
        *,
        attention_mask: jax.Array,
        positions: tuple[jax.Array, jax.Array],
        adapter_indices: jax.Array | None = None,
        kv_cache: tuple[jax.Array, jax.Array, int] | None = None,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        B, T, _ = x.shape

        # Query projection
        if self.q_lora_rank is None:
            q_states = self.q_proj(x, adapter_indices=adapter_indices)
        else:
            q_states = self.q_b_proj(
                self.q_a_layernorm(
                    self.q_a_proj(
                        x,
                        adapter_indices=adapter_indices,
                    )
                ),
                adapter_indices=adapter_indices,
            )

        q_states = q_states.reshape(B, T, self.num_heads, self.qk_head_dim)
        q_pass, q_rot = jnp.split(q_states, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x, adapter_indices=adapter_indices)
        k_pass, k_rot = jnp.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_rot = k_rot.reshape(B, T, 1, self.qk_rope_head_dim)

        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass), adapter_indices=adapter_indices)
        k_pass = k_pass.reshape(B, T, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_pass, v = jnp.split(k_pass, [self.qk_nope_head_dim], axis=-1)

        q_rot = apply_rope_interleave(q_rot, positions, self.qk_rope_head_dim, self.config.rope_theta)
        k_rot = apply_rope_interleave(k_rot, positions, self.qk_rope_head_dim, self.config.rope_theta)

        # Expand k_rot to all heads
        k_rot = jnp.broadcast_to(k_rot, (B, T, self.num_heads, self.qk_rope_head_dim))

        q = jnp.concatenate([q_pass, q_rot], axis=-1)
        k = jnp.concatenate([k_pass, k_rot], axis=-1)

        # Handle KV cache
        if kv_cache is not None:
            k_cache, v_cache, cache_position = kv_cache
            k = jax.lax.dynamic_update_slice(k_cache, k, (0, cache_position, 0, 0))
            v = jax.lax.dynamic_update_slice(v_cache, v, (0, cache_position, 0, 0))
        
        updated_cache = (k, v)

        # Jax attention expects v to have the same shape as k
        v = jnp.pad(v, ((0, 0), (0, 0), (0, 0), (0, self.qk_head_dim - self.v_head_dim)))

        attn_output = jax.nn.dot_product_attention(
            q,
            k,
            v,
            scale=self.scaling,
            mask=attention_mask[:, None, None, :].astype(bool),
            is_causal=kv_cache is None,
        )

        attn_output = attn_output[:, :, :, :self.v_head_dim].reshape(B, T, self.num_heads * self.v_head_dim)
        return self.o_proj(attn_output, adapter_indices=adapter_indices), updated_cache


class DeepseekV3MLP(nnx.Module):

    def __init__(
        self,
        dim: int,
        inter_dim: int,
        *,
        config: DeepseekV3Config,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ) -> None:
        self.dim = dim
        self.inter_dim = inter_dim
        self.gate_proj = LoRALinear(
            dim,
            inter_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("fsdp", "tp")),
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            rngs=rngs,
        )
        self.up_proj = LoRALinear(
            dim,
            inter_dim,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("fsdp", "tp")),
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            rngs=rngs,
        )
        self.down_proj = LoRALinear(
            inter_dim,
            dim,
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


class DeepseekV3TopkRouter(nnx.Module):
    """DeepseekV3 MoE routing gate. Returns raw router logits."""

    def __init__(self, config: DeepseekV3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.n_routed_experts = config.n_routed_experts

        self.weight = Param(
            config.hidden_size,
            config.n_routed_experts,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, None)),
            rngs=rngs,
        )

        self.e_score_correction_bias = nnx.Variable(jnp.zeros(config.n_routed_experts, dtype=dtype))

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        hidden_states = hidden_states.reshape(-1, self.config.hidden_size)
        router_logits = hidden_states.astype(jnp.float32) @ self.weight.value.astype(jnp.float32)
        return router_logits


class DeepseekV3NaiveMoe(nnx.Module):
    """Run NaiveMoe on selected expert groups."""

    def __init__(self, config: DeepseekV3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size

        # NOTE: Huggingface implementation uses a fused gate_up_proj, but the weights are keyed
        # by gate_proj and up_proj separately.
        self.gate_proj = LoRAExpert(
            self.num_experts,
            self.hidden_dim,
            self.intermediate_dim,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, "fsdp", "tp")),
            rngs=rngs,
        )
        self.up_proj = LoRAExpert(
            self.num_experts,
            self.hidden_dim,
            self.intermediate_dim,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, "fsdp", "tp")),
            rngs=rngs,
        )
        self.down_proj = LoRAExpert(
            self.num_experts,
            self.intermediate_dim,
            self.hidden_dim,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, "tp", "fsdp")),
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: jax.Array,
        top_k_index: jax.Array,
        top_k_weights: jax.Array,
        adapter_indices: jax.Array | None = None,
    ) -> jax.Array:
        num_experts_per_tok = top_k_index.shape[1]

        # Prepare for ragged_dot by sorting tokens based on their assigned expert
        selected_experts_flat = top_k_index.ravel()
        hidden_states_expanded = jnp.repeat(hidden_states, num_experts_per_tok, axis=0)
        adapter_indices_expanded = (
            jnp.repeat(adapter_indices, num_experts_per_tok) if adapter_indices is not None else None
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
        reshaped_out = unsorted_out.reshape(-1, num_experts_per_tok, self.hidden_dim)
        return jnp.sum(reshaped_out * top_k_weights[..., None], axis=1)


class DeepseekV3MoE(nnx.Module):
    """MoE layer for routing to top-k expert groups."""

    def __init__(self, config: DeepseekV3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor

        self.gate = DeepseekV3TopkRouter(config, dtype=dtype, rngs=rngs)
        self.experts = DeepseekV3NaiveMoe(config, dtype=dtype, rngs=rngs)

        inter_dim = config.moe_intermediate_size * config.n_shared_experts
        self.shared_experts = DeepseekV3MLP(
            config.hidden_size, inter_dim, config=config, dtype=dtype, rngs=rngs)

    def _compute_routing(self, router_logits: jax.Array) -> tuple[jax.Array, jax.Array]:
        num_tokens = router_logits.shape[0]
        num_experts = router_logits.shape[1]

        scores = nnx.sigmoid(router_logits)
        scores_with_bias = scores + self.gate.e_score_correction_bias.value

        experts_per_group = num_experts // self.n_group
        scores_grouped = scores_with_bias.reshape(num_tokens, self.n_group, experts_per_group)

        top2, _ = jax.lax.top_k(scores_grouped, 2)
        group_scores = jnp.sum(top2, axis=-1)

        _, top_group_indices = jax.lax.top_k(group_scores, self.topk_group)

        mask = jnp.ones((num_tokens, self.n_group), dtype=bool)
        batch_indices = jnp.arange(num_tokens)[:, None]
        mask = mask.at[batch_indices, top_group_indices].set(False)
        mask = jnp.broadcast_to(mask[:, :, None], scores_grouped.shape)

        scores_with_bias = jnp.where(mask, -jnp.inf, scores_grouped)
        scores_with_bias = scores_with_bias.reshape(num_tokens, num_experts)

        _, top_k_index = jax.lax.top_k(scores_with_bias, self.num_experts_per_tok)

        # Get weights from original scores
        top_k_weights = jnp.take_along_axis(scores, top_k_index, axis=-1)

        if self.norm_topk_prob:
            top_k_weights = top_k_weights / jnp.sum(top_k_weights, axis=-1, keepdims=True)

        top_k_weights = top_k_weights * self.routed_scaling_factor

        return top_k_weights.astype(router_logits.dtype), top_k_index

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

        router_logits = self.gate(hidden_states_flat)
        top_k_weights, top_k_index = self._compute_routing(router_logits)

        expert_output = self.experts(hidden_states_flat, top_k_index, top_k_weights, adapter_indices_flat)
        shared_output = self.shared_experts(
            hidden_states_flat.reshape(batch_size, seq_len, hidden_size),
            adapter_indices
        ).reshape(-1, hidden_size)
        expert_output = expert_output + shared_output

        return expert_output.reshape(batch_size, seq_len, hidden_size)


class DeepseekV3DecoderLayer(nnx.Module):

    def __init__(
        self, config: DeepseekV3Config, layer_idx: int, *, dtype: jnp.dtype, rngs: nnx.Rngs
    ) -> None:
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
        self.self_attn = DeepseekV3Attention(config, dtype=dtype, rngs=rngs)

        # Use dense MLP for initial layers, MoE for the rest
        if layer_idx >= config.first_k_dense_replace:
            self.mlp = DeepseekV3MoE(config, dtype=dtype, rngs=rngs)
        else:
            self.mlp = DeepseekV3MLP(
                config.hidden_size, config.intermediate_size, config=config, dtype=dtype, rngs=rngs)

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
