from flax import nnx
import jax
from jax import numpy as jnp
from jax.sharding import get_abstract_mesh

from tx.layers.lora import LoRAEmbed, LoRAExpert, LoRALinear
from tx.layers.rotary_embedding import get_rope
from tx.layers.util import Param, prepare_routing
from tx.layers.layernorm import RMSNorm
from tx.layers.attention import dot_product_attention
from tx.models.configs import Glm4Config
from tx.models.types import CausalLMOutput, ModelForCausalLM, ModelOutput
from tx.utils.generator import GeneratorMixin, KVCache
from tx.utils.logits_processor import LogitsProcessorMixin, LMHead


class Glm4Attention(nnx.Module):
    """Multi-head attention with Grouped Query Attention (GQA) support."""

    def __init__(self, config: Glm4Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads

        tp = get_abstract_mesh().shape.get("tp", 1)
        shard_attention_heads = config.shard_attention_heads
        if shard_attention_heads:
            assert self.num_heads % tp == 0, f"num_heads={self.num_heads} must be divisible by tp={tp}"
            assert self.num_kv_heads % tp == 0, f"num_kv_heads={self.num_kv_heads} must be divisible by tp={tp}"
        tp_shard = "tp" if shard_attention_heads else None

        self.head_dim = config.head_dim or config.hidden_size // self.num_heads

        self.q_proj = LoRALinear(
            in_features=config.hidden_size,
            out_features=self.num_heads * self.head_dim,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=config.attention_bias,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("fsdp", tp_shard)),
            rngs=rngs,
        )
        self.k_proj = LoRALinear(
            in_features=config.hidden_size,
            out_features=self.num_kv_heads * self.head_dim,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=config.attention_bias,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("fsdp", tp_shard)),
            rngs=rngs,
        )
        self.v_proj = LoRALinear(
            in_features=config.hidden_size,
            out_features=self.num_kv_heads * self.head_dim,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=config.attention_bias,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("fsdp", tp_shard)),
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
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (tp_shard, "fsdp")),
            rngs=rngs,
        )

        if self.config.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)

        self.rotary_dim = int(self.head_dim * self.config.partial_rotary_factor)
        self.rotary_emb, _ = get_rope(self.rotary_dim, config.rope_theta, config.rope_scaling)

    def __call__(
        self,
        x: jax.Array,
        *,
        attention_mask: jax.Array,
        positions: jax.Array,
        adapter_indices: jax.Array | None = None,
        kv_cache: tuple[jax.Array, jax.Array] | None = None,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        B, T, _ = x.shape

        q = self.q_proj(x, adapter_indices=adapter_indices).reshape(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x, adapter_indices=adapter_indices).reshape(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x, adapter_indices=adapter_indices).reshape(B, T, self.num_kv_heads, self.head_dim)

        if self.config.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Partial RoPE
        q_rot, q_pass = q[..., : self.rotary_dim], q[..., self.rotary_dim :]
        k_rot, k_pass = k[..., : self.rotary_dim], k[..., self.rotary_dim :]

        q_rot = self.rotary_emb(q_rot, positions)
        k_rot = self.rotary_emb(k_rot, positions)

        q = jnp.concatenate([q_rot, q_pass], axis=-1)
        k = jnp.concatenate([k_rot, k_pass], axis=-1)

        # Handle KV cache
        if kv_cache is not None:
            k, v = KVCache.update_layer(kv_cache, k, v, positions)

        updated_cache = (k, v)

        is_causal = kv_cache is None
        attn_output = dot_product_attention(q, k, v, attention_mask, is_causal, self.head_dim)

        output = attn_output.reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(output, adapter_indices=adapter_indices), updated_cache


class Glm4MLP(nnx.Module):

    def __init__(
        self,
        config: Glm4Config,
        *,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
        override_intermediate_size: int | None = None,
    ) -> None:
        self.config = config
        intermediate_size = override_intermediate_size or config.intermediate_size
        self.gate_proj = LoRALinear(
            config.hidden_size,
            intermediate_size,
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
            intermediate_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), ("fsdp", "tp")),
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            rngs=rngs,
        )
        self.down_proj = LoRALinear(
            intermediate_size,
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


class Glm4TopkRouter(nnx.Module):
    """GLM4 MoE routing gate. Returns raw router logits."""

    def __init__(self, config: Glm4Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config

        self.weight = Param(
            config.hidden_size,
            config.n_routed_experts,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, None)),
            rngs=rngs,
        )

        self.e_score_correction_bias = nnx.Variable(jnp.zeros(config.n_routed_experts, dtype=jnp.float32))

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        hidden_states = hidden_states.reshape(-1, self.config.hidden_size)
        router_logits = hidden_states.astype(jnp.float32) @ self.weight[...].astype(jnp.float32)
        return router_logits


class Glm4Experts(nnx.Module):

    def __init__(self, config: Glm4Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
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


class Glm4MoE(nnx.Module):
    """MoE layer with shared experts and group-based routing."""

    def __init__(self, config: Glm4Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.n_group = config.n_group

        self.gate = Glm4TopkRouter(config, dtype=dtype, rngs=rngs)
        self.experts = Glm4Experts(config, dtype=dtype, rngs=rngs)

        inter_dim = config.moe_intermediate_size * config.n_shared_experts
        self.shared_experts = Glm4MLP(config, dtype=dtype, rngs=rngs, override_intermediate_size=inter_dim)

    def _compute_routing(self, router_logits: jax.Array) -> tuple[jax.Array, jax.Array]:
        num_tokens = router_logits.shape[0]
        num_experts = router_logits.shape[1]

        scores = nnx.sigmoid(router_logits)
        scores_with_bias = scores + self.gate.e_score_correction_bias[...]

        experts_per_group = num_experts // self.n_group
        scores_grouped = scores_with_bias.reshape(num_tokens, self.n_group, experts_per_group)

        top2, _ = jax.lax.top_k(scores_grouped, 2)
        group_scores = jnp.sum(top2, axis=-1)

        _, top_group_indices = jax.lax.top_k(group_scores, self.config.topk_group)

        mask = jnp.ones((num_tokens, self.n_group), dtype=bool)
        batch_indices = jnp.arange(num_tokens)[:, None]
        mask = mask.at[batch_indices, top_group_indices].set(False)
        mask = jnp.broadcast_to(mask[:, :, None], scores_grouped.shape)

        scores_with_bias = jnp.where(mask, 0.0, scores_grouped)
        scores_with_bias = scores_with_bias.reshape(num_tokens, num_experts)

        _, top_k_index = jax.lax.top_k(scores_with_bias, self.config.num_experts_per_tok)

        # Get weights from original scores
        top_k_weights = jnp.take_along_axis(scores, top_k_index, axis=-1)

        if self.config.norm_topk_prob:
            top_k_weights = top_k_weights / jnp.sum(top_k_weights, axis=-1, keepdims=True)

        top_k_weights = top_k_weights * self.config.routed_scaling_factor

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
            hidden_states_flat.reshape(batch_size, seq_len, hidden_size), adapter_indices
        ).reshape(-1, hidden_size)
        expert_output = expert_output + shared_output

        return expert_output.reshape(batch_size, seq_len, hidden_size)


class Glm4DecoderLayer(nnx.Module):

    def __init__(self, config: Glm4Config, layer_idx: int, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
        self.self_attn = Glm4Attention(config, dtype=dtype, rngs=rngs)

        # Check if this is an MoE model and if this layer should use MoE
        if config.n_routed_experts and layer_idx >= config.first_k_dense_replace:
            self.mlp = Glm4MoE(config, dtype=dtype, rngs=rngs)
        else:
            self.mlp = Glm4MLP(config, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        hidden_states: jax.Array,
        *,
        attention_mask: jax.Array,
        positions: jax.Array,
        adapter_indices: jax.Array | None = None,
        kv_cache: tuple[jax.Array, jax.Array] | None = None,
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


class Glm4Model(nnx.Module):

    def __init__(self, config: Glm4Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
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
            [
                Glm4DecoderLayer(config, layer_idx=i, dtype=dtype, rngs=rngs)
                for i in range(config.num_hidden_layers)
            ]
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
                kv_cache=kv_cache and (kv_cache.keys[layer_idx], kv_cache.values[layer_idx]),
            )
            updated_keys.append(k)
            updated_values.append(v)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return ModelOutput(
            last_hidden_state=hidden_states,
            kv_cache=KVCache.update(kv_cache, updated_keys, updated_values, positions, attention_mask),
            hidden_states=all_hidden_states if output_hidden_states else None,
        )


class Glm4ForCausalLM(nnx.Module, ModelForCausalLM, GeneratorMixin, LogitsProcessorMixin):

    def __init__(self, config: Glm4Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.model = Glm4Model(config, dtype=dtype, rngs=rngs)

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
        else:
            self.lm_head = self.model.embed_tokens.T

    def get_lm_head(self) -> LMHead:
        """Return the lm_head callable for logits computation."""
        return self.lm_head

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
            positions = jnp.arange(attention_mask.shape[1])[None, :]

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            positions=positions,
            output_hidden_states=output_hidden_states,
            adapter_indices=adapter_indices,
            kv_cache=kv_cache,
        )

        return CausalLMOutput(
            last_hidden_state=outputs.last_hidden_state,
            kv_cache=outputs.kv_cache,
            hidden_states=outputs.hidden_states,
        )


Glm4MoeForCausalLM = Glm4ForCausalLM
