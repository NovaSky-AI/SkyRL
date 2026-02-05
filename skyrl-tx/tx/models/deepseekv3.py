from flax import nnx
import jax
from jax import numpy as jnp
from jax.sharding import get_abstract_mesh

from tx.layers.lora import LoRAEmbed, LoRAExpert, LoRALinear
from tx.layers.rotary_embedding import get_rope
from tx.layers.util import Param, prepare_routing, shard_map_ep
from tx.layers.layernorm import RMSNorm
from tx.models.configs import DeepseekV3Config
from tx.models.types import CausalLMOutput, ModelForCausalLM, ModelOutput
from tx.models.utils import create_stacked_layers, forward_layers
from tx.utils.generator import GeneratorMixin, KVCache
from tx.utils.logits_processor import LogitsProcessorMixin, LMHead


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
                sharding=("fsdp", tp_shard),
                max_lora_adapters=config.max_lora_adapters,
                max_lora_rank=config.max_lora_rank,
                dtype=dtype,
                param_dtype=dtype,
                use_bias=False,
                kernel_init=nnx.initializers.lecun_normal(),
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
                sharding=("fsdp", None),
                max_lora_adapters=config.max_lora_adapters,
                max_lora_rank=config.max_lora_rank,
                dtype=dtype,
                param_dtype=dtype,
                use_bias=config.attention_bias,
                kernel_init=nnx.initializers.lecun_normal(),
                rngs=rngs,
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
            self.q_b_proj = LoRALinear(
                in_features=self.q_lora_rank,
                out_features=self.num_heads * self.qk_head_dim,
                sharding=(None, tp_shard),
                max_lora_adapters=config.max_lora_adapters,
                max_lora_rank=config.max_lora_rank,
                dtype=dtype,
                param_dtype=dtype,
                use_bias=False,
                kernel_init=nnx.initializers.lecun_normal(),
                rngs=rngs,
            )

        self.kv_a_proj_with_mqa = LoRALinear(
            in_features=config.hidden_size,
            out_features=self.kv_lora_rank + self.qk_rope_head_dim,
            sharding=("fsdp", None),
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=config.attention_bias,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)

        self.kv_b_proj = LoRALinear(
            in_features=self.kv_lora_rank,
            out_features=self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            sharding=(None, tp_shard),
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=False,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )

        self.o_proj = LoRALinear(
            in_features=self.num_heads * self.v_head_dim,
            out_features=config.hidden_size,
            sharding=(tp_shard, "fsdp"),
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=config.attention_bias,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )

        self.rotary_emb, mscale = get_rope(self.qk_rope_head_dim, config.rope_theta, config.rope_scaling)
        self.scaling = self.qk_head_dim ** (-0.5) * mscale * mscale

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

        # Query projection
        if self.q_lora_rank is None:
            q_states = self.q_proj(x, adapter_indices=adapter_indices)
        else:
            y = self.q_a_proj(x, adapter_indices=adapter_indices)
            q_states = self.q_b_proj(self.q_a_layernorm(y), adapter_indices=adapter_indices)

        q_states = q_states.reshape(B, T, self.num_heads, self.qk_head_dim)
        q_pass, q_rot = jnp.split(q_states, [self.qk_nope_head_dim], axis=-1)

        compressed_kv = self.kv_a_proj_with_mqa(x, adapter_indices=adapter_indices)
        k_pass, k_rot = jnp.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_rot = k_rot.reshape(B, T, 1, self.qk_rope_head_dim)

        k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass), adapter_indices=adapter_indices)
        k_pass = k_pass.reshape(B, T, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_pass, v = jnp.split(k_pass, [self.qk_nope_head_dim], axis=-1)

        q_rot = self.rotary_emb(q_rot, positions)
        k_rot = self.rotary_emb(k_rot, positions)

        # Expand k_rot to all heads
        k_rot = jnp.broadcast_to(k_rot, (B, T, self.num_heads, self.qk_rope_head_dim))

        q = jnp.concatenate([q_pass, q_rot], axis=-1)
        k = jnp.concatenate([k_pass, k_rot], axis=-1)

        # Handle KV cache
        if kv_cache is not None:
            k, v = KVCache.update_layer(kv_cache, k, v, positions)

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

        attn_output = attn_output[:, :, :, : self.v_head_dim].reshape(B, T, self.num_heads * self.v_head_dim)
        return self.o_proj(attn_output, adapter_indices=adapter_indices), updated_cache


class DeepseekV3MLP(nnx.Module):

    def __init__(
        self,
        config: DeepseekV3Config,
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
            sharding=("fsdp", "tp"),
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.initializers.lecun_normal(),
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            rngs=rngs,
        )
        self.up_proj = LoRALinear(
            config.hidden_size,
            intermediate_size,
            sharding=("fsdp", "tp"),
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.initializers.lecun_normal(),
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            rngs=rngs,
        )
        self.down_proj = LoRALinear(
            intermediate_size,
            config.hidden_size,
            sharding=("tp", "fsdp"),
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.initializers.lecun_normal(),
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
        router_logits = hidden_states.astype(jnp.float32) @ self.weight[...].astype(jnp.float32)
        return router_logits


class DeepseekV3NaiveMoe(nnx.Module):
    """Run NaiveMoe on selected expert groups."""

    def __init__(self, config: DeepseekV3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config

        # NOTE: Huggingface implementation uses a fused gate_up_proj, but the weights are keyed
        # by gate_proj and up_proj separately.
        self.gate_proj = LoRAExpert(
            config.n_routed_experts,
            config.hidden_size,
            config.moe_intermediate_size,
            sharding=("ep", "fsdp", "tp"),
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )
        self.up_proj = LoRAExpert(
            config.n_routed_experts,
            config.hidden_size,
            config.moe_intermediate_size,
            sharding=("ep", "fsdp", "tp"),
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )
        self.down_proj = LoRAExpert(
            config.n_routed_experts,
            config.moe_intermediate_size,
            config.hidden_size,
            sharding=("ep", "tp", "fsdp"),
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            kernel_init=nnx.initializers.lecun_normal(),
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
        num_experts = self.config.n_routed_experts

        ep = get_abstract_mesh().shape.get("ep", 1)
        assert num_experts % ep == 0, f"num_experts={num_experts} must be divisible by ep={ep}"

        # Prepare for ragged_dot by sorting tokens based on their assigned expert
        hidden_expanded = jnp.repeat(hidden_states, num_experts_per_tok, axis=0)
        adapter_expanded = jnp.repeat(adapter_indices, num_experts_per_tok) if adapter_indices is not None else None
        hidden_sorted, group_sizes, unsort_indices, adapter_sorted = prepare_routing(
            hidden_expanded,
            top_k_index.ravel(),
            num_experts,
            adapter_indices=adapter_expanded,
        )

        def forward(experts, hidden_sorted, group_sizes, unsort_indices, adapter_sorted, top_k_weights):
            # Calculate local offset for this EP shard
            ep_rank = jax.lax.axis_index("ep")
            experts_per_rank = num_experts // jax.lax.axis_size("ep")
            group_offset = jnp.array([ep_rank * experts_per_rank], dtype=jnp.int32)

            # Expert computation with group_offset
            gate = experts.gate_proj(hidden_sorted, group_sizes, adapter_sorted, group_offset=group_offset)
            up = experts.up_proj(hidden_sorted, group_sizes, adapter_sorted, group_offset=group_offset)
            down = experts.down_proj(nnx.silu(gate) * up, group_sizes, adapter_sorted, group_offset=group_offset)

            # Unsort and combine
            out = down[unsort_indices].reshape(-1, num_experts_per_tok, self.config.hidden_size)
            local_out = jnp.sum(out * top_k_weights[..., None], axis=1)
            return jax.lax.psum(local_out, axis_name="ep")

        return shard_map_ep(self, forward, hidden_sorted, group_sizes, unsort_indices, adapter_sorted, top_k_weights)


class DeepseekV3MoE(nnx.Module):
    """MoE layer for routing to top-k expert groups."""

    def __init__(self, config: DeepseekV3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.n_group = config.n_group

        self.gate = DeepseekV3TopkRouter(config, dtype=dtype, rngs=rngs)
        self.experts = DeepseekV3NaiveMoe(config, dtype=dtype, rngs=rngs)

        inter_dim = config.moe_intermediate_size * config.n_shared_experts
        self.shared_experts = DeepseekV3MLP(config, dtype=dtype, rngs=rngs, override_intermediate_size=inter_dim)

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
        # _compute_routing uses float32 for softmax stability; cast back to model dtype
        # to maintain consistent dtypes through jax.lax.scan in forward_layers
        top_k_weights = top_k_weights.astype(hidden_states.dtype)

        expert_output = self.experts(hidden_states_flat, top_k_index, top_k_weights, adapter_indices_flat)
        shared_output = self.shared_experts(
            hidden_states_flat.reshape(batch_size, seq_len, hidden_size), adapter_indices
        ).reshape(-1, hidden_size)
        expert_output = expert_output + shared_output

        return expert_output.reshape(batch_size, seq_len, hidden_size)


class DeepseekV3DecoderLayer(nnx.Module):
    """Decoder layer supporting both dense MLP and sparse MoE."""

    def __init__(
        self,
        config: DeepseekV3Config,
        *,
        mlp_cls: type[DeepseekV3MLP] | type[DeepseekV3MoE],
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ) -> None:
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
        self.self_attn = DeepseekV3Attention(config, dtype=dtype, rngs=rngs)
        self.mlp = mlp_cls(config, dtype=dtype, rngs=rngs)

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


class DeepseekV3Model(nnx.Module):

    def __init__(self, config: DeepseekV3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.num_dense_layers = config.first_k_dense_replace
        self.num_moe_layers = config.num_hidden_layers - config.first_k_dense_replace
        self.num_layers = config.num_hidden_layers

        self.embed_tokens = LoRAEmbed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            sharding=("tp", None),
            dtype=dtype,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            param_dtype=dtype,
            embedding_init=nnx.initializers.normal(),
            rngs=rngs,
        )

        # Create stacked dense layers (layers 0 to first_k_dense_replace - 1)
        if self.num_dense_layers > 0:

            def create_dense_layer(rngs: nnx.Rngs) -> DeepseekV3DecoderLayer:
                return DeepseekV3DecoderLayer(config, mlp_cls=DeepseekV3MLP, dtype=dtype, rngs=rngs)

            self.dense_layers = create_stacked_layers(create_dense_layer, self.num_dense_layers, rngs)
        else:
            self.dense_layers = None

        # Create stacked MoE layers (layers first_k_dense_replace to num_hidden_layers - 1)
        if self.num_moe_layers > 0:

            def create_moe_layer(rngs: nnx.Rngs) -> DeepseekV3DecoderLayer:
                return DeepseekV3DecoderLayer(config, mlp_cls=DeepseekV3MoE, dtype=dtype, rngs=rngs)

            self.moe_layers = create_stacked_layers(create_moe_layer, self.num_moe_layers, rngs)
        else:
            self.moe_layers = None

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
        is_training: bool = False,
    ) -> ModelOutput:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        hidden_states = self.embed_tokens(input_ids, adapter_indices=adapter_indices)
        all_hidden_states: list[jax.Array] = []

        # Split KV cache for dense and MoE layers
        dense_kv_cache = None
        moe_kv_cache = None
        if kv_cache is not None:
            dense_kv_cache, moe_kv_cache = kv_cache.split(self.num_dense_layers)

        # Forward through dense layers
        dense_kv_result = None
        if self.dense_layers is not None:
            hidden_states, dense_hidden_states, dense_kv_result = forward_layers(
                self.dense_layers,
                hidden_states,
                self.num_dense_layers,
                attention_mask=attention_mask,
                positions=positions,
                adapter_indices=adapter_indices,
                kv_cache=dense_kv_cache,
                output_hidden_states=output_hidden_states,
                gradient_checkpointing=self.config.gradient_checkpointing,
                is_training=is_training,
            )
            all_hidden_states.extend(dense_hidden_states)

        # Forward through MoE layers
        moe_kv_result = None
        if self.moe_layers is not None:
            hidden_states, moe_hidden_states, moe_kv_result = forward_layers(
                self.moe_layers,
                hidden_states,
                self.num_moe_layers,
                attention_mask=attention_mask,
                positions=positions,
                adapter_indices=adapter_indices,
                kv_cache=moe_kv_cache,
                output_hidden_states=output_hidden_states,
                gradient_checkpointing=self.config.gradient_checkpointing,
                is_training=is_training,
            )
            all_hidden_states.extend(moe_hidden_states)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        # Merge KV caches from dense and MoE layers
        new_kv_cache = KVCache.concatenate(dense_kv_result, moe_kv_result)

        return ModelOutput(
            last_hidden_state=hidden_states,
            kv_cache=new_kv_cache,
            hidden_states=all_hidden_states if output_hidden_states else None,
        )


class DeepseekV3ForCausalLM(nnx.Module, ModelForCausalLM, GeneratorMixin, LogitsProcessorMixin):

    def __init__(self, config: DeepseekV3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.model = DeepseekV3Model(config, dtype=dtype, rngs=rngs)

        if not self.config.tie_word_embeddings:
            self.lm_head = LoRALinear(
                config.hidden_size,
                config.vocab_size,
                sharding=(None, "tp"),
                use_bias=False,
                dtype=dtype,
                param_dtype=dtype,
                kernel_init=nnx.initializers.lecun_normal(),
                max_lora_adapters=config.max_lora_adapters,
                max_lora_rank=config.max_lora_rank,
                rngs=rngs,
            )

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
        is_training: bool = False,
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
            is_training=is_training,
        )

        return CausalLMOutput(
            last_hidden_state=outputs.last_hidden_state,
            kv_cache=outputs.kv_cache,
            hidden_states=outputs.hidden_states,
        )


Glm4MoeLiteForCausalLM = DeepseekV3ForCausalLM
