from flax import nnx
import jax
from jax import numpy as jnp

from tx.layers.lora import LoRAExpert, LoRALinear
from tx.layers.util import prepare_routing
from tx.models.configs import Qwen3Config
from tx.layers.common import SwiGLUMLP
from tx.layers.layernorm import RMSNorm
from tx.models.llama3 import Llama3Attention, Llama3Model, Llama3ForCausalLM
from tx.models.types import CausalLMOutput
from tx.utils.generator import KVCache, compute_positions


class Qwen3Attention(Llama3Attention):
    """Qwen3 attention with QK normalization, inheriting from Llama3Attention.

    This class uses the _process_q_k hook to apply QK-Norm, avoiding code
    duplication of the full attention mechanism. See PR #657 review feedback.
    """

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        # Initialize parent Llama3Attention
        super().__init__(config, dtype=dtype, rngs=rngs)

        # Add QK-Norm layers specific to Qwen3
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)

    def _process_q_k(self, q: jax.Array, k: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Apply QK-Norm to query and key tensors (Qwen3-specific)."""
        return self.q_norm(q), self.k_norm(k)


class Qwen3Experts(nnx.Module):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.gate_proj = LoRAExpert(
            config.num_experts,
            config.hidden_size,
            config.moe_intermediate_size,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, None, "tp")),
            rngs=rngs,
        )
        self.up_proj = LoRAExpert(
            config.num_experts,
            config.hidden_size,
            config.moe_intermediate_size,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, None, "tp")),
            rngs=rngs,
        )
        self.down_proj = LoRAExpert(
            config.num_experts,
            config.moe_intermediate_size,
            config.hidden_size,
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, "tp", None)),
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
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, None)),
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
            self.mlp = SwiGLUMLP(
                config.hidden_size,
                config.intermediate_size,
                max_lora_adapters=config.max_lora_adapters,
                max_lora_rank=config.max_lora_rank,
                dtype=dtype,
                rngs=rngs,
            )

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


class Qwen3Model(Llama3Model):
    """Qwen3 model inheriting from Llama3Model with Qwen3-specific decoder layers.

    The only difference from Llama3Model is the use of Qwen3DecoderLayer (with QK-Norm)
    instead of Llama3DecoderLayer. Embedding and __call__ are inherited from parent.
    """

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        # Initialize parent (sets up embed_tokens and norm)
        super().__init__(config, dtype=dtype, rngs=rngs)

        # Override layers to use Qwen3DecoderLayer (with QK-Norm attention)
        self.layers = nnx.List(
            [Qwen3DecoderLayer(config, dtype=dtype, rngs=rngs) for _ in range(config.num_hidden_layers)]
        )


class Qwen3ForCausalLM(Llama3ForCausalLM):
    """Qwen3 for causal LM, inheriting from Llama3ForCausalLM with LoRA support for lm_head."""

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        # Use Qwen3Model instead of Llama3Model
        self.model = Qwen3Model(config, dtype=dtype, rngs=rngs)

        # Use LoRALinear for lm_head (Qwen3-specific)
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
            # Use LoRALinear lm_head with adapter_indices
            logits = self.lm_head(hidden_states, adapter_indices=adapter_indices)

        return CausalLMOutput(
            logits=logits,
            last_hidden_state=outputs.last_hidden_state,
            kv_cache=outputs.kv_cache,
            hidden_states=outputs.hidden_states,
        )


Qwen3MoeForCausalLM = Qwen3ForCausalLM
