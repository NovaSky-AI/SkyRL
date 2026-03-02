from __future__ import annotations

import math

from flax import nnx
import jax
from jax import numpy as jnp
from jax.sharding import get_abstract_mesh

from skyrl.tx.layers.attention import dot_product_attention
from skyrl.tx.layers.lora import LoRAEmbed, LoRAExpert, LoRALinear
from skyrl.tx.layers.rotary_embedding import apply_rope
from skyrl.tx.layers.util import Param, prepare_routing, shard_map_ep
from skyrl.tx.models.configs import Qwen3_5Config
from skyrl.tx.models.types import CausalLMOutput, ModelForCausalLM, ModelOutput
from skyrl.tx.utils.generator import GeneratorMixin, KVCache
from skyrl.tx.utils.logits_processor import LogitsProcessorMixin, LMHead


def apply_partial_rope(
    q: jax.Array,
    k: jax.Array,
    positions: jax.Array,
    rotary_dim: int,
    rope_theta: float,
) -> tuple[jax.Array, jax.Array]:
    if rotary_dim <= 0:
        return q, k
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot = apply_rope(q_rot, positions, rotary_dim, rope_theta)
    k_rot = apply_rope(k_rot, positions, rotary_dim, rope_theta)
    return jnp.concatenate([q_rot, q_pass], axis=-1), jnp.concatenate([k_rot, k_pass], axis=-1)


def l2norm(x: jax.Array, axis: int = -1, eps: float = 1e-6) -> jax.Array:
    inv_norm = jax.lax.rsqrt(jnp.sum(x * x, axis=axis, keepdims=True) + eps)
    return x * inv_norm


def apply_mask_to_padding_states(hidden_states: jax.Array, attention_mask: jax.Array | None) -> jax.Array:
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        hidden_states = hidden_states * attention_mask[..., None].astype(hidden_states.dtype)
    return hidden_states


def recurrent_gated_delta_rule(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    initial_state: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    dtype = query.dtype
    compute_dtype = dtype
    query = l2norm(query.astype(compute_dtype), axis=-1)
    key = l2norm(key.astype(compute_dtype), axis=-1)
    value = value.astype(compute_dtype)
    g = g.astype(compute_dtype)
    beta = beta.astype(compute_dtype)

    query = query * (1.0 / math.sqrt(query.shape[-1]))

    # [B, T, H, D] -> [T, B, H, D]
    query = jnp.swapaxes(query, 0, 1)
    key = jnp.swapaxes(key, 0, 1)
    value = jnp.swapaxes(value, 0, 1)
    g = jnp.swapaxes(g, 0, 1)
    beta = jnp.swapaxes(beta, 0, 1)

    batch_size = query.shape[1]
    num_heads = query.shape[2]
    k_head_dim = query.shape[3]
    v_head_dim = value.shape[3]

    if initial_state is None:
        initial_state = jnp.zeros((batch_size, num_heads, k_head_dim, v_head_dim), dtype=compute_dtype)
    else:
        initial_state = initial_state.astype(compute_dtype)

    def step_fn(
        state: jax.Array,
        inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        q_t, k_t, v_t, g_t, beta_t = inputs
        decay = jnp.exp(g_t)[..., None, None]
        state = state * decay
        kv_mem = jnp.sum(state * k_t[..., :, None], axis=-2)
        delta = (v_t - kv_mem) * beta_t[..., None]
        state = state + k_t[..., :, None] * delta[..., None, :]
        out_t = jnp.sum(state * q_t[..., :, None], axis=-2)
        return state, out_t

    final_state, outputs = jax.lax.scan(step_fn, initial_state, (query, key, value, g, beta))
    outputs = jnp.swapaxes(outputs, 0, 1).astype(dtype)
    return outputs, final_state.astype(dtype)


class Qwen3_5RMSNorm(nnx.Module):

    def __init__(self, dim: int, *, eps: float, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.eps = eps
        self.weight = Param(
            dim,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.zeros_init(), (None,)),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        compute_dtype = x.dtype
        out = x.astype(compute_dtype)
        out = out * jax.lax.rsqrt(jnp.mean(out * out, axis=-1, keepdims=True) + self.eps)
        out = out * (1.0 + self.weight[...].astype(compute_dtype))
        return out.astype(x.dtype)


class Qwen3_5RMSNormGated(nnx.Module):

    def __init__(self, dim: int, *, eps: float, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.eps = eps
        self.weight = Param(
            dim,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.ones_init(), (None,)),
            rngs=rngs,
        )

    def __call__(self, hidden_states: jax.Array, gate: jax.Array) -> jax.Array:
        input_dtype = hidden_states.dtype
        compute_dtype = hidden_states.dtype
        out = hidden_states.astype(compute_dtype)
        out = out * jax.lax.rsqrt(jnp.mean(out * out, axis=-1, keepdims=True) + self.eps)
        out = out * self.weight[...].astype(compute_dtype)
        out = out * nnx.silu(gate.astype(compute_dtype))
        return out.astype(input_dtype)


class Qwen3_5Attention(nnx.Module):

    def __init__(self, config: Qwen3_5Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
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
        rope_parameters = getattr(config, "rope_parameters", None)
        assert isinstance(rope_parameters, dict), "Qwen3_5Attention requires config.rope_parameters to be a dict."
        assert (
            "partial_rotary_factor" in rope_parameters
        ), "Qwen3_5Attention requires rope_parameters['partial_rotary_factor']."
        assert "rope_theta" in rope_parameters, "Qwen3_5Attention requires rope_parameters['rope_theta']."
        partial_rotary_factor = rope_parameters["partial_rotary_factor"]
        rope_theta = rope_parameters["rope_theta"]

        rotary_dim = int(self.head_dim * partial_rotary_factor)
        rotary_dim = min(self.head_dim, rotary_dim)
        self.rotary_dim = rotary_dim - (rotary_dim % 2)
        self.rope_theta = rope_theta

        self.q_proj = LoRALinear(
            in_features=config.hidden_size,
            out_features=self.num_heads * self.head_dim * 2,
            sharding=("fsdp", tp_shard),
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=getattr(config, "attention_bias", False),
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )
        self.k_proj = LoRALinear(
            in_features=config.hidden_size,
            out_features=self.num_kv_heads * self.head_dim,
            sharding=("fsdp", tp_shard),
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=getattr(config, "attention_bias", False),
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )
        self.v_proj = LoRALinear(
            in_features=config.hidden_size,
            out_features=self.num_kv_heads * self.head_dim,
            sharding=("fsdp", tp_shard),
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=getattr(config, "attention_bias", False),
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )
        self.o_proj = LoRALinear(
            in_features=self.num_heads * self.head_dim,
            out_features=config.hidden_size,
            sharding=(tp_shard, "fsdp"),
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=getattr(config, "attention_bias", False),
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        *,
        attention_mask: jax.Array,
        positions: jax.Array,
        adapter_indices: jax.Array | None = None,
        kv_cache: tuple[jax.Array, jax.Array] | None = None,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        bsz, seq_len, _ = x.shape

        q_all = self.q_proj(x, adapter_indices=adapter_indices).reshape(bsz, seq_len, self.num_heads, self.head_dim * 2)
        q, gate = jnp.split(q_all, 2, axis=-1)
        gate = gate.reshape(bsz, seq_len, self.num_heads * self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(
            self.k_proj(x, adapter_indices=adapter_indices).reshape(bsz, seq_len, self.num_kv_heads, self.head_dim)
        )
        v = self.v_proj(x, adapter_indices=adapter_indices).reshape(bsz, seq_len, self.num_kv_heads, self.head_dim)

        q, k = apply_partial_rope(q, k, positions, self.rotary_dim, self.rope_theta)

        if kv_cache is not None:
            k, v = KVCache.update_layer(kv_cache, k, v, positions)

        updated_cache = (k, v)
        is_causal = kv_cache is None
        attn_output = dot_product_attention(q, k, v, attention_mask, is_causal, self.head_dim)
        attn_output = attn_output.reshape(bsz, seq_len, self.num_heads * self.head_dim)
        attn_output = attn_output * nnx.sigmoid(gate)
        return self.o_proj(attn_output, adapter_indices=adapter_indices), updated_cache


class Qwen3_5GatedDeltaNet(nnx.Module):

    def __init__(self, config: Qwen3_5Config, layer_idx: int, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        # Keep linear-attention projections replicated across TP for simplicity/stability.
        projection_size_qkv = self.key_dim * 2 + self.value_dim
        self.in_proj_qkv = LoRALinear(
            self.hidden_size,
            projection_size_qkv,
            sharding=("fsdp", None),
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=False,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )
        self.in_proj_z = LoRALinear(
            self.hidden_size,
            self.value_dim,
            sharding=("fsdp", None),
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=False,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )
        self.in_proj_b = LoRALinear(
            self.hidden_size,
            self.num_v_heads,
            sharding=("fsdp", None),
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=False,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )
        self.in_proj_a = LoRALinear(
            self.hidden_size,
            self.num_v_heads,
            sharding=("fsdp", None),
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=False,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )

        # Stored as [kernel, 1, channels] so existing safetensors transpose logic round-trips with HF Conv1d.
        self.conv1d_weight = Param(
            self.conv_kernel_size,
            1,
            self.conv_dim,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (None, None, None)),
            rngs=rngs,
        )
        self.dt_bias = Param(
            self.num_v_heads,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.initializers.ones_init(), (None,)),
            rngs=rngs,
        )
        self.A_log = Param(
            self.num_v_heads,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(
                lambda key, shape, dtype: jnp.log(
                    jax.random.uniform(key, shape, dtype=dtype, minval=1e-3, maxval=16.0)
                ),
                (None,),
            ),
            rngs=rngs,
        )

        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
        self.out_proj = LoRALinear(
            self.value_dim,
            self.hidden_size,
            sharding=(None, "fsdp"),
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            dtype=dtype,
            param_dtype=dtype,
            use_bias=False,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs,
        )

    def _get_conv_kernel(self) -> jax.Array:
        # [kernel, 1, channels] -> [channels, 1, kernel]
        return self.conv1d_weight[...].transpose((2, 1, 0))

    def _causal_conv_prefill(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        # x: [B, C, T]
        kernel = self._get_conv_kernel()
        seq_len = x.shape[-1]
        left_pad = self.conv_kernel_size - 1
        x_padded = jnp.pad(x, ((0, 0), (0, 0), (left_pad, 0)))
        out = jax.lax.conv_general_dilated(
            x_padded,
            kernel,
            window_strides=(1,),
            padding="VALID",
            feature_group_count=self.conv_dim,
            dimension_numbers=("NCH", "OIH", "NCH"),
        )
        out = nnx.silu(out[..., :seq_len])

        state_pad = max(self.conv_kernel_size - seq_len, 0)
        conv_state = jnp.pad(x, ((0, 0), (0, 0), (state_pad, 0)))[..., -self.conv_kernel_size :]
        return out, conv_state

    def _causal_conv_decode(self, x: jax.Array, conv_state: jax.Array) -> tuple[jax.Array, jax.Array]:
        # x: [B, C, T], conv_state: [B, C, K]
        kernel = self._get_conv_kernel()
        seq_len = x.shape[-1]
        x_full = jnp.concatenate([conv_state, x], axis=-1)
        new_state = x_full[..., -self.conv_kernel_size :]
        out_full = jax.lax.conv_general_dilated(
            x_full,
            kernel,
            window_strides=(1,),
            padding="VALID",
            feature_group_count=self.conv_dim,
            dimension_numbers=("NCH", "OIH", "NCH"),
        )
        out = nnx.silu(out_full[..., -seq_len:])
        return out, new_state

    def __call__(
        self,
        hidden_states: jax.Array,
        *,
        attention_mask: jax.Array | None,
        adapter_indices: jax.Array | None = None,
        conv_state: jax.Array | None = None,
        recurrent_state: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        mixed_qkv = self.in_proj_qkv(hidden_states, adapter_indices=adapter_indices).transpose((0, 2, 1))
        z = self.in_proj_z(hidden_states, adapter_indices=adapter_indices).reshape(
            batch_size, seq_len, -1, self.head_v_dim
        )
        b = self.in_proj_b(hidden_states, adapter_indices=adapter_indices)
        a = self.in_proj_a(hidden_states, adapter_indices=adapter_indices)

        use_precomputed = conv_state is not None and recurrent_state is not None and seq_len == 1
        if use_precomputed:
            mixed_qkv, new_conv_state = self._causal_conv_decode(mixed_qkv, conv_state)
        else:
            mixed_qkv, new_conv_state = self._causal_conv_prefill(mixed_qkv)

        mixed_qkv = mixed_qkv.transpose((0, 2, 1))
        q_end = self.key_dim
        k_end = self.key_dim * 2
        query_flat = mixed_qkv[..., :q_end]
        key_flat = mixed_qkv[..., q_end:k_end]
        value_flat = mixed_qkv[..., k_end:]

        query = query_flat.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key_flat.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value_flat.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = nnx.sigmoid(b)
        g = -jnp.exp(self.A_log[...].astype(jnp.float32)) * jax.nn.softplus(
            a.astype(jnp.float32) + self.dt_bias[...].astype(jnp.float32)
        )

        if self.num_v_heads // self.num_k_heads > 1:
            repeats = self.num_v_heads // self.num_k_heads
            query = jnp.repeat(query, repeats, axis=2)
            key = jnp.repeat(key, repeats, axis=2)

        core_out, new_recurrent_state = recurrent_gated_delta_rule(query, key, value, g, beta, recurrent_state)

        z_shape = z.shape
        core_out = self.norm(core_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        core_out = core_out.reshape(z_shape).reshape(batch_size, seq_len, -1)
        out = self.out_proj(core_out, adapter_indices=adapter_indices)
        return out, new_conv_state, new_recurrent_state


class Qwen3_5MLP(nnx.Module):

    def __init__(
        self,
        config: Qwen3_5Config,
        *,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
        intermediate_size: int | None = None,
    ) -> None:
        hidden_size = config.hidden_size
        intermediate_size = intermediate_size or config.intermediate_size

        self.gate_proj = LoRALinear(
            hidden_size,
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
            hidden_size,
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
            hidden_size,
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
        return self.down_proj(
            nnx.silu(self.gate_proj(x, adapter_indices)) * self.up_proj(x, adapter_indices), adapter_indices
        )


class Qwen3_5Experts(nnx.Module):

    def __init__(self, config: Qwen3_5Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.gate_proj = LoRAExpert(
            config.num_experts,
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
            config.num_experts,
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
            config.num_experts,
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
        selected_experts: jax.Array,
        routing_weights: jax.Array,
        adapter_indices: jax.Array | None = None,
    ) -> jax.Array:
        num_experts = self.config.num_experts
        top_k = self.config.num_experts_per_tok
        hidden_size = self.config.hidden_size

        ep = get_abstract_mesh().shape.get("ep", 1)
        assert num_experts % ep == 0, f"num_experts={num_experts} must be divisible by ep={ep}"

        hidden_expanded = jnp.repeat(hidden_states, top_k, axis=0)
        adapter_expanded = jnp.repeat(adapter_indices, top_k) if adapter_indices is not None else None
        hidden_sorted, group_sizes, unsort_indices, adapter_sorted = prepare_routing(
            hidden_expanded, selected_experts.ravel(), num_experts, adapter_indices=adapter_expanded
        )

        def forward(experts, hidden_sorted, group_sizes, unsort_indices, adapter_sorted, routing_weights):
            ep_rank = jax.lax.axis_index("ep")
            experts_per_rank = num_experts // jax.lax.axis_size("ep")
            group_offset = jnp.array([ep_rank * experts_per_rank], dtype=jnp.int32)

            gate = experts.gate_proj(hidden_sorted, group_sizes, adapter_sorted, group_offset=group_offset)
            up = experts.up_proj(hidden_sorted, group_sizes, adapter_sorted, group_offset=group_offset)
            down = experts.down_proj(nnx.silu(gate) * up, group_sizes, adapter_sorted, group_offset=group_offset)

            out = down[unsort_indices].reshape(-1, top_k, hidden_size)
            local_out = jnp.sum(out * routing_weights[..., None], axis=1)
            return jax.lax.psum(local_out, axis_name="ep")

        return shard_map_ep(self, forward, hidden_sorted, group_sizes, unsort_indices, adapter_sorted, routing_weights)


class Qwen3_5SparseMoeBlock(nnx.Module):

    def __init__(self, config: Qwen3_5Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
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
        self.experts = Qwen3_5Experts(config, dtype=dtype, rngs=rngs)
        self.shared_expert = Qwen3_5MLP(
            config,
            dtype=dtype,
            rngs=rngs,
            intermediate_size=config.shared_expert_intermediate_size,
        )
        self.shared_expert_gate = LoRALinear(
            config.hidden_size,
            1,
            sharding=("fsdp", None),
            use_bias=False,
            dtype=dtype,
            param_dtype=dtype,
            kernel_init=nnx.initializers.lecun_normal(),
            max_lora_adapters=config.max_lora_adapters,
            max_lora_rank=config.max_lora_rank,
            rngs=rngs,
        )

    def __call__(self, hidden_states: jax.Array, adapter_indices: jax.Array | None = None) -> jax.Array:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.reshape(-1, hidden_dim)
        adapter_flat = jnp.repeat(adapter_indices, seq_len) if adapter_indices is not None else None

        router_logits = self.gate(hidden_states_flat)
        routing_weights = nnx.softmax(router_logits, axis=-1)
        routing_weights, selected_experts = jax.lax.top_k(routing_weights, k=self.config.num_experts_per_tok)
        routing_weights = routing_weights / jnp.sum(routing_weights, axis=-1, keepdims=True)
        routing_weights = routing_weights.astype(hidden_states_flat.dtype)

        expert_output = self.experts(hidden_states_flat, selected_experts, routing_weights, adapter_flat)
        shared_output = self.shared_expert(hidden_states_flat, adapter_indices=adapter_flat)
        shared_gate = nnx.sigmoid(self.shared_expert_gate(hidden_states_flat, adapter_indices=adapter_flat))

        final_hidden_states = expert_output + shared_gate * shared_output
        return final_hidden_states.reshape(batch_size, seq_len, hidden_dim)


class Qwen3_5DecoderLayer(nnx.Module):

    def __init__(self, config: Qwen3_5Config, layer_idx: int, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.layer_type = config.layer_types[layer_idx]
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs
        )

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx, dtype=dtype, rngs=rngs)
        else:
            self.self_attn = Qwen3_5Attention(config, dtype=dtype, rngs=rngs)

        use_moe = (
            layer_idx not in getattr(config, "mlp_only_layers", [])
            and getattr(config, "num_experts", 0) > 0
            and (layer_idx + 1) % getattr(config, "decoder_sparse_step", 1) == 0
        )
        if use_moe:
            self.mlp = Qwen3_5SparseMoeBlock(config, dtype=dtype, rngs=rngs)
        else:
            self.mlp = Qwen3_5MLP(config, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        hidden_states: jax.Array,
        *,
        attention_mask: jax.Array | None,
        positions: jax.Array,
        adapter_indices: jax.Array | None = None,
        kv_cache: tuple[jax.Array, jax.Array] | None = None,
        conv_state: jax.Array | None = None,
        recurrent_state: jax.Array | None = None,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array] | None, jax.Array | None, jax.Array | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states, new_conv_state, new_recurrent_state = self.linear_attn(
                hidden_states,
                attention_mask=attention_mask,
                adapter_indices=adapter_indices,
                conv_state=conv_state,
                recurrent_state=recurrent_state,
            )
            updated_kv = None
        else:
            hidden_states, updated_kv = self.self_attn(
                hidden_states,
                attention_mask=attention_mask,
                positions=positions,
                adapter_indices=adapter_indices,
                kv_cache=kv_cache,
            )
            new_conv_state = conv_state
            new_recurrent_state = recurrent_state

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, adapter_indices=adapter_indices)
        hidden_states = residual + hidden_states

        return hidden_states, updated_kv, new_conv_state, new_recurrent_state


class Qwen3_5Model(nnx.Module):

    def __init__(self, config: Qwen3_5Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
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

        layer_types = getattr(config, "layer_types", None)
        if layer_types is None:
            interval = getattr(config, "full_attention_interval", 4)
            layer_types = [
                "linear_attention" if (i + 1) % interval else "full_attention" for i in range(config.num_hidden_layers)
            ]
            config.layer_types = layer_types

        assert len(config.layer_types) == config.num_hidden_layers
        self.layer_types = tuple(config.layer_types)
        self.layers = nnx.List(
            [Qwen3_5DecoderLayer(config, i, dtype=dtype, rngs=rngs) for i in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)

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
        updated_keys: list[jax.Array] = []
        updated_values: list[jax.Array] = []
        updated_conv_states: list[jax.Array] = []
        updated_recurrent_states: list[jax.Array] = []

        batch_size = input_ids.shape[0]
        dtype = hidden_states.dtype

        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            layer_type = self.layer_types[layer_idx]
            if layer_type == "full_attention":
                layer_kv = (kv_cache.keys[layer_idx], kv_cache.values[layer_idx]) if kv_cache is not None else None
                hidden_states, updated_kv, new_conv_state, new_recurrent_state = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    positions=positions,
                    adapter_indices=adapter_indices,
                    kv_cache=layer_kv,
                )
                assert updated_kv is not None
                updated_keys.append(updated_kv[0])
                updated_values.append(updated_kv[1])

                if kv_cache is not None and kv_cache.conv_states is not None and kv_cache.recurrent_states is not None:
                    updated_conv_states.append(kv_cache.conv_states[layer_idx])
                    updated_recurrent_states.append(kv_cache.recurrent_states[layer_idx])
                else:
                    updated_conv_states.append(jnp.zeros((batch_size, 0, 0), dtype=dtype))
                    updated_recurrent_states.append(jnp.zeros((batch_size, 0, 0, 0), dtype=dtype))
            else:
                layer_conv_state = None
                layer_recurrent_state = None
                if kv_cache is not None and kv_cache.conv_states is not None and kv_cache.recurrent_states is not None:
                    layer_conv_state = kv_cache.conv_states[layer_idx]
                    layer_recurrent_state = kv_cache.recurrent_states[layer_idx]

                linear_mask = None if kv_cache is not None else attention_mask
                hidden_states, _, new_conv_state, new_recurrent_state = layer(
                    hidden_states,
                    attention_mask=linear_mask,
                    positions=positions,
                    adapter_indices=adapter_indices,
                    conv_state=layer_conv_state,
                    recurrent_state=layer_recurrent_state,
                )
                assert new_conv_state is not None and new_recurrent_state is not None
                updated_conv_states.append(new_conv_state)
                updated_recurrent_states.append(new_recurrent_state)

                if kv_cache is not None:
                    updated_keys.append(kv_cache.keys[layer_idx])
                    updated_values.append(kv_cache.values[layer_idx])
                else:
                    updated_keys.append(jnp.zeros((batch_size, 0, 0, 0), dtype=dtype))
                    updated_values.append(jnp.zeros((batch_size, 0, 0, 0), dtype=dtype))

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        if is_training:
            new_kv_cache = None
        else:
            new_kv_cache = KVCache.update(
                kv_cache,
                updated_keys,
                updated_values,
                positions,
                attention_mask,
                conv_states=updated_conv_states,
                recurrent_states=updated_recurrent_states,
            )

        return ModelOutput(
            last_hidden_state=hidden_states,
            kv_cache=new_kv_cache,
            hidden_states=all_hidden_states if output_hidden_states else None,
        )


class Qwen3_5Backbone(nnx.Module):

    def __init__(self, config: Qwen3_5Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        # Keep the nested `model.language_model.*` parameter structure used by HF checkpoints.
        self.language_model = Qwen3_5Model(config, dtype=dtype, rngs=rngs)


class Qwen3_5ForCausalLM(nnx.Module, ModelForCausalLM, GeneratorMixin, LogitsProcessorMixin):

    def __init__(self, config: Qwen3_5Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        model_config = config.get_text_config()
        self.config = model_config
        self.model = Qwen3_5Backbone(model_config, dtype=dtype, rngs=rngs)

        if model_config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = LoRALinear(
                model_config.hidden_size,
                model_config.vocab_size,
                sharding=(None, "tp"),
                use_bias=False,
                dtype=dtype,
                param_dtype=dtype,
                kernel_init=nnx.initializers.lecun_normal(),
                max_lora_adapters=model_config.max_lora_adapters,
                max_lora_rank=model_config.max_lora_rank,
                rngs=rngs,
            )

    def get_lm_head(self) -> LMHead:
        """Return the lm_head callable for logits computation."""
        return self.lm_head or self.model.language_model.embed_tokens.T

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

        outputs = self.model.language_model(
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


Qwen3_5ForConditionalGeneration = Qwen3_5ForCausalLM
Qwen3_5MoeForConditionalGeneration = Qwen3_5ForCausalLM
