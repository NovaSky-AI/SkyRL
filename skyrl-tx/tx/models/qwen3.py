from flax import nnx
import jax
from jax import numpy as jnp
from transformers import Qwen3Config


def Param(*shape: int, dtype: jnp.dtype, kernel_init: nnx.Initializer, rngs: nnx.Rngs):
    return nnx.Param(kernel_init(rngs.param(), shape, dtype))


class RMSNorm(nnx.Module):
    def __init__(self, size: int, *, eps: float = 1e-6, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.eps = eps
        self.weight = Param(size, dtype=dtype, kernel_init=nnx.with_partitioning(nnx.initializers.normal(), jax.P(None)), rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return self.weight * x / rms


class MultiHeadProj(nnx.Module):

    def __init__(self, subscripts: str, *shape: int, sharding: jax.P, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.subscripts = subscripts
        self.weight = Param(*shape, dtype=dtype, kernel_init=nnx.with_partitioning(nnx.initializers.normal(), sharding), rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.einsum(self.subscripts, x, self.weight)


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
        self.head_dim = config.head_dim
        
        self.q_proj = MultiHeadProj("BMK,KNH->BMNH", config.hidden_size, self.num_heads, self.head_dim, sharding=jax.P(None, "tp", None), dtype=dtype, rngs=rngs)
        self.k_proj = MultiHeadProj("BMK,KNH->BMNH", config.hidden_size, self.num_kv_heads, self.head_dim, sharding=jax.P(None, "tp", None), dtype=dtype, rngs=rngs)
        self.v_proj = MultiHeadProj("BMK,KNH->BMNH", config.hidden_size, self.num_kv_heads, self.head_dim, sharding=jax.P(None, "tp", None), dtype=dtype, rngs=rngs)
        self.o_proj = MultiHeadProj("BMKH,KHN->BMN", self.num_heads, self.head_dim, config.hidden_size, sharding=jax.P("tp", None, None), dtype=dtype, rngs=rngs)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        *,
        attention_mask: jax.Array | None = None,
        output_attentions: bool | None = None
    ) -> tuple[jax.Array, jax.Array | None]:
        q = self.q_norm(self.q_proj(x))
        k = self.k_norm(self.k_proj(x))
        v = self.v_proj(x)

        position_ids = jnp.arange(x.shape[1])[None, :].repeat(x.shape[0], axis=0)

        q = apply_rope(q, position_ids, self.head_dim, self.config.rope_theta)
        k = apply_rope(k, position_ids, self.head_dim, self.config.rope_theta)

        if self.num_kv_heads != self.num_heads:
            num_groups = self.num_heads // self.num_kv_heads
            k = jnp.repeat(k, num_groups, axis=2)
            v = jnp.repeat(v, num_groups, axis=2)

        attn_weights = jnp.einsum("BMNH,BTNH->BNMT", q, k)
        attn_weights = attn_weights / jnp.sqrt(self.head_dim)

        causal_mask = jnp.tril(jnp.ones((x.shape[1], x.shape[1])))[None, None, :, :]

        if attention_mask is not None:
            causal_mask *= attention_mask[:, None, None, :]

        attn_weights = jnp.where(causal_mask == 0, -jnp.inf, attn_weights)
        attn_weights = nnx.softmax(attn_weights, axis=-1)
        attn_output = jnp.einsum("BNMT,BTNH->BMNH", attn_weights, v)

        if not output_attentions:
            attn_weights = None

        return self.o_proj(attn_output), attn_weights
        

class Qwen3MLP(nnx.Module):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.gate_proj = nnx.Linear(
            config.hidden_size, config.intermediate_size, use_bias=False, dtype=dtype, param_dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.nn.linear.default_kernel_init, jax.P(None, "tp")), rngs=rngs
        )
        self.up_proj = nnx.Linear(
            config.hidden_size, config.intermediate_size, use_bias=False, dtype=dtype, param_dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.nn.linear.default_kernel_init, jax.P(None, "tp")), rngs=rngs
        )
        self.down_proj = nnx.Linear(
            config.intermediate_size, config.hidden_size, use_bias=False, dtype=dtype, param_dtype=dtype,
            kernel_init=nnx.with_partitioning(nnx.nn.linear.default_kernel_init, jax.P("tp", None)), rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.down_proj(nnx.silu(self.gate_proj(x)) * self.up_proj(x))
        

class Qwen3DecoderLayer(nnx.Module):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)
        self.self_attn = Qwen3Attention(config, dtype=dtype, rngs=rngs)
        self.mlp = Qwen3MLP(config, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        hidden_states: jax.Array,
        *,
        attention_mask: jax.Array | None = None,
        output_attentions: bool | None = None
    ) -> tuple[jax.Array, jax.Array | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, self_attn_weights


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
        self.layers = nnx.List([Qwen3DecoderLayer(config, dtype=dtype, rngs=rngs) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        input_ids: jax.Array,
        *,
        attention_mask: jax.Array | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None
    ) -> dict[str, jax.Array | list[jax.Array]]:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )

        hidden_states = self.embed_tokens(input_ids)

        all_hidden_states: list[jax.Array] = []
        all_self_attns: list[jax.Array] = []

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            hidden_states, self_attns = layer(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

            if output_attentions:
                all_self_attns.append(self_attns)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attns,
        }


class Qwen3ForCausalLM(nnx.Module):

    def __init__(self, config: Qwen3Config, *, dtype: jnp.dtype, rngs: nnx.Rngs) -> None:
        self.config = config
        self.model = Qwen3Model(config, dtype=dtype, rngs=rngs)
        if not self.config.tie_word_embeddings:
            self.lm_head = nnx.Linear(
                config.hidden_size, config.vocab_size, use_bias=False, dtype=dtype, param_dtype=dtype,
                kernel_init=nnx.with_partitioning(nnx.nn.linear.default_kernel_init, jax.P(None, "tp")), rngs=rngs
            )

    def __call__(
        self,
        input_ids: jax.Array,
        *,
        attention_mask: jax.Array | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None
    ) -> dict[str, jax.Array | list[jax.Array]]:
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        hidden_states = outputs["last_hidden_state"]
        if self.config.tie_word_embeddings:
            logits = hidden_states @ self.model.embed_tokens.embedding.value.T
        else:
            logits = self.lm_head(hidden_states)

        return {"logits": logits, **outputs}
