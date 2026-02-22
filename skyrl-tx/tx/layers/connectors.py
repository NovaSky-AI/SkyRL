from collections.abc import Callable
from typing import Any

from flax import nnx
import jax
from jax import numpy as jnp

from tx.layers.util import Param


def is_connector_path(path: tuple[Any, ...]) -> bool:
    normalized_path = tuple(p.key if hasattr(p, "key") else p.name if hasattr(p, "name") else p for p in path)
    return any(name in normalized_path for name in ("attn_connector", "mlp_connector"))


def _logit(x: jax.Array) -> jax.Array:
    """Inverse sigmoid: logit(x) = log(x / (1-x))."""
    x = jnp.clip(x, 1e-6, 1.0 - 1e-6)
    return jnp.log(x) - jnp.log(1.0 - x)


def default_b_pre(n: int, dtype: jnp.dtype = jnp.float32) -> jax.Array:
    """H_pre = sigmoid(b_pre) = 1/n: uniform aggregation across streams."""
    return _logit(jnp.array(1.0 / n, dtype=jnp.float32)).astype(dtype)


class LoRAConnector(nnx.Module):
    """
    Implementation of Manifold constrained HyperConnections (https://arxiv.org/pdf/2512.24880)

    Initialized as exact identity (standard residual): H_pre = 1/n, H_post = 1, M = I.
    Training discovers stream specialization through input-dependent routing (alpha = 1).
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion_rate: int,
        *,
        max_lora_adapters: int,
        sinkhorn_iters: int = 20,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ) -> None:
        self.expansion_rate = expansion_rate
        self.sinkhorn_iters = sinkhorn_iters
        n = expansion_rate
        C = hidden_dim

        # Phi matrices are zero-initialized so that alpha * x @ 0 + bias = bias at init.
        self.phi_pre = Param(
            max_lora_adapters,
            n * C,
            n,
            dtype=dtype,
            kernel_init=lambda _, shape, init_dtype: LoRAConnector._default_array("phi_pre", shape, init_dtype),
            rngs=rngs,
        )
        self.phi_post = Param(
            max_lora_adapters,
            n * C,
            n,
            dtype=dtype,
            kernel_init=lambda _, shape, init_dtype: LoRAConnector._default_array("phi_post", shape, init_dtype),
            rngs=rngs,
        )
        self.phi_res = Param(
            max_lora_adapters,
            n * C,
            n * n,
            dtype=dtype,
            kernel_init=lambda _, shape, init_dtype: LoRAConnector._default_array("phi_res", shape, init_dtype),
            rngs=rngs,
        )

        # H_pre = sigmoid(b_pre) = 1/n: uniform aggregation across streams
        self.b_pre = nnx.Param(LoRAConnector._default_array("b_pre", (max_lora_adapters, n), dtype))

        # H_post = 2 * sigmoid(b_post) ~= 1: narrow spectrum [0.8, 1.2] breaks stream
        # symmetry while preserving mean = 1 (standard residual behavior on average).
        self.b_post = nnx.Param(LoRAConnector._default_array("b_post", (max_lora_adapters, n), dtype))

        # M ~= I: identity mixing via Sinkhorn (minimal cross-stream leakage)
        self.b_res = nnx.Param(LoRAConnector._default_array("b_res", (max_lora_adapters, n, n), dtype))

        self.alpha_pre = nnx.Param(LoRAConnector._default_array("alpha_pre", (max_lora_adapters,), dtype))
        self.alpha_post = nnx.Param(LoRAConnector._default_array("alpha_post", (max_lora_adapters,), dtype))
        self.alpha_res = nnx.Param(LoRAConnector._default_array("alpha_res", (max_lora_adapters,), dtype))

    @staticmethod
    def _default_array(key_name: str, shape: tuple[int, ...], dtype: jnp.dtype) -> jax.Array:
        if key_name in {"alpha_pre", "alpha_post", "alpha_res"}:
            return jnp.full(shape, 1.0, dtype=dtype)
        if key_name in {"phi_pre", "phi_post", "phi_res"}:
            return jnp.zeros(shape, dtype=dtype)
        if key_name == "b_pre":
            n = shape[-1]
            return jnp.full(shape, default_b_pre(n, dtype), dtype=dtype)
        if key_name == "b_post":
            n = shape[-1]
            return jnp.broadcast_to(jnp.linspace(-0.2, 0.2, n, dtype=dtype), shape)
        if key_name == "b_res":
            n = shape[-1]
            return jnp.broadcast_to(10.0 * jnp.eye(n, dtype=dtype), shape)
        raise ValueError(f"Unknown connector key: {key_name}")

    @staticmethod
    def reset_adapter_slot(key_name: str, connector_slot: jax.Array) -> jax.Array:
        return LoRAConnector._default_array(key_name, connector_slot.shape, connector_slot.dtype)

    def _get_adapter_indices(self, batch_size: int, adapter_indices: jax.Array | None) -> jax.Array:
        if adapter_indices is None:
            return jnp.zeros((batch_size,), dtype=jnp.int32)
        return adapter_indices.astype(jnp.int32)

    @staticmethod
    def _sinkhorn_knopp(M: jax.Array, iters: int = 20) -> jax.Array:
        """Project a matrix onto the set of doubly stochastic matrices."""
        M = jnp.exp(M)

        def step(_, mat):
            mat = mat / mat.sum(axis=-1, keepdims=True)
            mat = mat / mat.sum(axis=-2, keepdims=True)
            return mat

        return jax.lax.fori_loop(0, iters, step, M)

    def pre(
        self,
        x: jax.Array,
        input_norm: Callable[[jax.Array], jax.Array],
        adapter_indices: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        B, T, n, C = x.shape
        if self.expansion_rate == 1:
            # Single-stream fast path: pre is identity on the residual stream.
            return x[..., 0, :], x.reshape(B, T, n * C)

        adapter_indices = self._get_adapter_indices(B, adapter_indices)
        # Apply input_norm independently to each of the n streams.
        x_norm = input_norm(x).reshape(B, T, n * C)

        alpha_pre = self.alpha_pre[adapter_indices]
        phi_pre = self.phi_pre[adapter_indices]
        b_pre = self.b_pre[adapter_indices]
        pre_logits = x_norm @ phi_pre
        tilde_H_pre = alpha_pre[:, None, None] * pre_logits + b_pre[:, None, :]

        H_pre = jax.nn.sigmoid(tilde_H_pre)
        x_agg = (H_pre[..., None] * x).sum(axis=-2)

        # Return residual norm for future use by post()
        return x_agg, x_norm

    def post(
        self, residual: jax.Array, output: jax.Array, residual_norm: jax.Array, adapter_indices: jax.Array | None = None
    ) -> jax.Array:
        B, T, n, C = residual.shape
        if self.expansion_rate == 1:
            # Single-stream fast path: plain residual connection.
            return residual + output[..., None, :]

        adapter_indices = self._get_adapter_indices(B, adapter_indices)

        alpha_post = self.alpha_post[adapter_indices]
        alpha_res = self.alpha_res[adapter_indices]
        phi_post = self.phi_post[adapter_indices]
        phi_res = self.phi_res[adapter_indices]
        b_post = self.b_post[adapter_indices]
        b_res = self.b_res[adapter_indices]

        post_logits = residual_norm @ phi_post
        tilde_H_post = alpha_post[:, None, None] * post_logits + b_post[:, None, :]
        res_logits = residual_norm @ phi_res
        tilde_H_res = alpha_res[:, None, None, None] * res_logits.reshape(B, T, n, n) + b_res[:, None, :, :]

        H_post = 2.0 * jax.nn.sigmoid(tilde_H_post)
        M = self._sinkhorn_knopp(tilde_H_res, self.sinkhorn_iters)

        y_dist = H_post[..., None] * output[..., None, :]
        x_mixed = M @ residual
        return x_mixed + y_dist
