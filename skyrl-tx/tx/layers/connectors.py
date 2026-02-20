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
        trainable: bool = False,
        sinkhorn_iters: int = 20,
        eps: float = 1e-5,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.expansion_rate = expansion_rate
        self.max_lora_adapters = max_lora_adapters
        self.trainable = trainable
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        n = expansion_rate
        C = hidden_dim

        # Phi matrices are zero-initialized so that alpha * x @ 0 + bias = bias at init.
        self.input_norm_weight = nnx.Param(jnp.ones((max_lora_adapters, n * C), dtype=dtype))
        self.phi_pre = Param(
            max_lora_adapters, n * C, n, dtype=dtype, kernel_init=nnx.initializers.zeros_init(), rngs=rngs
        )
        self.phi_post = Param(
            max_lora_adapters, n * C, n, dtype=dtype, kernel_init=nnx.initializers.zeros_init(), rngs=rngs
        )
        self.phi_res = Param(
            max_lora_adapters, n * C, n * n, dtype=dtype, kernel_init=nnx.initializers.zeros_init(), rngs=rngs,
        )

        # H_pre = sigmoid(b_pre) = 1/n: uniform aggregation across streams
        self.b_pre = nnx.Param(jnp.full((max_lora_adapters, n), default_b_pre(n), dtype=dtype))

        # H_post = 2 * sigmoid(b_post) = 1: standard residual for all streams at init.
        # Training discovers stream specialization (memory vs update roles).
        self.b_post = nnx.Param(jnp.zeros((max_lora_adapters, n), dtype=dtype))

        # M ~= I: strong identity mixing via Sinkhorn (minimal cross-stream leakage)
        self.b_res = nnx.Param(jnp.broadcast_to(10.0 * jnp.eye(n, dtype=dtype), (max_lora_adapters, n, n)))

        self.alpha_pre = nnx.Param(jnp.ones((max_lora_adapters,), dtype=dtype))
        self.alpha_post = nnx.Param(jnp.ones((max_lora_adapters,), dtype=dtype))
        self.alpha_res = nnx.Param(jnp.ones((max_lora_adapters,), dtype=dtype))

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

    def _norm(self, x_flat: jax.Array, adapter_indices: jax.Array) -> jax.Array:
        """Separate norm from layernorm.RMSNorm due to adapter indexing and trainability."""
        input_norm_weight = self.input_norm_weight[adapter_indices]
        rms = jnp.sqrt(jnp.mean(x_flat**2, axis=-1, keepdims=True) + self.eps)
        return (input_norm_weight[:, None, :] * x_flat) / rms

    def pre(self, x: jax.Array, adapter_indices: jax.Array | None = None) -> tuple[jax.Array, jax.Array]:
        B, T, n, C = x.shape
        if self.expansion_rate == 1:
            # Single-stream fast path: pre is identity on the residual stream.
            return x[..., 0, :], x.reshape(B, T, n * C)

        adapter_indices = self._get_adapter_indices(B, adapter_indices)
        x_flat = x.reshape(B, T, n * C)
        x_norm = self._norm(x_flat, adapter_indices)

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
        self,
        residual: jax.Array,
        output: jax.Array,
        residual_norm: jax.Array,
        adapter_indices: jax.Array | None = None
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
