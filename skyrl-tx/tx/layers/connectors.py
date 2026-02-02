"""Connection mechanisms for transformer layers (residual, learned connectors, etc.)."""

from flax import nnx
import jax
from jax import numpy as jnp

from tx.layers.util import Param
from tx.layers.layernorm import RMSNorm


class Connector(nnx.Module):

    def __init__(
        self,
        hidden_dim: int,
        expansion_rate: int,
        *,
        trainable: bool = False,
        sinkhorn_iters: int = 20,
        eps: float = 1e-5,
        dtype: jnp.dtype,
        rngs: nnx.Rngs,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.expansion_rate = expansion_rate
        self.trainable = trainable
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        n = expansion_rate
        C = hidden_dim

        self.norm = RMSNorm(hidden_dim, eps=eps, dtype=dtype, rngs=rngs)

        self.phi_pre = Param(n * C, n, dtype=dtype, kernel_init=nnx.initializers.normal(stddev=0.02), rngs=rngs)
        self.phi_post = Param(n * C, n, dtype=dtype, kernel_init=nnx.initializers.normal(stddev=0.02), rngs=rngs)
        self.phi_res = Param(n * C, n * n, dtype=dtype, kernel_init=nnx.initializers.normal(stddev=0.02), rngs=rngs)

        self.b_pre = Param(n, dtype=dtype, rngs=rngs, kernel_init=nnx.initializers.zeros_init())
        self.b_post = Param(n, dtype=dtype, rngs=rngs, kernel_init=nnx.initializers.zeros_init())
        self.b_res = Param(n, n, dtype=dtype, rngs=rngs, kernel_init=nnx.initializers.zeros_init())

        self.alpha_pre = nnx.Param(jnp.array(0.01, dtype=dtype))
        self.alpha_post = nnx.Param(jnp.array(0.01, dtype=dtype))
        self.alpha_res = nnx.Param(jnp.array(0.01, dtype=dtype))

    def _sinkhorn_knopp(self, M: jax.Array) -> jax.Array:
        M = jnp.exp(M)
        for _ in range(self.sinkhorn_iters):
            M = M / (M.sum(axis=-1, keepdims=True) + self.eps)
            M = M / (M.sum(axis=-2, keepdims=True) + self.eps)
        return M

    def pre(self, x: jax.Array) -> jax.Array:
        *batch_dims, n, C = x.shape

        x_flat = x.reshape(*batch_dims, n * C)
        rms = jnp.sqrt(jnp.mean(x_flat * x_flat, axis=-1, keepdims=True) + self.eps)
        x_norm = x_flat / rms

        tilde_H_pre = self.alpha_pre[...] * (x_norm @ self.phi_pre[...]) + self.b_pre[...]
        tilde_H_post = self.alpha_post[...] * (x_norm @ self.phi_post[...]) + self.b_post[...]
        tilde_H_res = self.alpha_res[...] * (x_norm @ self.phi_res[...]).reshape(*batch_dims, n, n) + self.b_res[...]

        H_pre = jax.nn.sigmoid(tilde_H_pre)
        self._H_post = 2.0 * jax.nn.sigmoid(tilde_H_post)
        self._M = self._sinkhorn_knopp(tilde_H_res)

        x_agg = jnp.einsum("...i,...ic->...c", H_pre, x)
        x_normed = self.norm(x_agg)

        return x_normed

    def post(self, residual: jax.Array, output: jax.Array) -> jax.Array:
        y_dist = self._H_post[..., None] * output[..., None, :]
        x_mixed = jnp.einsum("...ij,...jc->...ic", self._M, residual)
        return x_mixed + y_dist
