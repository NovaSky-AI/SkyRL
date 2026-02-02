"""Connection mechanisms for transformer layers (residual, learned connectors, etc.)."""

from flax import nnx
import jax
from jax import numpy as jnp

from tx.layers.util import Param
from tx.layers.layernorm import RMSNorm


class Connector(nnx.Module):
    """
    Implementation of Manifold constrained HyperConnections (https://arxiv.org/pdf/2512.24880)

    Weights initialized with identity mapping; Default behaviour equates to residual networks.
    """

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

        # Initialize biases for identity-like behavior:
        # H_pre = 1/n (uniform aggregation), H_post = 1 (full distribution), M = I (identity mixing)

        # H_pre = sigmoid(b_pre) = 1/n  =>  b_pre = logit(1/n)
        target_h_pre = jnp.array(1.0 / n, dtype=dtype)
        clamped = jnp.clip(target_h_pre, 1e-6, 1.0 - 1e-6)
        logit_1_over_n = jnp.log(clamped) - jnp.log(1.0 - clamped)
        self.b_pre = nnx.Param(jnp.full((n,), logit_1_over_n, dtype=dtype))

        # H_post = 2 * sigmoid(b_post) = 1  =>  b_post = 0
        self.b_post = nnx.Param(jnp.zeros((n,), dtype=dtype))

        # M = sinkhorn(exp(b_res)) = I  =>  b_res = large diagonal matrix
        self.b_res = nnx.Param(10.0 * jnp.eye(n, dtype=dtype))

        self.alpha_pre = nnx.Param(jnp.array(0.0, dtype=dtype))
        self.alpha_post = nnx.Param(jnp.array(0.0, dtype=dtype))
        self.alpha_res = nnx.Param(jnp.array(0.0, dtype=dtype))

    def _sinkhorn_knopp(self, M: jax.Array) -> jax.Array:
        M = jnp.exp(M)
        for _ in range(self.sinkhorn_iters):
            M = M / (M.sum(axis=-1, keepdims=True) + self.eps)
            M = M / (M.sum(axis=-2, keepdims=True) + self.eps)
        return M

    def _get_params(self):
        """Get all connector params, with stop_gradient applied if not trainable."""
        sg = (lambda x: x) if self.trainable else jax.lax.stop_gradient
        return (
            sg(self.alpha_pre[...]), sg(self.alpha_post[...]), sg(self.alpha_res[...]),
            sg(self.phi_pre[...]), sg(self.phi_post[...]), sg(self.phi_res[...]),
            sg(self.b_pre[...]), sg(self.b_post[...]), sg(self.b_res[...]),
            sg(self.norm.weight[...]),
        )

    def pre(self, x: jax.Array) -> jax.Array:
        *batch_dims, n, C = x.shape

        x_flat = x.reshape(*batch_dims, n * C)
        rms = jnp.sqrt(jnp.mean(x_flat * x_flat, axis=-1, keepdims=True) + self.eps)
        x_norm = x_flat / rms

        (alpha_pre, alpha_post, alpha_res, phi_pre, phi_post, phi_res,
         b_pre, b_post, b_res, norm_weight) = self._get_params()

        tilde_H_pre = alpha_pre * (x_norm @ phi_pre) + b_pre
        tilde_H_post = alpha_post * (x_norm @ phi_post) + b_post
        tilde_H_res = alpha_res * (x_norm @ phi_res).reshape(*batch_dims, n, n) + b_res

        H_pre = jax.nn.sigmoid(tilde_H_pre)
        self._H_post = 2.0 * jax.nn.sigmoid(tilde_H_post)
        self._M = self._sinkhorn_knopp(tilde_H_res)

        x_agg = jnp.einsum("...i,...ic->...c", H_pre, x)
        rms_norm = jnp.sqrt(jnp.mean(x_agg**2, axis=-1, keepdims=True) + self.norm.eps)
        x_normed = norm_weight * x_agg / rms_norm

        return x_normed

    def post(self, residual: jax.Array, output: jax.Array) -> jax.Array:
        y_dist = self._H_post[..., None] * output[..., None, :]
        x_mixed = jnp.einsum("...ij,...jc->...ic", self._M, residual)
        return x_mixed + y_dist
