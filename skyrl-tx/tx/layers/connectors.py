from flax import nnx
import jax
from jax import numpy as jnp

from tx.layers.util import Param, sinkhorn_knopp
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

        self.input_norm = RMSNorm(n * C, eps=eps, dtype=dtype, rngs=rngs)

        self.phi_pre = Param(n * C, n, dtype=dtype, kernel_init=nnx.initializers.normal(stddev=0.02), rngs=rngs)
        self.phi_post = Param(n * C, n, dtype=dtype, kernel_init=nnx.initializers.normal(stddev=0.02), rngs=rngs)
        self.phi_res = Param(n * C, n * n, dtype=dtype, kernel_init=nnx.initializers.normal(stddev=0.02), rngs=rngs)

        # Initialize biases for identity-like behavior:
        # H_pre = 1/n (uniform aggregation), H_post = 1 (full distribution), M = I (identity mixing)

        # H_pre = sigmoid(b_pre) = 1/n  =>  b_pre = inv_sigmoid(1/n)
        target_h_pre = jnp.array(1.0 / n, dtype=dtype)
        clamped = jnp.clip(target_h_pre, 1e-6, 1.0)
        inv_sigmoid = jnp.log(clamped) - jnp.log(1.0 - clamped)
        self.b_pre = nnx.Param(jnp.full((n,), inv_sigmoid, dtype=dtype))

        # H_post = 2 * sigmoid(b_post) = 1  =>  b_post = 0
        self.b_post = nnx.Param(jnp.zeros((n,), dtype=dtype))

        # M = sinkhorn(exp(b_res)) = I  =>  b_res = large diagonal matrix
        self.b_res = nnx.Param(10.0 * jnp.eye(n, dtype=dtype))

        self.alpha_pre = nnx.Param(jnp.array(0.0, dtype=dtype))
        self.alpha_post = nnx.Param(jnp.array(0.0, dtype=dtype))
        self.alpha_res = nnx.Param(jnp.array(0.0, dtype=dtype))

    def _get_params(self):
        """Get all connector params, with stop_gradient applied if not trainable."""
        sg = (lambda x: x) if self.trainable else jax.lax.stop_gradient
        return (
            sg(self.alpha_pre[...]), sg(self.alpha_post[...]), sg(self.alpha_res[...]),
            sg(self.phi_pre[...]), sg(self.phi_post[...]), sg(self.phi_res[...]),
            sg(self.b_pre[...]), sg(self.b_post[...]), sg(self.b_res[...]),
        )

    def pre(self, x: jax.Array) -> jax.Array:
        *batch_dims, n, C = x.shape

        x_norm = self.input_norm(x.reshape(*batch_dims, n * C))

        (alpha_pre, _, _, phi_pre, _, _, b_pre, _, _) = self._get_params()

        tilde_H_pre = alpha_pre * (x_norm @ phi_pre) + b_pre

        H_pre = jax.nn.sigmoid(tilde_H_pre)
        x_agg = (H_pre[..., None] * x).sum(axis=-2)
        return x_agg

    def post(self, residual: jax.Array, output: jax.Array) -> jax.Array:
        *batch_dims, n, C = residual.shape
        residual_norm = self.input_norm(residual.reshape(*batch_dims, n * C))

        (_, alpha_post, alpha_res, _, phi_post, phi_res, _, b_post, b_res) = self._get_params()
        tilde_H_post = alpha_post * (residual_norm @ phi_post) + b_post
        tilde_H_res = alpha_res * (residual_norm @ phi_res).reshape(*batch_dims, n, n) + b_res

        H_post = 2.0 * jax.nn.sigmoid(tilde_H_post)
        M = sinkhorn_knopp(tilde_H_res, self.sinkhorn_iters)

        y_dist = H_post[..., None] * output[..., None, :]
        x_mixed = M @ residual
        return x_mixed + y_dist
