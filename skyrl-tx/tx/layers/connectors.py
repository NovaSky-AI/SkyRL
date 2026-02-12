from flax import nnx
import jax
from jax import numpy as jnp

from tx.layers.util import Param, sinkhorn_knopp
from tx.layers.layernorm import RMSNorm


class LoRAConnector(nnx.Module):
    """
    Implementation of Manifold constrained HyperConnections (https://arxiv.org/pdf/2512.24880)

    Weights initialized with identity mapping; Default behaviour equates to residual networks.
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

        self.input_norm_weight = nnx.Param(jnp.ones((max_lora_adapters, n * C), dtype=dtype))
        self.phi_pre = Param(
            max_lora_adapters, n * C, n, dtype=dtype, kernel_init=nnx.initializers.normal(stddev=0.02), rngs=rngs
        )
        self.phi_post = Param(
            max_lora_adapters, n * C, n, dtype=dtype, kernel_init=nnx.initializers.normal(stddev=0.02), rngs=rngs
        )
        self.phi_res = Param(
            max_lora_adapters,
            n * C,
            n * n,
            dtype=dtype,
            kernel_init=nnx.initializers.normal(stddev=0.02),
            rngs=rngs,
        )

        target_h_pre = jnp.array(1.0 / n, dtype=dtype)
        clamped = jnp.clip(target_h_pre, 1e-6, 1.0 - 1e-6)
        inv_sigmoid = jnp.log(clamped) - jnp.log(1.0 - clamped)
        self.b_pre = nnx.Param(jnp.full((max_lora_adapters, n), inv_sigmoid, dtype=dtype))
        self.b_post = nnx.Param(jnp.zeros((max_lora_adapters, n), dtype=dtype))
        self.b_res = nnx.Param(jnp.broadcast_to(10.0 * jnp.eye(n, dtype=dtype), (max_lora_adapters, n, n)))

        self.alpha_pre = nnx.Param(jnp.zeros((max_lora_adapters,), dtype=dtype))
        self.alpha_post = nnx.Param(jnp.zeros((max_lora_adapters,), dtype=dtype))
        self.alpha_res = nnx.Param(jnp.zeros((max_lora_adapters,), dtype=dtype))

    def _get_adapter_indices(self, batch_size: int, adapter_indices: jax.Array | None) -> jax.Array:
        if adapter_indices is None:
            return jnp.zeros((batch_size,), dtype=jnp.int32)
        return adapter_indices.astype(jnp.int32)

    def _get_params(self, adapter_indices: jax.Array):
        sg = (lambda x: x) if self.trainable else jax.lax.stop_gradient
        input_norm_weight = sg(self.input_norm_weight[...])[adapter_indices]
        alpha_pre = sg(self.alpha_pre[...])[adapter_indices]
        alpha_post = sg(self.alpha_post[...])[adapter_indices]
        alpha_res = sg(self.alpha_res[...])[adapter_indices]
        phi_pre = sg(self.phi_pre[...])[adapter_indices]
        phi_post = sg(self.phi_post[...])[adapter_indices]
        phi_res = sg(self.phi_res[...])[adapter_indices]
        b_pre = sg(self.b_pre[...])[adapter_indices]
        b_post = sg(self.b_post[...])[adapter_indices]
        b_res = sg(self.b_res[...])[adapter_indices]
        return (
            input_norm_weight,
            alpha_pre,
            alpha_post,
            alpha_res,
            phi_pre,
            phi_post,
            phi_res,
            b_pre,
            b_post,
            b_res,
        )

    def _norm(self, x_flat: jax.Array, adapter_indices: jax.Array) -> jax.Array:
        input_norm_weight, *_ = self._get_params(adapter_indices)
        rms = jnp.sqrt(jnp.mean(x_flat**2, axis=-1, keepdims=True) + self.eps)
        return (input_norm_weight[:, None, :] * x_flat) / rms

    def pre(self, x: jax.Array, adapter_indices: jax.Array | None = None) -> jax.Array:
        B, T, n, C = x.shape
        adapter_indices = self._get_adapter_indices(B, adapter_indices)
        x_flat = x.reshape(B, T, n * C)
        x_norm = self._norm(x_flat, adapter_indices)

        (_, alpha_pre, _, _, phi_pre, _, _, b_pre, _, _) = self._get_params(adapter_indices)
        tilde_H_pre = alpha_pre[:, None, None] * jnp.einsum("btc,bcn->btn", x_norm, phi_pre) + b_pre[:, None, :]

        H_pre = jax.nn.sigmoid(tilde_H_pre)
        x_agg = (H_pre[..., None] * x).sum(axis=-2)
        return x_agg

    def post(self, residual: jax.Array, output: jax.Array, adapter_indices: jax.Array | None = None) -> jax.Array:
        B, T, n, C = residual.shape
        adapter_indices = self._get_adapter_indices(B, adapter_indices)
        residual_flat = residual.reshape(B, T, n * C)
        residual_norm = self._norm(residual_flat, adapter_indices)

        (_, _, alpha_post, alpha_res, _, phi_post, phi_res, _, b_post, b_res) = self._get_params(adapter_indices)

        tilde_H_post = alpha_post[:, None, None] * jnp.einsum("btc,bcn->btn", residual_norm, phi_post) + b_post[:, None, :]
        tilde_H_res = (
            alpha_res[:, None, None, None] * jnp.einsum("btc,bcn->btn", residual_norm, phi_res).reshape(B, T, n, n)
            + b_res[:, None, :, :]
        )

        H_post = 2.0 * jax.nn.sigmoid(tilde_H_post)
        M = sinkhorn_knopp(tilde_H_res, self.sinkhorn_iters)

        y_dist = H_post[..., None] * output[..., None, :]
        x_mixed = M @ residual
        return x_mixed + y_dist
