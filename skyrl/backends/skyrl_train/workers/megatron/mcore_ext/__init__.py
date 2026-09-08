"""Megatron-Core extensions that SkyRL carries ahead of the pinned ``megatron-core``.

Everything in this package is shaped for an upstream home in ``megatron.core``:

- ``hyper_connection``: backport of ``megatron/core/transformer/hyper_connection.py`` from
  Megatron-LM ``main`` (mHC, Manifold-Constrained Hyper-Connections). Delete once the
  ``megatron-core`` pin includes it and import from ``megatron.core.transformer`` instead.
- ``mhc_transformer_layer``: ``HyperConnectionTransformerLayer`` for the pinned
  ``TransformerLayer``, with MoE MLP support (upstream's rejects MoE) and block-boundary
  stream expand/contract handled at the layer level (the pinned ``TransformerBlock`` has no
  mHC hooks).
- ``kda``: ``KimiDeltaAttention`` (KDA) linear attention, the Kimi-Linear / GLM-5.3-Flash
  recurrent layer, as an ``experimental_attention_variant``-style module next to
  ``megatron.core.ssm.gated_delta_net``.
"""
