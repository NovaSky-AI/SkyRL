"""Megatron-Core extensions that SkyRL carries ahead of the pinned ``megatron-core``.

Removal plan (what retires each module, touchpoints outside this package, tests):
``skyrl/backends/skyrl_train/patches/megatron/README.md``.

Everything in this package is shaped for an upstream home in ``megatron.core``:

- ``hyper_connection``: ``RMSNormInputHyperConnectionModule``, megatron-core's mHC module with
  a standard-RMSNorm input normalization. Delete once ``TransformerConfig`` carries the
  input-norm knobs upstream.
- ``mhc_transformer_layer``: ``HyperConnectionTransformerLayer`` with MoE MLP support --
  megatron-core's own mHC layer rejects MoE sub-layers, and ``TransformerBlock`` owns the
  block-boundary stream expand/contract.
- ``kda``: ``KimiDeltaAttention`` (KDA) linear attention, the Kimi-Linear / GLM-5.3-Flash
  recurrent layer, as an ``experimental_attention_variant``-style module next to
  ``megatron.core.ssm.gated_delta_net``.
"""
