"""GLM-5.3-Flash (HF ``glm5_next``) for the Megatron backend.

Shaped for an upstream home in Megatron-Bridge (``megatron/bridge/models/glm/``):

- ``provider``: ``Glm5NextModelProvider`` (MLA provider + KDA / mHC / kpool fields).
- ``layer_specs``: per-layer KDA-or-DSA, dense-or-MoE block spec built on
  ``HyperConnectionTransformerLayer``.
- ``dsa``: the GLM-5.3-Flash flavour of ``DSAttention`` (NoPE MLA + kpool-compressed indexer).
- ``bridge``: ``Glm5NextBridge``, HF <-> Megatron parameter mapping for the language model of the
  ``Glm5NextForConditionalGeneration`` checkpoint. Importing it registers the bridge.

The Megatron-Core-side building blocks (mHC layer, KDA module) live in
``skyrl.backends.skyrl_train.workers.megatron.mcore_ext``.
"""
