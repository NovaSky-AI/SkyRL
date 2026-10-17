"""Qwen3.8-Flash-Next (HF ``qwen4_exp``) for the Megatron backend.

No Megatron-Bridge release ships this model yet (open: NVIDIA-NeMo/Megatron-Bridge#6123 /
#6354, on top of the unmerged NVIDIA/Megatron-LM#7393 / #7553). Shaped for an upstream home in
Megatron-Bridge (``megatron/bridge/models/qwen/``):

- ``modules``: gated low-rank hyper-connection, final mixer, PLE n-gram embedding (head-sharded
  over the expert-parallel group), sigmoid-gated GDN, frozen QSA indexer weights.
- ``layer_specs``: ``Qwen4ExpTransformerLayer`` and the GDN-or-attention / MoE block spec.
- ``provider``: ``Qwen4ExpModelProvider``.
- ``bridge``: ``Qwen4ExpBridge``; importing it registers the bridge.

Limits: TP=CP=PP=1; sequences up to ``indexer_budget + indexer_compress_ratio - 1`` tokens
(the QSA indexer is exactly dense there); no MTP training; text only; PLE table frozen.

Removal plan: ``skyrl/backends/skyrl_train/patches/megatron/README.md``.
"""
