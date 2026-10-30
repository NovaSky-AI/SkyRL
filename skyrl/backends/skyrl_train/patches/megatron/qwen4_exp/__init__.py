"""Qwen3.8-Flash-Next (HF ``qwen4_exp``) for the Megatron backend.

No Megatron-Bridge release ships this model yet (open: NVIDIA-NeMo/Megatron-Bridge#6123 /
#6354, on top of the unmerged NVIDIA/Megatron-LM#7393 / #7553). Shaped for an upstream home in
Megatron-Bridge (``megatron/bridge/models/qwen/``):

- ``modules``: gated low-rank hyper-connection, final mixer, PLE n-gram embedding (row-sharded
  over the expert-parallel group), sigmoid-gated GDN, the frozen QSA indexer and the masked
  attention over its selection.
- ``layer_specs``: ``Qwen4ExpTransformerLayer``, the QSA-aware core attention and the
  GDN-or-attention / MoE block spec.
- ``provider``: ``Qwen4ExpModelProvider``.
- ``bridge``: ``Qwen4ExpBridge``; importing it registers the bridge.

Limits: CP=PP=1, and TP > 1 needs sequence parallelism; no MTP training; text only; PLE table
frozen. Up to ``indexer_budget + indexer_compress_ratio - 1`` tokens attention is exactly dense
and stays on the fused TE kernel. Longer sequences (where the QSA indexer drops keys) need packed
(THD) input and run ``qsa_sparse_attention``: cuDNN frontend's QSA kernels on Blackwell when the
installed cudnn-frontend has them, else FlexAttention on CUDA, else a chunked, masked torch SDPA.

Removal plan: ``skyrl/backends/skyrl_train/patches/megatron/README.md``.
"""
