"""DeepSeek-V4.1-Flash (HF ``deepseek_v41``, text only) for the Megatron backend.

Built on megatron-core's DeepSeek-V4 ``dsv4_hybrid`` HybridModel and megatron-bridge's ``DeepSeekV4Bridge``;
parity reference is ``inference/model.py`` / ``inference/engram.py`` in the DeepSeek-V4.1-Flash-1017 release.

- ``attention``: V4.1 attention (ratio 1/2 compressors, cross-layer KV / index sharing, candidate blocks).
- ``engram``: frozen, row-sharded Engram hash tables and the trainable gate around them.
- ``model``: mHC delayed pre-mix, no ``hc_head``, Engram insertion, per-forward state.
- ``layer_specs``: hybrid stack spec with the V4.1 attention.
- ``bridge``: ``DeepSeekV41Bridge``; importing it registers the bridge and transformers config.
"""
