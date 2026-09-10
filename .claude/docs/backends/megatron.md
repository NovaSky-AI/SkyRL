# Megatron Backend

## Megatron-Bridge

SkyRL uses Megatron-Bridge for HF-to-Megatron model conversion. Installed from git with a pinned rev in `[tool.uv.sources]`.

## Key abstractions
- `MegatronConfig` in `skyrl/train/config.py`
- `MegatronWorker` in `skyrl/backends/skyrl_train/workers/megatron/megatron_worker.py`.
- Custom bridges in `skyrl/backends/skyrl_train/workers/megatron/model_bridges.py` (e.g., `GLM47FlashBridge`).
- GLM-5.3-Flash (`glm5_next`): `skyrl/backends/skyrl_train/workers/megatron/glm5_next/` (provider, block
  spec, bridge; Megatron-Bridge-shaped) on top of `skyrl/backends/skyrl_train/workers/megatron/mcore_ext/`
  (Megatron-Core-shaped: `hyper_connection.py` backport of Megatron-LM main's mHC module,
  `mhc_transformer_layer.py` mHC layer with MoE support, `kda.py` KDA linear attention). The DSA layers reuse
  megatron-core's `DSAttention` (NoPE MLA, token-level indexer) and are exact for sequences up to `index_topk`
  (2048) tokens; the k-pool indexer for longer sequences is not implemented and raises. Requires
  `language_model_only=True` (VL checkpoint) and packed sequences (`remove_microbatch_padding=True`).

## Parallelism Strategies

For picking TP/PP/EP/CP/SP sizes, invoke the `parallelism-strategies` skill.

Key strategies:
- **Tensor Parallelism (TP)**: Splits layers across GPUs within an NVLink domain. Use TP ≤ GPUs per node. Applicable for non-MoE linear layers.
- **Pipeline Parallelism (PP)**: Splits model layers across nodes. Use for cross-node scaling.
- **Data Parallelism (DP)**: Implicit — `world_size / (TP * PP)`. Each DP rank processes different data.
- **Sequence Parallelism (SP)**: Requires TP > 1. Splits along sequence dimension for LayerNorm/Dropout.
- **Context Parallelism (CP)**: For sequences > 8K tokens. Splits attention computation across GPUs.
- **Expert Parallelism (EP)**: For MoE models. Distributes experts across GPUs.
- **Expert Tensor Parallelism (ETP)**: For MoE models. Tensor parallelism for the expert layers.

Note: Sequence parallelism is auto-enabled when `tensor_model_parallel_size > 1` — there is no separate config field for it.

## Test Requirements

Megatron GPU tests need: `NVTE_FLASH_ATTN=0`
