# Megatron Backend

## Megatron-Bridge

SkyRL uses Megatron-Bridge for HF-to-Megatron model conversion. Installed from git with a pinned rev in `[tool.uv.sources]`.

- `MegatronConfig` in `skyrl/train/config.py`
- Custom bridges in `skyrl/backends/skyrl_train/workers/megatron/model_bridges.py` (e.g., `GLM47FlashBridge`).
- `MegatronWorker` in `skyrl/backends/skyrl_train/workers/megatron/megatron_worker.py`.

## Parallelism Strategies

Reference: https://docs.nvidia.com/nemo/megatron-bridge/latest/skills/perf-techniques/parallelism-strategies/SKILL.html

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

Megatron GPU tests need: `NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=0`
