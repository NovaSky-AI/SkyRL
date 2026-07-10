# Megatron Backend

## Megatron-Bridge

SkyRL uses Megatron-Bridge for HF-to-Megatron model conversion. Installed from git with a pinned rev in `[tool.uv.sources]`.

## LoRA

- Configured via `trainer.policy.model.lora` (rank/alpha/targets) plus `trainer.policy.megatron_config.lora_config` (`lora_type`, `merge_lora`, `experts_shared_outer_loras`).
- MoE grouped-expert LoRA: `experts_shared_outer_loras=true` gives per-expert inner matrices with shared outer matrices (Megatron-Bridge `SharedOuterGroupedExpertAdapter`); only `lora_type="lora"`.
- Weight sync: `merge_lora=true` (default) merges adapters into full weights; `merge_lora=false` exports adapter safetensors (`_save_lora_adapters_and_sync`), with fused-MoE layout conversion for vLLM in `_convert_moe_experts_lora_to_vllm`.

## Key abstractions
- `MegatronConfig` in `skyrl/train/config.py`
- `MegatronWorker` in `skyrl/backends/skyrl_train/workers/megatron/megatron_worker.py`.
- Custom bridges in `skyrl/backends/skyrl_train/workers/megatron/model_bridges.py` (e.g., `GLM47FlashBridge`).

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
