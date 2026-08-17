# Megatron Backend

## Megatron-Bridge

SkyRL uses Megatron-Bridge for HF-to-Megatron model conversion. Installed from git with a pinned rev in `[tool.uv.sources]`.

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

## Expert MXFP8

Set `trainer.policy.model.expert_mxfp8.enabled=true` on SM100/SM103 to apply
Transformer Engine MXFP8 only to routed expert linears. Set
`persistent=true` with `ddp_config.fp8_param_gather=true` to keep those
primary parameters in MXFP8 between optimizer steps.

Use `generator.inference_engine.serialized_weight_sync_mode=mxfp8`
with NCCL/CUDA-IPC to send expert E4M3 data and E8M0 scales to vLLM. Delta
sync remains a dense pre-quantization transport and cannot consume serialized
quantization chunks.

## Test Requirements

Megatron GPU tests need: `NVTE_FLASH_ATTN=0`
