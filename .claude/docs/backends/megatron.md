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

Set `trainer.policy.model.expert_mxfp8.enabled=true` on SM100/SM103 to run
routed expert GEMMs with Transformer Engine MXFP8 and configure vLLM for
expert-only online MXFP8. Checkpoints and weight sync remain high precision.
Set `trainer.policy.model.expert_mxfp8.persistent=true` with
`trainer.policy.megatron_config.ddp_config.fp8_param_gather=true` to keep
routed-expert primary parameters in MXFP8 between optimizer steps. Set
`generator.inference_engine.serialized_weight_sync_mode=serialized_mxfp8` to send
routed experts to vLLM as MXFP8 data and E8M0 scales. Unmerged LoRA is not
supported. Shared model layouts, MXFP8 format handling, and Megatron/vLLM
adapters live in `skyrl/backends/skyrl_train/quantization/`.

## Expert NVFP4

Set `trainer.policy.model.expert_nvfp4.enabled=true` to quantize routed
experts with Transformer Engine NVFP4. Serialized rollout updates require
`generator.inference_engine.serialized_weight_sync_mode=serialized_nvfp4`
and use ModelOpt NVFP4 loading in vLLM. Set
`expert_nvfp4.row_scaled_activation=true` for dynamically quantized per-token
W4A4 activations; otherwise rollout uses W4A16.

Row-scaled activations, scoped Four-Over-Six, and dequantized backward are
configured under `trainer.policy.model.expert_nvfp4`. These options require a
Transformer Engine build that exposes the corresponding NVFP4 recipe fields.

The Humans&-style RL recipe keeps BF16 primary parameters and uses all of
these settings together:

```text
trainer.policy.model.expert_nvfp4.enabled=true
trainer.policy.model.expert_nvfp4.training=true
trainer.policy.model.expert_nvfp4.persistent=false
trainer.policy.model.expert_nvfp4.backward_override=dequantized
trainer.policy.model.expert_nvfp4.row_scaled_activation=true
trainer.policy.model.expert_nvfp4.disable_rht=true
trainer.policy.model.expert_nvfp4.disable_stochastic_rounding=true
trainer.policy.model.expert_nvfp4.disable_2d_quantization=true
trainer.policy.model.expert_nvfp4.four_over_six_scope=all
trainer.policy.model.expert_nvfp4.four_over_six_e4m3_use_256_scope=all
trainer.policy.model.expert_nvfp4.four_over_six_error_mode=MSE
trainer.policy.model.expert_nvfp4.four_over_six_error_use_fast_math=true
trainer.policy.model.expert_nvfp4.disable_fp4_quant_fast_math=true
trainer.policy.model.expert_nvfp4.high_precision_last_layers=8
generator.inference_engine.serialized_weight_sync_mode=serialized_nvfp4
```

## Test Requirements

Megatron GPU tests need: `NVTE_FLASH_ATTN=0`
