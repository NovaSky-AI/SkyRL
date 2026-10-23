# Megatron Backend

## Megatron-Bridge

SkyRL uses Megatron-Bridge for HF-to-Megatron model conversion. Installed from git with a pinned rev in `[tool.uv.sources]`.

## Key abstractions
- `MegatronConfig` in `skyrl/train/config.py`
- `MegatronWorker` in `skyrl/backends/skyrl_train/workers/megatron/megatron_worker.py`.
- Custom bridges in `skyrl/backends/skyrl_train/workers/megatron/model_bridges.py` (e.g., `GLM47FlashBridge`).
- GLM-5.3-Flash (`glm5_next`): `skyrl/backends/skyrl_train/patches/megatron/glm5_next/` (provider, block
  spec, bridge; Megatron-Bridge-shaped) on top of `skyrl/backends/skyrl_train/patches/megatron/mcore_ext/`
  (Megatron-Core-shaped: `hyper_connection.py` backport of Megatron-LM main's mHC module,
  `mhc_transformer_layer.py` mHC layer with MoE support, `kda.py` KDA linear attention, `dsa_kpool.py` k-pool
  DSA indexer kernels). The DSA layers are megatron-core's `DSAttention` (NoPE MLA) with the k-pool top-k
  swapped in by `glm5_next/dsa.py`. Removal plan when bumping megatron: `patches/megatron/README.md`. Requires
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

## LoRA adapter import (`lora.init_adapter_path`)

- `skyrl/backends/skyrl_train/distributed/lora_import.py` (CPU, no megatron imports): parses a PEFT adapter
  into per-module pieces with their PEFT scale (`rank_pattern` / `alpha_pattern` / rslora), folds Tinker's
  `linear_attn.in_proj_q/k/v` into `in_proj_qkv`, stacks pieces block-diagonally, and folds per-module
  scales for vLLM's uniform `lora_alpha / r`.
- `skyrl/backends/skyrl_train/distributed/megatron/lora_import.py`: `SkyRLLoRA` (Megatron-Bridge `LoRA`
  with per-module `module_dims`), `plan_megatron_lora_import` (adapter module -> Megatron module via the
  bridge mapping registry, fused rank = sum of constituent ranks padded to TP), and
  `load_peft_adapter_into_megatron` (places tensors through the bridge's adapter conversion tasks; the
  fused `lora_B` row layout comes from running `_get_fused_adapter_linear_out_slices` on an index tensor).
- `distributed/megatron/lora_output_layer.py`: `enable_output_layer_attr_passthrough`, applied in the LoRA
  pre-wrap hook so a LoRA-wrapped `output_layer` answers `GPTModel._postprocess`'s reads/writes of
  `gather_output` / `sequence_parallel` with its base linear's attributes.
- Hooked in `MegatronWorker.configure_lora` (plan, targets, dims) and `make_megatron_module` (load after
  `provide_distributed_model`). MoE expert adapters and `canonical_lora` raise `NotImplementedError`.
- Test: `tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_lora_init_adapter.py` (HF+peft reference on
  a tiny Qwen3.6-style GDN model built by `megatron_tiny_models/create_qwen36_dense_tiny.py`).
