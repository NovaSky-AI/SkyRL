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
  `mhc_transformer_layer.py` mHC layer with MoE support, `kda.py` KDA linear attention). The DSA layers are
  megatron-core's `DSAttention` (NoPE MLA, k-pool indexer); `glm5_next/dsa.py` adds a long-sequence guard and
  the opt-in `SKYRL_DSA_INDEXER_TP_SHARD`. Removal plan when bumping megatron: `patches/megatron/README.md`. Requires
  `language_model_only=True` (VL checkpoint) and packed sequences (`remove_microbatch_padding=True`).
- Beam (`beam`): upstream Megatron-Bridge `BeamBridge` (`HybridModel`, attention + MLP hybrid layer per decoder
  layer) and vLLM 0.33's `BeamForCausalLM`, served with LoRA through `patches/vllm/patch_beam_lora.py` (a
  `vllm.general_plugins` entry point registering a `SupportsLoRA` subclass). SkyRL also carries the transformers
  config (`patches/transformers/configuration_beam.py`), the CP comm-type default, the fused LM-head softcap and the
  FP8 spec (`weight_sync/fp8/models/beam.py`); see `patches/megatron/README.md`. Tiny-model GPU rows: `beam-tiny_*` in
  `gpu_ci/megatron/test_megatron_{,lora_}models.py` (`megatron_tiny_models/create_beam_tiny.py`).

- Inkling2 (`inkling2_mm_model`): `skyrl/backends/skyrl_train/patches/megatron/inkling2/` (block
  attention residuals carried through `TransformerBlock` as one packed state tensor, relative-bias
  attention on FlexAttention, short convolutions, stock megatron-core MoE). The bridge loads the
  NVFP4 / MXFP8 release directly (experts dequantized to BF16 on import). Requires
  `language_model_only=True`; no PP or CP; TP needs sequence parallelism. `reference.py` is the
  plain-PyTorch oracle the tests compare against. Removal plan: `patches/megatron/README.md`.
  vLLM serves it through `patches/vllm/inkling2/` on the pinned Inkling2 vLLM build
  (`pyproject.toml`). LoRA syncs merged (`merge_lora=true`) or as an adapter
  (`merge_lora=false`, served by the build's `vllm/models/inkling2/lora.py`; use
  `normalize_moe_lora=true`, as the full-rank 384-expert adapter is ~19 GiB per vLLM GPU).

## Optimizer state on disk

`megatron_config.optimizer_offload_to_disk` (`skyrl/backends/skyrl_train/distributed/megatron/optimizer_offload_to_disk.py`):
`OffloadedDistributedOptimizer` subclasses megatron-core's `DistributedOptimizer`, keeps fp32 mains + Adam moments
for each rank's contiguous DDP-bucket shards in one file per rank and optimizer, and streams them through the GPU in
chunks during `step_with_ready_grads`. Its `_build_model_and_main_param_groups` is generated from the installed
megatron-core source by replacing one line (`shard_main_param = shard_model_param.clone().float()`); a megatron-core
bump that moves that line fails `tests/backends/skyrl_train/distributed/test_optimizer_offload_to_disk.py` (run with
`--extra megatron`) and the GPU test. Correctness test (bit-identity vs the regular optimizer, 2 GPUs):
`tests/backends/skyrl_train/gpu/gpu_ci/test_optimizer_offload_to_disk.py`. See also
`skyrl/backends/skyrl_train/patches/megatron/README.md`.

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
