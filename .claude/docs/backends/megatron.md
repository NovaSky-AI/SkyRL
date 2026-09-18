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

## Offload/backload under colocation

`MegatronStrategy.offload_to_cpu` / `backload_to_gpu` (`distributed/megatron/megatron_strategy.py`)
delegate to `offload_megatron_model_to_cpu` / `load_megatron_model_to_gpu` in `megatron_utils.py`,
passing `is_lora`. Behavior differs by mode:

- **Full-parameter**: the fused DDP param buffers (`model_chunk.buffers` + `expert_parallel_buffers`)
  are offloaded with Megatron's `buffer.offload_to_cpu(move_params=True)` and reloaded with
  `reload_from_cpu`.
- **LoRA** (`is_lora=True`): the fused buffers hold only grad-requiring params, i.e. the adapters, and
  are **skipped on both offload and backload**. Adapter params stay GPU-resident for the entire run,
  including while colocated vLLM engines generate. Frozen base weights live outside the buffers and
  are offloaded per-param: on first offload each one is written to a file under
  `SKYRL_FROZEN_OFFLOAD_DIR` (default `/data/skyrl/frozen-offload`, `0` disables), mapped with
  `torch.from_file`, and unlinked; the mapping is cached on `param._offload_cpu_data` and reused on
  later offloads (the GPU side is just freed). Failure falls back to pinned RAM for all frozen params.
  Grad buffers (`offload_grad_buffers`) and optimizer state are offloaded in both modes.

`WorkerDispatch.save_weights_for_sampler` (used by both the trainer and the Tinker API) picks one
of two colocated sync paths via `_is_lora_no_merge()` (#2064):

- **Classic** (`merge_lora=True`, or full-parameter): sleep the engines if awake, backload the full
  policy model (`need_model=True`; under LoRA this restores the frozen masters from their mmap/pinned
  copies), offload the optimizer per `offload_after_step`, `empty_cache`, wake engine weights,
  broadcast, offload the model again, wake the KV cache.
- **Adapter-only** (`merge_lora=False`): never backloads the masters. If a preceding forward/optim
  phase left any model or optimizer resident, offload it all, `empty_cache`, then export the adapter
  from the fused buffers (GPU-resident through offload, see above) and load it on vLLM.
  `_finish_weight_sync` is a no-op on this path. The Tinker backend (`skyrl_train_backend.py`,
  `_adapter_only_sync`) skips the eager post-create engine sleep here; training ops sleep the
  engines on demand at level 1 (`_sleep_inference_engines`), and a cold sample offloads the trainer
  (`offload_for_sampling`) and wakes them (`_wake_inference_engines_for_sampling`).

## Test Requirements

Megatron GPU tests need: `NVTE_FLASH_ATTN=0`
