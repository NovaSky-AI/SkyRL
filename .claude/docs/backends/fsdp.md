# FSDP Backend

## Overview

Default backend (`trainer.strategy=fsdp`). Uses PyTorch FSDP2 for distributed training.

- **FSDPConfig** in `skyrl/train/config.py`.
- **FSDPStrategy** in `skyrl/backends/skyrl_train/distributed/fsdp_strategy.py`.
- **FSDPWeightExtractor** for extracting weights from sharded parameters (in `skyrl/backends/skyrl_train/workers/fsdp/fsdp_worker.py`).
- **HFRouterReplay** for MoE Rollout Routing Replay (in `skyrl/backends/skyrl_train/utils/hf_router_replay.py`).

## CPU Offload

- `trainer.fsdp_config.cpu_offload=true` offloads optimizer states to CPU.
- Also available for reference model: `ref.fsdp_config.cpu_offload=true`.
- Useful when GPU memory is low but adds overhead.
- NOT to be confused with `offload_after_step`: This is for colocated training where training state is offloaded to CPU after a training step is complete, so that the inference workers can be loaded on the same GPUs.

## Sharding

- `FULL_SHARD` (default): Shards parameters, gradients, and optimizer states.
- `NO_SHARD`: Falls back when world_size=1.
- `fsdp_size`: Controls sharding group size. `-1` = auto (full world). For Hybrid Sharded Data Parallelism (HSDP), use `fsdp_size=<num_gpus_per_node>`

## MoE Routing Replay (R3)

- Enable: `trainer.policy.fsdp_config.moe_enable_routing_replay=true` + `generator.inference_engine.enable_return_routed_experts=true` + `generator.inference_engine.distributed_executor_backend=mp` (`ray` errors at startup — R3 can hang on it). Details: `docs/content/docs/algorithms/off_policy_correction.mdx`.
- `hf_router_replay.py` forward-hooks each softmax [`*TopKRouter` gate](https://github.com/huggingface/transformers/blob/v5.8.0/src/transformers/models/olmoe/modeling_olmoe.py#L353) (OlMoE, Qwen2/3-MoE), matched as `.mlp.gate` with `top_k`/`num_experts` attrs.
- Per-worker flag: the ref has its own `trainer.ref.fsdp_config.moe_enable_routing_replay`. Co-enable it when KL is on (`use_kl_loss`/`use_kl_in_reward`), else KL compares replay-routed policy logprobs against naturally-routed ref logprobs (a startup warning is logged in this case).
