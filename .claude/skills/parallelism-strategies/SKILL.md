---
name: parallelism-strategies
description: Operational guide for choosing and combining parallelism strategies (TP/PP/DP/CP/SP/EP) for the SkyRL Megatron backend. Use when sizing parallelism for a new model, debugging OOM/throughput on a given cluster topology, or extending an existing recipe to a new GPU count. Includes model-size sizing rules, hardware topology mapping, sequence-length thresholds, MoE-specific patterns, and pitfalls.
---

# Parallelism Strategy Selection

> **Source.** Adapted from NVIDIA Megatron-Bridge docs:
> `https://docs.nvidia.com/nemo/megatron-bridge/latest/skills/perf-techniques/parallelism-strategies/SKILL.html`
> Refreshed from Megatron-Bridge `b7b4e11b`.
>
> **SkyRL adaptation.** Upstream uses `cfg.model.<field>`. In SkyRL these are surfaced through `MegatronConfig` (`skyrl/train/config.py`) and set on the CLI as e.g. `trainer.megatron.tensor_model_parallel_size=...` for SFT and `trainer.policy.megatron_config.` for RL. Field names are otherwise identical.
>
> **Scope.** Megatron backend only. FSDP and JAX backends do not use TP/PP/EP — see `.claude/docs/backends/fsdp.md` and `.claude/docs/backends/jax.md`.

## Decision by Model Size

### Dense models

| Model size | GPUs | Recommended starting point |
| --- | --- | --- |
| < 1B | 1-8 | DP only |
| 1-10B | 8-16 | TP=2-4 + DP |
| 10-70B | 16-64 | TP=4-8 + PP=2-4 + DP |
| 70-175B | 64-256 | TP=8 + PP=4-8 + DP |
| 175-500B | 256-1024 | TP=8 + PP=8-16 + CP=2 + DP |

### MoE models

MoE parallelism differs from dense. Because only a fraction of parameters are active per token, TP can often stay at 1 or 2 — the active parameter shard already fits on a single GPU. EP is the primary scaling dimension, with PP handling cross-node layer distribution.

| Model (total / active) | TP | PP | EP | Notes |
| --- | --- | --- | --- | --- |
| OLMoE 7B / 1B | 1 | 1 | 8 | EP only, fits single node |
| Moonlight 16B / 3B | 2 | 1 | 8 | small TP for shared layers |
| DeepSeek-V2 236B / 21B | 1 | 4 | 32 | no TP at all |
| GLM-4.5 Air 106B / 12B | 1 | 4 | 8 | no TP at all |
| Qwen3 30B-A3B | 4 | 2 | 4 | |
| GLM-4.5 355B / 32B | 2 | 8 | 16 | |
| Qwen3 235B-A22B | 4 | 16 | 8 | CP=2 for pretrain |
| DeepSeek-V3 671B / 37B | 2 | 16 | 64 | TP=2, not 8 |
| Kimi-K2 1T | 2 | 16 | 32 | |

Key patterns:

- TP is sized by **active** params, not total params. A 671B MoE with 37B active needs far less TP than a 70B dense model.
- EP scales with expert count. Common: `EP = num_experts` or `num_experts / experts_per_gpu`.
- PP handles depth. Large MoE models use PP=8-16 across nodes.
- ETP (expert tensor parallelism) is rarely used. Llama 4 is an exception (ETP=4).

These are starting points, not hard rules. Always profile the first iteration to verify memory and communication.

## Decision by Hardware Topology

Single node with NVLink:

```python
cfg.model.tensor_model_parallel_size = 8
```

Multiple nodes with InfiniBand/ RoCE:

```python
cfg.model.tensor_model_parallel_size = 8
cfg.model.pipeline_model_parallel_size = N
```

Limited network (Ethernet):

```python
cfg.model.tensor_model_parallel_size = 4
cfg.model.pipeline_model_parallel_size = M
```

Stable rule: **keep TP within a single NVLink domain.** Use PP or DP for cross-node scaling. TP across nodes is almost always a performance loss.

## Decision by Sequence Length

| Sequence length | Recommendation |
| --- | --- |
| < 2K | standard TP + PP + DP |
| 2K-8K | add SP (`sequence_parallel=True`) |
| 8K-32K | add CP=2 |
| 32K+ | add CP=4-8, consider `a2a+p2p` for large CP |

## Combined Parallelism

3D parallelism (TP + PP + DP):

```python
cfg.model.tensor_model_parallel_size = 4
cfg.model.pipeline_model_parallel_size = 4
cfg.model.sequence_parallel = True
```

4D parallelism (TP + PP + CP + DP):

```python
cfg.model.tensor_model_parallel_size = 8
cfg.model.pipeline_model_parallel_size = 8
cfg.model.context_parallel_size = 2
cfg.model.sequence_parallel = True
```

MoE with EP + PP (e.g. DeepSeek-V2 236B on 128 GPUs):

```python
cfg.model.tensor_model_parallel_size = 1
cfg.model.pipeline_model_parallel_size = 4
cfg.model.expert_model_parallel_size = 32
cfg.model.sequence_parallel = False
```

MoE with small TP + PP + EP (e.g. DeepSeek-V3 671B on 256 GPUs):

```python
cfg.model.tensor_model_parallel_size = 2
cfg.model.pipeline_model_parallel_size = 16
cfg.model.expert_model_parallel_size = 64
cfg.model.sequence_parallel = True
```

DP size is always implicit:

```
data_parallel_size = world_size / (TP * PP * CP)
expert_data_parallel_size = world_size / (PP * EP * ETP)
```

## Minimum GPU Count

The dense `TP * CP` group and expert `EP * ETP` group share the GPUs in each pipeline
stage. Only pipeline stages use separate GPU sets:

```
minimum_gpus = PP * max(TP * CP, EP * ETP)
```

Do not multiply all five sizes together. For example, PP=1, TP=2, CP=1, EP=8,
ETP=1 needs 8 GPUs, not 16. Extra GPUs increase dense data parallelism and expert data
parallelism.

## Memory Estimation

Without parallelism (70B model, FP16):

```
parameters:       140 GB
gradients:        140 GB
optimizer states: 280 GB (Adam)
activations:       48 GB (batch=1, seq=4K)
total:            608 GB
```

With TP=4, PP=4, DP=4 (64 GPUs):

```
parameters:        8.75 GB per GPU
gradients:         8.75 GB per GPU
optimizer states: 17.50 GB per GPU
activations:       3.00 GB per GPU
total:           ~38    GB per GPU
```

## Code Anchors

Parallelism dimensions set in the model provider:

```python
model_config = GPTModelProvider(
    tensor_model_parallel_size=2,
    # ...
)
```

DP size:

```
data_parallel_size = world_size / (tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size)
```

Megatron-Bridge wires parallelism into process groups via:

```python
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=model_config.tensor_model_parallel_size,
    pipeline_model_parallel_size=model_config.pipeline_model_parallel_size,
    context_parallel_size=model_config.context_parallel_size,
    hierarchical_context_parallel_sizes=model_config.hierarchical_context_parallel_sizes,
    expert_model_parallel_size=model_config.expert_model_parallel_size,
    ...
)
```

## Pitfalls

1. **TP across nodes destroys throughput.** Always keep TP within a single NVLink domain.
2. **PP without interleaving has large pipeline bubbles.** Use `virtual_pipeline_model_parallel_size` when possible.
3. **SP requires `tensor_model_parallel_size > 1`.** Enabling SP alone without TP is a config error. (SkyRL note: SP is auto-enabled when TP > 1; no separate config knob — see `.claude/docs/backends/megatron.md`.)
4. **CP requires `seq_length % (2 * context_parallel_size) == 0`.**
5. **EP is only for MoE models.** Setting `expert_model_parallel_size` on a dense model is a no-op or error.
6. The model-size-to-parallelism table is a starting heuristic. Always profile the first iteration to check memory and communication.
7. `CUDA_DEVICE_MAX_CONNECTIONS` and related env vars interact with overlap settings.
8. For MoE models, minimum GPUs are `PP * max(TP * CP, EP * ETP)`, not the product of
   every parallelism size.
