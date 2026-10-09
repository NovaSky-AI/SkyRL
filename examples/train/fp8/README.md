# FP8 RL training + rollout examples

DAPO on AIME with FP8 across the performance-critical parts of the stack:
trainer linear-layer GEMMs, rollout weights, and the weight transfer between
them. All scripts use `fp8_weight_sync_mode=blockwise`, which sends
the trainer-produced FP8 payloads and block scales directly to vLLM instead of
re-quantizing a BF16 export — keeping the rollout policy numerically identical
to the trained one.

Prepare the dataset once:

```bash
bash examples/train/algorithms/dapo/prepare_dapo_data.sh
```

| Script | Hardware | Recipe | FP8 params |
| --- | --- | --- | --- |
| `run_fp8_hopper_blockwise_qwen35_9b.sh` | 8×H100 | blockwise, FP32 scales | — |
| `run_fp8_hopper_blockwise_fp8param_qwen35_9b.sh` | 8×H100 | blockwise, FP32 scales | E4M3 primary weights (~39% less parameter HBM) |
| `run_fp8_hopper_blockwise_qwen35_35b_a3b.sh` | 2×8×H100 | blockwise, FP32 scales | — |
| `run_fp8_hopper_blockwise_fp8param_qwen35_35b_a3b.sh` | 2×8×H100 | blockwise, FP32 scales | E4M3 primary weights (~42% less parameter HBM) |
| `run_fp8_blackwell_mxfp8_qwen35_9b.sh` | 8×B200 | `auto` → native MXFP8 | not yet supported on MXFP8 |
| `run_fp8_blackwell_mxfp8_qwen35_35b_a3b.sh` | 8×B200 | `auto` → native MXFP8 | not yet supported on MXFP8 |

Notes:

- **Colocated vs. non-colocated.** Every script defaults to
  `trainer.placement.colocate_all=true` (training and inference share GPUs).
  Run with `COLOCATE_ALL=false` and split the GPUs between
  `trainer.placement.policy_num_gpus_per_node` and the inference engines for a
  disaggregated placement.
- **Qwen3.5 runs text-only.** All scripts set `language_model_only=true` on the policy, ref and
  inference engine: Qwen3.5 otherwise loads through the VL bridge, which packs sequences inside
  its own forward and is rejected together with SkyRL sample packing.
- **GDN kernels on Blackwell.** The Blackwell scripts `export FLA_TILELANG=0` so fla uses its
  Triton GatedDeltaNet kernels; the TileLang packed backward aborts on B200 (it shows up as a
  CUDA "misaligned address" in the first backward). Leave it unset on Hopper, where the Triton
  backward is the broken one.
- **Recipe selection.** `fp8_recipe=auto` picks the architecture-native
  recipe: `blockwise` (FP32 scales) on Hopper, `mxfp8` on Blackwell/SM100+.
  The Hopper scripts pin `blockwise` explicitly; the Blackwell scripts use
  `auto`.
- **FP8 configuration surface.** The scripts use the top-level
  `megatron_config.fp8*` fields; the same keys under
  `transformer_config_kwargs` override them if you need to.
- **KV cache.** The scripts leave the rollout KV cache in BF16. To run it in FP8 too, add
  `generator.inference_engine.engine_init_kwargs.kv_cache_dtype=fp8_e4m3`.

## Choosing which modules run in FP8

Training and rollout precision are configured separately, and SkyRL does not check one against
the other: to keep a module in BF16 everywhere, list it on both sides.

**Training (Megatron)**, per worker (`trainer.policy` and `trainer.ref`):

| Field | What it sets |
| --- | --- |
| `megatron_config.fp8` | FP8 format: `e4m3`, or `hybrid` (E4M3 forward, E5M2 gradients) |
| `megatron_config.fp8_recipe` | `auto`, `blockwise` (128x128 weight tiles) or `mxfp8` (1x32 groups, E8M0 scales) |
| `megatron_config.fp8_exclude_modules` | Megatron module-path globs trained in BF16, e.g. `['*.shared_experts.*']` |
| `megatron_config.num_layers_at_start_in_bf16` / `num_layers_at_end_in_bf16` | First / last N transformer layers trained in BF16 |
| `megatron_config.te_precision_config_file` | A Megatron per-module precision YAML (`--te-precision-config-file` format) for anything finer, e.g. a different recipe per module |

**Rollout (vLLM)**:

| Field | What it sets |
| --- | --- |
| `generator.inference_engine.fp8_weight_sync_mode` | Wire format: `blockwise`, `mxfp8`, or `auto` (follows the policy's recipe) |
| `generator.inference_engine.fp8_weight_sync_exclude_modules` | HF module-name globs kept in BF16 on the rollout, e.g. `['*.layers.0.*','*.layers.39.*']`; SkyRL derives both the engine's ignore list and the sync from this one list |

For example, to keep the last two layers of Qwen3.5-35B-A3B (40 layers) in BF16 on both sides:

```bash
  trainer.policy.megatron_config.num_layers_at_end_in_bf16=2 \
  trainer.ref.megatron_config.num_layers_at_end_in_bf16=2 \
  "generator.inference_engine.fp8_weight_sync_exclude_modules=['*.layers.38.*','*.layers.39.*']" \
```

Module names differ between the two sides: Megatron uses `decoder.layers.N.self_attention.linear_qkv`,
`decoder.layers.N.mlp.experts.linear_fc1`, ...; the rollout uses HF checkpoint names such as
`model.language_model.layers.N.self_attn.q_proj` and `model.language_model.layers.N.mlp.experts`
(one module for all of a layer's routed experts). Megatron numbers layers within each pipeline
stage, so use the layer-count fields rather than layer-indexed `fp8_exclude_modules` patterns
under pipeline parallelism.
