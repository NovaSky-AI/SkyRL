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
- **KV cache.** FP8 KV cache for these hybrid-attention models is a separate
  compatibility PR; once available, add
  `generator.inference_engine.engine_init_kwargs.kv_cache_dtype=fp8_e4m3`.
