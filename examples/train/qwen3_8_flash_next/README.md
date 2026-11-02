# Qwen3.8-Flash-Next recipes

Megatron + colocated vLLM recipes for Qwen3.8-Flash-Next (HF `qwen4_exp`, ~180B MoE) on four nodes of 8xB200:

| script | what |
|---|---|
| `run_gsm8k_qwen3_8_flash_next_{lora,fullft}_4node.sh` | GSM8K, short context. The per-setting reasoning is in these scripts. |
| `run_dapo_qwen3_8_flash_next_{lora,fullft}_sync_4node.sh` | DAPO, 2K prompt / 8K response, sparse QSA regime, R3 router replay |
| `run_dapo_lora_262k_cp8_4node.sh` | DAPO LoRA at 262,144-token sequences with context parallelism |

## 262K context parallelism

`run_dapo_lora_262k_cp8_4node.sh` runs TP1 x CP8 x EP16 (DP4) with full recompute, `mlp_chunks_for_training=8`, 262K
micro-batches, and vLLM as 4 engines x TP8 with `max_model_len` 262,144. CP for `qwen4_exp` covers the QSA attention
(split-K cuDNN QSA over CP chunks with an LSE merge, Blackwell only, so install the cudnn-frontend wheel via
`QSA_CUDNN_WHEEL`), the PLE halo, and the QSA indexer's pooled-key all-gather. GDN needs TP x CP <= 16.

The stock DAPO prompts are short. To fill 262K, set `TRAIN_FILE` / `TEST_FILE` to a dataset of ~255K-token prompts.

Measured on 4 x 8 B200 (262,144-token sequences, one per DP rank):

| config | warm step | peak alloc GB |
|---|---|---|
| LoRA, TP1 x CP8 x EP16 (this script) | 38.5 s | 74.8 |
| Full FT, TP2 x CP8 x EP32, optimizer on GPU | 30.0 s | 107.4 (110 reserved) |

Full FT at 262K, as overrides to the same script (TP2 runs with sequence parallelism, DP2). At TP1, full FT needs
the optimizer offloaded to CPU.

```bash
MEGATRON_TP=2 MEGATRON_EP=32 LR=3e-6 RUN_NAME=qwen3_8_flash_next_dapo_sync_fullft_262k_tp2_cp8_ep32 \
  bash examples/train/qwen3_8_flash_next/run_dapo_lora_262k_cp8_4node.sh \
  trainer.policy.model.lora.rank=0 \
  trainer.policy.megatron_config.ddp_config.grad_reduce_in_fp32=false
```

### B200: fla GDN race workaround

With fla 0.5.2 and Triton 3.7.1, fla's autotuner picks 8 warps for `chunk_fwd_kernel_o` (the GDN chunk forward) on
sm_100. A Triton codegen race (triton-lang/triton#10590) then corrupts one 16-row block in about 40% of calls. This
hits every GDN model on B200, with or without CP. Two identical full-model forwards differ by 0.057 mean abs dlogp.

The 262K script pins that kernel to 4 warps, which costs under 1%:

```bash
export FLA_CACHE_MODE=default
export FLA_CONFIG_DIR=examples/train/qwen3_8_flash_next/fla_config   # absolute path, same on every node
```

`fla_config/chunk_fwd_kernel_o.json` holds the pinned config. Other kernels keep autotuning. SkyRL forwards
`FLA_CACHE_MODE` / `FLA_CONFIG_DIR` to the Ray workers. Export the same two variables to apply the workaround to the
other recipes. vLLM's vendored fla copy has no config-dir hook, so this pin covers only the trainer.

The permanent fix is fla >= 0.6.0 (it drops 8-warp configs on Blackwell with Triton < 3.8) or Triton >= 3.8. That
dependency bump will be a separate PR.
