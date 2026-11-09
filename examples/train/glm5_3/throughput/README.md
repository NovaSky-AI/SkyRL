# GLM-5.3 trainer throughput sweep (8 x 8xB200)

Systematic sweep of Megatron trainer throughput for GLM-5.3 (753B `glm_moe_dsa`, 40B active)
full fine-tuning, done 2026-10-02 on 8 nodes x 8 B200 (RoCE, 8x400G per node) with
megatron-core d476c21be, Megatron-Bridge 0.7.0, TransformerEngine 2.19.0, torch 2.13.


## Summary

| max response | baseline step | best step | speedup | best config |
|---|---|---|---|---|
| 8k (8.69M tokens/step) | 2045 s (TP8, TileLang) | 1144 s | **1.79x** | TP4/EP64, hybrid DSA, `OMP_NUM_THREADS=12`, bf16 |
| 16k (16.5M tokens/step) | 3443 s (TP8, TileLang) | 2645 s | **1.30x** | TP8/EP64, hybrid DSA, `OMP_NUM_THREADS=12`, bf16 |

MXFP8 on the trainer is ~5% slower than bf16 at both lengths (GEMMs are ~3% of the step). The
largest remaining cost is MoE all-to-all (~38% of the step); DeepEP / NCCL EP could not be made to
work on this cluster (see below).

## Method

`run_full_ctx_glm5p3_megatron.sh` (adapted from
`examples/train_scripts/full_context/run_full_ctx_megatron.sh`) runs DAPO-shaped training steps on
dummy rollouts instead of generating:

* 128 prompts x 12 samples = 1536 sequences per step, 4 mini-batches of 384 (4 optimizer steps).
* Prompts 128-512 tokens; responses uniform in [512, max] with 25% at the max (truncated). At
  8k max that is 8.69M tokens/step (mean 5.66k/sequence), close to the real DAPO runs.
* Token ids are windows of tokenized DAPO prompts, so MoE routing sees real text (the original
  full-context trainer repeats one token per batch, which sends every token to the same 8 experts).
* Lengths come from a per-step seed, so every config trains on identical batches.
* Trainer only (`trainer.skip_inference_engines=true`): no vLLM, no weight sync. A colocated run
  also keeps a sleeping vLLM engine (~9 GiB/GPU), so a config is usable in the real recipe only if
  its peak reserved memory stays below ~165 GiB.
* Each step logs `throughput/*`, `memory/*` and `timing/*` (incl. `policy_forward_backward` vs
  `policy_optim_step`) to W&B project `glm5p3_throughput`.

Step-to-step variance is ~1% (baseline: step 1 2045 s, step 2 1981 s; fwd+bwd 1517 vs 1507 s), so
most configs ran one step after the warm-up compile and were stopped.

Common settings (from `../run_dapo_glm5p3_fullft_fp8_rollout_8node.sh`): PP1/CP1/ETP1, SP on,
alltoall dispatcher, `max_tokens_per_microbatch=8192`, DSA indexer loss off, CPU-offloaded
precision-aware optimizer, `mlp_chunks_for_training=1`, recompute selective `[core_attn, moe]`.

## Results: bf16, 8k max response

| run | TP/EP | change | logprob fwd | fwd+bwd | optim | step | peak reserved | speedup |
|---|---|---|---|---|---|---|---|---|
| b0 | 8/64 | baseline (TileLang DSA) | 331 s | 1517 s | 194 s | 2045 s | 116 GiB | 1.00x |
| b1 | 8/64 | recompute `[core_attn]` only | 319 s | 1470 s | 192 s | 1985 s | 142 GiB | 1.03x |
| b2 | 8/64 | hybrid DSA (cuDNN attn + TileLang indexer) | 297 s | 1028 s | 189 s | 1517 s | 116 GiB | 1.35x |
| b3 | 4/64 | b2 + TP4 | 215 s | 807 s | 186 s | 1211 s | 140 GiB | 1.69x |
| b4 | 4/64 | b3 + 16k-token microbatches | 223 s | OOM | | | >169 GiB alloc | -- |
| b6 | 4/64 | b3 + `OMP_NUM_THREADS=12` (CPU optimizer threads) | 219 s | 835 s | 87 s | 1144 s | 140 GiB | **1.79x** |
| f1 | 4/64 | b6 + MXFP8 trainer | 223 s | 892 s | 82 s | 1200 s | 153 GiB | 1.70x |

Token throughput of the best config (b6): 7.6k tokens/s over the whole step (fwd 39.6k tok/s,
fwd+bwd 10.4k tok/s), vs 4.2k tokens/s for the baseline.

## DSA kernel backends

megatron-core resolves DSA's fused hooks (indexer top-k, absorbed sparse attention) from the
module named by `dsa_kernel_backend`. Single-node benchmark (`dsa_bench.py`, not checked in: real
Megatron worker `forward_backward` on a 5-layer GLM-5.3 -- 3 dense + 2 MoE layers -- at TP8/EP8,
16 DAPO-shaped sequences = 102k tokens in 13 packed 8k microbatches, recompute `[core_attn]`):

| backend | fwd+bwd | peak alloc | loss | top kernels (rank 0, one iteration) |
|---|---|---|---|---|
| `none` (PyTorch) | 12.09 s | 36.0 GiB | -- | fp32 SIMT sgemms for q.k (2.1 s+), masking elementwise |
| `tilelang` | 5.97 s | 21.0 GiB | 30746.824 | SparseMLA bwd 1871 ms (28.8 ms/call), SparseMLA fwd 1093 ms, `tl_indexer_fwd` 936 ms |
| `cudnn` (+FlashMLA fwd) | 5.22 s | 23.6 GiB | 30746.78-.84 (non-deterministic) | sparse-attn bwd 501 ms, FlashMLA fwd 227 ms, indexer: fp32 SIMT `torch.bmm` per head ~900 ms + ~1 s elementwise |
| **`cudnn` + TileLang indexer** | **4.48 s** | 23.6 GiB | 30746.811 | `tl_indexer_fwd` 937 ms, sparse-attn bwd 501 ms, FlashMLA fwd 227 ms |

* TileLang's SparseMLA backward runs at 5x its forward on SM100 (a well-tuned attention backward is
  ~2-2.5x); cuDNN's sparse-attention backward is 3.7x faster and FlashMLA's forward 4.8x faster.
* But on the packed THD path SkyRL trains on, the cuDNN backend's indexer top-k is a PyTorch
  fallback (`_indexer_topk_from_score_chunks`: one fp32 `torch.bmm` per indexer head on CUDA cores),
  slower than TileLang's fused indexer kernel. Its fused cuDNN indexer is only used without varlen.
* `skyrl/backends/skyrl_train/patches/megatron/patch_dsa_hybrid_indexer.py`
  (`SKYRL_DSA_INDEXER_BACKEND=tilelang` with `dsa_kernel_backend=cudnn`) resolves only the
  `run_fused_qk_topk` hook from TileLang. The hook contract is shared (TileLang returns
  `(indices, None)`, cuDNN's sparse attention then compacts and sorts them). GLM-5.3 shares top-k
  across layers (`index_topk_freq=4`), so every layer takes this split top-k -> sparse-attention
  path. Loss matches TileLang to 4e-7 relative.
* At full scale the hybrid cut fwd+bwd 1517 -> 1028 s (-32%), more than the 5-layer benchmark
  suggests, because the full model has 78 sparse-attention layers but computes the indexer in only
  ~22 of them.
* The cudnn backend needs FlashMLA (`nv_dev` branch, `flash_mla_sparse_fwd`), which is not on
  PyPI. The `megatron` extra now installs our build from
  [skyrl-wheels `cu13torch2.13-flashmla-b7643bd`](https://github.com/NovaSky-AI/skyrl-wheels/releases/tag/cu13torch2.13-flashmla-b7643bd)
  (x86_64, cp312-314, sm_90a + sm_100f; no aarch64 wheel yet). The sweep itself ran against an
  identical local build put on the workers' path with `SKYRL_PYTHONPATH_EXPORT=1 PYTHONPATH=...`;
  the release wheel gives a bit-identical loss on the 5-layer benchmark. To switch a run over:
  `trainer.policy.megatron_config.transformer_config_kwargs.dsa_kernel_backend=cudnn` and
  `export SKYRL_DSA_INDEXER_BACKEND=tilelang`.
* The earlier impression that TileLang made training 2x slower than the naive path came from a
  naive-kernel run that OOMed partway through step 1; on identical batches TileLang is 2x faster
  than naive.

## Where the time goes (full-scale profile)

torch.profiler on ranks 0 and 33 (different nodes) for one full step of the b3 config (TP4/EP64,
hybrid DSA) with a small batch (8 prompts x 12, one mini-batch: ~4-5 full 8k microbatches per
rank for the logprob forward and for training, plus one optimizer step). Rank 0, 115.6 s step:

| bucket | GPU time |
|---|---|
| NCCL SendRecv = MoE all-to-all (EP64 dispatch/combine) | 35.5 s (p50 4.7 ms/call, 5400 calls) |
| NCCL AllGather (TP4 sequence parallel) | 29.8 s, dominated by a few waits at phase boundaries (max 11.9 s; p50 119 us) |
| DSA: sparse attention fwd/bwd | 4.7 s |
| DSA: indexer + top-k + sorts | 4.2 s |
| elementwise / copies / reductions | 3.5 s |
| memcpy (optimizer D2H/H2D, staging) | 2.8 s |
| **all GEMMs (dense + grouped expert GEMMs)** | **2.8 s** |
| NCCL ReduceScatter (SP + gradient reduce-scatter) | 2.6 s |
| any kernel running | 73% of the step |
| a non-NCCL kernel running | 16% of the step |

Rank 33 looks the same (SendRecv 34.3 s, AllGather 24.0 s). Takeaways:

* **EP all-to-all is the largest cost, ~38% of the non-optimizer step.** A full 8k microbatch at
  TP4 sends ~200 MB per rank per dispatch (2.2k SP-local tokens x top-8 x 6144 x bf16), 7/8 of it
  across nodes; the median call (4.7 ms) is close to that bandwidth floor at ~22 GB/s/GPU
  effective, against a 50 GB/s (400 Gb/s) NIC per GPU. It is real traffic, not just waiting.
* **Compute is tiny.** GEMMs are 2.8 s of 115 s. GLM-5.3 has ~40.7B active matmul parameters plus
  ~22 GFLOP/token of sparse attention: ~414 GFLOP of useful work per token per step (logprob
  forward + forward/backward). b6 then runs at ~49 TFLOP/s/GPU, ~2% of B200 dense BF16.
  This is why MXFP8 does not help the trainer (below).
* **CPU optimizer:** `Optimizer.step#AdamW.step` was 21.7 s for one step on rank 0 (11.8B local
  params, mostly experts). SkyRL's policy actors request `num_cpus=1`, so Ray sets
  `OMP_NUM_THREADS=1` and the CPU Adam was single-threaded. `OMP_NUM_THREADS=12` (14 physical
  cores per GPU here) halved the optimizer time at full scale (186 -> 87 s per 4 updates). SkyRL now
  forwards `OMP_NUM_THREADS` from the driver to the workers.
* Host syncs: `aten::nonzero` (681 calls, 16 ms each) and `.item()` (2.5k calls) in the MoE
  permutation / packing paths serialize CPU and GPU; `cudaEventSynchronize` + `cudaStreamSynchronize`
  account for 45 s of CPU time on rank 0.

Traces: `/mnt/local_storage/etang_traces/p1/rank{0,33}_w0.pt.trace.json.gz` on b200-resv-ray-3 /
-5 (the rank-0/33 nodes for that run), ~95 MB each.

## DeepEP / NCCL EP

The profile puts EP all-to-all at ~38% of the step, so a node-aware dispatcher (send each token
once per destination node over RDMA and fan out over NVLink; at top-8 over 8 nodes a token touches
~5.3 nodes on average instead of 8 rank-sends, ~1.5x less RDMA traffic) plus FP8 dispatch payloads
could plausibly save 10-20% of the step. None of the routes worked on this cluster today:

* **DeepEP V1** (the API megatron-core's `moe_flex_dispatcher_backend=deepep` calls) needs NVSHMEM
  IBGDA, which needs the NVIDIA driver loaded with `NVreg_EnableStreamMemOPs=1` and
  `PeerMappingOverride=1` (here: `EnableStreamMemOPs: 0`, no `nvidia_peermem`). Changing that
  reloads the GPU driver on every node of a shared cluster; not done.
* **DeepEP V2** moved to NCCL GIN (needs NCCL >= 2.32.3) and an `EPBuffer` API that megatron-core
  does not call; it would need a new dispatcher integration.
* **TE 2.19 NCCL EP** (`moe_flex_dispatcher_backend=ncclep`, ships `libnccl_ep.so`, built on NCCL's
  device API/GIN) is the in-tree route. Tried in a worktree with `nvidia-nccl-cu13==2.32.3`
  (torch pins 2.29.7; TE needs >= 2.30.4):
  1. `NCCL EP requires NCCL Device API support` -- SkyRL forces `NCCL_CUMEM_ENABLE=0` when
     `weight_sync_backend=nccl`; the device API needs cuMem. Fixed by letting an explicit
     `NCCL_CUMEM_ENABLE=1` through.
  2. GIN picked the GDAKI transport (GPU drives the NIC): `ncclGinGdakiCreateContext ... DOCA
     Error 8` -- same driver-level requirement as IBGDA. `NCCL_GIN_TYPE=2` (CPU-proxy GIN) gets
     past it and NCCL EP bootstraps on all 64 ranks.
  3. NCCL EP JIT-compiles its kernels with nvcc on first use; inside the workers nvcc segfaulted
     (exit 139) on the HT scan kernel on several nodes (the same command succeeds by hand and in a
     plain Ray task), and the abort leaves `compile.failed` markers in `/tmp/nccl_ep/jit` that
     poison later runs.
  4. `CUDA_ERROR_ILLEGAL_ADDRESS` / `MISALIGNED_ADDRESS` in the local permute kernel. megatron-core
     bootstraps NCCL EP with `max_tokens_per_rank` from the first microbatch and marks varying
     per-rank token counts (packed THD) as a TODO; `patch_ncclep_max_tokens.py` (worktree only)
     fixes the bound, but the HT kernels evidently also assume equal per-rank token counts
     (megatron-core's HybridEP has `moe_hybridep_pad_uneven_dispatch_inputs` for exactly this).
     Making it work needs per-microbatch padding/masking of the dispatch input.

The ncclep experiment lives in a separate worktree (pyproject NCCL override + the two patches) and
is not on this branch.

## MXFP8

| run | config | fwd | fwd+bwd | optim | step | peak reserved |
|---|---|---|---|---|---|---|
| b6 | bf16 TP4/EP64, hybrid DSA, OMP 12 | 219 s | 835 s | 87 s | 1144 s | 140 GiB |
| f1 | same + MXFP8 (`fp8=e4m3`, `fp8_recipe=auto`) | 223 s | 892 s | 82 s | 1200 s | 153 GiB |

With TE 2.19 the MXFP8 trainer is now within 5% of bf16 (with TE 2.16 the full-FP8 DAPO run was
~1.6x slower than bf16), but it does not get faster: GEMMs are ~3% of the step, so faster FP8 GEMMs
cannot pay for the extra quantize/cast kernels, and FP8 adds 13 GiB of weight/transpose caches.
MXFP8 on the trainer is worth keeping for rollout/trainer numerics alignment (the full-FP8 DAPO run
learned best), not for trainer throughput.

## 16k max response

Same shape at `MAX_RESPONSE_LENGTH=16384`: 16.5M tokens/step (mean 10.7k/sequence).
`max_tokens_per_microbatch=16384`, since the longest sequences (up to 16.9k) get a microbatch of
their own anyway.

| run | config | fwd | fwd+bwd | optim | step | tokens/s | peak reserved |
|---|---|---|---|---|---|---|---|
| s1 | bf16 TP8/EP64, hybrid DSA, OMP 12 | 585 s | 1969 s | 86 s | **2645 s (1.30x)** | 6.24k | 135 GiB |
| s2 | s1 + MXFP8 | 619 s | 2098 s | 73 s | 2794 s | 5.91k | 145 GiB |
| s3 | bf16 TP4/EP64, hybrid DSA, OMP 12, 8k microbatches | 419 s | OOM | | | | >172 GiB alloc |
| s0 | 16k baseline (TP8, TileLang, single-threaded optimizer) | 611 s | 2639 s | 188 s | 3443 s | 4.79k | 130 GiB |

TP4 cannot hold a 16.9k-token microbatch (s3 OOMed in the first training mini-batch, as did 16k
packed microbatches at TP4 in b4), so 16k stays at TP8. The TP4 forward pass alone was 30% faster
(419 vs 585 s), so CP2 or more recompute at TP4 might be worth trying at 16k.

## Recommendations

For the DAPO recipe (`../run_dapo_glm5p3_fullft_fp8_rollout_8node.sh`), in order of payoff:

1. **Hybrid DSA backend** (`dsa_kernel_backend=cudnn` + `SKYRL_DSA_INDEXER_BACKEND=tilelang`,
   FlashMLA on the workers' path): -26% step time at no memory cost, numerically equivalent.
2. **TP4** at 8k: another -20%; peak 140 GiB reserved, inside the ~165 GiB colocated budget. Keep
   TP8 at 16k (TP4 OOMs on a 16.9k-token sequence). Untested in the colocated recipe: the weight
   sync peak was the binding constraint there at TP4 for LoRA (vLLM weights + reload buffers +
   trainer weights), though the eager-reload patch since cut vLLM's sync peak by ~20 GiB.
3. **`OMP_NUM_THREADS=12`** for the CPU-offloaded optimizer: halves the optimizer step (-6% of
   the step at 8k).
4. Keep MoE recompute (`[core_attn, moe]`): dropping it saves 3% for +26 GiB.
5. MXFP8 on the trainer costs ~5% throughput and ~10-13 GiB; use it for numerics, not speed.

Remaining headroom is in communication, not compute (GEMMs are ~3% of the step):

* EP dispatch (~38% of the step): a node-aware dispatcher (DeepEP/HybridEP/NCCL EP) needs either
  the driver settings for IBGDA/GDAKI or NCCL EP fixed for uneven per-rank token counts (pad each
  microbatch's dispatch input to the bootstrapped bound); FP8 dispatch payloads would halve bytes.
* EP32 (fewer cross-node hops) does not fit with fp32 main grads: experts are ~11.3B params per GPU
  at EP64 (23 GiB bf16 + 45 GiB fp32 grads), and EP32 doubles that. It would need
  `ddp_config.grad_reduce_in_fp32=false` (bf16 grad accumulation) -- not measured.
* Host syncs in the MoE path (`nonzero`, `.item()`) and CPU optimizer D2H/H2D
  (`overlap_cpu_optimizer_d2h_h2d=false` here) are the next smaller items.
