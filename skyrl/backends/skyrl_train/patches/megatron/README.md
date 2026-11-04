# Megatron patches and vendored upstream code

Everything under this folder is **temporary**: runtime patches for bugs in the pinned
`megatron-core` / `megatron-bridge`, and code vendored from upstream PRs that have not landed in
the pins yet. Each entry below says what upstream change retires it, how to tell whether a new pin
already contains that change, and every place *outside* this folder that has to change with it.

This file is the removal plan. When you bump `megatron-core` or `megatron-bridge` in
`pyproject.toml`, work through it top to bottom.

## Tests

Tests for this folder mirror its layout, so they are found and deleted together with the code:

- CPU: `tests/backends/skyrl_train/patches/megatron/` (marked `megatron`; run by the CPU megatron job)
- GPU: `tests/backends/skyrl_train/gpu/gpu_ci/patches/megatron/` (marked `megatron`; run by the
  Megatron GPU suites, ignored by the FSDP one)

| test | covers |
|---|---|
| `patches/megatron/glm5_next/test_dsa_kpool_math.py` (CPU) | megatron-core k-pool key compression vs HF |
| `patches/megatron/mcore_ext/test_hyper_connection_proj_rms.py` (CPU) | `mcore_ext/hyper_connection.py` projection: bitwise vs plain autograd, no saved FP32 copy |
| `patches/megatron/mcore_ext/test_mhc_moe_chunking.py` (CPU) | `mcore_ext/mhc_transformer_layer.py` `chunked_mlp`: chunked MLP/MoE matches one call |
| `gpu_ci/patches/megatron/glm5_next/test_dsa_kpool.py` | megatron-core k-pool top-k selection at GLM-5.3-Flash's shapes, query chunking |
| `gpu_ci/patches/megatron/mcore_ext/test_modules_vs_hf.py` | `mcore_ext/kda.py`, `mcore_ext/hyper_connection.py` vs HF |
| `gpu_ci/patches/megatron/test_dsa_index_share_recompute.py` | `patch_dsa_index_share.py` |
| `gpu_ci/patches/megatron/test_shared_expert_lora_tp.py` | `patch_shared_expert_lora_tp.py` |
| `gpu_ci/patches/megatron/test_dsa_head_full.py` | `patch_dsa_head_full.py` all-to-alls vs megatron-core's and an all-gather reference (TP 2 and 4) |
| `patches/megatron/test_sparse_mla_nope.py` (CPU) | `patch_sparse_mla_nope.py` padding/unpadding, fake kernel |
| `gpu_ci/patches/megatron/test_sparse_mla_nope.py` (H100) | `patch_sparse_mla_nope.py` vs dense reference, real TileLang kernel |
| `patches/megatron/test_dsa_hybrid_indexer.py` (CPU) | `patch_dsa_hybrid_indexer.py` hook resolution, fake backends |
| `patches/megatron/test_moe_release_dispatcher_probs.py` (CPU) | `patch_moe_release_dispatcher_probs.py`, fake layer state |
| `patches/megatron/test_distopt_mixed_dtype_group_index.py` (CPU) | `patch_distopt_mixed_dtype_group_index.py` index rebuild |
| `patches/megatron/test_clip_grads_mixed_dtype.py` (CPU) | `patch_clip_grads_mixed_dtype.py` dtype split and the `multi_tensor_applier` wrapper |
| `gpu_ci/patches/megatron/glm5_next/test_dsa_kpool_tp_shard.py` | `glm5_next/dsa.py` TP query sharding (two ranks) |
| `patches/megatron/qwen4_exp/test_qwen4_exp_modules.py` (CPU) | `qwen4_exp/modules.py` gated residual, final mixer, packed PLE vs HF; PLE row-sharded over EP and under sequence parallelism (gloo) |
| `patches/megatron/qwen4_exp/test_qwen4_exp_qsa.py` (CPU) | `qwen4_exp/modules.py` QSA indexer selection and sparse attention (outputs + grads, packed) vs HF |
| `patches/megatron/glm5_next/test_glm5_next_flops_fields.py` (CPU) | `glm5_next/bridge.py` sets the `kimi_*` FLOPs-formula fields |
| `patches/megatron/test_dist_optim_te_load_state_dict.py` (CPU) | `patch_dist_optim_te_load_state_dict.py` |
| `patches/megatron/test_dist_ckpt_exchange_wait.py` (CPU) | `patch_dist_ckpt_exchange_wait.py` |

The end-to-end GLM-5.3-Flash rows stay with the other models: `glm-5.3-flash-4layer_*` in
`gpu_ci/megatron/test_megatron_models.py` and `test_megatron_lora_models.py`. When removing a patch,
delete its tests here in the same change.

## Upgrade procedure

1. Bump the pins together. `megatron-core` must match the `3rdparty/Megatron-LM` submodule of the
   chosen `megatron-bridge` rev (`git ls-tree <bridge-rev> 3rdparty/Megatron-LM`). Regenerate
   `uv.lock` (`uv lock`).
2. For each entry below, run its **"Landed?"** check against the new pins (installed under
   `.venv/lib/python3.12/site-packages/megatron/`). Only remove an entry whose check passes.
   Partial landings happen: one PR can land without the others.
3. Remove it by following **Remove**, including every listed touchpoint outside this folder.
   Search for the module name afterwards (`grep -rn <module> skyrl tests examples .agents`); no
   references may remain.
4. Run the entry's **Verify** tests, then the full GLM-5.3-Flash GPU set in
   [Verification](#verification).
5. Update this file: delete the entry, or narrow it to whatever is still carried.

Do not keep a vendored module "just in case" once upstream has it. Two implementations drifting
apart is how the k-pool indexer once ended up silently unused in training.

## GLM-5.3-Flash (`glm5_next`)

GLM-5.3-Flash needs four upstream Megatron-LM PRs (#7522, the k-pool indexer, has landed) and a
Megatron-Bridge model. `mcore_ext/` holds
the megatron-core pieces and `glm5_next/` holds the Bridge model built on them. Once all of the
megatron-core pieces have landed, `mcore_ext/` is deleted entirely. Once Megatron-Bridge ships the
model, `glm5_next/` is deleted too.

### NVIDIA/Megatron-LM#7054: KDA (Kimi Delta Attention)

- **Carried as:** `mcore_ext/kda.py` (`KimiDeltaAttention`, `get_kda_module_spec`).
- **Landed?** megatron-core defines a KDA module or `experimental_attention_variant="kda"`, and
  `TransformerConfig` has `kda_gate_lower_bound`
  (`grep -rn "KimiDeltaAttention\|kda_gate_lower_bound" .venv/.../megatron/core`).
- **Remove:**
  - `glm5_next/layer_specs.py`: import the KDA spec from megatron-core instead of `mcore_ext.kda`.
  - `glm5_next/provider.py`: drop `kda_gate_lower_bound` once `TransformerConfig` declares it.
  - `glm5_next/bridge.py`: re-check the KDA parameter names (`q/k/v_conv1d`, `A_log`, `dt_bias`,
    `f_a/f_b/g_a/g_b_proj`) against upstream's module.
  - Delete `mcore_ext/kda.py`.
- **Verify:** `gpu_ci/patches/megatron/mcore_ext/test_modules_vs_hf.py::test_kda_matches_hf`; the GLM roundtrip rows.

### NVIDIA/Megatron-LM#7521: mHC (manifold-constrained hyper-connections)

Two pieces, which may land separately.

**a) Standard-RMSNorm input norm**
- **Carried as:** `mcore_ext/hyper_connection.py` (`RMSNormInputHyperConnectionModule`; its FP32
  projection is checkpointed so backward keeps the activation-dtype input, not the FP32 upcast:
  2 GiB per mHC site at 32k tokens per rank. #7521 still saves the upcast; worth proposing there).
- **Landed?** `TransformerConfig` has `mhc_norm_eps` / `mhc_norm_eps_inside_sqrt`, and
  `HyperConnectionModule` reads them.
- **Remove:**
  - `mcore_ext/mhc_transformer_layer.py` (or its upstream replacement): use megatron-core's
    `HyperConnectionModule` directly.
  - `glm5_next/provider.py`: drop `mhc_norm_eps_inside_sqrt` if it's now a `TransformerConfig`
    field. The bridge must still set it for GLM-5.3-Flash.
  - Delete `mcore_ext/hyper_connection.py`.

**b) MoE sub-layers in the mHC layer**
- **Carried as:** `mcore_ext/mhc_transformer_layer.py` (`HyperConnectionTransformerLayer`).
  - The layer replaces megatron-core's `_forward_mlp`, where `mlp_chunks_for_training` is applied,
    so it chunks the MLP/MoE itself (`chunked_mlp`), in eval forwards as well as training. On real
    data, routing sends some expert ranks several times their share of tokens; chunking bounds their
    dispatch/expert/combine buffers.
  - Its `forward` calls the private `TransformerLayer._maybe_unflatten_for_moe` /
    `_maybe_reflatten_from_moe` helpers; on every megatron-core bump, check their signatures against
    `TransformerLayer._forward_mlp`.
- **Landed?** megatron-core's `HyperConnectionTransformerLayer` accepts a MoE MLP submodule, with
  no `NotImplementedError` for MoE. Check that it honors `mlp_chunks_for_training` before switching.
  - Status at megatron-core `ed9c2c8c`: upstream's layer accepts MoE (it only rejects MoE CUDA-graph
    scopes), but it chunks the MLP in training only, and it builds megatron-core's
    `HyperConnectionModule`, so it can't be switched to until (a) lands too.
- **Remove:**
  - `glm5_next/layer_specs.py`: build the specs on megatron-core's layer.
  - `workers/megatron/megatron_worker.py`: the `enable_mhc_connections` block in `init_configs`
    downgrades `recompute_granularity="full"` to selective and drops `'mhc'`. It exists because
    megatron-core rejects mHC under full recompute, and our layer doesn't implement the mHC
    recompute managers that megatron-core's suggested alternative (`'mhc'` in
    `recompute_modules`) needs. With upstream's layer, keep a downgrade from full to selective
    **with** `'mhc'` in `recompute_modules`, or delete the block entirely if upstream now allows
    full recompute with mHC. SkyRL's default config is full recompute, and the GLM roundtrip
    tests run on defaults.
  - Delete `mcore_ext/mhc_transformer_layer.py`.
- **Verify:** `gpu_ci/patches/megatron/mcore_ext/test_modules_vs_hf.py::test_hyper_connection_matches_hf`,
  `patches/megatron/mcore_ext/test_mhc_moe_chunking.py`; the GLM roundtrip rows.
  These run with the default full recompute, so they exercise the worker block above.

### NVIDIA/Megatron-LM#7522: k-pool DSA indexer (landed)

Landed in megatron-core `ed9c2c8c`: `DSAIndexer` owns the k-pool gate/ape parameters and
`DSAttention.forward` selects through `fused_qk_topk_kpool`. The vendored kernels
(`mcore_ext/dsa_kpool.py`), `Glm5NextDSAIndexer` and `_forward_with_kpool_topk` are gone. What is
left in `glm5_next/`:

- `glm5_next/bridge.py` sets `dsa_indexer_weights_proj_output_dtype="fp32"` (and turns off its
  quantization) when `index_kpool > 1`. The vendored indexer always computed the head-gate
  projection in FP32 on the k-pool path; upstream defaults to bf16.
- `glm5_next/dsa.py` (`Glm5NextDSAttention`):
  - the `kpool <= 1` long-sequence guard;
  - opt-in `SKYRL_DSA_INDEXER_TP_SHARD=1`: `make_sharded_kpool_topk` wraps megatron-core's
    `_kpool_topk_from_score_chunks` so each tensor-parallel rank scores only its slice of query
    rows, then all-gathers the pool selections. The scoring is O(sq^2) and otherwise replicated on
    every TP rank, which all score the same gathered sequence with the same frozen indexer weights:
    at 512k tokens, TP8, 64 B200 (TCP NCCL) sharding cut warm fwd+bwd 1545 -> 1094 s.
    Bitwise-identical selections (`test_dsa_kpool_tp_shard.py`). Skipped when megatron-core already
    keeps the indexer rows sequence-local. On a megatron-core bump, check that
    `_kpool_topk_from_score_chunks` is still called with keyword arguments and that every
    per-query input is still indexed by query row along dim 0 (`mask` along dim 0 or 1).
- Upstream's score-chunk cap (`_KPOOL_SCORE_CHUNK_BYTES`) is 2 GiB, the default of the old
  `SKYRL_DSA_KPOOL_SCORE_CHUNK_ELEMS`, which is gone.
- **Verify** (all required on a megatron-core bump):
  - `patches/megatron/glm5_next/test_dsa_kpool_math.py` (CPU) and
    `gpu_ci/patches/megatron/glm5_next/test_dsa_kpool.py`, `test_dsa_kpool_tp_shard.py` (GPU).
  - `test_logprobs_matching_roundtrip[glm-5.3-flash-4layer_h100_tp2_ep4_kpool_beyond_topk]`. This
    is the only test that runs sequences past `index_topk`. Its logprob diff must not get worse
    than about 0.053 (token-level selection, i.e. no k-pool, gives about 0.059).

### NVIDIA/Megatron-LM#7523: FP8 wgrad

Not carried. GLM-5.3-Flash runs bf16 end to end.

### NVIDIA-NeMo/Megatron-Bridge#6044: the GLM-5.3-Flash model

- **Carried as:** `glm5_next/`: `provider.py`, `layer_specs.py`, `dsa.py`, and `bridge.py`
  (`Glm5NextBridge`, registered for `Glm5NextForConditionalGeneration` on import, plus the
  `HyperConnectionScaleMapping` / `HyperConnectionScaleSliceMapping` custom mappings).
- **Landed?** Megatron-Bridge registers a bridge for `Glm5NextForConditionalGeneration` /
  `model_type="glm5_next"` (e.g. under `megatron/bridge/models/glm*`).
- **Remove:**
  - `workers/megatron/model_bridges.py`: drop the `glm5_next` import that registers the bridge.
  - Tests that import from `glm5_next`: repoint them at Megatron-Bridge.
  - `.agents/docs/backends/megatron.md` and `docs/content/docs/.../supported_models.mdx`: update
    the model entry.
  - The `kimi_*` fields in `glm5_next/provider.py` and their assignments in `glm5_next/bridge.py`:
    check that the upstream provider sets the fields megatron-bridge's FLOPs formula
    (`flop_utils.num_floating_point_operations`) reads for KDA layers, then delete them with
    `test_glm5_next_flops_fields.py`.
  - Before deleting the local bridge, compare its behaviour with upstream's:
    - `language_model_only=True` handling;
    - the `head_dim=0` NoPE RoPE skip;
    - the k-pool field mapping, including the FP32 indexer weights projection;
    - the mHC scale mappings.
  - Move `Glm5NextDSAttention`'s opt-in TP sharding to wherever upstream builds the model, or
    drop it if upstream shards the k-pool scoring itself.
  - Delete `glm5_next/`.
- **Verify:** the full [Verification](#verification) set.

## Qwen3.8-Flash-Next (`qwen4_exp`)

### Upstream trials: NVIDIA/Megatron-LM#7393, NVIDIA-NeMo/Megatron-Bridge#6123, NVIDIA/Megatron-LM#7553, NVIDIA-NeMo/Megatron-Bridge#6354

These are independent attempts at `qwen4_exp`, not one stack: they use different model classes,
config fields and PLE designs, and only Megatron-Bridge#6123 builds on another of them
(Megatron-LM#7393). `qwen4_exp/` here is SkyRL's own implementation. Whichever trial lands first,
switch to it and delete `qwen4_exp/` as a whole; don't port pieces between implementations.

| Trial | What has to land | Shape (as of 2026-10) |
|---|---|---|
| Megatron-LM#7393 + Megatron-Bridge#6123 | both | `GPTModel`; TP + SP; PLE table vocab-parallel on GPU over TP |
| Megatron-Bridge#6354 | Bridge alone (it carries its own modules) | `HybridModel`; built through Builder + Config only (no provider); TP + SP, PP=CP=1; rejects activation recompute |
| Megatron-LM#7553 | core, plus a Bridge bridge on top | `HybridModel`; PP/VPP, CP, MTP, Triton QSA kernel, PLE table row-sharded over EP. Core alone does not retire `qwen4_exp/` |

- **Carried as:** `qwen4_exp/`: `modules.py` (gated low-rank hyper-connection, final mixer, PLE
  n-gram embedding, sigmoid-gated GDN, the frozen QSA indexer and its sparse attention:
  cuDNN frontend's QSA kernels on Blackwell, else FlexAttention on CUDA, else chunked SDPA),
  `layer_specs.py`,
  `provider.py`, and `bridge.py` (`Qwen4ExpBridge`, registered for
  `Qwen4ExpForConditionalGeneration` on import).
- **Landed?** The pinned Megatron-Bridge registers its own bridge for
  `Qwen4ExpForConditionalGeneration` / `model_type="qwen4_exp"` (under
  `megatron/bridge/models/qwen*`), from any of the trials above. megatron-core modules alone
  (Megatron-LM#7393 or #7553) are not enough.
- **Before switching**, check that the landed version runs the `examples/train/qwen3_8_flash_next/`
  recipes. Close a gap with a small patch on top of upstream or a recipe change, not by keeping
  `qwen4_exp/`:
  - Construction: SkyRL builds models through `bridge.to_megatron_provider()`. A Builder +
    Config-only bridge (Megatron-Bridge#6354) needs a builder path in `megatron_worker.py`.
  - Activation recompute: the DAPO full-FT recipe uses `recompute_granularity=full`.
  - PLE placement next to colocated vLLM at the recipes' TP (here: frozen, in pinned host memory,
    row-sharded over the EP group, loaded outside the bridge, never exported).
  - Packed (THD) sequences with the PLE n-gram context reset at every sequence boundary.
  - LoRA target module names and the vLLM adapter mapping.
  - Router replay (R3) through its MoE layer, including `chunked_mlp` if its layer bypasses
    `TransformerLayer._forward_mlp` as the mHC layers here do.
  - QSA top-k block selection past `indexer_budget + compress_ratio - 1` tokens (ties, dtype of
    the indexer scores), and the unexported MTP head.
- **Remove:**
  - `workers/megatron/model_bridges.py`: drop the `qwen4_exp` import.
  - `workers/megatron/megatron_worker.py`: re-check the `qwen4_exp` `language_model_only` guard
    against what upstream builds.
  - Delete `qwen4_exp/` and `tests/.../patches/megatron/qwen4_exp/`.
- **Verify:** the `qwen3.8-flash-next-4layer_*` rows of
  `gpu_ci/megatron/test_megatron_{,lora_}models.py`, `gpu_ci/megatron/test_qwen38_qsa_long_context.py`
  (~8K tokens, sparse regime) and the `megatron_qwen3_8_flash_next_lora` row of
  `gpu_ci/test_trainer_full_checkpointing.py`; then a few steps of the DAPO recipe.

## Standalone patches

### `patch_dsa_index_share.py` (+ `dsa_index_share_recompute.patch`): NVIDIA/Megatron-LM#6793

Per-forward DSA index-share carrier under activation recompute.
- **Landed?** The patch checks for itself: megatron-core's DSA module has
  `_dsa_index_share_carrier_scope`, and applying the patch logs a warning telling you to delete it.
- **Remove:** the `patch_dsa_index_share()` call in `MegatronWorker.make_megatron_module`, both
  files here, and the `*.patch` package-data entry in `pyproject.toml` if nothing else uses it.

### `patch_sparse_mla_nope.py`: NVIDIA/Megatron-LM#7617

Lets the TileLang SparseMLA kernel (`tilelang_dsa.fused_sparse_mla_absorbed`) take NoPE MLA
(q/k width 512) and top-k widths that aren't a multiple of 64, by zero-padding q/k to 576 and the
indices with -1. Exact. GLM-5.3-Flash needs both (width 512, k-pool selection 2048 + 3 = 2051).
Without it the kernel declines and megatron-core falls back to a dense `[heads, sq, sq]` FP32
softmax, which OOMs at 32k. Only active with `dsa_kernel_backend="tilelang"`.
- **Landed?** The patch checks for itself: it's a no-op, and logs a warning telling you to delete
  it, when `fused_sparse_mla_absorbed`'s source contains `query.size(-1) not in (512, 576)`.
- **Remove:** the `patch_sparse_mla_nope()` call in `MegatronWorker.make_megatron_module`, the
  module, and its CPU and GPU tests (`test_sparse_mla_nope.py`, and its line in
  `ci/gpu_ci_run_h100.sh`).

### `patch_moe_release_dispatcher_probs.py`: no upstream PR yet

`MoEAlltoAllTokenDispatcher.dispatch_preprocess` stores `self.probs` (the router output, with its
`grad_fn`) and only the same forward's `combine_preprocess` reads it. Kept until the next forward,
it pins that forward's autograd graph; under full recompute that is every MoE layer's recomputed
graph for the rest of backward (~21 GiB/GPU at 64k tokens on GLM-5.3-Flash). Generic MoE + full
recompute, not GLM- or mHC-specific (NVIDIA/Megatron-LM#7521 makes it reachable for megatron-core's
own mHC layers). The release wraps `MoELayer.postprocess`, where every MoE forward ends, so
patches that replace dispatcher methods can't drop it. Applied unconditionally in
`make_megatron_module`.
- **Landed?** megatron-core's dispatcher clears `self.probs` after the combine (or stops storing it).
- **Remove:** the module, its call in `make_megatron_module`, and
  `patches/megatron/test_moe_release_dispatcher_probs.py`.

### `patch_shared_expert_lora_tp.py`: Megatron-Bridge#6089

Shared-expert LoRA forward scaling under shared-expert overlap. It only activates when
`moe_shared_expert_overlap=True`.
- **Landed?** The patch checks for itself: it's a no-op when `ParallelLinearAdapter`'s source
  contains `_external_tp_reduce_scale`.
- **Remove:** the module-level `apply_shared_expert_lora_tp_patch()` call and import in
  `megatron_worker.py`, the module, and its two-rank GPU test.

### `patch_distopt_mixed_dtype_group_index.py`: NVIDIA/Megatron-LM#2872 (and #5835 on `dev`)

`DistributedOptimizer` records each model param's main-param position in grad-buffer order (bf16 buffer, then
fp32), then installs each optimizer group as `[*fp32 shards, *fp32-from-float16 shards]`. In a group that mixes
dtypes (DeepSeek-V4/V4.1 mHC `mapping_proj` and compressor `ape`, GLM-5.3 mHC) every lookup lands one param off,
and full-finetune checkpoint save fails with a shape mismatch in `sharded_param_state_*`. The patch rebuilds
`model_param_group_index_map` from `model_fp32_groups` / `model_float16_groups` after `__init__`.
- **Landed?** `DistributedOptimizer.__init__` rebuilds the map itself (#2872 on `main`; #5835 merged 2026-07-18
  into `dev` only, so not in a `main` pin).
- **Remove:** the `apply_distopt_mixed_dtype_group_index_patch()` call and import in `megatron_worker.py`, the
  module, and `patches/megatron/test_distopt_mixed_dtype_group_index.py`.

### `patch_clip_grads_mixed_dtype.py`: NVIDIA/Megatron-LM#7279 (issue #7338)

With the precision-aware optimizer the grads passed to `get_grad_norm_fp32` / `clip_grad_by_total_norm_fp32` keep
each param's dtype, so models with fp32-kept params hand one fused multi-tensor launch a mixed bf16/fp32 list and
the first step dies with "CUDA error: an illegal memory access". The patch wraps `clip_grads.multi_tensor_applier`
to launch once per dtype and recombine the L2 norm as sqrt(sum of squared group norms); #7279 does the same inside
the two functions (`_group_grads_by_dtype`).
- **Landed?** `clip_grads.py` has `_group_grads_by_dtype` (or otherwise splits by dtype before each launch).
- **Remove:** the `apply_clip_grads_mixed_dtype_patch()` call and import in `megatron_worker.py`, the module, and
  `patches/megatron/test_clip_grads_mixed_dtype.py`.

### `patch_packed_per_expert_sharded_state_dict.py`: Megatron-Bridge#6184

`PackedPerExpertLinear.sharded_state_dict` (the per-expert side of
`experts_shared_outer_loras=True` adapters) omits the required `pg_collection` kwarg, so
dist-checkpoint saving raises `TypeError`.
- **Landed?** The patch checks for itself: it's a no-op when
  `PackedPerExpertLinear.sharded_state_dict`'s source contains `pg_collection`. It's also a
  silent no-op if Bridge drops `PackedPerExpertLinear`, so check the class still exists.
- **Remove:** the `apply_packed_per_expert_sharded_state_dict_patch()` call and import in
  `megatron_worker.py`, and the module.

### `patch_dsa_hybrid_indexer.py`: SkyRL-only, no upstream counterpart

Opt-in (`SKYRL_DSA_INDEXER_BACKEND=tilelang` with `dsa_kernel_backend="cudnn"`): resolves the
`run_fused_qk_topk` DSA hook from the TileLang backend while sparse attention stays on
cuDNN/FlashMLA. On the packed THD path the cudnn backend's indexer top-k is a per-head fp32
`torch.bmm` fallback; TileLang's is a fused kernel. -32% trainer fwd+bwd for GLM-5.3 on B200.
Applies to stock `DSAttention` (GLM-5 / GLM-5.3 `glm_moe_dsa`, DeepSeek-V3.2); GLM-5.3-Flash's
k-pool indexer (`index_kpool > 1`) selects through megatron-core's `fused_qk_topk_kpool` and
bypasses the hook.
- **Landed?** Not an upstream fix: it is obsolete once megatron-core's cudnn backend has a fused
  varlen indexer top-k (`_indexer_topk_bshd` no longer falls back to `_indexer_topk_from_score_chunks`
  for packed inputs), or exposes per-hook backend selection.
- **Remove:** the module-level `apply_dsa_hybrid_indexer_patch()` call and import in
  `megatron_worker.py`, the `SKYRL_DSA_INDEXER_BACKEND` entry in `prepare_runtime_environment`,
  the module, and its CPU test.

### `patch_dist_ckpt_exchange_wait.py`: megatron-core fully-parallel checkpoint load

`exchange_loaded_tensors_broadcast` never waits on the async broadcasts of shards a rank only discards,
so their receive buffers stay allocated after the load. The patch waits on them.
- **Landed?** Self-checking: a no-op with a warning once upstream's source contains `.wait()`.
- **Remove:** the `apply_dist_ckpt_exchange_wait_patch()` call and import in `megatron_worker.py`, the
  module, and its CPU test.

### `patch_dist_optim_te_load_state_dict.py`: megatron-core distributed optimizer + TE FusedAdam

`DistributedOptimizer.load_state_dict` calls the inner `state_dict()`, which for TE's `FusedAdam` makes
fp32 copies of all optimizer state. With allocated FusedAdam state the patch restores only the param groups.
- **Landed?** `DistributedOptimizer.load_state_dict` no longer calls `self.optimizer.state_dict()`.
- **Remove:** the `apply_dist_optim_te_load_state_dict_patch()` call and import in `megatron_worker.py`,
  the module, and its CPU test.

### `patch_dsa_head_full.py` (+ `_block_copy_kernel.py`): SkyRL-only, no upstream counterpart

Opt-in (`SKYRL_DSA_HEAD_FULL=1`, with `SKYRL_DSA_HEAD_FULL_FAST_PERMUTE=1` and
`SKYRL_DSA_HEAD_FULL_ZIGZAG=1`): wraps the `run_fused_absorbed_sparse_attention` DSA hook in a
Ulysses-style all-to-all so each TP rank runs all heads on 1/TP of the query rows, instead of
`heads / TP` heads (padded to the kernel's 16- or 64/128-head tile) on every row. The fast-permute
flag does the all-to-all's layout moves with one Triton block copy; zigzag balances rows whose cost
grows with position. Applies to absorbed-MLA DSA (GLM-5 / GLM-5.3, DeepSeek-V3.2) under TP > 1.
- **Landed?** Not an upstream fix: obsolete once megatron-core's DSA kernels run efficiently at
  `heads / TP` heads per rank, or megatron-core shards absorbed sparse attention by query rows itself.
- **Remove:** the `apply_dsa_head_full_patch()` call and import in `megatron_worker.py`, the
  `SKYRL_DSA_HEAD_FULL*` entries in `prepare_runtime_environment`, and both modules.

### `patch_vision_attention_backend.py`: Megatron-Bridge `get_vision_model_config`

Qwen3-VL ViT attention-backend propagation.
- **Landed?** Megatron-Bridge's `get_vision_model_config` copies `attention_backend` from the
  language config.
- **Remove:** the `patch_vision_attention_backend()` call in `make_megatron_module`, and the module.

### `patch_mla_thd_v_pad.py`: currently not applied

Skips megatron-core's MLA THD value pad on Blackwell. **Nothing calls it:** the call was removed
from `megatron_worker.py` together with `disable_fa4_if_requested()`. Don't delete it silently: decide whether Blackwell MLA + CP training
still needs it. If yes, re-wire the call. If no, delete the module and this entry.

### `optimizer_offload_to_disk.py`: generated `_build_model_and_main_param_groups`, SkyRL-only

Not a module in this directory: `skyrl/backends/skyrl_train/distributed/megatron/optimizer_offload_to_disk.py`
generates `OffloadedDistributedOptimizer._build_model_and_main_param_groups` from the installed megatron-core source,
replacing the line `shard_main_param = shard_model_param.clone().float()` so the per-param fp32 main shards are
created with empty storage. Only the subclass gets the generated method; `DistributedOptimizer` is unchanged. It
raises if the line does not appear exactly once.
- **Landed?** Obsolete once `OptimizerConfig` has a field to defer main-param allocation (e.g.
  `defer_main_param_initialization`).
- **Remove:** `_deferred_build_model_and_main_param_groups` and its installation in
  `build_optimizer_offloaded_to_disk`, `test_deferred_build_method_is_generated_from_installed_megatron` in
  `tests/backends/skyrl_train/distributed/test_optimizer_offload_to_disk.py`, and the sentence on the generated
  method in `.agents/docs/backends/megatron.md`; set the config field instead.
- **Verify:** `tests/backends/skyrl_train/distributed/test_optimizer_offload_to_disk.py` and
  `tests/backends/skyrl_train/gpu/gpu_ci/test_optimizer_offload_to_disk.py` (2 GPUs).

## Verification

Use `RAY_ADDRESS=local` if a Ray cluster is already up on the box. Don't put `--` after
`uv run`: it breaks Ray's uv worker hook.

```bash
# CPU
uv run --isolated --extra dev --extra megatron pytest tests/backends/skyrl_train/patches/megatron/glm5_next/test_dsa_kpool_math.py

# GPU (4+ GPUs): k-pool kernels and TP sharding, KDA/mHC vs HF
uv run --isolated --extra dev --extra megatron pytest -s -v \
  tests/backends/skyrl_train/gpu/gpu_ci/patches/megatron/glm5_next/ \
  tests/backends/skyrl_train/gpu/gpu_ci/patches/megatron/mcore_ext/test_modules_vs_hf.py

# GPU (4 GPUs; needs ~/data/gsm8k from examples/train/gsm8k/gsm8k_dataset.py):
# the three GLM-5.3-Flash roundtrip rows vs vLLM. Run one roundtrip session per machine at a
# time: every session's vLLM server binds port 8000, so a second one on other GPUs sends its
# requests to the first one's server.
uv run --isolated --extra dev --extra megatron pytest -s -v -m "h100 or not h100" -k "glm-5.3" \
  tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_models.py \
  tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_lora_models.py
```

### Reference values

Mean |logprob diff| over the response tokens, as the tests print them. Compare a bump against the
newest column on the same GPU type and add a column for it. A column changes one pin at a time
where possible, so a shift can be pinned on Megatron or on vLLM.

| row / metric (threshold) | before 2026-10 | main 86c0bd69 | megatron bump | bump rebased on 7d4b02fe | vLLM 0.33 bump (#2604) |
|---|---|---|---|---|---|
| megatron-core / Megatron-Bridge | not recorded | 07147d69 / c4067f50 | ed9c2c8c / 7171584b | ed9c2c8c / 7171584b | ed9c2c8c / 7171584b |
| vLLM | not recorded | 0.32.0 | 0.32.0 | 0.32.0 | 0.33.0 |
| GPU | not recorded | 4x B200 | 4x B200 | 4x B200 | 4x H100 |
| `glm-5.3-flash-4layer_h100_tp2_ep4`: Megatron vs vLLM (< 0.1) | 0.0651 | 0.0634 | 0.0702, 0.0731 | 0.0654 | 0.0650 |
| ... vLLM before vs after weight sync (< 0.3) | - | 0.1321 | 0.1459, 0.1257 | 0.1338 | 0.1295 |
| `..._kpool_beyond_topk`: Megatron vs vLLM (< 0.1) | 0.0527 | 0.0553 | 0.0536 | 0.0583 | 0.0541 |
| ... vLLM before vs after weight sync (< 0.5) | 0.351 | 0.2867 | 0.2906 | 0.2835 | 0.2897 |
| `..._adapter` (`test_megatron_lora_models.py`): zero adapter (< 0.1) | 0.0620 | 0.0656 | 0.0632 | 0.0650 | 0.0659 |
| ... stale sampler vs perturbed Megatron (> 0.1) | - | 0.1532 | 0.1561 | 0.1532 | 0.1583 |
| ... updated adapter (< 0.1) | - | 0.0653 | 0.0618 | 0.0661 | 0.0651 |

Two values in a cell are two runs. The megatron bump sits about 0.008 above main on `tp2_ep4`
(reproduced, not investigated, well inside 0.1); the same pins with vLLM 0.31.0 gave 0.0630 there.
Rebased onto main 7d4b02fe the same row gave 0.0654. That run also passed the rows that landed on main
after the bump was cut: Qwen3.8-Flash-Next (Megatron vs vLLM 0.0487 / 0.0518, adapter rows 0.050-0.057)
and `kimi-k2.5-2layer-int4-qat` (0.0264).

The vLLM 0.33 column is the first on H100, so it has no same-GPU column to compare against yet; its
values sit within 0.006 of the newest B200 column (`bump rebased on 7d4b02fe`) on every row. A B200 column for vLLM 0.33 is still to be
recorded.
