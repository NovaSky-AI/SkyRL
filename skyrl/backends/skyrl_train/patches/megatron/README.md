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
| `patches/megatron/test_sparse_mla_nope.py` (CPU) | `patch_sparse_mla_nope.py` padding/unpadding, fake kernel |
| `gpu_ci/patches/megatron/test_sparse_mla_nope.py` (H100) | `patch_sparse_mla_nope.py` vs dense reference, real TileLang kernel |
| `patches/megatron/test_dsa_hybrid_indexer.py` (CPU) | `patch_dsa_hybrid_indexer.py` hook resolution, fake backends |
| `patches/megatron/test_moe_release_dispatcher_probs.py` (CPU) | `patch_moe_release_dispatcher_probs.py`, fake layer state |
| `gpu_ci/patches/megatron/glm5_next/test_dsa_kpool_tp_shard.py` | `glm5_next/dsa.py` TP query sharding (two ranks) |
| `gpu_ci/patches/megatron/test_checkpoint_without_output_frozen_inputs.py` | `patch_checkpoint_without_output_frozen_inputs.py`; the unpatched case doubles as the landed check |

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
  - Before deleting the local bridge, compare its behaviour with upstream's:
    - `language_model_only=True` handling;
    - the `head_dim=0` NoPE RoPE skip;
    - the k-pool field mapping, including the FP32 indexer weights projection;
    - the mHC scale mappings.
  - Move `Glm5NextDSAttention`'s opt-in TP sharding to wherever upstream builds the model, or
    drop it if upstream shards the k-pool scoring itself.
  - Delete `glm5_next/`.
- **Verify:** the full [Verification](#verification) set.

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

### `patch_checkpoint_without_output_frozen_inputs.py`: no upstream PR yet

`CheckpointWithoutOutput` (the `mhc` / `layernorm` / `moe_act` selective-recompute modules) saves its
inputs with `save_for_backward`, which keeps nothing when none of them requires grad, but its recompute
hook still fires once a later consumer is trainable. Under LoRA (frozen embeddings and early base weights,
trainable adapters after the first mHC / layernorm block) backward then runs off the end of
`ctx.saved_tensors`: `RuntimeError: coroutine raised StopIteration` (DeepSeek-V4 LoRA with
`recompute_modules=[..., mhc, layernorm]`). The patch keeps references to such inputs on the ctx and
recomputes from them. Applied unconditionally at `megatron_worker.py` import.
- **Landed?** megatron-core recomputes (or keeps) inputs that do not require grad: the GPU test
  `test_unpatched_recompute_fails_on_frozen_inputs` then fails because backward no longer raises.
- **Remove:** the module-level `apply_checkpoint_without_output_frozen_inputs_patch()` call and import in
  `megatron_worker.py`, the module, and
  `gpu_ci/patches/megatron/test_checkpoint_without_output_frozen_inputs.py`.

### `patch_vision_attention_backend.py`: Megatron-Bridge `get_vision_model_config`

Qwen3-VL ViT attention-backend propagation.
- **Landed?** Megatron-Bridge's `get_vision_model_config` copies `attention_backend` from the
  language config.
- **Remove:** the `patch_vision_attention_backend()` call in `make_megatron_module`, and the module.

### `patch_mla_thd_v_pad.py`: currently not applied

Skips megatron-core's MLA THD value pad on Blackwell. **Nothing calls it:** the call was removed
from `megatron_worker.py` together with `disable_fa4_if_requested()`. Don't delete it silently: decide whether Blackwell MLA + CP training
still needs it. If yes, re-wire the call. If no, delete the module and this entry.

## Verification

Use `RAY_ADDRESS=local` if a Ray cluster is already up on the box. Don't put `--` after
`uv run`: it breaks Ray's uv worker hook.

```bash
# CPU
uv run --isolated --extra dev --extra megatron pytest tests/backends/skyrl_train/patches/megatron/glm5_next/test_dsa_kpool_math.py

# GPU (4+ GPUs): kernels, KDA/mHC vs HF, and the three GLM roundtrip rows vs vLLM
uv run --isolated --extra dev --extra megatron pytest -s -v -m "h100 or not h100" \
  tests/backends/skyrl_train/gpu/gpu_ci/patches/megatron/glm5_next/test_dsa_kpool.py \
  tests/backends/skyrl_train/gpu/gpu_ci/patches/megatron/mcore_ext/test_modules_vs_hf.py \
  tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_models.py \
  -k "kpool_selects or glm5_next_modules or glm-5.3"
```

Expected Megatron-vs-vLLM logprob diffs (threshold 0.1):
- `glm-5.3-flash-4layer_h100_tp2_ep4`: 0.0651
- `..._kpool_beyond_topk`: 0.0527
- `..._lora`: 0.0620
