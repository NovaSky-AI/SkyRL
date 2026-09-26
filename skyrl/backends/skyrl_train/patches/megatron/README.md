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
| `patches/megatron/mcore_ext/test_dsa_kpool_math.py` (CPU) | `mcore_ext/dsa_kpool.py` key compression vs HF |
| `gpu_ci/patches/megatron/mcore_ext/test_dsa_kpool.py` | `mcore_ext/dsa_kpool.py` pooled top-k selection |
| `gpu_ci/patches/megatron/mcore_ext/test_modules_vs_hf.py` | `mcore_ext/kda.py`, `mcore_ext/hyper_connection.py` vs HF |
| `gpu_ci/patches/megatron/test_dsa_index_share_recompute.py` | `patch_dsa_index_share.py` |
| `gpu_ci/patches/megatron/test_shared_expert_lora_tp.py` | `patch_shared_expert_lora_tp.py` |

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
apart is how the k-pool indexer ended up silently unused in training (see
`glm5_next/dsa.py`, `Glm5NextDSAttention._forward_with_kpool_topk`).

## GLM-5.3-Flash (`glm5_next`)

GLM-5.3-Flash needs four upstream Megatron-LM PRs and a Megatron-Bridge model. `mcore_ext/` holds
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
- **Carried as:** `mcore_ext/hyper_connection.py` (`RMSNormInputHyperConnectionModule`).
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
- **Landed?** megatron-core's `HyperConnectionTransformerLayer` accepts a MoE MLP submodule, with
  no `NotImplementedError` for MoE.
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
- **Verify:** `gpu_ci/patches/megatron/mcore_ext/test_modules_vs_hf.py::test_hyper_connection_matches_hf`; the GLM roundtrip rows.
  These run with the default full recompute, so they exercise the worker block above.

### NVIDIA/Megatron-LM#7522: k-pool DSA indexer

This is the riskiest entry. A wrong k-pool selection doesn't raise. It silently attends to
different tokens than vLLM once a sequence is longer than `dsa_indexer_topk` (2048).

- **Carried as:**
  - `mcore_ext/dsa_kpool.py`: the six k-pool kernels, copied verbatim from #7522.
  - `glm5_next/dsa.py`:
    - `Glm5NextDSAIndexer`: k-pool gate/ape parameters and the gate score, hand-merged onto the
      pinned `DSAIndexer`;
    - `Glm5NextDSAttention._forward_with_kpool_topk`: swaps the pinned `DSAttention.forward`'s
      token-level top-k for `fused_qk_topk_kpool`, and raises if the pooled selection doesn't run
      exactly once;
    - the `kpool <= 1` long-sequence guard.
  - `glm5_next/layer_specs.py`: the `core_attention.module` / `submodules.indexer.module` swaps.
  - `glm5_next/provider.py`: `dsa_indexer_kpool`, `dsa_indexer_kpool_always_select_tail`.
- **Landed?** megatron-core's `experimental_attention_variant/dsa.py` defines
  `fused_qk_topk_kpool`, `DSAttention.forward` dispatches on the indexer's `index_kpool`, and
  `TransformerConfig` has `dsa_indexer_kpool`.
- **Remove:**
  - `glm5_next/layer_specs.py`: stop swapping in `Glm5NextDSAIndexer`, and in
    `Glm5NextDSAttention` if nothing else is left in it.
  - `glm5_next/dsa.py`: delete `Glm5NextDSAIndexer` and `_forward_with_kpool_topk`. Delete the
    whole module if `Glm5NextDSAttention` only has the guard left.
  - `glm5_next/provider.py`: drop the k-pool fields if `TransformerConfig` declares them.
  - `glm5_next/bridge.py`: keep reading `index_kpool` / `index_kpool_always_select_tail` from the
    HF config into the provider. Upstream's parameter names for the compress gate/ape must match
    the bridge mapping.
  - Tests:
    - `tests/backends/skyrl_train/patches/megatron/mcore_ext/test_dsa_kpool_math.py` imports
      `mcore_ext.dsa_kpool._kpool_compress_keys`. Repoint it at megatron-core.
    - `tests/.../gpu_ci/patches/megatron/mcore_ext/test_dsa_kpool.py` (GPU kernel checks): same.
  - Delete `mcore_ext/dsa_kpool.py`.
- **Verify** (all required):
  - `patches/megatron/mcore_ext/test_dsa_kpool_math.py` (CPU) and `gpu_ci/patches/megatron/mcore_ext/test_dsa_kpool.py` (GPU).
  - `test_logprobs_matching_roundtrip[glm-5.3-flash-4layer_h100_tp2_ep4_kpool_beyond_topk]`. This
    is the only test that runs sequences past `index_topk`. Its logprob diff must not get worse
    than with the vendored code (about 0.053; token-level selection, i.e. no k-pool, gives about
    0.059).

### NVIDIA/Megatron-LM#7523: FP8 wgrad

Not carried. GLM-5.3-Flash runs bf16 end to end.

### Megatron-Bridge: the GLM-5.3-Flash model

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
    - the k-pool field mapping;
    - the mHC scale mappings.
  - Delete `glm5_next/`.
- **Verify:** the full [Verification](#verification) set.

## Standalone patches

### `patch_dsa_index_share.py` (+ `dsa_index_share_recompute.patch`): NVIDIA/Megatron-LM#6793

Per-forward DSA index-share carrier under activation recompute.
- **Landed?** The patch checks for itself: megatron-core's DSA module has
  `_dsa_index_share_carrier_scope`, and applying the patch logs a warning telling you to delete it.
- **Remove:** the `patch_dsa_index_share()` call in `MegatronWorker.make_megatron_module`, both
  files here, and the `*.patch` package-data entry in `pyproject.toml` if nothing else uses it.

### `patch_shared_expert_lora_tp.py`: Megatron-Bridge#6089

Shared-expert LoRA forward scaling under shared-expert overlap. It only activates when
`moe_shared_expert_overlap=True`.
- **Landed?** The patch checks for itself: it's a no-op when `ParallelLinearAdapter`'s source
  contains `_external_tp_reduce_scale`.
- **Remove:** the module-level `apply_shared_expert_lora_tp_patch()` call and import in
  `megatron_worker.py`, the module, and its two-rank GPU test.

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
uv run --isolated --extra dev --extra megatron pytest tests/backends/skyrl_train/patches/megatron/mcore_ext/test_dsa_kpool_math.py

# GPU (4+ GPUs): kernels, KDA/mHC vs HF, and the three GLM roundtrip rows vs vLLM
uv run --isolated --extra dev --extra megatron pytest -s -v -m "h100 or not h100" \
  tests/backends/skyrl_train/gpu/gpu_ci/patches/megatron/mcore_ext/test_dsa_kpool.py \
  tests/backends/skyrl_train/gpu/gpu_ci/patches/megatron/mcore_ext/test_modules_vs_hf.py \
  tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_models.py \
  -k "kpool_selects or glm5_next_modules or glm-5.3"
```

Expected Megatron-vs-vLLM logprob diffs (threshold 0.1):
- `glm-5.3-flash-4layer_h100_tp2_ep4`: 0.0651
- `..._kpool_beyond_topk`: 0.0527
- `..._lora`: 0.0620
