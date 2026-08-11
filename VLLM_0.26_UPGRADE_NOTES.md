# vLLM 0.24 → 0.26 upgrade: required fixes

Working notes for the `vllm_0.24.0` branch bump to `vllm==0.26.0` (cu129 wheel,
torch 2.11 + cu128). Every entry below is a change that was *required* to get a
CI job from red to green, with the underlying cause, so reviewers can tell an
upstream bug from a SkyRL misuse.

## Summary of changes

| Area | File | Why |
|---|---|---|
| `xgrammar` pin | `pyproject.toml` | 0.2.4/0.2.5 have no cp312 linux-x86_64 wheel |
| flashinfer pins | `pyproject.toml` | 0.6.12 rejects vLLM 0.26's MNNVL `layout_code` |
| FA4 import guard | `skyrl/_compat.py`, `skyrl/__init__.py` | cutlass-dsl 4.6 breaks `flash_attn.cute`, killing megatron-core |
| `create_server_socket` | `inference_servers/vllm_server_actor.py` | new required kwarg `reuse_port` |
| TCPStore port | `inference_servers/vllm_server_actor.py`, `common.py` | RayExecutorV2 binds privileged port 100 when DP is off |
| `mm_serde` path | `skyrl/backends/renderer.py`, `tests/.../test_renderer.py` | module moved under `entrypoints.scale_out` |
| GLM MLA test model | `tests/.../megatron/test_megatron_models.py` | 0.26 allowlists MLA head-dim triples; tiny fixture was off-spec |
| NCCL weight sync | `inference_servers/new_inference_worker_wrap.py` | `receive_weights` lost its `load_weights` callback |
| delta weight sync | `weight_sync/delta_engine.py` | engine constructor + `receive_weights` signature changed |

Numbered sections below. #1-#5 are install/startup blockers, #6-#8 are API drift
that only bites specific code paths.

## 1. `xgrammar` has no cp312 linux-x86_64 wheel (install-blocking)

```
error: Distribution `xgrammar==0.2.4 @ registry+https://pypi.org/simple` can't be
installed because it doesn't have a source distribution or wheel for the current platform
```

`xgrammar` 0.2.4 shipped cp310/cp311 manylinux x86_64 wheels but only
aarch64 + macOS for cp312; 0.2.5 currently has cp310 only. vLLM 0.26 requires
`xgrammar>=0.2.1,<1.0.0`, so 0.2.3 (which has the cp312 x86_64 wheel) satisfies
it. Added to `[tool.uv.override-dependencies]`:

```toml
"xgrammar==0.2.3",
```

**Upstream:** publish gap on the xgrammar side. Revisit when a release ≥0.2.4
has a cp312 x86_64 wheel.

## 2. `flash_attn.cute` (FA4) breaks every megatron-core import

```
File ".../flash_attn/cute/utils.py", line 44, in <module>
    smem: cute.Tensor, thr_mma: cute.core.ThrMma, ...
AttributeError: module 'cutlass.cute.core' has no attribute 'ThrMma'
```

flash-attn 2.8.3 bundles `flash_attn/cute` written against the
nvidia-cutlass-dsl 4.0/4.1 API. vLLM 0.26 **hard-pins `nvidia-cutlass-dsl==4.6.0`**
(as does quack-kernels), and `cutlass.cute.core.ThrMma` no longer exists there,
so importing the subpackage raises `AttributeError`.

megatron-core probes FA4 with a bare `except ImportError`
(`megatron/core/transformer/attention.py`), so the `AttributeError` escapes and
takes down `megatron.core.transformer.attention` — i.e. *all* of megatron-core.
7 test modules failed at collection.

Fix: `skyrl/_compat.py::disable_flash_attn_cute()` sets
`sys.modules["flash_attn.cute"] = None`, which turns the probe into a plain
`ImportError` so megatron-core falls back to `HAVE_FA4 = False`. Called from
`skyrl/__init__.py` so it runs before any SkyRL submodule imports megatron.
Nothing in SkyRL uses FA4.

**Alternative if this ever needs removing:** rebuild the flash-attn 2.8.3 fork
wheel with `flash_attn/cute` stripped. Downgrading cutlass-dsl is not an option
(vLLM's pin is exact).

## 3. `create_server_socket()` gained a required `reuse_port` kwarg

```
TypeError: create_server_socket() missing 1 required keyword-only argument: 'reuse_port'
```

0.26 signature is `create_server_socket(addr, *, reuse_port: bool)`. SkyRL runs
one uvicorn per port (no `api_server_count` fan-out), matching vLLM's own
single-server path in `run_server_worker`, so `reuse_port=False`.

## 4. RayExecutorV2 binds privileged port 100 when DP is disabled

```
torch.distributed.DistNetworkError: The server socket has failed to listen on any
local network address. port: 100, ..., EADDRINUSE
```

vLLM 0.26 makes `RayExecutorV2` the default for
`distributed_executor_backend="ray"` (`VLLM_USE_RAY_V2_EXECUTOR_BACKEND=True`).
It derives the worker `torch.distributed` TCPStore port as:

```python
start_port = master_port + 100 + local_dp_rank * window   # ray_executor_v2._select_tcpstore_port
```

With DP disabled, `ParallelConfig.__post_init__` takes its
"fall back to env vars" branch, where `VLLM_DP_MASTER_PORT` defaults to **0** and
`VLLM_DP_RANK_LOCAL` to **0** — so `start_port` is 100, a privileged port, and
every engine dies before loading the model.

Fix: seed `VLLM_DP_MASTER_PORT` with a free ephemeral port per server actor
(`pick_free_port()` in `inference_servers/common.py`). Ignored when DP is on,
because vLLM assigns the master port itself on that path.

**This looks like an upstream vLLM bug** — `_select_tcpstore_port` never guards
`master_port == 0`, so any non-DP ray-backend deployment on 0.26.0 should hit it.
Worth filing upstream; the workaround is env-local and harmless once fixed.

## 5. flashinfer must be ≥ 0.6.13 (MNNVL `layout_code`)

```
RuntimeError: Worker failed with error 'MNNVL AllReduce does not support
quantization fusion and thus no layout_code'
```

vLLM 0.26's allreduce+RMSNorm fusion pass
(`compilation/passes/fusion/allreduce_rms_fusion.py`) passes
`QuantizationSFLayout.SWIZZLED_128x4` as `layout_code` for both the `trtllm` and
`mnnvl` workspaces. flashinfer 0.6.12 raises when MNNVL gets a non-`None`
`layout_code`; 0.6.13 defaults and honours it.

Pins moved 0.6.12 → **0.6.13** for `flashinfer-python`, `flashinfer-jit-cache`
and `flashinfer-cubin`, in the `fsdp` and `megatron` extras and the overrides.

**Why 0.6.13 and not the 0.6.14 vLLM's metadata asks for:** `flashinfer-cubin`
was never released at 0.6.14 (PyPI stops at 0.6.13, and it is not on the
flashinfer index at all), and flashinfer hard-errors on a version skew:

```
RuntimeError: flashinfer-cubin version (0.6.13) does not match flashinfer version (0.6.14).
```

0.6.13 is also the version vLLM 0.26.0's own source comments reference.

## 6. `mm_serde` moved (`entrypoints.serve.disagg` → `entrypoints.scale_out.token_in_token_out`)

`skyrl/backends/renderer.py` imports `decode_mm_kwargs_item` for the multimodal
render path. In 0.26 it lives at
`vllm.entrypoints.scale_out.token_in_token_out.mm_serde`; the old
`vllm.entrypoints.serve.disagg` package is gone. The import now tries the new
path first and falls back to the old one, and `tests/.../test_renderer.py`
stubs the new module path.

Only reachable with multimodal input, so no GPU test caught it — found by
resolving every `vllm.*` symbol SkyRL imports against 0.26.

## 7. GLM MLA test fixture rejected by the new MLA prefill selector

```
ValueError: No valid MLA prefill backend found with MLAPrefillSelectorConfig(
  dtype=torch.bfloat16, mla_dimensions=(qk_nope_head_dim=64, qk_rope_head_dim=64, v_head_dim=64)).
  Reasons: {FLASH_ATTN: [Model does not have supported MLA dimensions ...]}
```

0.26 added `v1/attention/backends/mla/prefill/selector.py`, which allowlists MLA
head-dim triples per backend. On Hopper and older the only candidate is
`FLASH_ATTN`, which accepts exactly:

| Triple (qk_nope / qk_rope / v) | Model family |
|---|---|
| 128 / 64 / 128 | DeepSeek |
| 192 / 64 / 256 | GLM |
| 64 / 64 / 128 | Mistral-S4 |

`eatang/glm-4.7-flash-tiny-random` was scaled down to 64/64/64, which is not in
the list, so it can no longer serve on vLLM. The real `zai-org/GLM-4.7-Flash`
uses the GLM triple (192/64/256) and is unaffected.

Per maintainer direction the tiny fixture was dropped and MLA coverage moved to
the real 31B checkpoint in the h100 job:

- `glm-4.7-flash_h100_tp4_ep4` — TP=4/EP=4/ETP=1, vLLM TP=4 colocated,
  thresholds 3e-1 (post-sync) / 5e-2 (megatron-vs-vLLM), `marks=pytest.mark.h100`.
- Added to `is_large_moe` (skips DistributedOptimizer init — 31B MoE OOMs at
  init otherwise) and to `_engine_overrides_for_model` (`gpu_memory_utilization=0.5`
  plus `max_model_len=4096`, since GLM-4.7-Flash's 202k default context would
  size the KV pool past what is left beside the colocated policy shard).
- Removed the now-unused `_skip_mla_on_pre_hopper` mark.

Verified on 4xH100: megatron diff 0.0205, post-sync diff 0.118.

## 8. Weight-transfer engine API changed (`receive_weights`, engine constructor)

```
Exception: Call to collective_rpc method failed: Worker failed with error
'NCCLWeightTransferEngine.receive_weights() got an unexpected keyword argument 'load_weights''
```

Two related breaks in `vllm.distributed.weight_transfer`:

**a) `receive_weights` lost its `load_weights` callback.** The 0.26 contract is
`receive_weights(self, update_info)`, and engines load through
`self.model.load_weights`. Callers who need to interpose now retarget the engine
with the new base-class hooks:

```python
engine.set_weight_update_target(model, model_config)   # ...  engine.reset_weight_update_target()
```

SkyRL's `NewInferenceWorkerWrap.update_weights_nccl` still passed
`load_weights=_load_weights`, so **every non-colocated NCCL weight sync failed**
(colocated sync goes through the IPC path, which does its own loading — that is
why only the non-colocated params broke). It now retargets the engine at a small
`_LoadWeightsProxy` that forwards everything except `load_weights`, preserving
both the `set_current_vllm_config` wrap and the spec-decode drafter reload.

**b) Engine constructor is now `(config, vllm_config, device, model)`.**
`WeightTransferEngineFactory.create_engine` calls
`engine_cls(config, vllm_config, device, model)`; SkyRL's
`DeltaWeightTransferEngine.__init__` took `(config, parallel_config, model)`, so
`weight_sync_backend="delta"` would `TypeError` at engine creation. Updated to
the 4-arg signature (still duck-typed, not subclassed, so the module stays
importable without vLLM) plus `set/reset_weight_update_target` and the new
`receive_weights` signature.

**c) The engine must implement vLLM's full lifecycle contract.** 0.26's
`Worker` drives `start_weight_update()` → `update_weights(dict)` →
`finish_weight_update()` on the engine, plus `reset_weight_update_target()` and
`supports_draft_weight_update`. Previously the native `/update_weights` endpoint
called `receive_weights` with its own `load_weights` callback, so a duck-typed
engine could get away with fewer methods. `test_delta_weight_sync_sparse_update_e2e`
failed with:

```
500 "Call to collective_rpc method failed: Worker failed with error
''DeltaWeightTransferEngine' object has no attribute 'update_weights''"
```

`update_weights` now mirrors the base (parse → receive → synchronize).
`start_weight_update` / `finish_weight_update` are documented no-ops, because
SkyRL's delta flow drives layerwise reload from the worker extension
(`skyrl_start_weight_update` / `skyrl_finish_weight_update` — see
`DeltaWeightTransferSender._apply_receiver_update`); doing it in the engine too
would double-initialize. `supports_draft_weight_update = False`.

This one *is* covered: `test_delta_weight_sync_e2e.py` is parametrized
`[fsdp, megatron]`, and the `[megatron]` param caught it.

## CI job status

| Job | Command | Result |
|---|---|---|
| `skyrl_train_megatron_models` | `pytest -m megatron_models tests/backends/skyrl_train/gpu/gpu_ci` | 3 passed, 4 skipped |
| `skyrl-train-gpu-ci-h100` (megatron half) | `pytest -m h100 megatron/test_megatron_models.py megatron/test_router_replay.py` | **5 passed** (36m) |
| `skyrl-train-gpu-ci-h100` (fsdp half) | `pytest -m h100 test_policy_local_engines_e2e.py` | **2 passed** (13m, after fix #8) |
| `skyrl-train-gpu-ci-megatron` | `ci/gpu_ci_run_skyrl_train_megatron.sh` | 129 passed / 1 failed / 2 skipped (3h08m); the failure was #8c, re-run green after the fix |
| `skyrl-train-gpu-ci` | `ci/gpu_ci_run_skyrl_train.sh` | **112 passed, 9 skipped** (2h26m) |
| cpu `not vllm` leg | `--extra skyrl-train --extra dev pytest tests/train tests/backends/skyrl_train` | 1464 passed, 5 skipped |
| cpu `vllm` leg | `--extra fsdp --extra dev pytest ... -m vllm` | 5 passed |

h100 megatron-models measurements (4xH100, TP=4/EP=4/ETP=1, vLLM TP=4 colocated):

| Param | megatron-vs-vLLM diff (thr) | post-sync diff (thr) |
|---|---|---|
| `glm-4.7-flash_h100_tp4_ep4` | 0.0205 (5e-2) | 0.118 (3e-1) |
| `nemotron3-nano_tp4_ep4_h100` | 0.0339 (5e-2) | 0.212 (5e-1) |
| `qwen3.5-35b-a3b_h100_tp4_ep4` | 0.0088 (5e-2) | 0.0216 (3e-1) |

Router replay (`megatron/test_router_replay.py`, Moonlight-16B-A3B, TP2/PP2/EP2)
still behaves correctly under 0.26 — replay reduces divergence from the rollout
as the test asserts: with replay 0.0071 vs without replay 0.0121.

### Reproducing locally (4xH100)

```bash
# megatron model-parity job
uv run --directory . --isolated --extra dev --extra megatron \
  pytest -s tests/backends/skyrl_train/gpu/gpu_ci -m megatron_models

# h100 job (both halves, as ci/gpu_ci_run_h100.sh runs them)
uv run --directory . --isolated --extra dev --extra fsdp \
  pytest -s -vvv -m h100 tests/backends/skyrl_train/gpu/gpu_ci/test_policy_local_engines_e2e.py
uv run --directory . --isolated --extra dev --extra megatron \
  pytest -s -vvv -m h100 tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_models.py \
                          tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_router_replay.py

# megatron marker job
uv run --directory . --isolated --extra dev --extra megatron \
  pytest -s tests/backends/skyrl_train/gpu/gpu_ci -m megatron

# fsdp / non-megatron job
uv run --directory . --isolated --extra dev --extra fsdp \
  pytest -s tests/backends/skyrl_train/gpu/gpu_ci -m "not (integrations or megatron)" \
  --ignore=tests/backends/skyrl_train/gpu/gpu_ci/megatron
```

The h100 job pulls ~225GB of checkpoints (GLM-4.7-Flash, Nemotron-3-Nano-30B,
Qwen3.5-35B-A3B, Moonlight-16B); pre-warming the HF cache makes reruns much faster.

### Worth reporting upstream / revisiting

| Issue | Where | Action |
|---|---|---|
| `_select_tcpstore_port` doesn't guard `master_port == 0`, so every non-DP ray-backend engine tries privileged port 100 | vLLM `v1/executor/ray_executor_v2.py` | file upstream; drop `_seed_dp_master_port()` once fixed |
| megatron-core's FA4 probe catches only `ImportError`, so a broken `flash_attn.cute` takes down all of megatron-core | Megatron-LM `core/transformer/attention.py` | file upstream (`except (ImportError, AttributeError)`); or rebuild the flash-attn fork wheel without `flash_attn/cute` and drop `skyrl/_compat.py` |
| vLLM 0.26 requires `flashinfer-python==0.6.14` but `flashinfer-cubin` has no 0.6.14 release, and flashinfer hard-errors on the skew | flashinfer releases | re-pin all three to 0.6.14 when cubin 0.6.14 ships |
| `xgrammar` 0.2.4/0.2.5 missing cp312 linux-x86_64 wheels | xgrammar releases | drop the `==0.2.3` override once a newer release has the wheel |

### Static checks worth repeating on the next vLLM bump

Two cheap greps caught things no test did:

1. Resolve every `vllm.*` symbol SkyRL imports (`from vllm... import X`) against
   the installed version — this is what surfaced the `mm_serde` move (#6).
2. Assert every key SkyRL `setattr`s onto the parsed vLLM CLI namespace in
   `inference_servers/utils.py::build_vllm_cli_args` still exists. Those are
   plain `setattr` calls, so a *removed* vLLM arg is silently ignored rather than
   raising — a silent behavior change. All 37 keys SkyRL sets still exist in
   0.26 (`enable_return_routed_experts`, `language_model_only`,
   `kv_cache_metrics`, `weight_transfer_config`, `worker_extension_cls`, ...).

All skips are pre-existing and environment-gated — the same ones CI produces:

- `megatron_models` job: `qwen3.5-moe_tp2_ep2` carries its own
  `pytest.mark.skip` (tiny-Qwen3.5 correctness), and the three `h100`-marked
  params are auto-skipped outside `-m h100` by
  `tests/backends/skyrl_train/gpu/conftest.py`.
- `megatron` job: 2 skips, incl. `megatron_fully_reshardable_optimizer_cpu_offload`
  (documented upstream megatron-core bugs).
- `skyrl_train` job: 9 skips behind `requires_local_vllm`
  (`SKYRL_LOCAL_VLLM != "1"`), which no CI script sets either.

**Bottom line: all four GPU CI jobs and both CPU legs pass on this branch, with
no test disabled or threshold loosened to get there.** The only test-side change
is #7, which swaps an unservable tiny MLA fixture for the real checkpoint at
maintainer direction.
