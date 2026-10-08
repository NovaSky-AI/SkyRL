# Transformer Engine patches

Runtime adjustments for the `transformer-engine==2.19.0` pin.

## `patch_fa2_head_dim.py`: FA2 `head_dim > 192` compute-capability allowlist

TE rejects FlashAttention 2 for `head_dim > 192` unless the compute capability is
one of `(8,0), (9,0), (10,0), (12,0)`, keeping only FA2's real limits (`<= 256`,
`% 8 == 0`) out of the picture. On an excluded arch a `head_dim` of 256 (Gemma
2/3) drops to unfused attention, which materializes the quadratic attention
matrix — upstream reported a 202 GiB allocation and OOM at 65k tokens.

[#3360](https://github.com/NVIDIA/TransformerEngine/pull/3360) removes it, and is
**still open and unmerged**. So this backport is still required at 2.19.0.

### The clause moved, which makes "is it fixed yet?" easy to get wrong

Upstream history, from `git log -S` on
`transformer_engine/pytorch/attention/dot_product_attention/utils.py`:

| what | where |
|---|---|
| introduced as `head_dim_qk > 192` | `8ace813c` (#1704), a refactor |
| removed | `5535b098` ([#2836](https://github.com/NVIDIA/TransformerEngine/pull/2836)), merged 2026-06-04 |
| reintroduced as `fa2_padded_head_dim > 192` | before v2.17 shipped |
| removed again | [#3360](https://github.com/NVIDIA/TransformerEngine/pull/3360) — **open, never merged** |

`fa2_padded_head_dim` is `max(head_dim_qk, head_dim_v)`; the gate is otherwise
character-identical. Verified against the published meta wheels:

| release | gate present |
|---|---|
| 2.16.0, 2.16.1 | yes, as `head_dim_qk` |
| 2.17.0, 2.17.1, 2.18.0, 2.19.0 | yes, as `fa2_padded_head_dim` |

A grep for the 2.16 spelling finds nothing from 2.17.0 on, which reads exactly
like "upstream fixed it" and is not. `patch_fa2_head_dim.py` matches both
spellings for that reason.

### Deleting this when it is genuinely fixed

Check the source, not a PR number or changelog:

```bash
uv run --isolated --extra megatron python -c "import torch, inspect; \
from transformer_engine.pytorch.attention.dot_product_attention import utils as u; \
print('device_compute_capability not in' in inspect.getsource(u.get_attention_backend))"
```

`False` means the gate is gone and this module should be deleted. `True` means it
is still there in some spelling — if the patch stops matching, add the new
spelling to `_CLAUSE_VARIANTS` rather than assuming it was fixed. The patch warns
loudly (and reports which case it found) instead of no-opping quietly.

`verify_fa2_head_dim.py` runs the same check and then measures the effect:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv   # must not be 8.0/9.0/10.0/12.0
uv run --isolated --extra megatron python \
    skyrl/backends/skyrl_train/patches/te/verify_fa2_head_dim.py
```

Phase A (backend selection) should show:

```
 head_dim    FA2 before     FA2 after
      128         2.8.3         2.8.3
      192         2.8.3         2.8.3
      200          None         2.8.3   <- fixed
      256          None         2.8.3   <- fixed
      264          None          None   <- still rejected, correctly
```

Phase B forces FA2 and diffs fwd+bwd against TE's unfused reference — the part
that proves the allowlist was stale rather than protecting a broken kernel.
Reference from a validated H100 run (`--force`, bf16, 4k causal): `max |out diff|
= 0.01562`, `max |dq diff| = 0.01562`, PASS. A crash or CUDA error instead means
#3360 itself is unsafe on that arch — report it upstream, don't ship.

### Why a Python patch

TE ships as a prebuilt wheel and `uv run --isolated` builds a fresh venv per
invocation, so a site-packages edit does not survive a run (nor reach other Ray
nodes). The gate is an inline clause in a ~1000-line function, but
`get_attention_backend` is undecorated and its only caller resolves it as a
module attribute (`dpa_utils.get_attention_backend`), so recompiling that one
function into TE's own `__dict__` is enough. The patch also invalidates
`_attention_backends`, which memoizes backend selection.

Guarded to no-op on sm80/90/100/120, and idempotent.

### Wired in at

`MegatronWorker.make_megatron_module()`, before `provide_distributed_model()` —
the choke point both the policy and ref workers pass through, and the last hook
before any TE attention layer exists. Megatron only; FSDP uses HF transformers.
Inert on H100.

## `disable_fa4.py`: FlashAttention 4 opt-out

`disable_fa4_if_requested()` flips `FlashAttentionUtils.v4_is_installed` to
`False` when `SKYRL_DISABLE_FA4` is set, so TE falls back to FA2 or cuDNN fused
attention through its own backend selection. It is the runtime half of the `fa4`
extra in `pyproject.toml`: the extra decides whether FA4 is installed, this
decides whether an installed FA4 is used, without re-resolving the venv.

Mainly for A/B-ing FA2 against FA4 on an otherwise identical environment, and as
an escape hatch if an FA4 kernel misbehaves. Nothing currently calls it from
`megatron_worker.py` — wire the call in if you need the switch.

## `pin_nvrtc.py`: NVRTC paired with the headers TE compiles against

TE JIT-compiles some kernels with NVRTC (e.g. the MXFP8 RMSNorm forward). Its
loader takes `libnvrtc` from `NVRTC_HOME` / `CUDA_HOME` / `/usr/local/cuda`
before the pip `nvidia-cuda-nvrtc` wheel, and 2.19 also points
`NVTE_CUDA_INCLUDE_DIR` at the pip `nvidia/cu13` headers. On a box whose system
toolkit is CUDA 12.x, NVRTC 12.9 then compiles against CUDA 13 headers and
fails with `NVRTC_ERROR_COMPILATION` in `rmsnorm_fwd_kernel.cu`. This is a
regression from 2.16: that release left the include dir unset, so the system
NVRTC and the system headers matched. It only shows on MXFP8 (Blackwell) paths.

`pin_te_nvrtc_to_pip_cuda()` sets `NVRTC_HOME` to the pip `nvidia/cu{major}`
(the major comes from the installed `transformer-engine-cu{major}` wheel). It
leaves the environment alone if the user set `NVTE_CUDA_INCLUDE_DIR`, if
`NVRTC_HOME` already holds a `libnvrtc`, or if TE is already imported. The
`Run-time NVRTC version:` line in TE's compile log shows which NVRTC TE loaded.

**Wired in at** `skyrl/backends/skyrl_train/__init__.py`: TE reads `NVRTC_HOME`
once, at import, and every SkyRL module that imports TE or Megatron lives under
that package. A script that imports `transformer_engine` without going through
SkyRL is not covered.

**Delete it** when `_load_cuda_library` in `transformer_engine/common/__init__.py`
prefers the pip wheel. `test_installed_te_still_prefers_system_nvrtc` in
`tests/backends/skyrl_train/patches/te/test_pin_nvrtc.py` fails at that point.
