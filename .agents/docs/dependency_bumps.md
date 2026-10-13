# Dependency Bumps

What to check and which tests to run when bumping a pinned dependency.

## Every bump

- Regenerate `uv.lock` and check in its diff that only the packages you meant to change actually changed.
- If the dependency has patches under `skyrl/backends/skyrl_train/patches/<dep>/`, read that
  directory's README. A patch whose text match no longer finds its target can **silently stop applying**,
  so check each patch against the new source.
- Rebase onto `main` before you run GPU CI. A stale base causes failures unrelated to the bump.

## Megatron-related bumps

For megatron-core and megatron-bridge, follow `skyrl/backends/skyrl_train/patches/megatron/README.md`.

## Transformer Engine

TE sits under every Megatron kernel path, so run **all** Megatron GPU CI. Apply the PR labels
(see `ci.md`):

| Label | Suite |
|---|---|
| `run_megatron_gpu_ci` | `SkyRL-GPU-Megatron` (`megatron` marker) |
| `run_megatron_gpu_ci_models` | `Megatron-Model-GPU-CI` (`megatron_models` marker) |
| `run_h100_gpu_ci` | `H100-GPU-CI` (`h100` marker: FP8 blockwise rows, router replay, LoRA models, ...) |

CI has no Blackwell runners. Run the MXFP8 rows **manually on a B200** box:

```bash
RAY_ADDRESS=local uv run --isolated --extra dev --extra megatron \
  pytest -m b200 -k mxfp8 -v -s tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_models.py \
  > /tmp/mxfp8.log 2>&1
```

Put the B200 results in the PR description.

Also check:
- The TE patches in `patches/te/` (see that directory's README), such as the FA2 `head_dim` gate.
- **The NVRTC runtime the TE JIT uses.** TE loads `/usr/local/cuda`'s NVRTC ahead of the pip one,
  and since 2.19 it compiles against the pip CUDA headers. On a box with a CUDA 12.x toolkit that mismatch
  fails MXFP8 RMSNorm with `NVRTC_ERROR_COMPILATION`. `patches/te/pin_nvrtc.py` pins `NVRTC_HOME` to the
  pip CUDA. Its tripwire test tells you when the patch can be deleted.
