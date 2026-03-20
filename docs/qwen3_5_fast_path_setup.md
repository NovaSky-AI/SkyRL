# Qwen3.5 Fast Path Setup for FSDP Training

## Background

Qwen3.5 is a hybrid architecture with both standard attention layers and **linear attention** (gated delta rule) layers. The linear attention layers require two specialized libraries for fast forward and backward passes:

1. **`flash-linear-attention` (fla)** - Triton kernels for `chunk_gated_delta_rule` and `fused_recurrent_gated_delta_rule`
2. **`causal-conv1d`** - CUDA kernels for causal 1D convolution (`causal_conv1d_fn`, `causal_conv1d_bwd`)

Without both libraries, HuggingFace transformers falls back to pure PyTorch implementations (`torch_chunk_gated_delta_rule` and `F.silu(self.conv1d(...))`) which are **significantly slower** for both forward and backward passes. The check is at:

```python
# transformers/models/qwen3_5/modeling_qwen3_5.py:295
is_fast_path_available = all(
    (causal_conv1d_fn, causal_conv1d_update, chunk_gated_delta_rule, fused_recurrent_gated_delta_rule)
)
```

## Current State

- `flash-linear-attention` is vendored at `./flash-linear-attention/` and properly wired as a uv source in `pyproject.toml`. It installs cleanly (pure Python/Triton, no CUDA compilation).
- `causal-conv1d` is vendored at `./causal-conv1d/` but is **blocked** from installation by an override in `pyproject.toml`:
  ```toml
  "causal-conv1d; sys_platform == 'never'",
  ```
  This was originally added to suppress transitive resolution from `flash-linear-attention[conv1d]`.

## The Problem with `causal-conv1d`

`causal-conv1d` requires CUDA compilation (it has `.cu` files in `csrc/`). This creates issues with uv + Ray:

1. **uv build isolation**: `causal-conv1d`'s `pyproject.toml` only declares `setuptools`, `wheel`, `torch` as build requirements, but with `match-runtime = true` for torch, uv can't resolve because `causal-conv1d` doesn't declare static metadata.

2. **`no-build-isolation`**: Works locally but fails on Ray workers because they create fresh venvs that lack setuptools.

3. **Local wheel path**: Ray's working directory packaging doesn't include the 160MB wheel file (excluded by `.gitignore` or size limits), so `path = "./dist/..."` fails on workers.

## Solution: Pre-built Wheel via URL

The correct approach (matching how `flash-attn` is handled) is to host a pre-built wheel and reference it by URL.

### Step 1: Build the wheel locally

```bash
# From repo root, with the venv that has torch installed:
uv build ./causal-conv1d --no-build-isolation --python .venv/bin/python --wheel --out-dir ./dist/
```

This produces `dist/causal_conv1d-1.6.1-cp312-cp312-linux_x86_64.whl` (~160MB).

### Step 2: Host the wheel

Upload to a URL accessible by Ray workers. Options:
- GitHub release (like flash-attn uses `mjun0812/flash-attention-prebuild-wheels`)
- S3/GCS bucket
- Any HTTP server accessible from the cluster

Example: if uploaded to `https://github.com/YOUR_ORG/prebuild-wheels/releases/download/v1.0/causal_conv1d-1.6.1-cp312-cp312-linux_x86_64.whl`

### Step 3: Update `pyproject.toml`

```toml
# In [project.optional-dependencies]
fsdp = [
    # ... existing deps ...
    "causal-conv1d>=1.4.0; sys_platform == 'linux'",
    "flash-linear-attention>=0.4.2; sys_platform == 'linux'",
]

# In override-dependencies: REMOVE the sys_platform == 'never' line for causal-conv1d
# Replace with a real constraint:
"causal-conv1d>=1.4.0; sys_platform == 'linux'",

# In [tool.uv.sources]
causal-conv1d = { url = "https://YOUR_URL/causal_conv1d-1.6.1-cp312-cp312-linux_x86_64.whl", marker = "sys_platform == 'linux'" }
flash-linear-attention = { path = "./flash-linear-attention", editable = true }
```

### Step 4: Verify

```python
python -c "
from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available
print('is_fast_path_available:', is_fast_path_available)
"
# Should print: is_fast_path_available: True
```

The model_wrapper.py logging (added in this branch) will also confirm:
```
INFO  | Qwen3.5 fast path is ENABLED (causal-conv1d + flash-linear-attention)
```

## Alternative: Docker Image

For production, the cleanest approach is to include `causal-conv1d` in the Docker image:

```dockerfile
# In docker/Dockerfile, after CUDA toolkit is installed:
COPY causal-conv1d /tmp/causal-conv1d
RUN cd /tmp/causal-conv1d && pip install --no-build-isolation . && rm -rf /tmp/causal-conv1d
```

This avoids wheel hosting entirely since all Ray workers share the same Docker image.

## Quick Local-Only Test (no Ray workers)

If you just want to verify the fast path works locally (single-node, no Ray worker venvs):

```bash
# Install directly into the venv
uv pip install -e ./causal-conv1d --python .venv/bin/python --no-build-isolation

# Verify
.venv/bin/python -c "from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available; print(is_fast_path_available)"
# True
```

## Notes

- `causal-conv1d` v1.6.1 does have a `CachedWheelsCommand` that auto-downloads prebuilt wheels from [Dao-AILab/causal-conv1d releases](https://github.com/Dao-AILab/causal-conv1d/releases), but as of March 2026, no wheels exist for torch 2.10.0 + CUDA 12.8.
- The `flash-linear-attention` fla package lists `causal-conv1d>=1.4.0` as an optional dependency under the `conv1d` extra. The `sys_platform == 'never'` override in SkyRL's pyproject.toml suppresses this.
- Prime-RL does not install either `fla` or `causal-conv1d` — they don't have Qwen3.5 linear attention support.
- The torch fallback `torch_chunk_gated_delta_rule` has a known issue with CPU tensor creation during gradient checkpointing recomputation (see commented-out Patch 2 in `model_wrapper.py`). The fla fast path avoids this entirely.
