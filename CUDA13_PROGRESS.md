# CUDA 13 Upgrade — Review Response & CI Validation

Branch: `eric/cuda-13` → PR [NovaSky-AI/SkyRL#2040](https://github.com/NovaSky-AI/SkyRL/pull/2040)

## 1. Review comments — both INVALID (no code change needed)

Both comments came from `gemini-code-assist[bot]`, flagged "high priority". Both assert that
a package/extra does not exist on PyPI. Both assertions are false, verified directly against
the PyPI JSON API and by real dependency resolution.

### 1.1 `jax[cuda13]` — comment claims the `cuda13` extra does not exist

**Claim:** "JAX does not publish a `cuda13` extra on PyPI ... Using `cuda13` will cause
installation of the `gpu` extra to fail." Suggested `jax[cuda12]`.

**Verdict: false.** `jax` has published a `cuda13` extra since well before our floor of
`>=0.7.2`.

```
$ curl -s https://pypi.org/pypi/jax/0.7.2/json | jq -r '.info.provides_extra[]'
minimum-jaxlib, cpu, ci, tpu, cuda, cuda12, cuda13, cuda12-local, cuda13-local, rocm, k8s, xprof
```

The extra resolves to a real, published plugin:

```
jaxlib<=0.7.2,>=0.7.2;              extra == "cuda13"
jax-cuda13-plugin[with-cuda]<=0.7.2,>=0.7.2; extra == "cuda13"
```

`jax 0.8.0` lists the same extra. Our committed `uv.lock` already contains the resolved
`jax-cuda13-plugin` (0.10.2 / 0.11.0) and `jax-cuda13-pjrt` entries, i.e. resolution of the
`gpu` extra demonstrably succeeds today — `uv lock --check` passes with 464 packages resolved.

**Why the suggested change would be worse:** `jax[cuda12]` pulls the `nvidia-*-cu12` CUDA
runtime stack. This project pins `torch==2.11.0+cu130` from the `pytorch-cu130` index, so
taking the suggestion would install a *second*, redundant CUDA runtime (cu12 alongside cu13)
into every GPU environment — the exact duplication the rest of this PR removes (see the
`nixl-cu12; sys_platform == 'never'` exclusion in `pyproject.toml`).

### 1.2 `nvidia-cudnn-cu13` — comment claims the package does not exist

**Claim:** "NVIDIA does not publish a `nvidia-cudnn-cu13` package on PyPI ... will result in
a package resolution error and break the Docker build." Suggested `nvidia-cudnn-cu12>=9.3`.

**Verdict: false.** The package exists and is actively released (currently 9.24.0.43, with
releases back through 9.12.0.46).

The exact `docker/Dockerfile.megatron` install line resolves cleanly against a fresh target:

```
$ uv pip install --dry-run --python-platform linux --python-version 3.12 \
    --target /tmp/dryrun_cudnn "torch==2.11.0" "nvidia-cudnn-cu13>=9.3" \
    --extra-index-url https://download.pytorch.org/whl/cu130
Resolved 29 packages
 + nvidia-cudnn-cu13==9.19.0.56
 + torch==2.11.0+cu130
```

`nvidia-cudnn-cu13` is also already present in the committed `uv.lock`.

**Why the suggested change would be worse:** same reason as above — `nvidia-cudnn-cu12`
carries `nvidia-*-cu12` runtime deps, so it would drag a mismatched cuDNN/CUDA-12 runtime
into a cu13 image alongside the cu13 one.

**Conclusion:** no code changes made for either comment. Replied on the PR with the evidence
above.

## 2. Real blocker found while setting up GPU CI

The published Docker Hub images the GPU CI job specs point at **do not exist**.

Every `ci/anyscale_gpu_*.yaml` on this branch was retargeted from `cu12.8` to `cu13.0`:

```yaml
image_uri: novaskyai/skyrl-train-ray-2.56.0-py3.12-cu13.0          # 8 job specs
image_uri: novaskyai/skyrl-train-ray-2.56.0-py3.12-cu13.0-megatron # 4 job specs
```

Docker Hub `novaskyai` has 9 repositories. It contains
`skyrl-train-ray-2.56.0-py3.12-cu12.8` and `...-cu12.8-megatron`, but **no `cu13.0`
repositories at all** (`GET /v2/repositories/novaskyai/skyrl-train-ray-2.56.0-py3.12-cu13.0/`
→ `object not found`). There is no image build/push automation anywhere in the repo
(`.github/`, `ci/`, `docker/` contain no `docker build`/`docker push`/`buildx`), so these are
published out-of-band by a maintainer.

Consequence: those 12 GPU CI jobs cannot start — the cluster fails the image pull before the
entrypoint ever runs. Note this failure mode is *also* what
`ci/submit_anyscale_job.sh` treats as a GPU-capacity failure (job fails with zero job runs),
so it will burn all 5 submission attempts, 300s apart, before reporting failure.

The one exception is `ci/anyscale_gpu_ci_h100.yaml`, which already points at an
Anyscale-registry cu130 image that does exist:
`anyscale/image/skyrl-train-ray-2.56.0-slim-py312-cu130-megatron-2.10-te-efa-1.47:1`.

## 3. Environment available for validation

- This Anyscale workspace: 4×H100 80GB, driver 580.126.09, CUDA 13.0.
- `anyscale` CLI present, `ANYSCALE_CLI_TOKEN` set → can submit the real CI job specs.
- GitHub token is `erictang000` with **admin** on `NovaSky-AI/SkyRL` → can apply the
  `run_*_gpu_ci` gating labels to trigger the real workflows.
- **No** docker/podman/buildah binary in this workspace → cannot build or push the
  `novaskyai` Docker Hub images from here.

Relevant: the CI runner scripts build their own environment at runtime
(`uv run --isolated --extra dev --extra fsdp ...`) from `pyproject.toml` + `uv.lock`. The
docker image supplies the base layer (Ray, CUDA system libs, NCCL headers), so the
dependency substance of this PR can be validated locally on this workspace's H100s
independently of the missing images.

## 4. Status

| Item | Status |
|------|--------|
| Verify review comment 1 (`jax[cuda13]`) | Done — invalid |
| Verify review comment 2 (`nvidia-cudnn-cu13`) | Done — invalid |
| Reply to review comments on PR | In progress |
| Local validation of GPU CI suites on 4×H100 | In progress |
| Missing `novaskyai` cu13.0 images | **Blocked** — needs maintainer push or Anyscale image build |
| Full GPU CI green | Pending the above |
