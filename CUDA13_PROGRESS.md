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

### 2b. Second missing image: the H100 CI Anyscale image (confirmed by a real CI run)

This branch also renamed the H100 job's Anyscale-registry image:

```diff
-image_uri: anyscale/image/skyrl-train-ray-2.56.0-slim-py312-cu128-megatron-2.10-te-efa-1.47:1
+image_uri: anyscale/image/skyrl-train-ray-2.56.0-slim-py312-cu130-megatron-2.10-te-efa-1.47:1
```

The cu130 image was never built. I applied the `run_h100_gpu_ci` label to PR #2040 to run the
real workflow; it failed in 7 seconds
([run 32056729715](https://github.com/NovaSky-AI/SkyRL/actions/runs/32056729715)):

```
Error: API Exception (404) from POST /api/v2/builds/get_or_create_build_from_image_uri
{"error":{"detail":"Cluster environment with name skyrl-train-ray-2.56.0-slim-py312-cu130-megatron-2.10-te-efa-1.47 not found"}}
```

This is *not* a credential-scope artifact: GitHub Actions used the real
`secrets.ANYSCALE_CLI_TOKEN` and got the identical 404 that a manual submit from this
workspace got. Meanwhile the old cu128 image resolves fine
(`anyscale image get ...cu128-megatron-2.10-te-efa-1.47:1` → `status: SUCCEEDED`), confirming
the rename is aspirational rather than the token being wrong.

Anyscale does not expose the containerfile behind the existing cu128 image
(`anyscale image get` returns only uri/status/ray_version), and the name encodes
build inputs not present in this repo (an EFA 1.47 install layer on top of a megatron stack),
so the cu130 equivalent cannot be faithfully reconstructed from the repo alone — whoever
built the cu128 image needs to rebuild it for cu130.

### 2c. Pre-existing (not caused by this branch): `SkyRL-GPU` cannot run on fork PRs

`SkyRL-GPU` ([run 32055764447](https://github.com/NovaSky-AI/SkyRL/actions/runs/32055764447))
fails on this PR with:

```
Error: Your user credentials are invalid. Please go to https://console.anyscale.com/v2/api-keys ...
```

Cause: `.github/workflows/gpu_skyrl.yaml` triggers on `pull_request` (not
`pull_request_target`, which the other GPU workflows use). GitHub does not pass repository
secrets to `pull_request` runs from a fork, so `ANYSCALE_CLI_TOKEN` arrives empty. PR #2040 is
from the `erictang000/SkyRL` fork, hence the failure. The same workflow passes on push to
`main` (run 32053268078 on 22b668f). This will fail identically for *any* fork PR and is
unrelated to CUDA 13 — flagging it rather than fixing it here, since changing that trigger is
a security-relevant decision outside this PR's scope.

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
| Verify review comment 1 (`jax[cuda13]`) | Done — invalid, no change |
| Verify review comment 2 (`nvidia-cudnn-cu13`) | Done — invalid, no change |
| Reply to both review comments on PR | Done |
| FSDP env resolves/builds/imports on cu13 | Done — 121 tests collected |
| Megatron env resolves/builds/imports on cu13 | Done — 368 pkgs incl. TE cu13 + megatron-bridge |
| FSDP GPU CI suite on 4×H100 locally | Running |
| Megatron GPU CI suite locally | Queued behind FSDP (GPU contention) |
| Missing `novaskyai` cu13.0 Docker Hub images | **Blocked on you** — you're pushing them |
| Missing cu130 H100 Anyscale image | **Blocked on you** — needs rebuild, containerfile not in repo |
| Full GPU CI green | Blocked on both images above |

### What I can and cannot do from here

- **Cannot** submit the CI job specs directly: attempted, and it fails for the same reason CI
  does (missing images). Separately, this workspace's token cannot see the `l4_ci` /
  `k8s-single-node-4h100` compute configs, though it *is* in the same Anyscale org (it can
  read the cu128 image).
- **Can** trigger the real workflows via the `run_*_gpu_ci` labels on PR #2040 (my GitHub
  token has admin). This is the working route and is what I used for the H100 run above.
- **Can** run the CI suites' pytest commands locally on this workspace's 4×H100, which
  validates the actual dependency substance of the PR (the runner scripts build their env from
  `pyproject.toml`/`uv.lock` at runtime anyway).

### Next steps once you've pushed the images

1. Ping me when `novaskyai/skyrl-train-ray-2.56.0-py3.12-cu13.0` and `-cu13.0-megatron` are
   live, and when the cu130 H100 Anyscale image is registered.
2. I re-apply each `run_*_gpu_ci` label (remove + re-add, since `pull_request_target` fires on
   the `labeled` event) and drive all suites to green.
