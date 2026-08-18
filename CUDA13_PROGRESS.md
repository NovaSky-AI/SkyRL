# CUDA 13 Upgrade — Review Response & Local GPU CI Report

Branch: `eric/cuda-13` → PR [NovaSky-AI/SkyRL#2040](https://github.com/NovaSky-AI/SkyRL/pull/2040)

**Bottom line:** both review comments are factually wrong and need no code change. Locally,
**292 GPU tests pass** across all six pytest-based GPU CI suites; the only 3 failures are an
artifact of this workspace being a CUDA 12.8 box, not a defect in the PR. No source changes
were needed — the branch is unmodified apart from this report.

---

## 1. Review comments — both INVALID, no code changed

Both are from `gemini-code-assist[bot]`, both marked "high priority", and both assert that a
package or extra does not exist on PyPI. Both assertions are false. Replied on the PR with the
evidence below.

### 1.1 `jax[cuda13]` (`pyproject.toml:31`)

Claim: "JAX does not publish a `cuda13` extra on PyPI ... Using `cuda13` will cause
installation of the `gpu` extra to fail." Suggested `jax[cuda12]`.

**False.** `jax` 0.7.2 (our floor) and 0.8.0 both list it:

```
$ curl -s https://pypi.org/pypi/jax/0.7.2/json | jq -r '.info.provides_extra[]'
minimum-jaxlib, cpu, ci, tpu, cuda, cuda12, cuda13, cuda12-local, cuda13-local, rocm, k8s, xprof
```

It resolves to a real published plugin (`jax-cuda13-plugin[with-cuda]`), the committed
`uv.lock` already contains the resolved `jax-cuda13-plugin` / `jax-cuda13-pjrt` entries, and
`uv lock --check` passes (464 packages).

**Confirmed at runtime, not just in resolution:** the `tests/tx/gpu` CI suite installs the
`gpu` extra — i.e. `jax[cuda13]` — and **12/12 tests pass on GPU** (§2). The extra it claims
cannot install is the one that just ran.

### 1.2 `nvidia-cudnn-cu13` (`docker/Dockerfile.megatron:40`)

Claim: "NVIDIA does not publish a `nvidia-cudnn-cu13` package on PyPI ... will result in a
package resolution error and break the Docker build." Suggested `nvidia-cudnn-cu12>=9.3`.

**False.** The package exists and is actively released (9.24.0.43 today, back through
9.12.0.46). The flagged line resolves cleanly against a fresh target:

```
$ uv pip install --dry-run --python-platform linux --python-version 3.12 \
    --target /tmp/dryrun_cudnn "torch==2.11.0" "nvidia-cudnn-cu13>=9.3" \
    --extra-index-url https://download.pytorch.org/whl/cu130
Resolved 29 packages
 + nvidia-cudnn-cu13==9.19.0.56
 + torch==2.11.0+cu130
```

### Why both suggestions would actively regress the PR

Each proposed fix swaps a cu13 package for its cu12 twin, which drags the `nvidia-*-cu12`
runtime stack in beside torch's cu130 — two CUDA runtimes in one environment. That is not
hypothetical: §3 below is a failure caused by exactly that condition. The suggestions would
create more instances of the very bug class, and the PR already removes one deliberately
(`nixl-cu12; sys_platform == 'never'`).

---

## 2. Local GPU CI results — 292 passed, 3 failed

Run on this workspace (4×H100 80GB, driver 580.126.09), executing each `ci/gpu_ci_run_*.sh`
pytest invocation verbatim, serially.

| # | Suite (source script) | Result | Time |
|---|---|---|---|
| 1 | FSDP GPU CI (`gpu_ci_run_skyrl_train.sh`) | **112 passed**, 9 skipped, 36 deselected | 2:27:07 |
| 2 | `tests/tx/gpu` (`gpu_ci_run.sh`, `--extra gpu`) | **12 passed** | 0:00:26 |
| 3 | Megatron GPU CI (`gpu_ci_run_skyrl_train_megatron.sh`) | **130 passed**, 2 skipped, 148 deselected | 3:14:07 |
| 4 | Megatron models (`..._megatron_models.sh`) | **3 passed**, 4 skipped, 273 deselected | 0:16:58 |
| 5a | H100 FSDP (`gpu_ci_run_h100.sh`, part 1) | **2 passed**, 6 deselected | 0:14:38 |
| 5b | H100 Megatron (`gpu_ci_run_h100.sh`, part 2) | **3 failed**, 2 passed, 4 deselected | 0:30:53 |
| 6 | Tinker↔Megatron backend (`..._tinker_skyrl_train_backend.sh`) | **31 passed** | 0:13:08 |

Both dependency stacks resolve, build and import cleanly on CUDA 13 (FSDP: 121 tests
collected; Megatron: 368 packages including `transformer-engine-cu13` 2.16 and
`megatron-bridge` built from git). Since the CI runner scripts build their environment from
`pyproject.toml` + `uv.lock` at runtime, this exercises the actual dependency substance of the
PR rather than a prebuilt image.

---

## 3. The 3 failures are a property of this workspace, not of the PR

All three (`test_logprobs_matching_roundtrip[glm-4.7-flash_h100_tp4_ep4]`,
`test_router_replay::test_logprobs[tp2_pp2_ep2]`,
`test_router_replay::test_forward_backward[tp2_pp2_ep2]`) die in Transformer Engine's cuDNN
fused-attention path with:

```
RuntimeError: Multiple libcudart libraries found: libcudart.so.12 and libcudart.so.13
```

**Where the cu12 runtime comes from — not from any Python package.** A scan of the megatron
environment's `site-packages` finds exactly one CUDA runtime, `nvidia/cu13/lib/libcudart.so.13`.
The cu12 copy comes from the operating system:

```
$ ls -ld /usr/local/cuda            →  /usr/local/cuda -> /usr/local/cuda-12.8/
$ echo $LD_LIBRARY_PATH             →  /opt/cudnn/lib:/usr/local/cuda/lib64
$ readlink -f /usr/local/cuda/lib64 →  /usr/local/cuda-12.8/targets/x86_64-linux/lib
```

This Anyscale workspace runs a CUDA 12.8 base image with no CUDA 13 toolkit, so
`libcudart.so.12` sits on the loader path. TE's fused attention refuses to run when it sees
two libcudart versions, and the pip stack correctly supplies the cu13 one — hence the clash.

**The CI image does not have this problem, by construction.** `docker/Dockerfile.megatron`
installs the CUDA 13.0.2 toolkit, which repoints `/usr/local/cuda` at 13.0 (the Dockerfile's
own comment says so explicitly):

```dockerfile
RUN wget https://developer.download.nvidia.com/compute/cuda/13.0.2/local_installers/cuda_13.0.2_580.95.05_linux.run \
    && sudo sh cuda_13.0.2_580.95.05_linux.run --silent --toolkit && rm -rf cuda_13.0.2_580.95.05_linux.run
```

So in CI, `/usr/local/cuda/lib64` resolves to the 13.0 tree and only `libcudart.so.13` is
visible. The cu12.8 remnants of the base layer stay at `/usr/local/cuda-12.8` and off the
path. The h100 job additionally runs on a cu130 image rather than this one.

**Not reproducible-away locally.** Ray workers inherit the raylet's environment, not the
caller's, so overriding `LD_LIBRARY_PATH` for pytest does not reach the workers (verified:
a probe actor still reports `/opt/cudnn/lib:/usr/local/cuda/lib64` and can load
`libcudart.so.12`). Forcing a private Ray cluster with a clean path segfaults against the
managed workspace cluster. Properly validating these 3 tests needs an image whose
`/usr/local/cuda` is 13.0 — i.e. the CI image.

### A wrong turn, recorded

I first blamed `nvidia-cutlass-dsl==4.6.0`, which depends on `nvidia-cutlass-dsl-libs-cu12`
unconditionally while adding `-cu13` only under its `cu13` extra, so both land in the
environment. I added an override dropping the cu12 variant, re-locked, and re-ran: **the
failure was identical**. Inspecting the wheel shows why — `nvidia-cutlass-dsl-libs-cu12`
contains 19 entries and no cudart at all (just `_cutlass_ir.cu12*.so` and
`libcute_dsl_runtime.so`), so it could never have been the source. That change is
**reverted**; `pyproject.toml` and `uv.lock` are untouched on this branch. Dropping the
redundant cu12 CUTLASS libs may still be a reasonable cleanup on its own merits, but it fixes
nothing here and I have not validated it, so it is not proposed.

---

## 4. Still blocked (needs you)

### 4.1 The cu13.0 images do not exist

Every GPU CI job spec was retargeted to images that were never published, so the workflows
cannot start — they fail at image pull, before the entrypoint runs.

- **Docker Hub (12 job specs):** `novaskyai/skyrl-train-ray-2.56.0-py3.12-cu13.0` and
  `-cu13.0-megatron`. The `novaskyai` org has 9 repos, including the `cu12.8` pair and no
  `cu13.0` anything. There is no build/push automation in the repo — these are published
  out-of-band. You said you'd push these.
- **Anyscale registry (h100 job):** the branch renamed
  `anyscale/image/...-cu128-megatron-2.10-te-efa-1.47:1` → `...-cu130-...`, which was never
  built. I applied the `run_h100_gpu_ci` label to PR #2040; the real workflow failed in 7
  seconds ([run 32056729715](https://github.com/NovaSky-AI/SkyRL/actions/runs/32056729715))
  with a 404 on that image name. Not a credentials artifact — CI's own
  `secrets.ANYSCALE_CLI_TOKEN` produced the identical error a manual submit did, and the old
  cu128 image still resolves. Anyscale does not expose the containerfile behind it, and its
  name encodes an EFA 1.47 layer absent from this repo, so it needs a rebuild by whoever built
  the cu128 one.

Note for whoever re-triggers: `pull_request_target` fires on the `labeled` event, so a label
already present must be removed and re-added.

### 4.2 The 6 e2e suites need wandb

`gsm8k_colocate`, `gsm8k_fully_async`, `gsm8k_colocate_megatron`, `sft_tulu3_megatron`,
`gsm8k_tinker`, `gsm8k_tinker_fully_async` all verify through wandb (`get_summary.py` /
`check_sft_trend.py` read run history via `wandb.Api()`), and no `WANDB_API_KEY` is configured
here. The first three honor a `LOGGER` env var, so their *training* could be run with
`LOGGER=console` to exercise the stack on cu13, but their accuracy/token/logprob threshold
assertions cannot be checked without wandb. `sft_tulu3_megatron` hardcodes `logger=wandb`;
the two tinker suites drive wandb natively through `tinker-cookbook`. **None were run** —
say the word if you want the console-only training variants.

### 4.3 Pre-existing, unrelated: `SkyRL-GPU` cannot run on fork PRs

[Run 32055764447](https://github.com/NovaSky-AI/SkyRL/actions/runs/32055764447) fails with
"Your user credentials are invalid" because `.github/workflows/gpu_skyrl.yaml` triggers on
`pull_request` rather than the `pull_request_target` the other GPU workflows use, and GitHub
withholds secrets from fork `pull_request` runs. PR #2040 is from the `erictang000` fork. The
same workflow passes on push to `main` (run 32053268078). This affects every fork PR and has
nothing to do with CUDA 13; changing that trigger is a security decision outside this PR's
scope, so it is flagged, not fixed.

---

## 5. Verification status

| Item | Status |
|---|---|
| Review comment 1 (`jax[cuda13]`) | Invalid — refuted, and the extra runs green on GPU |
| Review comment 2 (`nvidia-cudnn-cu13`) | Invalid — refuted by fresh-target resolution |
| Both replied to on PR | Done |
| FSDP / Megatron envs resolve + import on cu13 | Verified |
| 6 pytest GPU CI suites run locally | Verified — 292 passed |
| 3 h100 megatron tests | **Unverified** — needs a `/usr/local/cuda`→13.0 image |
| 6 e2e training suites | **Not run** — need `WANDB_API_KEY` |
| Full GPU CI green | Blocked on the missing cu13.0 images (§4.1) |
