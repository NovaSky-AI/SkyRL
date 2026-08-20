# CUDA 13 Upgrade — Review Response & Local GPU CI Report

Branch: `eric/cuda-13` → PR [NovaSky-AI/SkyRL#2040](https://github.com/NovaSky-AI/SkyRL/pull/2040)

**Bottom line:** both review comments are factually wrong and need no code change. Locally,
**292 GPU tests pass** across all six pytest-based GPU CI suites; the only 3 failures are an
artifact of this workspace being a CUDA 12.8 box, not a defect in the PR.

The branch has since also been bumped **Ray 2.56.0 → 2.57.0** (§6). Note the 292-test run
above predates that bump and **cannot be repeated on this workspace**, because its Ray cluster
is 2.56.0 and a 2.57.0 client refuses to connect.

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

So in the images built from those Dockerfiles, `/usr/local/cuda/lib64` resolves to the 13.0
tree and only `libcudart.so.13` is visible; the base layer's `cu12.8` remnants stay at
`/usr/local/cuda-12.8`, off the loader path. (The base sets
`LD_LIBRARY_PATH=/usr/local/cuda/lib64` — the symlink — confirmed from its registry config, so
repointing the symlink is sufficient.)

**But this is a live risk for the h100 job specifically, not merely local noise.** This
workspace's image is
`anyscale/image/skyrl-train-ray-2.56.0-slim-py312-cu128-megatron-2.10-te-efa-1.47:2` — the
**cu128 counterpart of the h100 CI image itself** (§4.1 wants the `cu130` variant of that same
name). That image is *not* built from `docker/Dockerfile.megatron`, so it never gets the
13.0.2 runfile step, and its cu128 build has no CUDA 13 toolkit at all. Hence: if the cu130
rebuild is produced by renaming a cu128-shaped image rather than genuinely moving to CUDA 13,
the h100 suite will fail in CI with the exact error seen here. The rebuild must base on a
cu130 tag or add the 13.0.2 runfile step. `CUDA13_HANDOFF.md` §2 carries the verification
commands.

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

- **Docker Hub (12 job specs):** `novaskyai/skyrl-train-ray-2.57.0-py3.12-cu13.0` and
  `-cu13.0-megatron` (names now carry Ray 2.57.0). The `novaskyai` org has 9 repos, including the `cu12.8` pair and no
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

---

## 6. Ray 2.56.0 → 2.57.0

Requested as a low-risk addition to this PR, and the dependency graph agrees: the lock diff is
**one package, 17 lines**.

```
$ uv lock
Resolved 464 packages in 35.65s
Updated ray v2.56.0 -> v2.57.0
$ git diff uv.lock | grep -E '^[-+](name|version) = ' | sort | uniq -c
      1 -version = "2.56.0"
      1 +version = "2.57.0"
```

Nothing else in the graph moved, and nothing constrains `ray` besides our own pin.

### What changed

- `pyproject.toml`: `ray[default]==2.57.0` and `ray==2.57.0`.
- All 12 `ci/anyscale_*.yaml`: `ray_version: "2.57.0"`. Anyscale requires this to match the
  Ray actually installed in the image.
- **Image names, which embed the Ray version:** `novaskyai/skyrl-train-ray-2.57.0-py3.12-cu13.0`,
  `...-cu13.0-megatron`, and the h100 job's
  `anyscale/image/skyrl-train-ray-2.57.0-slim-py312-cu130-megatron-2.10-te-efa-1.47:1`. These
  are the names to build (§4.1) — the 2.56.0 names are now obsolete and were never published
  either.
- `docker/Dockerfile`, `docker/Dockerfile.megatron`: base tag →
  `anyscale/ray:2.57.0-slim-py312-cu128`. `docker/Dockerfile.amd`: `ARG RAY_VERSION=2.57.0`.
- Docs and the modal example: image tags and the "we suggest Ray x.y.z" prose.

### Deliberately left alone

- **`docs/.../installation.mdx` "Ray < 2.56.0" section.** The `< 2.56.0` threshold there is a
  real behavioural boundary — below it you must install vllm in the base pip environment for
  the uv + Ray integration to work. That is a property of Ray, not of our pin, so bumping the
  number would make the docs assert something false about 2.56.0.
- **`skyrl-agent/pyproject.toml`** (`override-dependencies = ["ray==2.56.0", ...]`). It is a
  separate project with its own `uv.lock`, not a uv workspace member, and this PR does not
  otherwise touch it. Bumping it means re-locking an unrelated package. Flagged, not done.

### Base image: moved to the full cu130 tag

The Dockerfiles previously based on `anyscale/ray:2.56.0-slim-py312-cu128` and relied on the
CUDA 13.0.2 runfile repointing `/usr/local/cuda` to keep the base layer's CUDA 12 tree off the
loader path. Both now base on the real thing:

```
FROM anyscale/ray:2.57.0-py312-cu130
```

At 2.57.0 Anyscale's `slim` variants stop at cu129, so a genuinely-CUDA-13 base means giving
up `slim`:

| Tag | Size (compressed, amd64) |
|---|---|
| `2.57.0-slim-py312-cu128` | 3.19 GB |
| `2.57.0-py312-cu130` | **8.00 GB** |

That size is a deliberate trade for a base with **no CUDA 12 tree at all** — nothing for
Transformer Engine's libcudart scan to trip over, rather than a CUDA 12 tree merely hidden
behind a symlink. Given §3, that robustness is worth paying for.

#### Two things this switch required / raised

1. **`ENV PATH=/home/ray/.local/bin:${PATH}` — required, not cosmetic.** The slim images put
   `~/.local/bin` on `PATH`; the cu130 image does **not**:

   | | slim cu128 | full cu130 |
   |---|---|---|
   | `PATH` includes `~/.local/bin` | yes | **no** |
   | `User` | `ray` | `1000` |

   The uv installer writes its binary there, so without this `ENV` the `RUN uv pip install
   --system ...` step in `Dockerfile.megatron` fails with `uv: command not found`, and the CI
   entrypoints' `uv run` would fail at runtime too. Caught by reading the base images' registry
   configs, not by a build.

2. **The CUDA runfile install was dropped.** Both Dockerfiles previously did

   ```dockerfile
   RUN wget .../cuda_13.0.2_580.95.05_linux.run && sudo sh ... --silent --toolkit
   ```

   which existed only to get a CUDA 13 `nvcc` for deepspeed onto a cu128 base. The cu130 base
   already ships one — verified in the image itself:

   ```
   $ docker run --rm anyscale/ray:2.57.0-py312-cu130 nvcc --version
   Cuda compilation tools, release 13.0, V13.0.88
   ```

   (Note V13.0.88 is the 13.0.2-level compiler, despite the base's `CUDA_VERSION=13.0.0`, so
   nothing regresses versus the runfile.) Keeping it would have layered a runfile install over
   the base's apt-managed toolkit in the same `/usr/local/cuda-13.0` prefix for ~4 GB and no
   gain. Removing it roughly cancels the larger base: +4.8 GB of base, −4 GB of toolkit.

3. **Watch item: the cu130 base bundles its own cuDNN.** It sets
   `NV_CUDNN_PACKAGE=libcudnn9-cuda-13=9.12.0.46-1`, which the slim cu128 base did not.
   `Dockerfile.megatron` also pip-installs `nvidia-cudnn-cu13>=9.3` and prepends
   `/opt/cudnn/lib` to `LD_LIBRARY_PATH`, so there are now two cuDNN 9 copies with the pip one
   taking precedence. Both are cuDNN 9 for CUDA 13, so no cross-major conflict is expected,
   but given how much time §3 cost, verify on first build:
   `python -c "import transformer_engine.pytorch"` plus one fused-attention test.

### Validation status — weaker than for the CUDA 13 work

Verified: dependency resolution (above), and the CPU suites that CPU CI runs.

**Not** verified: any GPU test, because this workspace cannot run one against Ray 2.57.0. Its
cluster is 2.56.0, so a 2.57.0 client dies with `RuntimeError: Version mismatch`, and starting
an isolated second cluster segfaults on this Anyscale node (it does so on 2.56.0 too, so that
is an environment constraint rather than anything to do with 2.57.0). The first real GPU CI
run is therefore the first genuine test of this bump.
