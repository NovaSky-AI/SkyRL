# CUDA 13 PR — Handoff: build & publish the cu13.0 images

**For:** an agent/engineer on a machine **with a working Docker daemon** and push rights to
the `novaskyai` Docker Hub org. The prior session ran on a Kubernetes pod with no docker
binary, so the image builds could not be done there.

**PR:** [NovaSky-AI/SkyRL#2040](https://github.com/NovaSky-AI/SkyRL/pull/2040) —
`erictang000:eric/cuda-13` → `NovaSky-AI:main`. State: open, `mergeable: true`,
`mergeable_state: unstable`. `main` is **not** branch-protected, so there are no required
status checks and no red check can physically block the merge button.

**The whole remaining task:** three container images referenced by CI were renamed on this
branch (to cu13.0, and to Ray 2.57.0) but never built. Build and publish them, then let GPU
CI run.

> **The image names embed the Ray version.** This branch also bumps Ray 2.56.0 → 2.57.0, so
> the images to build are `...ray-2.57.0-...`, **not** `...ray-2.56.0-...`. The 12 job specs
> already declare `ray_version: "2.57.0"`, and Anyscale requires that to match the Ray
> actually installed in the image. The Dockerfiles install Ray via their base tag
> (`anyscale/ray:2.57.0-slim-py312-cu128`), so building from this branch gets this right
> automatically — just don't hand-edit the tags back.

---

## 0. Read this first — do NOT "fix" the two bot review comments

`gemini-code-assist[bot]` left two high-priority comments claiming that `jax[cuda13]` and
`nvidia-cudnn-cu13` do not exist on PyPI. **Both are hallucinations.** They have already been
refuted in replies on the PR. Verify in seconds if you want:

```bash
curl -s https://pypi.org/pypi/jax/0.7.2/json | jq -r '.info.provides_extra[]'   # includes cuda13
curl -s https://pypi.org/pypi/nvidia-cudnn-cu13/json | jq -r '.info.version'    # 9.24.0.43
```

Applying either suggested change would **regress** the PR: each swaps a cu13 package for its
cu12 twin, pulling the `nvidia-*-cu12` runtime stack in beside torch's cu130 and putting two
CUDA runtimes in one env. Leave `pyproject.toml` and `docker/Dockerfile.megatron` alone.

---

## 1. Build and push the two Docker Hub images

Both Dockerfiles are **fully self-contained** — neither has a `COPY` or `ADD`, so the build
context is irrelevant and no branch code is baked in. They only provision the base
environment (CUDA 13.0.2 toolkit, uv, NCCL headers, cuDNN, torch). Project dependencies are
installed at *runtime* by the CI entrypoints via `uv run --isolated`, from the
`pyproject.toml`/`uv.lock` that the Anyscale job uploads as its `working_dir`.

Both images are x86_64 — pass `--platform linux/amd64` if you are on an arm64 machine.

```bash
git clone https://github.com/erictang000/SkyRL.git && cd SkyRL
git checkout eric/cuda-13

docker login   # needs push rights to the novaskyai org

# 1) base image -- 6 job specs depend on it
docker build --platform linux/amd64 \
  -f docker/Dockerfile \
  -t novaskyai/skyrl-train-ray-2.57.0-py3.12-cu13.0 .
docker push novaskyai/skyrl-train-ray-2.57.0-py3.12-cu13.0

# 2) megatron image -- 5 job specs depend on it
docker build --platform linux/amd64 \
  -f docker/Dockerfile.megatron \
  -t novaskyai/skyrl-train-ray-2.57.0-py3.12-cu13.0-megatron .
docker push novaskyai/skyrl-train-ray-2.57.0-py3.12-cu13.0-megatron
```

Expect these to be slow: each downloads and silently installs the 13.0.2 CUDA toolkit runfile
(~4 GB), and the megatron one additionally installs torch + cuDNN.

**Verify both are actually public before touching CI** (this is the step whose absence caused
the last CI failure):

```bash
for r in skyrl-train-ray-2.57.0-py3.12-cu13.0 skyrl-train-ray-2.57.0-py3.12-cu13.0-megatron; do
  curl -s "https://hub.docker.com/v2/repositories/novaskyai/$r/" | jq -r '.name // .message'
done
# Must print the two repo names. "object not found" means the push did not land.
```

For reference, the existing cu12.8 pair (`...-py3.12-cu12.8` and `...-cu12.8-megatron`) are
the only ones present today; there is no cu13.0 repo of any kind, and there is **no
build/push automation anywhere in the repo** (`.github/`, `ci/`, `docker/` contain no
`docker build`/`push`/`buildx`), so this has always been a manual, out-of-band step.

---

## 2. The third image is an Anyscale-registry image and needs a decision

`ci/anyscale_gpu_ci_h100.yaml` was renamed to:

```
anyscale/image/skyrl-train-ray-2.57.0-slim-py312-cu130-megatron-2.10-te-efa-1.47:1
```

That image does not exist. Applying `run_h100_gpu_ci` to the PR failed in 7 seconds
([run 32056729715](https://github.com/NovaSky-AI/SkyRL/actions/runs/32056729715)):

```
API Exception (404) from POST /api/v2/builds/get_or_create_build_from_image_uri
{"error":{"detail":"Cluster environment with name skyrl-train-ray-2.56.0-slim-py312-cu130-megatron-2.10-te-efa-1.47 not found"}}
```

This is not a credentials problem: GitHub Actions used the real `secrets.ANYSCALE_CLI_TOKEN`
and got the same 404 as a manual `anyscale job submit`, while the old cu128 image still
resolves (`anyscale image get --name ...cu128-megatron-2.10-te-efa-1.47:1` → `SUCCEEDED`).

It is a Docker Hub push away from being unblocked **only if** you decide to point the h100
job at the megatron image from §1. Note the constraints before choosing:

- Anyscale does not expose the containerfile behind the existing cu128 image
  (`anyscale image get` returns only uri/status/ray_version), and its name encodes build
  inputs absent from this repo — `efa-1.47`, an EFA (Elastic Fabric Adapter) layer, and a
  `slim` base. So it cannot be faithfully reconstructed from the repo alone.
- **Option A (needs the original author):** whoever built the cu128 image rebuilds it for
  cu130 and registers it under the same name with `:1`. Preserves EFA and the existing spec.
- **Option B (repo-only, needs judgement):** repoint `ci/anyscale_gpu_ci_h100.yaml` at
  `novaskyai/skyrl-train-ray-2.57.0-py3.12-cu13.0-megatron`. Its compute config is
  `k8s-single-node-4h100:1` — a **single node**, so EFA (a multi-node interconnect) is
  plausibly unnecessary. This is a real change to CI topology, though: confirm with the
  maintainers rather than assuming, and be aware it changes what that job actually tests.

Ask before picking. Do not silently switch the h100 job to a different image.

### Whichever way you go: that image MUST put CUDA 13 on the loader path

This is the one real trap. The h100 image is **not** built from `docker/Dockerfile.megatron`,
so it does not inherit that file's CUDA 13.0.2 toolkit step. Its existing cu128 build has no
CUDA 13 toolkit at all — verified by inspecting a workspace running exactly that image
(`anyscale/image/skyrl-train-ray-2.56.0-slim-py312-cu128-megatron-2.10-te-efa-1.47:2`): only
`/usr/local/cuda-12.8` is present and `/usr/local/cuda` points at it.

Run the h100 suite on such an image with this branch's cu13 pip stack and 3 tests fail:

```
RuntimeError: Multiple libcudart libraries found: libcudart.so.12 and libcudart.so.13
```

because the OS supplies `libcudart.so.12` via `LD_LIBRARY_PATH=.../usr/local/cuda/lib64` while
pip supplies `libcudart.so.13`, and TE's cuDNN fused attention refuses to run with both
visible. That is precisely what happened during local validation (see §5), and it will happen
in CI too if the cu130 rebuild is a cu128 image with a new name.

So the rebuild must genuinely be CUDA 13 — base it on a cu130 tag, as
`docker/Dockerfile[.megatron]` now do (`anyscale/ray:2.57.0-py312-cu130`), or otherwise install
a CUDA 13 toolkit. Verify before wiring it into CI:

```bash
# inside the built image
readlink -f /usr/local/cuda        # want /usr/local/cuda-13.0 (NOT cuda-12.8)
echo "$LD_LIBRARY_PATH"
ls /usr/local/cuda/lib64/libcudart.so.*   # want only libcudart.so.13
```

The two Docker Hub images in §1 are fine on this point by construction: they now base on
`anyscale/ray:2.57.0-py312-cu130`, which has no CUDA 12 tree at all. (They previously used a
cu128 slim base and depended on the runfile repointing `/usr/local/cuda`; the switch to a real
cu130 base removes that dependency.) Still worth confirming with the same three commands on
first build, since no one has built them yet.

Note one consequence of that base for the build: it does **not** put `~/.local/bin` on `PATH`
the way the slim images do, so the Dockerfiles now set `ENV PATH=/home/ray/.local/bin:${PATH}`
after installing uv. Without it the build dies at `uv pip install` with `uv: command not
found`. Do not "tidy" that line away.

---

## 3. Trigger GPU CI once the images are live

Five workflows are label-gated on the PR. Apply the label; **`pull_request_target` fires on
the `labeled` event, so a label already present must be removed and re-added** or nothing
happens. `run_h100_gpu_ci` is currently already applied (from the failed run above).

| Label | Workflow | Job spec | Image needed |
|---|---|---|---|
| `run_train_gpu_ci` | SkyRL-Train-GPU | `ci/anyscale_gpu_ci_skyrl_train.yaml` | cu13.0 |
| `run_train_megatron_gpu_ci` | SkyRL-Train-GPU-Megatron | `..._skyrl_train_megatron.yaml` | cu13.0-megatron |
| `run_train_megatron_gpu_ci_models` | Megatron-Model-GPU-CI | `..._megatron_models.yaml` | cu13.0-megatron |
| `run_tinker_skyrl_train_backend_gpu_ci` | Tinker-SkyRL-Train-Backend-GPU | `ci/anyscale_tinker_skyrl_train_backend_gpu.yaml` | cu13.0-megatron |
| `run_h100_gpu_ci` | H100-GPU-CI | `ci/anyscale_gpu_ci_h100.yaml` | the §2 image |

```bash
export GH_TOKEN=...   # needs push/admin on NovaSky-AI/SkyRL
R=NovaSky-AI/SkyRL; PR=2040; L=run_train_gpu_ci
curl -sX DELETE -H "Authorization: Bearer $GH_TOKEN" "https://api.github.com/repos/$R/issues/$PR/labels/$L"
curl -sX POST   -H "Authorization: Bearer $GH_TOKEN" "https://api.github.com/repos/$R/issues/$PR/labels" -d "{\"labels\":[\"$L\"]}"
```

The **6 e2e workflows are nightly `schedule` + `workflow_dispatch` only** — they never run on
a PR, so they cannot block the merge, but they *will* break the next nightly if the images
are missing. Trigger them manually with `workflow_dispatch` if you want pre-merge signal.

---

## 4. Gotchas that will waste your time

- **Verify the image exists before labeling.** `ci/submit_anyscale_job.sh` classifies "job
  failed with zero job runs" as a GPU-capacity failure and resubmits — up to 5 attempts,
  300 s apart. A missing image looks exactly like that, so it burns ~25 minutes before
  reporting failure.
- **`SkyRL-GPU` is red on this PR and that is pre-existing, not yours.**
  `.github/workflows/gpu_skyrl.yaml` triggers on `pull_request` (not `pull_request_target`
  like the others), and GitHub withholds secrets from fork `pull_request` runs, so
  `ANYSCALE_CLI_TOKEN` arrives empty → "Your user credentials are invalid". It fails for any
  fork PR and passes on push to `main`. `main` is unprotected so it blocks nothing. Fixing
  the trigger is a security decision outside this PR's scope. Alternatively, moving the
  branch onto `NovaSky-AI/SkyRL` instead of the fork would give the run its secrets.
- **The 6 e2e suites need `WANDB_API_KEY`.** They verify through `wandb.Api()` in
  `get_summary.py` / `check_sft_trend.py`. The first three honor a `LOGGER` env var, so
  training can run with `LOGGER=console`, but then the accuracy/token/logprob threshold
  assertions do not run. `sft_tulu3_megatron.sh` hardcodes `logger=wandb`.

---

## 5. Already verified locally — do not redo

Run on a 4×H100 workspace, executing each `ci/gpu_ci_run_*.sh` pytest invocation verbatim.
**292 tests passed.** See `CUDA13_PROGRESS.md` on this branch for the full evidence.

> **Caveat: that run was on Ray 2.56.0, before the 2.57.0 bump, and could not be repeated.**
> This workspace's Ray cluster is 2.56.0, so a 2.57.0 client refuses to connect
> (`RuntimeError: Version mismatch`), and starting a second, isolated cluster segfaults on
> this Anyscale node. So the Ray bump itself is validated only by dependency resolution (a
> clean one-package lock diff, `ray v2.56.0 -> v2.57.0`) plus the CPU suites — **not** by any
> GPU test. Treat the first CI run as the real test of the Ray upgrade, and do not assume the
> 292 figure below still holds for it.

| Suite | Result |
|---|---|
| FSDP GPU CI | 112 passed, 9 skipped |
| `tests/tx/gpu` (`--extra gpu`, i.e. `jax[cuda13]`) | 12 passed |
| Megatron GPU CI | 130 passed, 2 skipped |
| Megatron models | 3 passed, 4 skipped |
| H100 FSDP | 2 passed |
| H100 Megatron | **3 failed**, 2 passed |
| Tinker↔Megatron backend | 31 passed |

The 3 h100-megatron failures are an **artifact of that workspace**, not a PR defect. They die
in TE's cuDNN fused attention with `Multiple libcudart libraries found: libcudart.so.12 and
libcudart.so.13`. Cause: that box is a CUDA 12.8 image (`/usr/local/cuda ->
/usr/local/cuda-12.8/`, `LD_LIBRARY_PATH=/opt/cudnn/lib:/usr/local/cuda/lib64`), so a *system*
`libcudart.so.12` sits on the loader path beside the pip cu13 runtime. `site-packages` itself
contains only `libcudart.so.13`. Both Dockerfiles install the CUDA 13.0.2 toolkit, which
repoints `/usr/local/cuda` at 13.0, so a CI container does not have the collision.

**These 3 tests are therefore still genuinely unverified** — the CI run in §3 is their first
real test. If they fail *in CI*, that is new information and worth investigating; do not
dismiss it as the workspace artifact described here.

(A dead end already explored, so you don't repeat it: `nvidia-cutlass-dsl` does pull
`nvidia-cutlass-dsl-libs-cu12` unconditionally while adding `-cu13` only under its `cu13`
extra, so both land in the env — but that wheel contains no cudart at all, and overriding it
away left the failure byte-identical. It is not the cause.)

---

## 6. Before merging: delete these two files

`CUDA13_PROGRESS.md` and `CUDA13_HANDOFF.md` are working notes for this handoff, not
repository content. They are the only non-cuda13 changes on the branch (3 commits). Drop them
so the branch is purely the CUDA 13 upgrade:

```bash
git rm CUDA13_PROGRESS.md CUDA13_HANDOFF.md && git commit -m "chore: drop working notes"
# or: git rebase -i main  and drop the three docs commits
```

Anything worth keeping belongs in the PR description or a PR comment.
