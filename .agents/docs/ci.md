# CI

- **Workflows**: `.github/workflows/{cpu,gpu,tinker}_*.yaml`.
- **Runner glue**: `ci/anyscale_*.yaml` (Anyscale job spec) → `ci/gpu_*_run*.sh` (pytest invocation).

## Workflow namespaces

One naming stem per pipeline, used by the workflow file, the Anyscale job spec, the
runner script and the Anyscale job name. `train` is not part of any of them -- it
survives only where it names a real thing (the `skyrl_train` package, the
`novaskyai/skyrl-train-ray-*` image).

| Check name | Workflow | Anyscale spec / runner | Covers |
|---|---|---|---|
| `SkyRL-CPU` | `cpu_skyrl.yaml` | — | pre-commit, `tests/train`, `tests/backends/skyrl_train` (CPU), `tests/tinker`, `tests/utils`, `skyrl-gym` |
| `SkyRL-GPU` | `gpu_skyrl.yaml` | `anyscale_gpu_ci_skyrl.yaml` / `gpu_ci_run_skyrl.sh` | `tests/backends/skyrl_train/gpu/gpu_ci` |
| `SkyRL-GPU-Megatron` | `gpu_skyrl_megatron.yaml` | `anyscale_gpu_ci_skyrl_megatron.yaml` / `gpu_ci_run_skyrl_megatron.sh` | the `megatron` marker |
| `Megatron-Model-GPU-CI` | `gpu_skyrl_megatron_models.yaml` | `anyscale_gpu_ci_skyrl_megatron_models.yaml` / `gpu_ci_run_skyrl_megatron_models.sh` | the `megatron_models` marker |
| `H100-GPU-CI` | `gpu_ci_h100.yaml` | `anyscale_gpu_ci_h100.yaml` / `gpu_ci_run_h100.sh` | H100-only suites |
| `Tinker-SkyRL-Backend-GPU` | `tinker_skyrl_backend_gpu.yaml` | `anyscale_tinker_skyrl_backend_gpu.yaml` / `gpu_ci_run_tinker_skyrl_backend.sh` | `tests/tinker/skyrl_train` |
| `SkyRL-JAX-CPU` | `cpu_jax.yaml` | — | `tests/tx`, `tests/backends/test_jax_backend.py`, engine benchmark |
| `SkyRL-JAX-GPU` | `gpu_jax.yaml` | `anyscale_gpu_ci_jax.yaml` / `gpu_ci_run_jax.sh` | `tests/tx/gpu` |
| `SkyRL-GPU-E2E-CI*` | `gpu_e2e_ci*.yaml` | `anyscale_gpu_e2e_test*.yaml` / `gpu_e2e_test_run*.sh` | end-to-end training runs |

## CPU vs GPU

- **CPU workflows** (`cpu_*.yaml`) run on `ubuntu-latest`, auto-trigger on push to `main`/`rc/*` and on PRs. Run lint + the CPU pytest suites from AGENTS.md.
- **GPU workflows** (`gpu_*.yaml`, `tinker_*.yaml`) run on `ubuntu-latest` but submit to Anyscale via `anyscale job submit -f ci/<config>.yaml --timeout 12000`. **Label-gated** on PRs (except `SkyRL-JAX-GPU`, which is path-gated).

### GPU gating labels

Apply one of these to a PR to launch the corresponding Anyscale job. The label must
already exist in repo settings -- a workflow referencing a label nobody can apply
never runs, and fails silently.

| Label | Launches |
|---|---|
| `run_gpu_ci` | `SkyRL-GPU` |
| `run_megatron_gpu_ci` | `SkyRL-GPU-Megatron` |
| `run_megatron_gpu_ci_models` | `Megatron-Model-GPU-CI` |
| `run_h100_gpu_ci` | `H100-GPU-CI` |
| `run_tinker_skyrl_backend_gpu_ci` | `Tinker-SkyRL-Backend-GPU` |

## Anyscale

- Compute config: `l4_ci` (referenced from `ci/anyscale_*.yaml`).
- Cloud: `sky-anyscale-aws-us-east-1`.
- Image: `novaskyai/skyrl-train-ray-2.57.0-py3.12-cu13.0` (varies per workflow).
- Logs: visit the Anyscale job page linked from the GitHub Actions step output. Stderr from Ray workers shows up under the head node logs, not the entrypoint logs.

## Adding a New Test to CI

1. Decide CPU or GPU. CPU is free; GPU costs Anyscale credits per run.
2. CPU: just add the test under `tests/` — `cpu_skyrl.yaml` already globs the suite.
3. GPU: add the test, then either (a) extend an existing `ci/gpu_*_run*.sh` to include it, or (b) add a new workflow + runner pair if it needs a different extras combo or a different compute config.

## Gotchas

- The `paths:` filter on each workflow gates whether CPU CI even runs. Touching only `docs/` or `examples/` skips CI.
- The `SkyRL-JAX-*` workflows are deliberately scoped to the JAX/tx code path (`skyrl/tx`, `skyrl/backends/{backend,jax,ray_jax}.py`, `skyrl/utils`, `skyrl/tinker/types.py`, `tests/tx`), so most PRs never run them. If you touch tx, check that your files are in those `paths:` lists — and note the `push:` and `pull_request:` lists are duplicated, because GitHub Actions does not support YAML anchors.
