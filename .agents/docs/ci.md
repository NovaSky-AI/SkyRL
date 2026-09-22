# CI

- **Workflows**: `.github/workflows/{cpu,gpu,tinker}_*.yaml`.
- **Runner glue**: `ci/anyscale_*.yaml` (Anyscale job spec) → `ci/gpu_*_run*.sh` (pytest invocation).

## Workflow namespaces

| Check name | File | Covers |
|---|---|---|
| `SkyRL-CPU` | `cpu_skyrl_train.yaml` | pre-commit, `tests/train`, `tests/backends/skyrl_train` (CPU), `tests/tinker`, `tests/utils`, `skyrl-gym` |
| `SkyRL-GPU` | `gpu_skyrl_train.yaml` | `tests/backends/skyrl_train/gpu/gpu_ci` on Anyscale |
| `SkyRL-JAX-CPU` | `cpu_jax.yaml` | `tests/tx`, `tests/backends/test_jax_backend.py`, engine benchmark |
| `SkyRL-JAX-GPU` | `gpu_jax.yaml` | `tests/tx/gpu` on Anyscale |

## CPU vs GPU

- **CPU workflows** (`cpu_*.yaml`) run on `ubuntu-latest`, auto-trigger on push to `main`/`rc/*` and on PRs. Run lint + the CPU pytest suites from AGENTS.md.
- **GPU workflows** (`gpu_*.yaml`, `tinker_*.yaml`) run on `ubuntu-latest` but submit to Anyscale via `anyscale job submit -f ci/<config>.yaml --timeout 12000`. **Label-gated** on PRs (except `SkyRL-JAX-GPU`, which is path-gated).

## Anyscale

- Compute config: `l4_ci` (referenced from `ci/anyscale_*.yaml`).
- Cloud: `sky-anyscale-aws-us-east-1`.
- Image: `novaskyai/skyrl-train-ray-2.57.0-py3.12-cu13.0` (varies per workflow).
- Logs: visit the Anyscale job page linked from the GitHub Actions step output. Stderr from Ray workers shows up under the head node logs, not the entrypoint logs.

## Adding a New Test to CI

1. Decide CPU or GPU. CPU is free; GPU costs Anyscale credits per run.
2. CPU: just add the test under `tests/` — `cpu_skyrl_train.yaml` already globs the suite.
3. GPU: add the test, then either (a) extend an existing `ci/gpu_*_run*.sh` to include it, or (b) add a new workflow + runner pair if it needs a different extras combo or a different compute config.

## Gotchas

- The `paths:` filter on each workflow gates whether CPU CI even runs. Touching only `docs/` or `examples/` skips CI.
- The `SkyRL-JAX-*` workflows are deliberately scoped to the JAX/tx code path (`skyrl/tx`, `skyrl/backends/{backend,jax,ray_jax}.py`, `skyrl/utils`, `skyrl/tinker/types.py`, `tests/tx`), so most PRs never run them. If you touch tx, check that your files are in those `paths:` lists — and note the `push:` and `pull_request:` lists are duplicated, because GitHub Actions does not support YAML anchors.
