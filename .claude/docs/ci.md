# CI

- **Workflows**: `.github/workflows/{cpu,gpu,tinker}_*.yaml`.
- **Runner glue**: `ci/anyscale_*.yaml` (Anyscale job spec) → `ci/gpu_*_run*.sh` (pytest invocation).

## CPU vs GPU

- **CPU workflows** (`cpu_skyrl*.yaml`) run on `ubuntu-latest`, auto-trigger on push to `main`/`rc/*` and on every PR. Run lint + the CPU pytest suites from CLAUDE.md.
- **GPU workflows** (`gpu_*.yaml`, `tinker_*.yaml`) run on `ubuntu-latest` but submit to Anyscale via `anyscale job submit -f ci/<config>.yaml --timeout 12000`. **Label-gated** on PRs.

## Anyscale

- Compute config: `l4_ci` (referenced from `ci/anyscale_*.yaml`).
- Cloud: `sky-anyscale-aws-us-east-1`.
- Image: `novaskyai/skyrl-train-ray-2.56.0-py3.12-cu12.8` (varies per workflow).
- Logs: visit the Anyscale job page linked from the GitHub Actions step output. Stderr from Ray workers shows up under the head node logs, not the entrypoint logs.

## Adding a New Test to CI

1. Decide CPU or GPU. CPU is free; GPU costs Anyscale credits per run.
2. CPU: just add the test under `tests/` — `cpu_skyrl_train.yaml` already globs the suite.
3. GPU: add the test, then either (a) extend an existing `ci/gpu_*_run*.sh` to include it, or (b) add a new workflow + runner pair if it needs a different extras combo or a different compute config.

## Gotchas

- The `paths:` filter on each workflow gates whether CPU CI even runs. Touching only `docs/` or `examples/` skips CI.
