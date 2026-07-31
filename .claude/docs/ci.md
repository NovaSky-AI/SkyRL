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

## Slack Alerts (`#skyrl-ci-alerts`)

`CI-Slack-Digest` (`.github/workflows/ci_slack_digest.yaml`) posts one message a day at 14:00 UTC summarising `main` CI health: workflows that are red right now, and workflows that went green again since the last digest. Logic lives in `.github/scripts/ci_slack_digest.py`.

- **Secret**: `SLACK_CI_ALERTS_WEBHOOK_URL` — a Slack incoming webhook bound to `#skyrl-ci-alerts`. Without it the job still runs and renders the digest into the step log instead of posting.
- **Monitored workflows**: the `MONITORED_WORKFLOWS` env block in the workflow. Add new workflows there — the digest footer flags any workflow file whose `name:` is missing from the list, so drift is visible rather than silent.
- **Dry run**: `gh workflow run CI-Slack-Digest -f dry_run=true`, then read the step log. Use this to check a change before it reaches the channel.
- **Classifier tests**: `.github/scripts/test_ci_slack_digest.py`, run by the digest job itself before it posts. They live next to the script (not under `tests/`) so editing them does not trip the `tests/**` path filters that gate the GPU workflows.

Classification rules worth knowing:

- Only `main` counts. PR runs carry the PR's head branch, so they never trigger an alert.
- `cancelled` and `skipped` runs are ignored entirely — they neither alert nor break a failure/recovery streak. This matters because `cancel-in-progress` concurrency cancels a meaningful share of `main` runs.
- "Failing now" is state-based (derived from the latest run, not the 24h window), so a missed or delayed digest never drops a still-broken workflow off the report.
- The window defaults to 25h rather than 24h because GitHub's scheduled triggers drift; the overlap can re-report a recovery once, which is cheaper than missing one.

## Gotchas

- The `paths:` filter on each workflow gates whether CPU CI even runs. Touching only `docs/` or `examples/` skips CI.
- `ci/**` is in the `paths:` filter of nearly every GPU workflow, so editing anything there triggers the full Anyscale GPU suite. Keep non-runner CI helpers in `.github/scripts/` instead.
