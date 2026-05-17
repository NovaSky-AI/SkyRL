# Development

For full details, see `docs/content/docs/getting-started/development.mdx`.

## Pre-commit checklist

- Ensure you've run pre-commit hooks before every commit.
- Ensure relevant CPU and GPU tests pass.
- Sign off on every commit with `git commit -s`.

## Do

- `uv run --isolated` for GPU tests and training to get clean environments.
- `uv run --extra dev --extra <backend>` for running with specific backend extras.

## Don't

- Using `python ...` for executing python scripts