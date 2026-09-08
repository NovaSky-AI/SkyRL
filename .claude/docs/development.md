# Development

The opt-in B300 FA2/FA4 image profile lives in `docker/attention/b300-fa4/`.
Its generated manifest is installed at the image's SkyRL root; ordinary installs
keep the root manifest and lock. Regenerate the profile whenever the root
manifest changes. Build/release and GPU validation commands are documented in
`docker/attention/b300-fa4/qualification.md`.

For full details, see `docs/content/docs/getting-started/development.mdx`.

## Pre-commit checklist

- Ensure you've run pre-commit hooks before every commit (More instructions in the doc above)
- Ensure relevant CPU and GPU tests pass.
- Sign off on every commit with `git commit -s`.

## Do

- `uv run --isolated` for GPU tests and training to get clean environments.
- `uv run --extra dev --extra <backend>` for running with specific backend extras.

## Don't

- Using `python ...` for executing python scripts
