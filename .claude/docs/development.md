# Development

## Package Manager: uv

- **Always use `uv`**. Never use bare `python`, `pip`, or `pip install`.
- `uv run --isolated` for GPU tests and training to get clean environments.
- `uv run --extra dev --extra <backend>` for running with specific backend extras.

## Extras and Conflicts

Backend extras are mutually exclusive (defined in `[tool.uv]` conflicts):
- `fsdp`, `megatron`, `jax`, `flashrl` — never combine these in one environment.
- Other extras: `dev`, `gpu`, `tpu`, `miniswe`, `tinker`, `harbor`, `aws`, `gcp`, `azure`.

## Key Dependencies

- `skyrl-gym` is an editable sub-package linked via `[tool.uv.sources]`.
- Custom package indexes: `pytorch-cu128`, `flashinfer-cu128`, `jax-tpu`.
- `no-build-isolation-package` for `transformer-engine` and related CUDA packages.
- `override-dependencies` in `[tool.uv]` pins or disables transitive deps.

## Formatting and Linting

```bash
# Run all pre-commit hooks (ruff + black + gitleaks)
bash format.sh
# Or directly
pre-commit run --all-files
```
