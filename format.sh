set -e

if command -v uv >/dev/null 2>&1; then
    uv pip install -q pre-commit
    # pre-commit run --all-files always runs from the root directory.
    uv run pre-commit run --all-files --config .pre-commit-config.yaml
else 
    pip install -q pre-commit
    # pre-commit run --all-files always runs from the root directory.
    pre-commit run --all-files --config .pre-commit-config.yaml
fi