#!/usr/bin/env bash
set -xeuo pipefail

export CI=true

# Run GPU-specific tests
uv run --extra gpu --extra tinker --extra dev pytest tests/gpu
