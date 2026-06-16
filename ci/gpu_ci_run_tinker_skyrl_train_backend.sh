#!/usr/bin/env bash
set -xeuo pipefail

export CI=true

# End-to-end multi-LoRA tests: spin up a real Tinker API server backed by
# SkyRL-Train Megatron and exercise per-adapter swap, signature gating,
# v1 single-tenant sample guard, per-adapter Adam step isolation, and
# delete-then-train continuity.
uv run --directory . --isolated --extra tinker --extra megatron --with pytest --with pytest-timeout \
    pytest -s --timeout=600 tests/tinker/skyrl_train/
