#!/usr/bin/env bash
uv run --directory . --isolated --extra dev --extra vllm pytest -s tests/gpu/gpu_ci