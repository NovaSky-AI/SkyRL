# SkyRL tx: Tests

## Run all tests
uv run pytest -v

## Run Specific tests
uv run pytest -v -s tests/models/test_qwen3_generate.py::test_qwen3_generate_speed