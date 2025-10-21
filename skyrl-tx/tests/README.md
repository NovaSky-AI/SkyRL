# SkyRL tx: Tests

## Run all tests
uv run pytest -v

## Run Specific tests
uv run pytest -v -s tests/models/<file>::<test_name> e.g. test_qwen3_generate.py::test_qwen3_generate_speed