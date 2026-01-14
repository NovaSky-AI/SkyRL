# SkyRL tx: Tests

## Run all tests
uv run --extra dev --extra tinker pytest -v

## Run Specific tests
uv run --extra dev --extra tinker pytest -v -s tests/models/test_qwen3_generate.py::test_qwen3_generate_speed

## Run Cutile tests
TX_USE_CUTILE_LORA=1 uv run pytest tests/test_cutile_lora_equivalence.py -v