# SkyRL tx: Tests

## Run all tests
uv run pytest -v

## Run Specific tests
uv run pytest -v -s tests/models/<file>::<test_name> e.g. test_qwen3_generate::test_qwen3_generate_speed

tests/models/test_qwen3.py
- test_qwen3_forward - Forward pass matches HuggingFace
- test_qwen3_with_kv_cache - KV cache correctness

tests/models/test_qwen3_generate.py
- test_qwen3_generate - Generation matches HuggingFace
- test_qwen3_generate_speed - Profile generation performance

tests/models/test_qwen3_lora_training.py
- test_qwen3_lora_training - LoRA training reduces loss