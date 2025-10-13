# Flash RL Examples

## Installation

The `flashrl` extra does not automatically install vLLM due to PyPI package distribution constraints. You must install the custom vLLM wheel manually.

## Install custom vLLM

```bash
# Install the custom vLLM wheel
uv pip install https://github.com/NovaSky-AI/SkyRL/releases/download/skyrl_train-v0.1.0/vllm-0.1.dev7509+gcc487699a.d20250821-cp312-cp312-linux_x86_64.whl
```

## Running Examples

After installing the custom vLLM wheel, you can run the Flash RL examples:

```bash
# Example: DAPO with FlashRL using 0.5B model with FP8
bash examples/flash_rl/run_dapo_gsm8k_flashrl_0.5b_fp8.sh

# Example: DAPO with FlashRL using 0.5B model with INT8
bash examples/flash_rl/run_dapo_gsm8k_flashrl_0.5b_int8.sh

# Example: DAPO with FlashRL using 32B model with INT8
bash examples/flash_rl/run_dapo_gsm8k_flashrl_32b_int8.sh
```