# SAPO Trainer

# SAPO Trainer

`run_sapo_qwen3_4b_aime.sh` is a bash script that launches the Qwen3-4B SAPO job using `uv`. It uses the `examples/algorithms/dapo/main_dapo.py` script, but with the SAPO policy loss and sequence mean loss reduction.

## Quick Start

1. Set/Export your `WANDB_API_KEY`.
2. Ensure you have `uv` installed and the environment is set up in `skyrl-train`.
3. Submit the job:

   ```bash
   bash examples/algorithms/sapo/run_sapo_qwen3_4b_aime.sh
   ```