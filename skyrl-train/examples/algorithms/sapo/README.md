# SAPO Trainer

# SAPO Trainer

`run_dapo_aime_qwen3_4b_aime.sh` is a Slurm script that launches the Qwen3-4B SAPO job using `uv`.

## Quick Start

1. Set/Export your `WANDB_API_KEY`.
2. Ensure you have `uv` installed and the environment is set up in `skyrl-train`.
3. Submit the job:

   ```bash
   cd examples/algorithms/sapo
   sbatch run_dapo_aime_qwen3_4b_aime.sh
   ```

Logs land in `logs/sapo/30B/` (you can change it) and checkpoints under `$DATA_ROOT/checkpoint/…`. Monitor the job with `squeue -u <user>` or check the `.out/.err` files.

Happy trainings :)
