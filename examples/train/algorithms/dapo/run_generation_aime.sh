set -x

# Generation-only eval of DeepSeek-R1-Distill-Qwen-1.5B on AIME 2025 with the `aime` environment.
#
# Sampling follows MathArena's settings for this model: temperature 0.6, top_p 0.95, up to 32000
# generated tokens. N_SAMPLES samples per problem (default 16) are graded by the `aime` env, which
# reads the last `Answer: X` line of the response (reward +1 if correct, -1 otherwise).
#
# Metrics are logged per data source, e.g. for AIME 2025:
#   eval/aime-2025/mean_positive_reward  avg@N_SAMPLES accuracy
#   eval/aime-2025/avg_score             mean reward = 2 * accuracy - 1
#   eval/aime-2025/pass_at_<N_SAMPLES>   fraction of problems with at least one correct sample
# AIME 2024 (`aime-2024-cleaned.parquet`) is reported under `eval/math_dapo/...`. Per-sample
# responses and rewards are written to `$HOME/exports/dumped_evals/eval_only/` (`trainer.export_path`).
#
# Reference result (N_SAMPLES=16, NUM_GPUS=2 on B200): mean_positive_reward 0.125,
# avg_score -0.75, pass_at_16 0.333. MathArena reports 20.0% for this model. Most of the gap is
# answer format: the model often puts its final answer only in \boxed{} without an `Answer:`
# line, which the `aime` env scores as incorrect.
#
# Usage (`prepare_dapo_data.sh` writes `aime-2025-cleaned.parquet` with
# `data_preprocess_aime2025.py`; that script can also be run on its own):
#
# bash examples/train/algorithms/dapo/prepare_dapo_data.sh
# bash examples/train/algorithms/dapo/run_generation_aime.sh
#
# To evaluate on AIME 2024 and AIME 2025 together:
#
# bash examples/train/algorithms/dapo/run_generation_aime.sh \
#   data.val_data="['$HOME/data/dapo/aime-2024-cleaned.parquet', '$HOME/data/dapo/aime-2025-cleaned.parquet']"

DATA_DIR=${DATA_DIR:-"$HOME/data/dapo"}
NUM_GPUS=${NUM_GPUS:-1}
N_SAMPLES=${N_SAMPLES:-16}
LOGGER="console"  # change to "wandb" to log to Weights & Biases

uv run --isolated --extra fsdp \
  -m skyrl.train.entrypoints.main_generate \
  data.val_data="['$DATA_DIR/aime-2025-cleaned.parquet']" \
  trainer.policy.model.path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
  trainer.logger="$LOGGER" \
  trainer.max_prompt_length=1024 \
  trainer.placement.colocate_all=false \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.gpu_memory_utilization=0.85 \
  generator.eval_n_samples_per_prompt=$N_SAMPLES \
  generator.eval_sampling_params.max_generate_length=32000 \
  generator.eval_sampling_params.temperature=0.6 \
  generator.eval_sampling_params.top_p=0.95 \
  environment.env_class=aime \
  "$@"
