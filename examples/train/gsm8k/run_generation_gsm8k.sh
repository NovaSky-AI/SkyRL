set -x

# Generation only for for Qwen2.5-0.5B-Instruct on GSM8K.

# uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/gsm8k/run_generation_gsm8k.sh

DATA_DIR="$HOME/data/gsm8k"
NUM_GPUS=1
LOGGER="wandb"  # change to "console" to print to stdout

INFERENCE_BACKEND="vllm"  # or "sglang"

uv run --isolated --extra $INFERENCE_BACKEND \
  -m skyrl_train.entrypoints.main_generate \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.policy.model.path="Qwen/Qwen2.5-0.5B-Instruct" \
  trainer.logger="$LOGGER" \
  trainer.placement.colocate_all=false \
  generator.backend=$INFERENCE_BACKEND \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  generator.gpu_memory_utilization=0.9 \
  generator.eval_sampling_params.max_generate_length=1024 \
  generator.eval_sampling_params.temperature=0.7 \
  environment.env_class=gsm8k \
  "$@"
