set -x

# Generation only for for Qwen2.5-0.5B-Instruct on GSM8K.

# uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/rlm/run_generation_rlm.sh

# By default, the RLM tool will use the local HTTP endpoint to the trainer model for inference.
# To run via any OpenAI compatible server, just set the rlm_api_key and appropriate base_url and model 
# in the default.yaml config file. Set the http_endpoint flag to false to disable the local HTTP 
# endpoint (for OpenAI API).

DATA_DIR="$HOME/data/gsm8k"
NUM_GPUS=1
LOGGER="console"  # change to "console" to print to stdout

INFERENCE_BACKEND="vllm"  # or "sglang"

uv run --isolated --extra $INFERENCE_BACKEND \
  -m skyrl_train.entrypoints.main_generate \
  data.val_data="['$DATA_DIR/validation_rlm.parquet']" \
  trainer.policy.model.path="Qwen/Qwen2.5-7B-Instruct" \
  trainer.logger="$LOGGER" \
  trainer.placement.colocate_all=false \
  trainer.eval_batch_size=1 \
  generator.n_samples_per_prompt=1 \
  generator.backend=$INFERENCE_BACKEND \
  generator.enable_http_endpoint=true \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  generator.gpu_memory_utilization=0.9 \
  generator.eval_sampling_params.max_generate_length=1024 \
  generator.eval_sampling_params.temperature=0.7 \
  environment.env_class=rlm_ex \
  "$@"
