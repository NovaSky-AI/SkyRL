# Launches SkyRL training on the Verifiers environment.
#
# Example:
#   bash integrations/verifiers/run_verifiers.sh
#
set -x

# Specify environment ID from Environments Hub in form "org/name@version" (e.g., will/wordle@0.1.4)
ENV_ID="will/wordle"
DATA_DIR="$HOME/data/$ENV_ID"
NUM_GPUS=1
LOGGER="wandb"  # change to "console" to print to stdout

uv run --isolated --with verifiers --extra vllm -m integrations.verifiers.entrypoints.main_verifiers \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  generator.n_samples_per_prompt=5 \
  trainer.epochs=20 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.train_batch_size=128 \
  trainer.policy_mini_batch_size=128 \
  trainer.micro_forward_batch_size_per_gpu=32 \
  trainer.micro_train_batch_size_per_gpu=32 \
  trainer.max_prompt_length=8192 \
  generator.max_input_length=8192 \
  generator.sampling_params.max_generate_length=1024 \
  generator.enable_http_endpoint=true \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  environment.env_class="$ENV_ID" \
  trainer.project_name="verifiers" \
  trainer.run_name="verifiers_test" \
  trainer.ckpt_interval=-1 \
  trainer.ckpt_path="$HOME/ckpts/verifiers_ckpt"
  $@