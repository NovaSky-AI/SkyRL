set -x
# Helper that converts Environment Hub ID to required uv flags
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/env_resolver.sh"

# Specify environment from Environments Hub in form "org/name@version" (e.g., will/wordle@0.1.4)
ENV_ID="primeintellect/reverse-text"

DATA_DIR="$HOME/data/$ENV_ID"
NUM_GPUS=1
LOGGER="console"  # change to "console" to print to stdout

ENV_UV_INSTALL_FLAGS="$(verifiers_env_to_uv_flags "$ENV_ID")"
uv run --isolated --with verifiers $ENV_UV_INSTALL_FLAGS --extra vllm -m examples.verifiers.main_verifiers \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  generator.n_samples_per_prompt=2 \
  trainer.policy.model.path="Qwen/Qwen2.5-0.5B-Instruct" \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=20 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.train_batch_size=2 \
  trainer.policy_mini_batch_size=2 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=8192 \
  generator.max_input_length=8192 \
  generator.sampling_params.max_generate_length=1024 \
  generator.enable_http_endpoint=true \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  environment.env_class="$ENV_ID" \
  trainer.project_name="verifiers" \
  trainer.run_name="verifiers_test" \
  trainer.ckpt_interval=10 \
  trainer.ckpt_path="$HOME/ckpts/verifiers_ckpt"
  $@