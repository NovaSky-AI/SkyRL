set -x

# Colocated GRPO training+generation for Qwen2.5-1.5B-Instruct on GSM8K with HTTP endpoint.

# uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# bash examples/inference_http_endpoint/run_gsm8k_with_inference_http_endpoint.sh

# NOTE (charlie): The only difference between this and the original run_gsm8k.sh is that we set
# `generator.enable_http_endpoint` to true and set the HTTP endpoint host and port.
# Besides, we run `main.py` in this folder which uses the SkyRLGymHTTPGenerator, a simple wrapper
# of SkyRLGymGenerator that uses the HTTP endpoint for rollout as a demonstration.

DATA_DIR="$HOME/data/gsm8k"
NUM_GPUS=4
LOGGER="console"  # change to "console" to print to stdout

INFERENCE_BACKEND="vllm"
# INFERENCE_BACKEND="sglang"
uv run --isolated --extra $INFERENCE_BACKEND -m examples.inference_http_endpoint.main \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=20 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=1024 \
  trainer.policy_mini_batch_size=256 \
  trainer.micro_forward_batch_size_per_gpu=64 \
  trainer.micro_train_batch_size_per_gpu=64 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.8 \
  generator.enable_http_endpoint=true \
  generator.http_endpoint_host="127.0.0.1" \
  generator.http_endpoint_port=8000 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k" \
  trainer.run_name="gsm8k_test" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/gsm8k_1.5B_ckpt" \
  $@