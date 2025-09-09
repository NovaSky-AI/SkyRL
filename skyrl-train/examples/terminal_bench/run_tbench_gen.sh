set -x

# WORK IN PROGRESS
# Colocated GRPO training+generation for Qwen2.5-1.5B-Instruct on TerminalBench tasks.

# export WANDB_API_KEY=<your_key_here>
# bash examples/terminal_bench/run_tbench.sh

NUM_GPUS=1
LOGGER="console"  # change to "console" to print to stdout
TBENCH_CONFIG_DIR="examples/terminal_bench"
SANDBOXES_DIR="sandboxes" # TODO: For now, `sandboxes` is cloned into SkyRL/skyrl-train.

uv run --isolated --extra vllm --extra sandboxes --with "sandbox@./sandboxes" -m examples.terminal_bench.entrypoints.main_tbench_generate \
  hydra.searchpath=[file://$TBENCH_CONFIG_DIR] \
  +terminal_bench_config=terminal_bench \
  terminal_bench_config.max_episodes=16 \
  terminal_bench_config.sandboxes_dir=$SANDBOXES_DIR \
  trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  generator.enable_http_endpoint=true \
  generator.http_endpoint_host="127.0.0.1" \
  generator.http_endpoint_port=8000 \
  generator.sampling_params.max_generate_length=4096 \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.gpu_memory_utilization=0.8 \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.logger="$LOGGER" \
  $@