set -x

# My key
export DAYTONA_API_KEY=YOUR_KEY_HERE
export WANDB_API_KEY=YOUR_KEY_HERE

# Got after hf download open-thoughts/OpenThoughts-Agent-v1-RL --repo-type=dataset
# cd into the downloaded folder, say /path/to/.cache/huggingface/hub/datasets--open-thoughts--OpenThoughts-Agent-v1-RL/snapshots/hash_code
# python extract_parquet_tasks.py tasks_new.parquet ./extracted_tasks
TRAIN_DATA="['/home/ray/.cache/huggingface/hub/datasets--open-thoughts--OpenThoughts-Agent-v1-RL/snapshots/39ab71434e90d8f87d2cd69c13b6d8a0cb2c238f/extracted_tasks']"

CHAT_TEMPLATE_PATH="/home/ray/default/SkyRLHarbor3/skyrl-train/skyrl_train/utils/templates/qwen3_acc_thinking.jinja2"
TRIALS_DIR="/home/ray/trials_run"

NUM_GPUS=4

uv run --isolated --extra vllm --extra harbor -m examples.terminal_bench.entrypoints.main_tbench_generate \
  data.train_data=$TRAIN_DATA \
  hydra.searchpath=['file://examples/terminal_bench'] \
  +terminal_bench_config=default \
  ++terminal_bench_config.trials_dir=$TRIALS_DIR \
  ++terminal_bench_config.trial_name="dummy" \
  trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
  generator.served_model_name="Qwen2.5-1.5B-Instruct" \
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
  +generator.engine_init_kwargs.chat_template=$CHAT_TEMPLATE_PATH \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.placement.colocate_all=false \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.train_batch_size=$NUM_GPUS \
  trainer.policy_mini_batch_size=$NUM_GPUS \
  trainer.logger=console \
  $@