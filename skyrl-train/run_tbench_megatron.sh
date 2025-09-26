set -x

# WORK IN PROGRESS
# Colocated GRPO training+generation for Qwen3-8B on TerminalBench tasks with Megatron on 4 GPUs.

# Prep (examples):
# uv run examples/terminal_bench/prepare_dataset.py --task_dir $HOME/data/terminal_bench/tasks --output_dir $HOME/data/terminal_bench --output_name train
# export WANDB_API_KEY=<your_key_here>
# bash examples/terminal_bench/run_tbench_megatron.sh

# DATA_DIR="$HOME/ez_apex_100"
DATA_DIR="$HOME/tmp"
NUM_NODES=1
NUM_GPUS=8
LOGGER="wandb"  # change to "console" to print to stdout
TBENCH_CONFIG_DIR="examples/terminal_bench"
SANDBOXES_DIR="/root/SkyRL/skyrl-train/sandboxes"
MODEL_NAME="Qwen/Qwen3-Coder-30B-A3B-Instruct"

# Inference backend (for rollout generation)
INFERENCE_BACKEND="vllm"  # currently only vLLM is supported for Megatron in this setup

# Megatron parallelism (4 GPUs total => 2x TP, 2x PP, 1x CP)
MEGATRON_TP=2
MEGATRON_PP=2
MEGATRON_CP=2

MEGATRON_EP=4
MEGATRON_ETP=1

FLASH_ATTN=true
NUM_INFERENCE_ENGINES=1
INFERENCE_ENGINE_TP=8

# Torch profiler (optional)
ENABLE_TORCH_PROFILER=false
RANKS_TO_PROFILE="[0]"
SAVE_PATH="$HOME/megatron_prof/tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}_${MODEL_NAME}"

export SKYRL_PYTHONPATH_EXPORT=1
# data.train_data="['$DATA_DIR/train.parquet']" \
uv run --isolated --extra $INFERENCE_BACKEND --extra sandboxes --extra mcore --with "sandboxes@./sandboxes" -m examples.terminal_bench.entrypoints.main_tbench \
  data.train_data="['$DATA_DIR']" \
  hydra.searchpath=[file://$TBENCH_CONFIG_DIR] \
  +terminal_bench_config=terminal_bench \
  +terminal_bench_config.sandboxes_dir=$SANDBOXES_DIR \
  terminal_bench_config.max_episodes=16 \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL_NAME" \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.ref_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$INFERENCE_ENGINE_TP \
  generator.enable_http_endpoint=true \
  generator.http_endpoint_host="127.0.0.1" \
  generator.http_endpoint_port=8000 \
  trainer.policy.megatron_config.torch_profiler_config.enable=$ENABLE_TORCH_PROFILER \
  trainer.policy.megatron_config.torch_profiler_config.ranks=$RANKS_TO_PROFILE \
  trainer.policy.megatron_config.torch_profiler_config.save_path=$SAVE_PATH \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.ref.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.ref.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.ref.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.ref.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.ref.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.use_sample_packing=true \
  trainer.flash_attn=$FLASH_ATTN \
  trainer.epochs=1 \
  trainer.eval_batch_size=32 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=8 \
  trainer.policy_mini_batch_size=4 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=-1000000 \
  trainer.max_prompt_length=16000 \
  generator.sampling_params.max_generate_length=16000 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=terminal_bench \
  generator.n_samples_per_prompt=4 \
  generator.gpu_memory_utilization=0.85 \
  trainer.logger="$LOGGER" \
  trainer.project_name="terminal_bench" \
  trainer.run_name="terminal_bench_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}_${MODEL_NAME}" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/ez_apex_ckpt" \
  trainer.hf_save_interval=1000000 \
  $@


# environment.env_class=gsm8k \

# terminal_bench_config.max_episodes=16 \