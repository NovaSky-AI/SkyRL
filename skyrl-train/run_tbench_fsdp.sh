set -x

# WORK IN PROGRESS
# Colocated GRPO training+generation for Qwen3-8B on TerminalBench tasks with Megatron on 4 GPUs.

# Prep (examples):
# uv run examples/terminal_bench/prepare_dataset.py --task_dir $HOME/data/terminal_bench/tasks --output_dir $HOME/data/terminal_bench --output_name train
# export WANDB_API_KEY=<your_key_here>
# bash examples/terminal_bench/run_tbench_megatron.sh

export WANDB_API_KEY="854a2b39e99ffee11c76d1003eb8a777045687e9"
export WANDB_ENTITY="bespoke-labs"
export PYTHONPATH="$PWD"

DATA_DIR="/data/ez_apex_281"
NUM_NODES=2
NUM_GPUS=8
LOGGER="wandb"  # change to "console" to print to stdout
TBENCH_CONFIG_DIR="examples/terminal_bench"
SANDBOXES_DIR="$HOME/SkyRL/skyrl-train/sandboxes"
MODEL_NAME="Qwen/Qwen3-Coder-30B-A3B-Instruct"

FLASH_ATTN=true
NUM_INFERENCE_ENGINES=16
INFERENCE_ENGINE_TP=1

export SKYRL_PYTHONPATH_EXPORT=1
export SKYRL_LD_LIBRARY_PATH_EXPORT=1
export PYTHONPATH="$HOME/SkyRL/skyrl-train/"
export CUDNN_PATH="/opt/cudnn"
export CPATH="$CUDNN_PATH/include:${CPATH:-}"
export LD_LIBRARY_PATH="$CUDNN_PATH/lib:${LD_LIBRARY_PATH:-}"
export RAY_worker_register_timeout_seconds=1800
export TOKENIZERS_PARALLELISM=false

# Inference backend (for rollout generation)
INFERENCE_BACKEND="vllm"  # currently only vLLM is supported for Megatron in this setup



# data.train_data="['$DATA_DIR/train.parquet']" \
uv run --isolated --extra $INFERENCE_BACKEND --extra sandboxes --with "sandboxes@./sandboxes" -m examples.terminal_bench.entrypoints.main_tbench \
  data.train_data="['$DATA_DIR']" \
  hydra.searchpath=[file://$TBENCH_CONFIG_DIR] \
  +terminal_bench_config=terminal_bench \
  +terminal_bench_config.sandboxes_dir=$SANDBOXES_DIR \
  terminal_bench_config.max_episodes=64 \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL_NAME" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.cpu_offload=true \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.policy.fsdp_config.fsdp_size=2 \
  trainer.ref.fsdp_config.fsdp_size=2 \
  trainer.policy.sequence_parallel_size=8 \
  trainer.ref.sequence_parallel_size=8 \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.ref_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.use_sample_packing=true \
  trainer.flash_attn=$FLASH_ATTN \
  trainer.epochs=20 \
  trainer.eval_batch_size=32 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=1 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=32000 \
  trainer.logger="$LOGGER" \
  trainer.project_name="terminal_bench" \
  trainer.run_name="fsdp" \
  trainer.resume_mode=latest \
  trainer.hf_save_interval=4 \
  trainer.ckpt_interval=1 \
  trainer.ckpt_path="/data/ckpt_fsdp" \
  trainer.export_path="/data/hf_ckpt_fsdp" \
  trainer.algorithm.eps_clip_low=0.2 \
  trainer.algorithm.eps_clip_high=0.28 \
  trainer.algorithm.loss_reduction="token_mean" \
  trainer.gradient_checkpointing_use_reentrant=true \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=false \
  generator.sampling_params.max_generate_length=32000 \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$INFERENCE_ENGINE_TP \
  generator.enable_http_endpoint=true \
  generator.http_endpoint_host="127.0.0.1" \
  generator.http_endpoint_port=8000 \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  generator.n_samples_per_prompt=8 \
  generator.gpu_memory_utilization=0.8 \
  generator.apply_overlong_filtering=true \
  environment.env_class=terminal_bench \
  $@
