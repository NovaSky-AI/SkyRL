set -x

# WORK IN PROGRESS
# Colocated GRPO training+generation for Qwen3-8B on TerminalBench tasks with Megatron on 4 GPUs.

# Prep (examples):
# uv run examples/terminal_bench/prepare_dataset.py --task_dir $HOME/data/terminal_bench/tasks --output_dir $HOME/data/terminal_bench --output_name train
# export WANDB_API_KEY=<your_key_here>
# bash examples/terminal_bench/run_tbench_megatron.sh
export WANDB_API_KEY="854a2b39e99ffee11c76d1003eb8a777045687e9"
export WANDB_ENTITY="bespoke-labs"

DATA_DIR="$HOME/ez_apex_250"
NUM_NODES=2
NUM_GPUS=8
LOGGER="wandb"  # change to "console" to print to stdout
TBENCH_CONFIG_DIR="examples/terminal_bench"
SANDBOXES_DIR="$HOME/SkyRL/skyrl-train/sandboxes"
MODEL_NAME="Qwen/Qwen3-Coder-30B-A3B-Instruct"

# Inference backend (for rollout generation)
INFERENCE_BACKEND="vllm"  # currently only vLLM is supported for Megatron in this setup

# Megatron parallelism (4 GPUs total => 2x TP, 2x PP, 1x CP)
MEGATRON_TP=2
MEGATRON_PP=2
MEGATRON_CP=4

MEGATRON_EP=8
MEGATRON_ETP=1

FLASH_ATTN=true
NUM_INFERENCE_ENGINES=8
INFERENCE_ENGINE_TP=2

# Torch profiler (optional)
ENABLE_TORCH_PROFILER=false
RANKS_TO_PROFILE="[0]"
SAVE_PATH="$HOME/megatron_prof/tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}_${MODEL_NAME}"

export NCCL_SOCKET_IFNAME=enp210s0f0np0
export GLOO_SOCKET_IFNAME=enp210s0f0np0
export NCCL_IB_HCA=mlx5_ib0
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3     
export NCCL_DEBUG=INFO

export SKYRL_PYTHONPATH_EXPORT=1
export PYTHONPATH="$HOME/SkyRL/skyrl-train/"
# export CUDNN_PATH="$(python -c 'import inspect, os, nvidia.cudnn as c; print(os.path.dirname(inspect.getfile(c)))')"
# export CUDNN_PATH="/home/user/SkyRL/skyrl-train/.venv/lib/python3.12/site-packages/nvidia/cudnn"
# export CUDNN_PATH="/opt/cudnn"
export CPATH="$CUDNN_PATH/include:${CPATH:-}"
export LD_LIBRARY_PATH="$CUDNN_PATH/lib:${LD_LIBRARY_PATH:-}"
export RAY_worker_register_timeout_seconds=1800
# export GLOO_SOCKET_IFNAME=enp27s0f0np0  #TODO: Make sure to change this for other nodes. Use this to check: ip -4 route
# export NCCL_SOCKET_IFNAME=enp27s0f0np0
# export CUDA_LAUNCH_BLOCKING=1
# export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
# export CUDA_COREDUMP_SHOW_PROGRESS=1
# export CUDA_COREDUMP_GENERATION_FLAGS='skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory'
# export CUDA_COREDUMP_FILE="/root/SkyRL/skyrl-train/cuda_coredump_%h.%p.%t"
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=INFO

export TOKENIZERS_PARALLELISM=false

# data.train_data="['$DATA_DIR/train.parquet']" \
uv run --isolated --extra $INFERENCE_BACKEND --extra sandboxes --extra mcore --with "sandboxes@./sandboxes" -m examples.terminal_bench.entrypoints.main_tbench \
  data.train_data="['$DATA_DIR']" \
  hydra.searchpath=[file://$TBENCH_CONFIG_DIR] \
  +terminal_bench_config=terminal_bench \
  +terminal_bench_config.sandboxes_dir=$SANDBOXES_DIR \
  terminal_bench_config.max_episodes=1 \
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
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity="full" \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_method="uniform" \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=1 \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=false \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=false \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=false \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=0.0 \
  trainer.use_sample_packing=true \
  trainer.flash_attn=$FLASH_ATTN \
  trainer.epochs=20 \
  trainer.eval_batch_size=32 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=1 \
  trainer.policy_mini_batch_size=1 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=32000 \
  generator.sampling_params.max_generate_length=32000 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=false \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=terminal_bench \
  generator.n_samples_per_prompt=8 \
  generator.gpu_memory_utilization=0.7 \
  trainer.logger="$LOGGER" \
  trainer.project_name="terminal_bench" \
  trainer.run_name="terminal_bench_megatron_H200_4_small" \
  trainer.ckpt_path="/data/ez_apex_250_small" \
  trainer.export_path="/data/hf_ckpt_small" \
  trainer.hf_save_interval=1 \
  trainer.ckpt_interval=1 \
  trainer.algorithm.eps_clip_low=0.2 \
  trainer.algorithm.eps_clip_high=0.28 \
  trainer.algorithm.loss_reduction="token_mean" \
  generator.apply_overlong_filtering=true \
  trainer.resume_mode=latest \
  $@


# ./run_tbench_megatron.sh 2>&1 | tee -a run_tbench_megatron.log
# trainer.resume_mode=latest \

