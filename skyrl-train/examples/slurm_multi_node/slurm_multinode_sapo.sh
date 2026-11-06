#!/bin/bash
#SBATCH --job-name=sapo-4B
#SBATCH --partition=main
#SBATCH --nodes=2                # Number of nodes
#SBATCH --ntasks-per-node=1      # One task per node
#SBATCH --cpus-per-task=128      # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=500:00:00
#SBATCH --output=logs/sapo/4B/dapo_math/%x_%j.out
#SBATCH --error=logs/sapo/4B/dapo_math/%x_%j.err

# This script runs the training of RL on multi-nodes. It does resume automatically from latest checkpoint if the run crashes.
# Example run with Qwen3-30B SAPO with new model engine

set -x

# Colocated DAPO training+generation for Qwen3-4B-Base on DAPO training data and validate on AIME 2024.
# sbatch slurm_dapo_multinode.sh

# Determine the script source directory
unset VIRTUAL_ENV

# Determine script source directory
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "$SLURM_SUBMIT_DIR"
    SCRIPT_SOURCE_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_SOURCE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi

REPO_ROOT=$(cd "$SCRIPT_SOURCE_DIR/../../.." && pwd)

# The skyrl-train directory is where pyproject.toml lives.
PROJECT_DIR="$REPO_ROOT/skyrl-train"

# Ray + uv REQUIRE working_dir == directory containing pyproject.toml
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

PYTHON="$REPO_ROOT/skyrl/bin/python"

echo "[INFO] REPO_ROOT=$REPO_ROOT"
echo "[INFO] PYTHON=$PYTHON"
echo "[INFO] PYTHONPATH=$PYTHONPATH"


# can make training faster depending on clusters
# this setting is optimized for H200 GPUs with NVLink and InfiniBand
export NCCL_IBEXT_DISABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_IB_HCA=mlx5
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1

# for triton
export TRITON_ALLOW_HOPPER=1

############################################

# Determine how many nodes were allocated. 
NNODES=${SLURM_JOB_NUM_NODES}
export NNODES

# Determine how many GPUs we actually have on the master node.
# Carefull! Assumes all nodes have same number of GPUs! 
# SLURM sets SLURM_GPUS_PER_NODE only when #SBATCH --gpus-per-node is used, not with --gres.
# uncomment below line to manually set number of gpus per node if not using --gpus-per-node
# export SLURM_GPUS_PER_NODE=8
# SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-$(nvidia-smi -L | wc -l)} # 8
# export SLURM_GPUS_PER_NODE
echo "SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "REPO_ROOT: $REPO_ROOT"

############################################
# 1.          Experiment Params            #
############################################

MODEL_NAME="Qwen/Qwen3-4B-Base"
DATA_DIR="$HOME/data/dapo"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"

# --- Resource Configuration ---
NUM_NODES=$NNODES                   # Use the SLURM count defined earlier
NUM_GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
NUM_INFERENCE_ENGINES=16
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=1

# --- DAPO / Algorithm Parameters ---
ADVANTAGE_ESTIMATOR="grpo"
POLICY_LOSS_TYPE="sapo"
LOSS_REDUCTION="sequence_mean" # must be sequence_mean for SAPO
USE_KL_LOSS=false
TAU_POS=1.0
TAU_NEG=1.05
CLIP_RATIO_C=10.0

# Overlong punishment settings for DAPO
APPLY_OVERLONG_FILTERING=false
OVERLONG_BUFFER_LEN=$((1024 * 4))
OVERLONG_BUFFER_PENALTY_FACTOR=1.0

# --- Generator / Sampling Parameters ---
TEMPERATURE=1.0
TOP_P=1.0
EVAL_TOP_P=0.7
MAX_PROMPT_LENGTH=$((1024 * 2))
MAX_RESPONSE_LENGTH=$((1024 * 8))
GENERATOR_BACKEND="vllm"
GPU_MEMORY_UTILIZATION=0.8
RUN_ENGINES_LOCALLY=true
WEIGHT_SYNC_BACKEND="nccl"
ASYNC_ENGINE=false
BATCHED=true
ENV_CLASS="aime"

# --- Training Hyperparameters ---
TRAIN_BATCH_SIZE=256
MINI_BATCH_SIZE=32
MICRO_FORWARD_BATCH_SIZE=8
MICRO_TRAIN_BATCH_SIZE=4
EPOCHS=20
LR=1e-6
NUM_WARMUP_STEPS=160
WEIGHT_DECAY=0.1
MAX_GRAD_NORM=1.0
STRATEGY="fsdp2"
COLOCATE_ALL=true
ENFORCE_EAGER=true # cuda graphs can cause some instability

# --- Evaluation & Samples ---
N_SAMPLES_PER_PROMPT=16
EVAL_N_SAMPLES_PER_PROMPT=32
EVAL_BATCH_SIZE=1024
EVAL_BEFORE_TRAIN=false
EVAL_INTERVAL=10
UPDATE_EPOCHS_PER_BATCH=1

# --- Logging & Checkpointing ---
export WANDB_API_KEY=.... # set your WANDB_API_KEY before running the script if using wandb
LOGGER="wandb"  # change to "console" to print to stdout
PROJECT_NAME="RL-PRs"
RUN_NAME="qwen3-4B-base-sapo-fsdp2-skyrl"
EXPORT_PATH="$HOME/exports/dapo_qwen3_4b_base_skyrl"
CKPT_PATH="$HOME/ckpts/dapo_qwen3_4b_base_skyrl"
CKPT_INTERVAL=-1
HF_SAVE_INTERVAL=-1
RESUME_MODE="latest"
MAX_CKPTS_TO_KEEP=3

############################################
# 2.           Start Ray cluster           #
############################################

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export RAY_memory_monitor_refresh_ms=0
export RAY_LOGGING_LEVEL=DEBUG
export HYDRA_FULL_ERROR=1

# Let Ray know how many nodes to expect
export RAY_NUM_NODES=$NNODES

############################################
# 3.          Start Ray Head               #
############################################

# Get head node and its IP
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Convert to IPv4 if needed
if [[ "$head_node_ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"$head_node_ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
  else
    head_node_ip=${ADDR[0]}
  fi
  echo "IPV6 address detected. Using IPV4: $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=$port
export ip_head

echo "Starting Ray HEAD at $head_node ($ip_head)"
until nvidia-smi > /dev/null 2>&1; do
  echo "Waiting for GPU visibility..."
  sleep 2
done
srun --nodes=1 --ntasks=1 -w "$head_node" \
  uv run --python "$PYTHON" --active -- ray start --head --node-ip-address="$head_node_ip" \
  --port=$port --num-cpus "$SLURM_CPUS_PER_TASK" --num-gpus "$SLURM_GPUS_PER_NODE" --block &

sleep 10

############################################
# 4.          Start Ray Workers            #
############################################

worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "Starting WORKER $i at $node_i"
  until nvidia-smi > /dev/null 2>&1; do
    echo "Waiting for GPU visibility..."
    sleep 2
  done
  srun --nodes=1 --ntasks=1 -w "$node_i" \
    uv run --python "$PYTHON" --active -- ray start --address "$ip_head" \
    --num-cpus "$SLURM_CPUS_PER_TASK" --num-gpus "$SLURM_GPUS_PER_NODE" --block &
  sleep 5
done

# Final launch barrier
sleep 10

############################################
# 5.    Confirm Ray cluster resources      #
############################################

uv run --python "$PYTHON" --active -- python - << 'EOF'
import ray
ray.init(address="auto")
print("Cluster resources:", ray.cluster_resources())
EOF

############################################
# 6.    Launch training script via srun    #
############################################

echo "Using $SLURM_NNODES nodes for training..."

srun --overlap --nodes=$NNODES --ntasks=1 -w "$head_node" \
    uv run --python "$PYTHON" --active --extra vllm \
    -m examples.algorithms.dapo.main_dapo \
    data.train_data="['$TRAIN_FILE']" \
    data.val_data="['$TEST_FILE']" \
    trainer.algorithm.advantage_estimator="$ADVANTAGE_ESTIMATOR" \
    trainer.algorithm.policy_loss_type="$POLICY_LOSS_TYPE" \
    +trainer.algorithm.overlong_buffer.len=$OVERLONG_BUFFER_LEN \
    +trainer.algorithm.overlong_buffer.penalty_factor=$OVERLONG_BUFFER_PENALTY_FACTOR \
    trainer.algorithm.loss_reduction=$LOSS_REDUCTION \
    generator.enforce_eager=$ENFORCE_EAGER \
    generator.apply_overlong_filtering=$APPLY_OVERLONG_FILTERING \
    generator.sampling_params.temperature=$TEMPERATURE \
    generator.sampling_params.top_p=$TOP_P \
    generator.eval_sampling_params.top_p=$EVAL_TOP_P \
    generator.eval_sampling_params.temperature=$TEMPERATURE \
    trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
    trainer.algorithm.clip_ratio_c=$CLIP_RATIO_C \
    trainer.policy.model.path="$MODEL_NAME" \
    trainer.placement.colocate_all=$COLOCATE_ALL \
    trainer.strategy="$STRATEGY" \
    trainer.placement.policy_num_nodes=$NUM_NODES \
    trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
    trainer.policy.fsdp_config.fsdp_size=$NUM_GPUS_PER_NODE \
    generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
    generator.inference_engine_tensor_parallel_size=$INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE \
    trainer.epochs=$EPOCHS \
    trainer.algorithm.tau_pos=$TAU_POS \
    trainer.algorithm.tau_neg=$TAU_NEG \
    trainer.eval_batch_size=$EVAL_BATCH_SIZE \
    trainer.eval_before_train=$EVAL_BEFORE_TRAIN \
    trainer.eval_interval=$EVAL_INTERVAL \
    trainer.update_epochs_per_batch=$UPDATE_EPOCHS_PER_BATCH \
    trainer.train_batch_size=$TRAIN_BATCH_SIZE \
    trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
    trainer.micro_forward_batch_size_per_gpu=$MICRO_FORWARD_BATCH_SIZE \
    trainer.micro_train_batch_size_per_gpu=$MICRO_TRAIN_BATCH_SIZE \
    trainer.ckpt_interval=$CKPT_INTERVAL \
    trainer.max_prompt_length=$MAX_PROMPT_LENGTH \
    generator.sampling_params.max_generate_length=$MAX_RESPONSE_LENGTH \
    trainer.policy.optimizer_config.lr=$LR \
    trainer.policy.optimizer_config.num_warmup_steps=$NUM_WARMUP_STEPS \
    trainer.policy.optimizer_config.weight_decay=$WEIGHT_DECAY \
    trainer.policy.optimizer_config.max_grad_norm=$MAX_GRAD_NORM \
    generator.backend="$GENERATOR_BACKEND" \
    generator.run_engines_locally=$RUN_ENGINES_LOCALLY \
    generator.weight_sync_backend="$WEIGHT_SYNC_BACKEND" \
    generator.async_engine=$ASYNC_ENGINE \
    generator.batched=$BATCHED \
    environment.env_class="$ENV_CLASS" \
    generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
    generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES_PER_PROMPT \
    generator.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    trainer.logger="$LOGGER" \
    trainer.project_name="$PROJECT_NAME" \
    trainer.run_name="$RUN_NAME" \
    trainer.export_path="$EXPORT_PATH" \
    trainer.hf_save_interval=$HF_SAVE_INTERVAL \
    trainer.resume_mode="$RESUME_MODE" \
    trainer.max_ckpts_to_keep=$MAX_CKPTS_TO_KEEP \
    trainer.ckpt_path="$CKPT_PATH" \
    $@