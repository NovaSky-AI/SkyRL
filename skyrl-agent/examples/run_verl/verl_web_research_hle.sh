#!/bin/bash
set -x

# Basic environment setup
export PYTHONUNBUFFERED=1
export RUST_BACKTRACE=1
export HYDRA_FULL_ERROR=1
# increase ulimit for ray if needed
# ulimit -n 65535


# Default data/checkpoint directories (override via env if needed)
DATA_DIR=${DATA_DIR:-"$HOME/data/web_research"}
mkdir -p "$DATA_DIR"

DATA_FILE=${DATA_FILE:-"$DATA_DIR/textbook_stem_5k_with_instance.parquet"}
VAL_FILE=${VAL_FILE:-"$DATA_DIR/hle_webthinker_converted.parquet"}

TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR:-"$DATA_DIR/ckpts"}
ROLLOUT_DIR=${ROLLOUT_DIR:-"$DATA_DIR/rollouts/trainwithoss20b"}
VAL_ROLLOUT_DIR=${VAL_ROLLOUT_DIR:-"$DATA_DIR/rollouts/web_research_val_tigerb_hlestart_70ckpt"}
mkdir -p "$TRAIN_OUTPUT_DIR" "$ROLLOUT_DIR" "$VAL_ROLLOUT_DIR"

# Load environment variables from .env file
if [ -f .env ]; then
    set -a  # automatically export all variables
    source .env
    set +a
fi

# Set a writable, shared web cache directory for the web_browser tool
mkdir -p "${SKYAGENT_WEB_CACHE_DIR:-$DATA_DIR/web_cache}" 

# Optimize HuggingFace model loading
# export HF_HUB_OFFLINE=1  # Commented out to allow model download if needed
# export TRANSFORMERS_OFFLINE=1  # Commented out to allow online checks
# export HF_HOME=~/.cache/huggingface  # Use default cache location where models exist

uv run --isolated --frozen --extra verl --env-file .env -m skyrl_agent.integrations.verl.verl_main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${DATA_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.max_prompt_length=8192 \
    data.prompt_key=prompt \
    data.return_raw_chat=true \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.train_batch_size=64 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.prompt_length=8192 \
    actor_rollout_ref.rollout.response_length=32000 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.clip_ratio_low=0.4 \
    actor_rollout_ref.actor.clip_ratio_high=0.4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=11000 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=14000 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=14000 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=8 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    algorithm.use_kl_in_reward=False \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='skyagent_websearch' \
    trainer.experiment_name='train-64n8-withoss20b-verl' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=50 \
    data.val_batch_size=500 \
    trainer.total_epochs=10 \
    trainer.ray_wait_register_center_timeout=900 \
    trainer.default_local_dir="${TRAIN_OUTPUT_DIR}" \
    trainer.rollout_data_dir="${ROLLOUT_DIR}" \
    trainer.validation_data_dir="${VAL_ROLLOUT_DIR}" \
    +skyrl_agent.task_yaml="./examples/run_verl/verl_web_research_hle.yaml" \
    +skyrl_agent.num_trajectories=8 "$@"
