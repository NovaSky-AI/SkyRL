set -x

DATA_DIR="/mnt/shared_storage/reasoning_data/train_filtered_80_math/"
ONLINE_EVAL_DATA_DIR="/mnt/shared_storage/reasoning_data/online_eval_math/"

# needed for efa ... LD_LIBRARY_PATH gets reset even if placed in .env with uv rn
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/opt/amazon/efa/lib"

export train_files="[$(
  ls "${DATA_DIR}/"/*.parquet \
    | xargs -n1 basename \
    | sed "s|^|'${DATA_DIR}|;s|$|'|" \
    | paste -sd, -
)]"
echo "train_files = $train_files"

export test_files="[$(
  ls "${ONLINE_EVAL_DATA_DIR}/"/*.parquet \
    | xargs -n1 basename \
    | sed "s|^|'${ONLINE_EVAL_DATA_DIR}|;s|$|'|" \
    | paste -sd, -
)]"

echo "test_files = $test_files"

uv run --isolated --extra verl --env-file .env -m skyrl_agent.integrations.verl.verl_main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.train_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=35000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='S-ckpt2' \
    trainer.experiment_name='qwen3-8b_math_skyagent' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=2 \
    trainer.test_freq=2 \
    trainer.total_epochs=10 \
    trainer.default_local_dir=/mnt/shared_storage/checkpoints/qwen3-8b_math_skyagent_large_batch \
    +skyrl_agent.task_yaml="./examples/run_verl/verl_reasoning.yaml" \
    +skyrl_agent.num_trajectories=8 \
    +trainer.remote_anyscale_upload=True \
    +trainer.remote_upload_dir=remote_ckpts/qwen3-8b_math_skyagent_large_batch $@
