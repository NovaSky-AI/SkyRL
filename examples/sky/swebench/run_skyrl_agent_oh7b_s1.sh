PROJECT_NAME='oh-7b-training'
EXPERIMENT_NAME='SkyRL-Agent-7b-v0-stage1'
DATA_PATH="<path_to_swegym_dataset>"
SFT_MODEL_PATH='NovaSky-AI/SWE-Gym-OpenHands-7B-Agent'
CKPT_PATH='<path_to_ckpt>'


BATCH_SIZE=16
MAX_NUM_ITERS=15
NUM_TRAJ=16
MAX_PARALLEL_AGENTS=64
SAVE_FREQ=1

USE_KL_LOSS=True
KL_LOSS_COEF=0.001
KL_LOSS_TYPE=low_var_kl
ENTROPY_COEFF=0
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.2

GPU_MEM_UTIL=0.8
TP_SIZE=1
# Assumes a h200 node
# For 2xH100: change tp size -> 2, sequence parallel size -> 2, nnodes -> 2
NNODES=1
SP_SIZE=1
TEMPERATURE=0.5
TOP_P=0.95



PYTHONUNBUFFERED=1 uv run --isolated --extra swebench --directory . --frozen --env-file .env.swebench -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=["$DATA_PATH/train.parquet"] \
    data.val_files=["$DATA_PATH/validation.parquet"] \
    data.train_batch_size=$BATCH_SIZE \
    data.max_prompt_length=31232 \
    data.max_response_length=1536 \
    data.truncation='error' \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP_SIZE \
    actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW \
    actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.name=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.n_trajectories=$NUM_TRAJ \
    actor_rollout_ref.rollout.sampling_params.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.sampling_params.top_p=$TOP_P \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    actor_rollout_ref.rollout.max_parallel_agents=$MAX_PARALLEL_AGENTS \
    actor_rollout_ref.rollout.max_iterations=$MAX_NUM_ITERS \
    actor_rollout_ref.rollout.enable_memory_saver=True \
    actor_rollout_ref.rollout.qwen3_enable_thinking=False \
    +actor_rollout_ref.rollout.max_starting_message_length=12000 \
    actor_rollout_ref.actor.masking=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager="swebench" \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME \
    trainer.resume_mode=auto \
    trainer.max_actor_ckpt_to_keep=10 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$NNODES \
    trainer.save_freq=$SAVE_FREQ \
    data.dataloader_num_workers=0 \
    actor_rollout_ref.exchange_size=500000000 \
    trainer.test_freq=-1 \
    trainer.total_epochs=100 $@