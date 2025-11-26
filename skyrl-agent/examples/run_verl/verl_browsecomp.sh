set -x

train_data="/mnt/shared_storage/browsecomp_plus/browsecomp_plus_train.parquet"
test_data="/mnt/shared_storage/browsecomp_plus/browsecomp_plus_skyagent.parquet"
# test_data="/mnt/shared_storage/browsecomp_plus/browsecomp_plus_test.parquet"


# MODEL=Qwen/Qwen3-8B
# MODEL=Qwen/Qwen3-32B
MODEL=/mnt/cluster_storage/ckpt/skyrl-agent-swe/step_125
NNODES=1
SP_SIZE=4
TP_SIZE=4

 
 uv run --isolated --extra verl --env-file .env -m skyrl_agent.integrations.verl.verl_main_ppo \
    algorithm.adv_estimator=loop \
    data.train_files=$train_data \
    data.val_files=$test_data \
    data.dataloader_num_workers=0 \
    data.train_batch_size=64 \
    data.val_batch_size=830 \
    data.max_prompt_length=8000 \
    data.max_response_length=32768 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=true \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=False \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='skyagent-32b-r2e' \
    trainer.experiment_name='browsecomp-eval-base' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$NNODES \
    trainer.max_actor_ckpt_to_keep=10 \
    trainer.save_freq=1 \
    trainer.default_local_dir=/mnt/local_storage/ckpts/skyagent-32b-r2e-verl/browsecomp-eval \
    trainer.test_freq=20 \
    trainer.total_epochs=15 \
    +skyrl_agent.task_yaml="./examples/run_verl/verl_browsecomp.yaml" \
    +skyrl_agent.num_trajectories=1 \
    +trainer.remote_anyscale_upload=True \
    +trainer.remote_upload_dir=remote_ckpts/skyagent-32b-r2e-verl/browsecomp-eval $@
