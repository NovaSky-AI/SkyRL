set -x

DATA_DIR="/mnt/shared_storage/datasets/r2e-all"
TRAIN_DATA="${DATA_DIR}/train.parquet"
VAL_DATA="${DATA_DIR}/validation.parquet"

CKPT_DIR=$HOME/ckpts
EXPORT_DIR=$HOME/exports


MODEL=Qwen/Qwen3-32B
NNODES=2
SP_SIZE=4
TP_SIZE=4
NUM_GPUS=8
NUM_INFERENCE_ENGINES=4
BATCH_SIZE=64
LOGGER=wandb
INFERENCE_BACKEND=vllm
seed=1

# export LD_LIBRARY_PATH="/opt/amazon/efa/lib:$LD_LIBRARY_PATH"

uv run --isolated --env-file .env --extra skyrl-train \
    --with vllm==0.9.2 \
    --with transformers==4.53.0 \
    --with torch==2.7.0 \
    --with "flash-attn@https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
    -m skyrl_agent.integrations.skyrl_train.skyrl_train_main  \
  data.train_data="['$TRAIN_DATA']" \
  data.val_data="['$VAL_DATA']" \
  trainer.algorithm.advantage_estimator="loop" \
  trainer.policy.model.path=$MODEL \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  trainer.placement.policy_num_nodes=$NNODES \
  trainer.placement.ref_num_nodes=$NNODES \
  generator.inference_engine_tensor_parallel_size=$TP_SIZE \
  +generator.task="./examples/run_skyrl/skyrl_swe.yaml" \
  trainer.epochs=10 \
  trainer.seed=$seed \
  trainer.eval_batch_size=128 \
  trainer.eval_before_train=false \
  trainer.eval_interval=10 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$BATCH_SIZE \
  trainer.policy_mini_batch_size=$BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=2 \
  trainer.max_ckpts_to_keep=20 \
  trainer.max_prompt_length=8000 \
  generator.sampling_params.max_generate_length=32768 \
  generator.enforce_eager=false \
  generator.enable_prefix_caching=true \
  trainer.algorithm.policy_loss_type="dual_clip" \
  trainer.policy.optimizer_config.lr=1e-6 \
  trainer.policy.sequence_parallel_size=$SP_SIZE \
  trainer.ref.sequence_parallel_size=$SP_SIZE \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.algorithm.eps_clip_low=0.2 \
  trainer.algorithm.eps_clip_high=0.28 \
  trainer.algorithm.loss_reduction="seq_mean_token_sum_norm" \
  trainer.algorithm.grpo_norm_by_std=false \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=null \
  generator.n_samples_per_prompt=8 \
  generator.eval_n_samples_per_prompt=1 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="skyagent-32b-r2e-skyrl" \
  trainer.run_name="skyagent-skyrl-32b-r2e-4500-loop-tool" \
  trainer.ckpt_path="$CKPT_DIR" \
  trainer.export_path="$EXPORT_DIR" \
  trainer.dump_data_batch=true \
  generator.max_num_batched_tokens=16384 \
  $@
