set -x

# Colocated GRPO training+generation for Qwen2.5-Coder-7B-Instruct on SkyRL-SQL-653 data using Megatron and LoRA.
# Uses 1 node with 8 GPUs.
# huggingface-cli download NovaSky-AI/SkyRL-SQL-653-data-newfmt --local-dir $HOME/data/sql --repo-type dataset
# export WANDB_API_KEY=<your_key_here>
# bash examples/text_to_sql/run_skyrl_sql_megatron_lora.sh

# change these paths to your own
DATA_DIR="$HOME/data/sql"
DB_PATH="$HOME/data/sql/db_files/data"
CKPT_PATH="$HOME/ckpts/skyrl_sql_7B_ckpt"

LORA_RANK=32
LORA_ALPHA=64
LORA_A_INIT_METHOD="kaiming"

NUM_NODES=1
NUM_GPUS=8
NUM_INFERENCE_ENGINES=2
TP_SIZE=4
MAX_INPUT_LENGTH=29000
MAX_GENERATE_LENGTH=3000
TRAIN_BATCH_SIZE=256

# Megatron parameters
MEGATRON_TP=4
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=1
MEGATRON_ETP=null

# TIS parameters
TIS_IMP_RATIO_CAP=2.0
TIS_TYPE=token

uv run --isolated --extra mcore -m skyrl_train.entrypoints.main_base \
  trainer.algorithm.advantage_estimator="grpo" \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.policy.model.path="Qwen/Qwen2.5-Coder-7B-Instruct" \
  trainer.policy.model.lora.rank=$LORA_RANK \
  trainer.policy.model.lora.alpha=$LORA_ALPHA \
  trainer.policy.model.lora.init_method=$LORA_A_INIT_METHOD \
  trainer.epochs=30 \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$TP_SIZE \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.max_prompt_length=6000 \
  generator.max_input_length=$MAX_INPUT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_GENERATE_LENGTH \
  trainer.policy.optimizer_config.lr=3.0e-5 \
  trainer.policy_mini_batch_size=256 \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.off_policy_correction.tis_ratio_type=$TIS_TYPE \
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=$TIS_IMP_RATIO_CAP \
  trainer.ckpt_interval=60 \
  trainer.hf_save_interval=30 \
  trainer.dump_data_batch=true \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  environment.env_class=text2sql \
  generator.use_conversation_multi_turn=false \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.7 \
  generator.max_turns=6 \
  generator.sampling_params.temperature=0.6 \
  generator.sampling_params.top_p=0.95 \
  generator.sampling_params.stop='["</sql>", "</solution>"]' \
  generator.eval_sampling_params.stop='["</sql>", "</solution>"]' \
  environment.skyrl_gym.text2sql.db_path=$DB_PATH \
  trainer.logger="wandb" \
  trainer.project_name="skyrlsql" \
  trainer.run_name="skyrlsql_repro_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}_lora_rank${LORA_RANK}_alpha${LORA_ALPHA}" \
  trainer.resume_mode=latest \
  trainer.ckpt_path=$CKPT_PATH \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  $@