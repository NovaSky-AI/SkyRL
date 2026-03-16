set -x

# GOLD Cross-Tokenizer Distillation: Qwen Teacher -> Llama Student
# Uses Llama-3.2-3B-Instruct as the student model and Qwen3-4B-Instruct as the teacher model
# bash examples/train/algorithms/dapo/prepare_dapo_data.sh
# bash examples/train/gold_distillation/run_gold_distill_qwen_to_llama.sh

DATA_DIR="${DATA_DIR:-$HOME/data/dapo}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/dapo-math-17k-cleaned.parquet}"
TEST_FILE="${TEST_FILE:-$DATA_DIR/aime-2024-cleaned.parquet}"
LOGGER=wandb

# Cross-tokenizer distillation args
# set this to the huggingface/local path of your teacher model
TEACHER_MODEL="Qwen/Qwen3-4B-Instruct-2507"
STUDENT_MODEL="meta-llama/Llama-3.2-3B-Instruct"

# GOLD supervised distillation — no RL rewards, no KL loss
ADVANTAGE_ESTIMATOR="no_op"
USE_KL_IN_REWARD=false
USE_KL_LOSS=false

# Placement args
: "${NUM_GPUS_PER_NODE:=2}"
: "${NUM_INFERENCE_ENGINES:=2}"
INFERENCE_ENGINE_TP_SIZE=1

# sampling params
TEMPERATURE=1.0
TOP_P=1.0
EVAL_TOP_P=0.7

# training parameters
TRAIN_BATCH_SIZE=128
MINI_BATCH_SIZE=128
N_SAMPLES_PER_PROMPT=4
EVAL_N_SAMPLES_PER_PROMPT=8
ENFORCE_EAGER=true
LR=1e-5

uv run --isolated --extra fsdp -m examples.train.gold_distillation.main_gold_distill \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$TEST_FILE']" \
  trainer.algorithm.advantage_estimator=$ADVANTAGE_ESTIMATOR \
  trainer.policy.model.path=$STUDENT_MODEL \
  trainer.ref.model.path=$TEACHER_MODEL \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TP_SIZE \
  trainer.epochs=20 \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.ckpt_interval=-1 \
  trainer.max_prompt_length=512 \
  generator.inference_engine.enforce_eager=$ENFORCE_EAGER \
  generator.sampling_params.max_generate_length=1024 \
  generator.sampling_params.temperature=$TEMPERATURE \
  generator.sampling_params.top_p=$TOP_P \
  generator.eval_sampling_params.temperature=$TEMPERATURE \
  generator.eval_sampling_params.top_p=$EVAL_TOP_P \
  generator.eval_sampling_params.max_generate_length=1024 \
  generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES_PER_PROMPT \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.num_warmup_steps=0 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.use_kl_in_reward=$USE_KL_IN_REWARD \
  trainer.algorithm.gold_beta=0.0 \
  trainer.algorithm.gold_distillation_weight=1.0 \
  trainer.algorithm.gold_matched_weight=1.0 \
  trainer.algorithm.gold_unmatched_weight=1.0 \
  trainer.algorithm.gold_crossentropy_weight=0.0 \
  trainer.algorithm.gold_student_temperature=1.0 \
  trainer.algorithm.gold_teacher_temperature=1.0 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.async_engine=false \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gold_distillation" \
  trainer.run_name="gold_qwen3_4b_to_llama_3b" \
  trainer.resume_mode=latest \
  trainer.export_path="$HOME/exports/gold_qwen_to_llama" \
  trainer.hf_save_interval=-1 \
  trainer.max_ckpts_to_keep=-1 \
  trainer.ckpt_path="$HOME/ckpts/gold_qwen_to_llama" \
  $@
