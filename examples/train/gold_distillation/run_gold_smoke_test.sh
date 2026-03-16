set -x

# GOLD Cross-Tokenizer Distillation Smoke Test
# Minimal run to verify the training loop works end-to-end.
#
# For Modal:
#   cd examples/train_integrations/modal && \
#   MODAL_GPU=A100:1 modal run main.py \
#     --command "cd /root/SkyRL && \
#       uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir /root/data/gsm8k && \
#       bash examples/train/gold_distillation/run_gold_smoke_test.sh"

# Data - defaults to Modal paths, override with env vars
: "${DATA_DIR:=/root/data/gsm8k}"
: "${CHECK_POINT_DIR:=/root/data/ckpts/gold_smoke_test}"
: "${EXPORT_PATH:=/root/data/export/gold_smoke_test}"
: "${LOGGER:=wandb}"

# Cross-tokenizer distillation args
TEACHER_MODEL="Qwen/Qwen3-4B-Instruct-2507"
STUDENT_MODEL="meta-llama/Llama-3.2-3B-Instruct"

# GOLD supervised distillation — no RL rewards, no KL loss
ADVANTAGE_ESTIMATOR="no_op"
USE_KL_IN_REWARD=false
USE_KL_LOSS=false

# Placement args
: "${NUM_GPUS:=1}"

# Small smoke test parameters
TRAIN_BATCH_SIZE=16
MINI_BATCH_SIZE=16
N_SAMPLES_PER_PROMPT=2
TEMPERATURE=1.0
LR=1e-5

uv run --isolated --extra fsdp -m examples.train.gold_distillation.main_gold_distill \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator=$ADVANTAGE_ESTIMATOR \
  trainer.policy.model.path=$STUDENT_MODEL \
  trainer.ref.model.path=$TEACHER_MODEL \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  trainer.epochs=2 \
  trainer.eval_batch_size=16 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.ckpt_interval=-1 \
  trainer.max_prompt_length=512 \
  generator.inference_engine.enforce_eager=true \
  generator.sampling_params.max_generate_length=256 \
  generator.sampling_params.temperature=$TEMPERATURE \
  generator.sampling_params.top_p=1.0 \
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
  generator.inference_engine.gpu_memory_utilization=0.5 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gold_distillation" \
  trainer.run_name="gold_smoke_test" \
  trainer.resume_mode=null \
  trainer.ckpt_path=$CHECK_POINT_DIR \
  trainer.export_path=$EXPORT_PATH \
  $@
