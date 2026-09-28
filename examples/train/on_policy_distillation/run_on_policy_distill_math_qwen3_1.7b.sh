set -x

# On-policy distillation for math on the DAPO dataset, eval on AIME 2024.
# Student: Qwen3-1.7B-Base. Teacher: a Qwen3-family model that shares the student's tokenizer,
# launched by the job (TEACHER_BACKEND=skyrl, default) or a dedicated Fireworks deployment
# (TEACHER_BACKEND=fireworks); see run_on_policy_distill_math_qwen3_4b.sh for the placements.
#
# bash examples/train/algorithms/dapo/prepare_dapo_data.sh
# bash examples/train/on_policy_distillation/run_on_policy_distill_math_qwen3_1.7b.sh
# export FIREWORKS_API_KEY=<your_key_here>
# TEACHER_BACKEND=fireworks TEACHER_MODEL=accounts/<account>/deployments/<id> bash examples/train/on_policy_distillation/run_on_policy_distill_math_qwen3_1.7b.sh

DATA_DIR="$HOME/data/dapo"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"
LOGGER=wandb

STUDENT_MODEL="Qwen/Qwen3-1.7B-Base"
TEACHER_MAX_CONCURRENCY=64

# Teacher placement (see the header). NUM_GPUS_PER_NODE is the student's share of the node; with the
# launched teacher the rest of the node is the teacher's (TEACHER_NUM_ENGINES * TEACHER_TP_SIZE GPUs).
TEACHER_BACKEND="${TEACHER_BACKEND:-skyrl}"
case "$TEACHER_BACKEND" in
  skyrl)
    TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-32B}"
    TEACHER_NUM_ENGINES="${TEACHER_NUM_ENGINES:-1}"
    TEACHER_TP_SIZE="${TEACHER_TP_SIZE:-4}"
    NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-4}"
    TEACHER_ARGS=(
      trainer.teacher.backend=skyrl
      trainer.teacher.model="$TEACHER_MODEL"
      trainer.teacher.inference_engine.num_engines="$TEACHER_NUM_ENGINES"
      trainer.teacher.inference_engine.tensor_parallel_size="$TEACHER_TP_SIZE"
    )
    ;;
  fireworks)
    TEACHER_MODEL="${TEACHER_MODEL:?set TEACHER_MODEL to a Fireworks model or deployment id}"
    NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
    TEACHER_ARGS=(
      trainer.teacher.backend=fireworks
      trainer.teacher.model="$TEACHER_MODEL"
    )
    ;;
  *)
    echo "TEACHER_BACKEND must be 'skyrl' (launched by the job) or 'fireworks', got '$TEACHER_BACKEND'" >&2
    exit 1
    ;;
esac

# On-policy distillation args
KL_COEF=1.0             # advantages -= KL_COEF * (log pi_student - log pi_teacher)
USE_TASK_REWARD=false   # false: pure distillation; true: teacher term on top of the GRPO advantages

# Placement args (student; colocated, one engine per GPU)
NUM_INFERENCE_ENGINES="$NUM_GPUS_PER_NODE"
INFERENCE_ENGINE_TP_SIZE=1

# sampling params
TEMPERATURE=1.0
TOP_P=1.0
EVAL_TOP_P=0.7

# repro run parameters
TRAIN_BATCH_SIZE=512
MINI_BATCH_SIZE=512
N_SAMPLES_PER_PROMPT=16
EVAL_N_SAMPLES_PER_PROMPT=32
ENFORCE_EAGER=false
LR=1e-5

uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_opd \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$TEST_FILE']" \
  trainer.policy.model.path=$STUDENT_MODEL \
  "${TEACHER_ARGS[@]}" \
  trainer.teacher.max_concurrency=$TEACHER_MAX_CONCURRENCY \
  trainer.algorithm.opd.kl_coef=$KL_COEF \
  trainer.algorithm.opd.use_task_reward=$USE_TASK_REWARD \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TP_SIZE \
  trainer.epochs=20 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$MINI_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.max_prompt_length=2048 \
  generator.inference_engine.enforce_eager=$ENFORCE_EAGER \
  generator.sampling_params.max_generate_length=8192 \
  generator.sampling_params.temperature=$TEMPERATURE \
  generator.sampling_params.top_p=$TOP_P \
  generator.eval_sampling_params.temperature=$TEMPERATURE \
  generator.eval_sampling_params.top_p=$EVAL_TOP_P \
  generator.eval_sampling_params.max_generate_length=8192 \
  generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES_PER_PROMPT \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.policy.optimizer_config.num_warmup_steps=0 \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.batched=true \
  environment.env_class=aime \
  generator.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="aime_on_policy_distillation" \
  trainer.run_name="on_policy_distillation_aime_qwen3_1.7b_base" \
  trainer.resume_mode=latest \
  trainer.export_path="$HOME/exports/aime_on_policy_distill_1.7b_base" \
  trainer.hf_save_interval=10 \
  trainer.max_ckpts_to_keep=3 \
  trainer.ckpt_interval=10 \
  trainer.ckpt_path="$HOME/ckpts/aime_on_policy_distill_1.7b_base" \
  $@
