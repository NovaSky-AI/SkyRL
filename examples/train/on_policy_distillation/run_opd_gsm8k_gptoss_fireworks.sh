set -exo pipefail

# On-policy distillation smoke test with a fully serverless Fireworks teacher.
#   student: unsloth/gpt-oss-20b-BF16 (local, FSDP)   teacher: gpt-oss-120b on Fireworks serverless
# Both use the gpt-oss (o200k_harmony) tokenizer, which is what lets the teacher score the student's
# token ids verbatim. Same gpt-oss training flags as examples/train/gptoss/run_gsm8k_gptoss.sh.
#
# Serverless is fine for checking the plumbing; the startup self-test refuses a teacher whose
# logprobs differ across replicas by more than trainer.algorithm.opd.self_test_max_abs_diff nats.
# For a real run use a dedicated deployment (run_on_policy_distill_math_qwen3_4b.sh).
#
# uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export FIREWORKS_API_KEY=<your_key_here>
# export WANDB_API_KEY=<your_key_here>
# bash examples/train/on_policy_distillation/run_opd_gsm8k_gptoss_fireworks.sh

DATA_DIR="$HOME/data/gsm8k"
NUM_GPUS=8
LOGGER="wandb"  # change to "console" to print to stdout

STUDENT_MODEL="unsloth/gpt-oss-20b-BF16"
TEACHER_MODEL="accounts/fireworks/models/gpt-oss-120b"

uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_opd \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.policy.model.path="$STUDENT_MODEL" \
  trainer.teacher.backend=fireworks \
  trainer.teacher.model="$TEACHER_MODEL" \
  trainer.teacher.max_concurrency=32 \
  trainer.algorithm.opd.kl_coef=1.0 \
  trainer.algorithm.opd.use_task_reward=false \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=2 \
  generator.inference_engine.tensor_parallel_size=4 \
  trainer.flash_attn=false \
  trainer.remove_microbatch_padding=false \
  trainer.epochs=1 \
  trainer.eval_batch_size=32 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=32 \
  trainer.policy_mini_batch_size=32 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=5 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=2048 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.batched=false \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=4 \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  generator.chat_template_kwargs={reasoning_effort:'low'} \
  trainer.logger="$LOGGER" \
  trainer.project_name="opd_gsm8k_gptoss" \
  trainer.run_name="opd_gptoss_20b_from_120b_fireworks" \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$HOME/ckpts/opd_gsm8k_gptoss" \
  $@
