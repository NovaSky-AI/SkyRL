set -x

# SRPO (https://arxiv.org/abs/2604.02288) on the SDPO paper's Section 3 benchmarks:
# SciKnowEval biology, chemistry, material, and physics, and ToolAlpaca tool use.
# Same recipe as run_sdpo.sh, with sample routing: correct rollouts take a GRPO
# update, failed rollouts with a successful sibling take an entropy-weighted SDPO
# update. Uses 4 GPUs: 2 FSDP training, 2 vLLM. Sized for 141 GB GPUs (H200); on
# 80 GB GPUs, Qwen3-8B needs more training GPUs (NUM_POLICY_GPUS).
#
# Usage:
#
# git clone https://github.com/lasgroup/SDPO $HOME/SDPO
# uv run --isolated --extra fsdp -m examples.train.sdpo.reproduction.prepare_data \
#   --sdpo-repo $HOME/SDPO --output-dir $HOME/data/sdpo
# export WANDB_API_KEY=<your_key_here>
# bash examples/train/sdpo/reproduction/run_srpo.sh chemistry [extra overrides...]
#
# Tasks: biology, chemistry, material, physics, tooluse.
# For OLMo-3, set MODEL=allenai/Olmo-3-7B-Instruct.

TASK=${1:-chemistry}
shift || true
case "$TASK" in biology|chemistry|material|physics|tooluse) ;; *) echo "Unknown task: $TASK" >&2; exit 1 ;; esac

DATA_DIR=${DATA_DIR:-"$HOME/data/sdpo"}
MODEL=${MODEL:-"Qwen/Qwen3-8B"}
SEED=${SEED:-42}
OUTPUT_DIR=${OUTPUT_DIR:-"$HOME/outputs/srpo/$TASK/seed_$SEED"}
LOGGER=${LOGGER:-wandb}

NUM_POLICY_GPUS=${NUM_POLICY_GPUS:-2}
NUM_INFERENCE_ENGINES=${NUM_INFERENCE_ENGINES:-2}

uv run --isolated --extra fsdp -m examples.train.sdpo.reproduction.main_reproduction \
  data.train_data="['$DATA_DIR/$TASK/train.jsonl']" \
  data.val_data="['$DATA_DIR/$TASK/test.jsonl']" \
  data.dataloader.num_workers=0 \
  trainer.policy.model.path=$MODEL \
  trainer.seed=$SEED \
  trainer.strategy=fsdp \
  trainer.placement.colocate_all=false \
  trainer.placement.colocate_policy_ref=false \
  trainer.placement.policy_num_gpus_per_node=$NUM_POLICY_GPUS \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.55 \
  generator.inference_engine.engine_init_kwargs='{"max_model_len":18944,"max_num_batched_tokens":18944,"disable_custom_all_reduce":true}' \
  trainer.epochs=30 \
  trainer.train_batch_size=32 \
  trainer.policy_mini_batch_size=32 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.remove_microbatch_padding=false \
  trainer.update_epochs_per_batch=1 \
  trainer.max_prompt_length=2048 \
  trainer.policy.optimizer_config.lr=1e-5 \
  trainer.policy.optimizer_config.num_warmup_steps=10 \
  trainer.policy.optimizer_config.scheduler=constant_with_warmup \
  trainer.policy.optimizer_config.weight_decay=0.01 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.policy.optimizer_config.adam_betas='[0.9,0.999]' \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.distillation.use_sample_routing=true \
  trainer.distillation.use_entropy_weighting=true \
  trainer.distillation.entropy_beta=1.0 \
  trainer.distillation.teacher.kind=ema \
  trainer.distillation.teacher.update_rate=0.05 \
  trainer.distillation.teacher.storage=cuda \
  trainer.distillation.loss=distribution \
  trainer.distillation.alpha=0.5 \
  trainer.distillation.topk=100 \
  trainer.distillation.add_tail=true \
  trainer.distillation.reduction=upstream_microbatch_mean \
  trainer.distillation.rollout_is_clip=2.0 \
  trainer.distillation.success_reward_threshold=0.5 \
  trainer.distillation.exclude_self_demonstration=true \
  trainer.distillation.remove_thinking=true \
  trainer.distillation.include_environment_feedback=false \
  trainer.distillation.filter_ineligible_demonstrations=false \
  trainer.distillation.max_teacher_prompt_length=10240 \
  trainer.distillation.max_teacher_sequence_length=18944 \
  trainer.distillation.overflow=truncate_right \
  trainer.distillation.chat_template_kwargs='{"enable_thinking":false}' \
  trainer.distillation.reprompt_template='"{prompt}{solution}{feedback}\n\nCorrectly solve the original question."' \
  trainer.distillation.solution_template='"\nCorrect solution:\n\n{successful_previous_attempt}"' \
  trainer.distillation.feedback_template='"\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}"' \
  generator.n_samples_per_prompt=8 \
  generator.max_turns=1 \
  generator.batched=false \
  generator.chat_template_kwargs='{"enable_thinking":false}' \
  generator.apply_overlong_filtering=false \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.top_k=-1 \
  generator.sampling_params.max_generate_length=8192 \
  generator.sampling_params.logprobs=0 \
  generator.eval_n_samples_per_prompt=16 \
  generator.eval_sampling_params.temperature=0.6 \
  generator.eval_sampling_params.top_p=0.95 \
  generator.eval_sampling_params.top_k=-1 \
  generator.eval_sampling_params.max_generate_length=8192 \
  trainer.eval_batch_size=32 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.ckpt_interval=-1 \
  trainer.hf_save_interval=-1 \
  trainer.resume_mode=none \
  trainer.logger=$LOGGER \
  trainer.project_name=sdpo \
  trainer.run_name=srpo-$TASK-seed$SEED \
  trainer.ckpt_path=$OUTPUT_DIR/checkpoints \
  trainer.export_path=$OUTPUT_DIR/exports \
  trainer.log_path=$OUTPUT_DIR/logs \
  "$@"
