set -exo pipefail

# Colocated GRPO training+generation for GPT-OSS-20B on multi-turn GSM8K.
# This is the minimal GPT-OSS multi-turn recipe in the repo: it reuses the built-in
# `gsm8k_multi_turn` environment instead of adding extra infra like search or SQL servers.
#
# GPT-OSS-specific notes:
# - GPT-OSS currently requires flash attention to be disabled because attention sinks are not
#   supported there yet: https://github.com/Dao-AILab/flash-attention/issues/1797
# - We also disable sample packing for the same reason.
# - GPT-OSS support in SkyRL requires a Transformers version new enough to expose `GptOssConfig`
#   (the runtime patch path currently gates on >= 4.56.2).
#
# Dataset setup:
#   uv run examples/train/turn_level_rewards/gsm8k_multi_turn_dataset.py \
#     --output_dir $HOME/data/gsm8k_multi_turn \
#     --max_turns 5
#
# Usage:
#   export WANDB_API_KEY=<your_key_here>
#   bash examples/train/gptoss/run_gsm8k_multi_turn_gptoss.sh
#
# NOTE: If you generated the dataset with a different `--max_turns`, keep MAX_TURNS below in sync.

DATA_DIR="$HOME/data/gsm8k_multi_turn"
CKPT_PATH="$HOME/ckpts/gsm8k_multi_turn_gptoss_ckpt"
NUM_GPUS=8
MAX_TURNS=5
MAX_INPUT_LENGTH=4096
MAX_GENERATE_LENGTH=1024
TRAIN_BATCH_SIZE=16
LOGGER="wandb"  # change to "console" to print to stdout
INFERENCE_BACKEND="vllm"

uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="unsloth/gpt-oss-20b-BF16" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=2 \
  trainer.flash_attn=false \
  trainer.use_sample_packing=false \
  generator.inference_engine.tensor_parallel_size=4 \
  generator.inference_engine.enforce_eager=true \
  trainer.epochs=20 \
  trainer.eval_batch_size=16 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$TRAIN_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=5 \
  trainer.max_prompt_length=512 \
  generator.max_input_length=$MAX_INPUT_LENGTH \
  generator.max_turns=$MAX_TURNS \
  generator.sampling_params.max_generate_length=$MAX_GENERATE_LENGTH \
  generator.use_conversation_multi_turn=true \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=false \
  environment.env_class=gsm8k_multi_turn \
  generator.n_samples_per_prompt=2 \
  generator.inference_engine.gpu_memory_utilization=0.7 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k_multi_turn_gptoss" \
  trainer.run_name="gsm8k_multi_turn_gptoss_low" \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$CKPT_PATH" \
  generator.chat_template_kwargs={reasoning_effort:'low'} \
  trainer.dump_data_batch=true \
  $@
