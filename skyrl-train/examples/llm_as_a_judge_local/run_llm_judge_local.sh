#!/bin/bash
set -x

# ============================================================================
# LLM-as-a-Judge with LOCAL vLLM reward model — GRPO on GSM8K
# ============================================================================
#
# This is the "local" variant of examples/llm_as_a_judge. Instead of calling
# an external API (e.g. OpenAI) for reward scoring, it uses a locally-hosted
# vLLM reward model managed by a FrozenRewardInferenceClient (a subclass of
# InferenceEngineClient with frozen weights — no weight sync, no sleep mode).
# No changes to SkyRL core are required.
#
# Prerequisites:
#   1. Prepare the dataset:
#      uv run examples/llm_as_a_judge_local/gsm8k_dataset_local.py \
#          --output_dir $HOME/data/gsm8k_llm_judge_local
#
#   2. Run training:
#      bash examples/llm_as_a_judge_local/run_llm_judge_local.sh
#
# What happens:
#   - main_llm_judge_local.py starts a RewardInferenceService Ray actor
#     wrapping a FrozenRewardInferenceClient
#   - The reward engine has no weight sync, no sleep — always active,
#     with automatic load balancing across engines
#   - Each env instance scores rewards via the named Ray actor — no HTTP,
#     no port conflicts, no stale subprocess issues
#   - No OPENAI_API_KEY needed; no external API costs
#
# GPU layout (2 nodes × 1 GPU each):
#   - Node 1: FSDP policy + vLLM inference (colocated, sleep/wake)
#   - Node 2: Frozen vLLM reward engine (scoring)
#
# Scaling the reward model:
#   Increase REWARD_NUM_ENGINES — each engine claims its own GPU and
#   load balancing is automatic via FrozenRewardInferenceClient.
#
# Works on any GPU with ≥16 GB VRAM and bfloat16 support (e.g. L4, A10G, A100).
# ============================================================================

DATA_DIR="$HOME/data/gsm8k_llm_judge_local"
CKPT_PATH="$HOME/ckpts/llm_judge_local"

# -- Cluster layout: 2 × 1 GPU (colocated policy+inference) --
NUM_POLICY_GPUS=1          # FSDP policy (dp_size=1, single GPU)
NUM_INFERENCE_ENGINES=1    # Policy generation engine
TP_SIZE=1                  # TP per engine
LOGGER=wandb

# -- Reward model (frozen, uses FrozenRewardInferenceClient — no weight sync) --
REWARD_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
REWARD_NUM_ENGINES=1       # Number of frozen vLLM engines (scale-out knob)
REWARD_MAX_MODEL_LEN=4096

uv run --isolated --extra vllm -m examples.llm_as_a_judge_local.main_llm_judge_local \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen2.5-0.5B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.placement.colocate_policy_ref=false \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_POLICY_GPUS \
  trainer.placement.critic_num_gpus_per_node=0 \
  trainer.placement.ref_num_gpus_per_node=0 \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$TP_SIZE \
  trainer.epochs=20 \
  trainer.eval_batch_size=2 \
  trainer.eval_before_train=false \
  trainer.eval_interval=50 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=8 \
  trainer.micro_forward_batch_size_per_gpu=8 \
  trainer.micro_train_batch_size_per_gpu=8 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=false \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  generator.n_samples_per_prompt=4 \
  generator.gpu_memory_utilization=0.7 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k" \
  trainer.run_name="gsm8k_llm_judge_local" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT_PATH" \
  environment.env_class=llm_as_a_judge_local \
  +environment.skyrl_gym.llm_as_a_judge_local.model="$REWARD_MODEL" \
  +environment.skyrl_gym.llm_as_a_judge_local.num_reward_engines=$REWARD_NUM_ENGINES \
  +environment.skyrl_gym.llm_as_a_judge_local.max_model_len=$REWARD_MAX_MODEL_LEN \
  +environment.skyrl_gym.llm_as_a_judge_local.temperature=0.0 \
  +environment.skyrl_gym.llm_as_a_judge_local.max_tokens=512 \
  "$@"
