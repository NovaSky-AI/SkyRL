set -x

# Colocated GRPO training+generation for Qwen/Qwen3.5-35B-A3B on GSM8K with
# FSDP sample-support replay. Runs on one node of 8xB200s.
#
# Packed inputs are enabled by default; set
# REMOVE_MICROBATCH_PADDING=false to disable them.
#
# uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/train/top_p_replay/run_qwen35b_top_p_replay_fsdp.sh

DATA_DIR="${DATA_DIR:-/root/data/gsm8k}"
LOGGER="${LOGGER:-wandb}"  # change to "console" to print to stdout
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-35B-A3B}"

NUM_NODES=1
NUM_GPUS=8

NUM_INFERENCE_ENGINES=1
INFERENCE_ENGINE_TP=8

# Train Qwen3.5's text backbone.
LANGUAGE_MODEL_ONLY=true
REMOVE_MICROBATCH_PADDING="${REMOVE_MICROBATCH_PADDING:-true}"
ENGINE_INIT_KWARGS='{"gdn_prefill_backend": "triton", "kernel_config": {"moe_backend": "triton"}}'

# Use Triton GDN on Blackwell.
export FLA_TILELANG=0

# Allow model download and compilation during engine startup.
export SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=3600

# Capture the bounded post-filter support set for replay.
SAMPLE_SUPPORT_REPLAY="${SAMPLE_SUPPORT_REPLAY:-true}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MAX_TRAINING_STEPS="${MAX_TRAINING_STEPS:-40}"
RUN_LABEL="${RUN_LABEL:-top_p${TOP_P}_top_k${TOP_K}_sample_support_${SAMPLE_SUPPORT_REPLAY}}"
DISTRIBUTED_EXECUTION_BACKEND="${DISTRIBUTED_EXECUTION_BACKEND:-mp}"
CKPT_PATH="${CKPT_PATH:-$HOME/ckpts/gsm8k_fsdp_top_p_replay_ckpt}"

SKYRL_RAY_PG_TIMEOUT_IN_S=300 uv run --isolated --extra fsdp --with blobfile -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.policy.language_model_only=$LANGUAGE_MODEL_ONLY \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TP \
  trainer.algorithm.enable_sample_support_replay=$SAMPLE_SUPPORT_REPLAY \
  generator.inference_engine.distributed_executor_backend=$DISTRIBUTED_EXECUTION_BACKEND \
  generator.inference_engine.enable_return_sample_support_set=$SAMPLE_SUPPORT_REPLAY \
  "generator.inference_engine.engine_init_kwargs=$ENGINE_INIT_KWARGS" \
  trainer.remove_microbatch_padding=$REMOVE_MICROBATCH_PADDING \
  trainer.flash_attn=true \
  trainer.gradient_checkpointing=true \
  trainer.logprobs_chunk_size=256 \
  trainer.epochs=20 \
  trainer.max_training_steps=$MAX_TRAINING_STEPS \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=32 \
  trainer.max_tokens_per_microbatch=8192 \
  trainer.ckpt_interval=100 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  generator.sampling_params.top_p=$TOP_P \
  generator.sampling_params.top_k=$TOP_K \
  generator.use_conversation_multi_turn=true \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=false \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.inference_engine.gpu_memory_utilization=0.6 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k_top_p_replay" \
  trainer.run_name="gsm8k_fsdp8_packed${REMOVE_MICROBATCH_PADDING}_qwen35b-a3b_${RUN_LABEL}" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$CKPT_PATH" \
  generator.inference_engine.language_model_only=$LANGUAGE_MODEL_ONLY \
  $@
