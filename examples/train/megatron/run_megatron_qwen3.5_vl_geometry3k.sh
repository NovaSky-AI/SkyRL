set -x

# Multi-turn GRPO for Qwen3.5-9B (a VLM, vision tower trained) on Geometry-3K with Megatron,
# with sequence packing and context parallelism. Should run on 1 node of 8xH100s.
#
# uv run examples/train/geometry3k/geometry_3k_dataset.py --output_dir $HOME/data/geometry_3k
# export WANDB_API_KEY=<your_key_here>
# bash examples/train/megatron/run_megatron_qwen3.5_vl_geometry3k.sh

MODEL_NAME="Qwen/Qwen3.5-9B"
DATA_DIR="$HOME/data/geometry_3k"
NUM_GPUS=8
LOGGER="wandb"  # change to "console" to print to stdout

# --- VLM flags (Megatron) ---
# Train the vision tower too. With true, the model is the text-only GPTModel and images are ignored.
LANGUAGE_MODEL_ONLY=false
# Sequence packing (THD). Supported for VLMs on Megatron-Bridge's Qwen3VLModel (Qwen3-VL, Qwen3.5),
# which rebuilds mRoPE positions per packed sample. Required for context parallelism.
REMOVE_MICROBATCH_PADDING=true
# Token budget per packed microbatch: several samples share one row up to this many tokens.
# -1 packs one sample per microbatch.
MAX_TOKENS_PER_MICROBATCH=12288
# Context parallelism. Every CP rank gets the full packed stream and the model applies the CP split.
# With CP > 1, Megatron-Bridge's Qwen-VL providers turn on per-token loss mode, which needs
# overlap_grad_reduce=false and doesn't support MTP (trainer.mtp.enabled must stay false).
MEGATRON_CP=2
OVERLAP_GRAD_REDUCE=false
# Compute logprobs without materializing the full [tokens, vocab] logits; needed for long VLM rows.
FUSED_LM_HEAD_LOGPROB=true

# megatron config
MEGATRON_TP=2
MEGATRON_PP=1

# Qwen3.5 flags
ENGINE_INIT_KWARGS='{"gdn_prefill_backend": "triton"}' # see https://github.com/vllm-project/vllm/issues/36921#issuecomment-4109702738

if [ ! -f "$DATA_DIR/train.parquet" ]; then
  uv run examples/train/geometry3k/geometry_3k_dataset.py --output_dir "$DATA_DIR"
fi

uv run --isolated --extra megatron --with pylatexenc \
  python examples/train/geometry3k/geometry3k_entrypoint.py \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/test.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL_NAME" \
  trainer.policy.language_model_only=$LANGUAGE_MODEL_ONLY \
  generator.inference_engine.language_model_only=$LANGUAGE_MODEL_ONLY \
  trainer.remove_microbatch_padding=$REMOVE_MICROBATCH_PADDING \
  trainer.max_tokens_per_microbatch=$MAX_TOKENS_PER_MICROBATCH \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.ddp_config.overlap_grad_reduce=$OVERLAP_GRAD_REDUCE \
  trainer.fused_lm_head_logprob=$FUSED_LM_HEAD_LOGPROB \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.engine_init_kwargs="$ENGINE_INIT_KWARGS" \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.7 \
  generator.vision_language_generator=true \
  generator.batched=false \
  generator.max_turns=3 \
  generator.n_samples_per_prompt=4 \
  generator.sampling_params.max_generate_length=2048 \
  environment.env_class=geometry3k \
  trainer.epochs=6 \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=128 \
  trainer.policy_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.max_prompt_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.loss_reduction=token_mean \
  trainer.ckpt_interval=10 \
  trainer.logger="$LOGGER" \
  trainer.project_name="geometry3k" \
  trainer.run_name="qwen3.5_9b_vl_megatron_tp${MEGATRON_TP}_cp${MEGATRON_CP}_packed" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/qwen3.5_9b_vl_geometry3k_megatron" \
  trainer.export_path="$HOME/exports/qwen3.5_9b_vl_geometry3k_megatron" \
  "$@"
