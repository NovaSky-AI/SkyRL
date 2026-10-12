#!/usr/bin/env bash
set -euo pipefail
set -x

# Minimal GLM-5.2-FP8 LoRA training proof on two 8xB300 nodes.
#
# The topology is disaggregated: Megatron uses one full node and vLLM uses
# the other. Both nodes must already be members of the same Ray cluster.
#
# Prepare GSM8K before running:
#   uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir "${DATA_DIR}"
#
# DATA_DIR, CKPT_DIR, and LORA_SYNC_DIR must be visible from both nodes.
# The defaults assume a shared home directory.

MODEL_NAME="${MODEL_NAME:-zai-org/GLM-5.2-FP8}"
DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k}"
CKPT_DIR="${CKPT_DIR:-$HOME/ckpts/glm5_2_744b_grpo_megatron}"
LORA_SYNC_DIR="${LORA_SYNC_DIR:-$CKPT_DIR/lora-sync}"
LOGGER="${LOGGER:-console}"

TRAIN_FILE="${DATA_DIR}/train.parquet"
test -f "${TRAIN_FILE}"
mkdir -p "${CKPT_DIR}" "${LORA_SYNC_DIR}"

POLICY_NUM_NODES=1
GPUS_PER_NODE=8
MEGATRON_TP=8
MEGATRON_EP=8
INFERENCE_ENGINE_TP=8
LORA_RANK=32

uv run --isolated --extra megatron -m skyrl.train.entrypoints.main_base \
  "data.train_data=['${TRAIN_FILE}']" \
  'data.val_data=[]' \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.use_kl_loss=false \
  trainer.policy.model.path="${MODEL_NAME}" \
  trainer.policy.language_model_only=true \
  trainer.policy.model.lora.rank=${LORA_RANK} \
  trainer.policy.model.lora.alpha=${LORA_RANK} \
  'trainer.policy.model.lora.target_modules=["linear_q_down_proj","linear_q_up_proj","linear_kv_down_proj","linear_kv_up_proj","linear_proj","linear_fc1","linear_fc2"]' \
  trainer.policy.model.lora.lora_sync_path="${LORA_SYNC_DIR}" \
  trainer.strategy=megatron \
  trainer.placement.colocate_all=false \
  trainer.placement.policy_num_nodes=${POLICY_NUM_NODES} \
  trainer.placement.policy_num_gpus_per_node=${GPUS_PER_NODE} \
  trainer.policy.megatron_config.tensor_model_parallel_size=${MEGATRON_TP} \
  trainer.policy.megatron_config.pipeline_model_parallel_size=1 \
  trainer.policy.megatron_config.context_parallel_size=1 \
  trainer.policy.megatron_config.expert_model_parallel_size=${MEGATRON_EP} \
  trainer.policy.megatron_config.expert_tensor_parallel_size=1 \
  trainer.policy.megatron_config.moe_token_dispatcher_type=alltoall \
  trainer.policy.megatron_config.moe_router_load_balancing_type=none \
  trainer.policy.megatron_config.moe_grouped_gemm=true \
  trainer.policy.megatron_config.moe_router_score_function=sigmoid \
  trainer.policy.megatron_config.lora_config.merge_lora=false \
  trainer.policy.megatron_config.ddp_config.average_in_collective=false \
  trainer.policy.megatron_config.transformer_config_kwargs.qk_pos_emb_head_dim=64 \
  trainer.policy.megatron_config.transformer_config_kwargs.dsa_indexer_topk_freq=4 \
  trainer.policy.megatron_config.transformer_config_kwargs.dsa_indexer_skip_topk_offset=3 \
  trainer.policy.megatron_config.transformer_config_kwargs.dsa_indexer_rope_interleaved=true \
  trainer.policy.megatron_config.transformer_config_kwargs.dsa_indexer_rotate_activation=false \
  trainer.policy.megatron_config.transformer_config_kwargs.dsa_indexer_k_norm_epsilon=1.0e-6 \
  trainer.policy.megatron_config.transformer_config_kwargs.mtp_num_layers=0 \
  trainer.policy.megatron_config.transformer_config_kwargs.mtp_use_repeated_layer=false \
  trainer.policy.megatron_config.transformer_config_kwargs.calculate_per_token_loss=true \
  trainer.policy.megatron_config.transformer_config_kwargs.gradient_accumulation_fusion=false \
  trainer.policy.megatron_config.transformer_config_kwargs.sequence_parallel=true \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity=full \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_method=uniform \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=1 \
  trainer.fused_lm_head_logprob=true \
  trainer.remove_microbatch_padding=false \
  trainer.max_tokens_per_microbatch=2048 \
  trainer.logprobs_chunk_size=2048 \
  trainer.epochs=1 \
  trainer.max_training_steps=2 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.ckpt_interval=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=4 \
  trainer.policy_mini_batch_size=4 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=256 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.offload_after_step=false \
  generator.n_samples_per_prompt=4 \
  generator.sampling_params.max_generate_length=256 \
  generator.batched=true \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.language_model_only=true \
  generator.inference_engine.num_engines=1 \
  generator.inference_engine.tensor_parallel_size=${INFERENCE_ENGINE_TP} \
  generator.inference_engine.pipeline_parallel_size=1 \
  generator.inference_engine.data_parallel_size=1 \
  generator.inference_engine.distributed_executor_backend=mp \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.enforce_eager=false \
  generator.inference_engine.gpu_memory_utilization=0.75 \
  generator.inference_engine.max_num_seqs=8 \
  generator.inference_engine.max_num_batched_tokens=2048 \
  generator.inference_engine.enable_prefix_caching=false \
  generator.inference_engine.enable_chunked_prefill=true \
  generator.inference_engine.engine_init_kwargs.max_model_len=2048 \
  generator.inference_engine.engine_init_kwargs.kv_cache_dtype=fp8 \
  generator.inference_engine.engine_init_kwargs.disable_custom_all_reduce=true \
  generator.inference_engine.engine_init_kwargs.linear_backend=triton \
  generator.inference_engine.engine_init_kwargs.moe_backend=triton \
  'generator.inference_engine.engine_init_kwargs.lora_target_modules=["fused_qkv_a_proj","q_a_proj","q_b_proj","q_proj","kv_a_proj_with_mqa","kv_b_proj","o_proj","gate_up_proj","down_proj","experts"]' \
  generator.inference_engine.engine_init_kwargs.trust_remote_code=true \
  environment.env_class=gsm8k \
  trainer.logger="${LOGGER}" \
  trainer.project_name=glm5_2_744b_grpo \
  trainer.run_name=glm5_2_744b_grpo_megatron_2n \
  trainer.resume_mode=null \
  trainer.ckpt_path="${CKPT_DIR}" \
  "$@"
