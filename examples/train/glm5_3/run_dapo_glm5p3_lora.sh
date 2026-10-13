#!/usr/bin/env bash
# BF16 GLM-5.3 DAPO LoRA: four colocated 8xB300 nodes, TP8/EP8 per node.
# Start a Ray cluster and prepare DAPO/AIME data first; see README.md.
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-zai-org/GLM-5.3-BF16}
DATA_DIR=${DATA_DIR:-$HOME/data/dapo}
: "${RUN_ROOT:?Set RUN_ROOT to a shared directory for checkpoints and logs}"
NUM_NODES=${NUM_NODES:-4}

# A fresh run must not mix its checkpoints with a previous run's retention queue.
if [[ -e "$RUN_ROOT/checkpoints" ]]; then
  echo "Use a fresh RUN_ROOT; $RUN_ROOT/checkpoints already exists." >&2
  exit 1
fi

export SKYRL_WORKER_NCCL_TIMEOUT_IN_S=${SKYRL_WORKER_NCCL_TIMEOUT_IN_S:-21600}

uv run --isolated --extra megatron -m examples.train.algorithms.dapo.main_dapo \
  data.train_data="['$DATA_DIR/dapo-math-17k-cleaned.parquet']" \
  data.val_data="['$DATA_DIR/aime-2024-cleaned.parquet']" \
  environment.env_class=aime \
  trainer.strategy=megatron \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.policy.language_model_only=true \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_nodes="$NUM_NODES" \
  trainer.placement.policy_num_gpus_per_node=8 \
  trainer.policy.megatron_config.tensor_model_parallel_size=8 \
  trainer.policy.megatron_config.expert_model_parallel_size=8 \
  trainer.policy.megatron_config.expert_tensor_parallel_size=1 \
  trainer.policy.megatron_config.pipeline_model_parallel_size=1 \
  trainer.policy.megatron_config.context_parallel_size=1 \
  trainer.policy.megatron_config.moe_token_dispatcher_type=alltoall \
  trainer.policy.megatron_config.moe_router_load_balancing_type=none \
  trainer.policy.megatron_config.moe_router_score_function=sigmoid \
  trainer.policy.megatron_config.moe_grouped_gemm=true \
  trainer.policy.megatron_config.moe_enable_routing_replay=true \
  trainer.policy.megatron_config.ddp_config.average_in_collective=false \
  trainer.policy.megatron_config.transformer_config_kwargs='{"moe_router_bias_update_rate":0.0,"dsa_kernel_backend":"tilelang","calculate_per_token_loss":true,"gradient_accumulation_fusion":false,"sequence_parallel":true,"recompute_granularity":"full","recompute_method":"uniform","recompute_num_layers":1,"recompute_modules":[]}' \
  trainer.policy.megatron_config.lora_config.merge_lora=false \
  trainer.policy.model.lora.rank=32 \
  trainer.policy.model.lora.alpha=32 \
  trainer.policy.model.lora.max_loras=1 \
  trainer.policy.model.lora.share_expert_adapters=true \
  trainer.policy.model.lora.sync_mode=memory \
  trainer.policy.model.lora.target_modules='[linear_q_down_proj,linear_q_up_proj,linear_kv_down_proj,linear_kv_up_proj,linear_proj,linear_fc1,linear_fc2]' \
  trainer.fused_lm_head_logprob=true \
  trainer.logprobs_chunk_size=8192 \
  trainer.remove_microbatch_padding=false \
  trainer.max_tokens_per_microbatch=-1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  generator.inference_engine.num_engines="$NUM_NODES" \
  generator.inference_engine.tensor_parallel_size=8 \
  generator.inference_engine.expert_parallel_size=8 \
  generator.inference_engine.distributed_executor_backend=mp \
  generator.inference_engine.model_dtype=bfloat16 \
  generator.inference_engine.language_model_only=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.enable_return_routed_experts=true \
  generator.inference_engine.enable_prefix_caching=true \
  generator.inference_engine.enable_chunked_prefill=true \
  generator.inference_engine.enforce_eager=false \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  generator.inference_engine.max_num_seqs=32 \
  generator.inference_engine.max_num_batched_tokens=32768 \
  generator.inference_engine.router_init_kwargs.request_timeout_secs=21600 \
  generator.inference_engine.router_init_kwargs.queue_size=4096 \
  generator.inference_engine.router_init_kwargs.queue_timeout_secs=21600 \
  generator.inference_engine.engine_init_kwargs='{"disable_custom_all_reduce":true,"linear_backend":"triton","moe_backend":"triton","max_model_len":32768,"kv_cache_dtype":"bfloat16","enable_flashinfer_autotune":false,"attention_config":{"mla_prefill_backend":"FLASH_ATTN"},"kernel_config":{"ir_op_priority":{"rms_norm":["vllm_c"],"fused_add_rms_norm":["vllm_c"]}},"compilation_config":{"pass_config":{"fuse_allreduce_rms":false}}}' \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.policy_loss_type=dual_clip \
  trainer.algorithm.eps_clip_low=0.2 \
  trainer.algorithm.eps_clip_high=0.28 \
  trainer.algorithm.clip_ratio_c=10.0 \
  trainer.algorithm.loss_reduction=token_mean \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.dynamic_sampling.type=null \
  trainer.algorithm.overlong_buffer_len=4096 \
  trainer.algorithm.overlong_buffer_penalty_factor=1.0 \
  trainer.algorithm.off_policy_correction.tis_ratio_type=token \
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=2.0 \
  generator.apply_overlong_filtering=true \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.max_generate_length=8192 \
  generator.eval_sampling_params.temperature=1.0 \
  generator.eval_sampling_params.top_p=0.7 \
  generator.eval_sampling_params.max_generate_length=8192 \
  generator.n_samples_per_prompt=16 \
  generator.eval_n_samples_per_prompt=32 \
  trainer.max_prompt_length=2048 \
  trainer.train_batch_size=128 \
  trainer.policy_mini_batch_size=32 \
  trainer.update_epochs_per_batch=1 \
  trainer.epochs=20 \
  trainer.max_training_steps=20 \
  trainer.policy.optimizer_config.lr=1e-5 \
  trainer.policy.optimizer_config.num_warmup_steps=40 \
  trainer.policy.optimizer_config.scheduler=constant_with_warmup \
  trainer.policy.optimizer_config.weight_decay=0.1 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.eval_batch_size=30 \
  trainer.ckpt_interval=1 \
  trainer.max_ckpts_to_keep=3 \
  trainer.resume_mode=null \
  trainer.dump_data_batch=true \
  trainer.ckpt_path="$RUN_ROOT/checkpoints" \
  trainer.export_path="$RUN_ROOT/exports" \
  trainer.log_path="$RUN_ROOT/logs" \
  trainer.logger=wandb \
  trainer.project_name=glm53-dapo \
  trainer.run_name=glm53-dapo-lora \
  "$@"
