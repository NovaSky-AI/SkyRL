#!/bin/bash
set -x

# ==============================================================================
# GLM-4.7-Flash GRPO Training on 4x8xH200 (32 GPUs)
#
# Phase 6: Full-scale training with 32 GPUs
#   - TP=4, EP=8, DP=4 (4 data-parallel replicas)
#   - Batch=1024, n_samples=16
#   - Full 8K context
#
# Prerequisites:
#   - Run 00_setup_and_sanity_check.sh on ALL 4 nodes
#   - Set NODE0_IP below (head node)
#   - Start ray workers on nodes 1-3:
#       cd $HOME/SkyRL && .venv/bin/ray start --address=<NODE0_IP>:6379
# ==============================================================================

NODE0_IP="${NODE0_IP:?Set NODE0_IP to the head node IP}"

cd "$HOME/SkyRL"

MODEL_NAME="zai-org/GLM-4.7-Flash"
DATA_DIR="$HOME/data/gsm8k"
LOGGER="console"  # change to "wandb" for real runs

export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export RAY_memory_usage_threshold=1.0

# Start ray head
.venv/bin/ray stop --force 2>/dev/null; sleep 3
.venv/bin/ray start --head --node-ip-address=$NODE0_IP
sleep 5

echo "============================================"
echo " Waiting for 3 worker nodes to join"
echo "============================================"
echo "On each worker node, run:"
echo "  cd \$HOME/SkyRL && .venv/bin/ray start --address=$NODE0_IP:6379"
echo ""
echo "Press Enter once all 3 worker nodes have joined..."
read -r

# Verify 4 nodes
.venv/bin/ray status
echo ""

echo "============================================"
echo " Phase 6: 32-GPU training (4x8xH200)"
echo "============================================"

.venv/bin/python -m skyrl.train.entrypoints.main_base \
    data.train_data="['$DATA_DIR/train.parquet']" \
    data.val_data="['$DATA_DIR/validation.parquet']" \
    trainer.algorithm.advantage_estimator=grpo \
    trainer.algorithm.policy_loss_type=regular \
    trainer.algorithm.eps_clip_low=0.2 \
    trainer.algorithm.eps_clip_high=0.28 \
    trainer.algorithm.use_kl_loss=false \
    trainer.policy.model.path=$MODEL_NAME \
    trainer.placement.colocate_all=true \
    trainer.strategy=megatron \
    trainer.placement.policy_num_nodes=4 \
    trainer.placement.policy_num_gpus_per_node=8 \
    generator.inference_engine.num_engines=4 \
    generator.inference_engine.tensor_parallel_size=8 \
    generator.inference_engine.enforce_eager=true \
    generator.inference_engine.async_engine=true \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.engine_init_kwargs.max_model_len=8704 \
    trainer.policy.megatron_config.tensor_model_parallel_size=4 \
    trainer.policy.megatron_config.pipeline_model_parallel_size=1 \
    trainer.policy.megatron_config.context_parallel_size=1 \
    trainer.policy.megatron_config.expert_model_parallel_size=8 \
    trainer.policy.megatron_config.expert_tensor_parallel_size=1 \
    trainer.policy.megatron_config.empty_cuda_cache=true \
    trainer.flash_attn=true \
    trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=true \
    trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=1.0 \
    trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=true \
    trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=true \
    trainer.policy.megatron_config.optimizer_config_kwargs.adam_beta1=0.9 \
    trainer.policy.megatron_config.optimizer_config_kwargs.adam_beta2=0.98 \
    trainer.policy.megatron_config.transformer_config_kwargs.moe_token_dispatcher_type=alltoall \
    trainer.policy.megatron_config.transformer_config_kwargs.moe_grouped_gemm=true \
    trainer.policy.megatron_config.transformer_config_kwargs.moe_permute_fusion=true \
        trainer.policy.megatron_config.transformer_config_kwargs.moe_router_load_balancing_type=seq_aux_loss \
        trainer.policy.megatron_config.transformer_config_kwargs.moe_aux_loss_coeff=0.0 \
        trainer.policy.megatron_config.transformer_config_kwargs.no_rope_fusion=true \
    trainer.policy.megatron_config.transformer_config_kwargs.moe_router_score_function=sigmoid \
    trainer.policy.megatron_config.transformer_config_kwargs.moe_router_pre_softmax=true \
    trainer.policy.megatron_config.transformer_config_kwargs.moe_router_enable_expert_bias=true \
    trainer.policy.megatron_config.transformer_config_kwargs.moe_router_bias_update_rate=0 \
    trainer.policy.megatron_config.transformer_config_kwargs.moe_router_topk_scaling_factor=1.8 \
    trainer.policy.megatron_config.transformer_config_kwargs.moe_router_dtype=fp32 \
    trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity=full \
    trainer.policy.megatron_config.transformer_config_kwargs.recompute_method=uniform \
    trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=1 \
    trainer.policy.megatron_config.transformer_config_kwargs.accumulate_allreduce_grads_in_fp32=true \
    trainer.policy.megatron_config.transformer_config_kwargs.make_vocab_size_divisible_by=64 \
    trainer.epochs=20 \
    trainer.eval_batch_size=2048 \
    trainer.eval_before_train=false \
    trainer.eval_interval=5 \
    trainer.update_epochs_per_batch=1 \
    trainer.train_batch_size=1024 \
    trainer.policy_mini_batch_size=256 \
    trainer.micro_forward_batch_size_per_gpu=1 \
    trainer.micro_train_batch_size_per_gpu=1 \
    trainer.max_prompt_length=512 \
    generator.sampling_params.max_generate_length=8192 \
    generator.n_samples_per_prompt=16 \
    generator.inference_engine.gpu_memory_utilization=0.7 \
    generator.batched=true \
    environment.env_class=gsm8k \
    trainer.policy.optimizer_config.lr=1.0e-6 \
    trainer.policy.optimizer_config.weight_decay=0.1 \
    trainer.policy.optimizer_config.max_grad_norm=1.0 \
    trainer.policy.optimizer_config.num_warmup_steps=0 \
    trainer.logger=$LOGGER \
    trainer.project_name=glm47_h200 \
    trainer.run_name=phase6_32gpu_4x8 \
    trainer.resume_mode=null \
    trainer.ckpt_interval=10 \
    trainer.ckpt_path="$HOME/ckpts/phase6" \
    2>&1 | tee /tmp/phase6.log

echo "Phase 6 done. Check /tmp/phase6.log"
