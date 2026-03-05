#!/bin/bash
set -x

# ==============================================================================
# GLM-4.7-Flash GRPO Training on 2x8xH200 (16 GPUs)
#
# Runs 3 phases of increasing ambition:
#   Phase 3: 8-GPU single-node (proven config, should be trivial on H200)
#   Phase 4: 8-GPU with full slime settings (8K context, batch=256)
#   Phase 5: 16-GPU multi-node (slime's 16-GPU config)
#
# Prerequisites:
#   - Run 00_setup_and_sanity_check.sh on BOTH nodes first
#   - Set NODE0_IP and NODE1_IP below
#   - Ensure both nodes can reach each other (ssh, NCCL)
# ==============================================================================

# ---- EDIT THESE ----
NODE0_IP="${NODE0_IP:?Set NODE0_IP to the head node IP}"
NODE1_IP="${NODE1_IP:?Set NODE1_IP to the second node IP}"
# --------------------

cd "$HOME/SkyRL"

MODEL_NAME="zai-org/GLM-4.7-Flash"
DATA_DIR="$HOME/data/gsm8k"
LOGGER="console"  # change to "wandb" for real runs

# Common env vars
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export RAY_memory_usage_threshold=1.0

# ==============================================================================
# Phase 3: 8-GPU single-node (H200 should make this trivial)
#   - Same as our proven A100 config but with gpu_memory_utilization=0.7
#   - Short context (1024 gen) to verify basic operation quickly
# ==============================================================================
run_phase3() {
    echo "============================================"
    echo " Phase 3: 8-GPU single-node quick test"
    echo "============================================"

    .venv/bin/ray stop --force 2>/dev/null; sleep 3
    .venv/bin/ray start --head
    sleep 2

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
        trainer.placement.policy_num_nodes=1 \
        trainer.placement.policy_num_gpus_per_node=8 \
        generator.inference_engine.num_engines=2 \
        generator.inference_engine.tensor_parallel_size=4 \
        generator.inference_engine.enforce_eager=true \
        generator.inference_engine.async_engine=true \
        generator.inference_engine.run_engines_locally=true \
        generator.inference_engine.weight_sync_backend=nccl \
        generator.inference_engine.backend=vllm \
        generator.inference_engine.engine_init_kwargs.max_model_len=1536 \
        trainer.policy.megatron_config.tensor_model_parallel_size=1 \
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
        trainer.epochs=1 \
        trainer.eval_before_train=false \
        trainer.eval_interval=999 \
        trainer.update_epochs_per_batch=1 \
        trainer.train_batch_size=64 \
        trainer.policy_mini_batch_size=64 \
        trainer.micro_forward_batch_size_per_gpu=1 \
        trainer.micro_train_batch_size_per_gpu=1 \
        trainer.max_prompt_length=512 \
        generator.sampling_params.max_generate_length=1024 \
        generator.n_samples_per_prompt=8 \
        generator.inference_engine.gpu_memory_utilization=0.7 \
        generator.batched=true \
        environment.env_class=gsm8k \
        trainer.policy.optimizer_config.lr=1.0e-6 \
        trainer.policy.optimizer_config.weight_decay=0.1 \
        trainer.policy.optimizer_config.max_grad_norm=1.0 \
        trainer.policy.optimizer_config.num_warmup_steps=0 \
        trainer.logger=$LOGGER \
        trainer.project_name=glm47_h200 \
        trainer.run_name=phase3_8gpu_quick \
        trainer.resume_mode=null \
        trainer.ckpt_interval=1 \
        trainer.ckpt_path="$HOME/ckpts/phase3" \
        2>&1 | tee /tmp/phase3.log

    echo "Phase 3 done. Check /tmp/phase3.log"
}

# ==============================================================================
# Phase 4: 8-GPU with full slime settings (8K context, batch=256)
#   - This OOMed on A100 (80GB) but should work on H200 (141GB)
#   - Full slime-matching config
# ==============================================================================
run_phase4() {
    echo "============================================"
    echo " Phase 4: 8-GPU full slime match (8K context)"
    echo "============================================"

    .venv/bin/ray stop --force 2>/dev/null; sleep 3
    .venv/bin/ray start --head
    sleep 2

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
        trainer.placement.policy_num_nodes=1 \
        trainer.placement.policy_num_gpus_per_node=8 \
        generator.inference_engine.num_engines=2 \
        generator.inference_engine.tensor_parallel_size=4 \
        generator.inference_engine.enforce_eager=true \
        generator.inference_engine.async_engine=true \
        generator.inference_engine.run_engines_locally=true \
        generator.inference_engine.weight_sync_backend=nccl \
        generator.inference_engine.backend=vllm \
        generator.inference_engine.engine_init_kwargs.max_model_len=8704 \
        trainer.policy.megatron_config.tensor_model_parallel_size=1 \
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
        trainer.eval_batch_size=1024 \
        trainer.eval_before_train=false \
        trainer.eval_interval=5 \
        trainer.update_epochs_per_batch=1 \
        trainer.train_batch_size=256 \
        trainer.policy_mini_batch_size=256 \
        trainer.micro_forward_batch_size_per_gpu=1 \
        trainer.micro_train_batch_size_per_gpu=1 \
        trainer.max_prompt_length=512 \
        generator.sampling_params.max_generate_length=8192 \
        generator.n_samples_per_prompt=8 \
        generator.inference_engine.gpu_memory_utilization=0.7 \
        generator.batched=true \
        environment.env_class=gsm8k \
        trainer.policy.optimizer_config.lr=1.0e-6 \
        trainer.policy.optimizer_config.weight_decay=0.1 \
        trainer.policy.optimizer_config.max_grad_norm=1.0 \
        trainer.policy.optimizer_config.num_warmup_steps=0 \
        trainer.logger=$LOGGER \
        trainer.project_name=glm47_h200 \
        trainer.run_name=phase4_8gpu_slime_match \
        trainer.resume_mode=null \
        trainer.ckpt_interval=10 \
        trainer.ckpt_path="$HOME/ckpts/phase4" \
        2>&1 | tee /tmp/phase4.log

    echo "Phase 4 done. Check /tmp/phase4.log"
}

# ==============================================================================
# Phase 5: 16-GPU multi-node (2x8xH200)
#   - Matches slime's 16-GPU config: TP=4, EP=8, colocated
#   - Uses all 16 GPUs for both training and inference
#   - Larger batch (512) to utilize all GPUs
#
# Multi-node setup:
#   On NODE0 (head): NODE0_IP=x.x.x.x NODE1_IP=y.y.y.y bash 01_train_2x8_h200.sh phase5
#   On NODE1 (worker): ray start --address=<NODE0_IP>:6379
# ==============================================================================
run_phase5() {
    echo "============================================"
    echo " Phase 5: 16-GPU multi-node (2x8xH200)"
    echo "============================================"
    echo "Head node: $NODE0_IP"
    echo "Worker node: $NODE1_IP"

    .venv/bin/ray stop --force 2>/dev/null; sleep 3
    .venv/bin/ray start --head --node-ip-address=$NODE0_IP
    sleep 5

    echo "Waiting for worker node to join ray cluster..."
    echo "On the worker node, run:"
    echo "  cd \$HOME/SkyRL && .venv/bin/ray start --address=$NODE0_IP:6379"
    echo ""
    echo "Press Enter once the worker node has joined..."
    read -r

    # Verify 2 nodes are connected
    .venv/bin/ray status
    echo ""

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
        trainer.placement.policy_num_nodes=2 \
        trainer.placement.policy_num_gpus_per_node=8 \
        generator.inference_engine.num_engines=2 \
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
        trainer.eval_batch_size=1024 \
        trainer.eval_before_train=false \
        trainer.eval_interval=5 \
        trainer.update_epochs_per_batch=1 \
        trainer.train_batch_size=512 \
        trainer.policy_mini_batch_size=256 \
        trainer.micro_forward_batch_size_per_gpu=1 \
        trainer.micro_train_batch_size_per_gpu=1 \
        trainer.max_prompt_length=512 \
        generator.sampling_params.max_generate_length=8192 \
        generator.n_samples_per_prompt=8 \
        generator.inference_engine.gpu_memory_utilization=0.7 \
        generator.batched=true \
        environment.env_class=gsm8k \
        trainer.policy.optimizer_config.lr=1.0e-6 \
        trainer.policy.optimizer_config.weight_decay=0.1 \
        trainer.policy.optimizer_config.max_grad_norm=1.0 \
        trainer.policy.optimizer_config.num_warmup_steps=0 \
        trainer.logger=$LOGGER \
        trainer.project_name=glm47_h200 \
        trainer.run_name=phase5_16gpu_2x8 \
        trainer.resume_mode=null \
        trainer.ckpt_interval=10 \
        trainer.ckpt_path="$HOME/ckpts/phase5" \
        2>&1 | tee /tmp/phase5.log

    echo "Phase 5 done. Check /tmp/phase5.log"
}

# ==============================================================================
# Run the requested phase
# ==============================================================================
PHASE="${1:-phase3}"
echo "Running $PHASE..."
case "$PHASE" in
    phase3) run_phase3 ;;
    phase4) run_phase4 ;;
    phase5) run_phase5 ;;
    all)
        run_phase3
        echo "Phase 3 complete. Starting Phase 4 in 10s..."
        sleep 10
        run_phase4
        echo "Phase 4 complete. Phase 5 requires manual multi-node setup."
        echo "Run: bash 01_train_2x8_h200.sh phase5"
        ;;
    *)
        echo "Usage: $0 {phase3|phase4|phase5|all}"
        echo "  phase3 - 8-GPU quick test (short context, 1 epoch)"
        echo "  phase4 - 8-GPU full slime match (8K context, 20 epochs)"
        echo "  phase5 - 16-GPU multi-node (2x8, requires NODE0_IP and NODE1_IP)"
        echo "  all    - Run phase3 then phase4 sequentially"
        exit 1
        ;;
esac
