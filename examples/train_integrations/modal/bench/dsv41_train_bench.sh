#!/bin/bash
# Trainer-only throughput of the full DeepSeek-V4.1-Flash LoRA RL step on one 8-GPU node, depthfirst's rollout shape
# (multi-turn: TURNS assistant spans with OBS-token loss-masked observations between them), with the throughput
# branch's long-context settings: stateful full recompute with layer inputs in pinned host memory, the replayed
# indexer top-k, old-logprob forward skipped, token-balanced DP, fused LM-head logprob, fused indexer scoring.
# Real released weights (FP8 / MXFP4, dequantized to BF16 on load), LoRA r32 on the recipe's targets, R3, EP8.
#
# UV_RUN overrides the launcher (e.g. "uv run --no-sync" with UV_PROJECT_ENVIRONMENT set).
#   MODEL_PATH=<dir> OUT=<dir> bash dsv41_train_bench.sh <name> <max response tokens> [extra overrides]
#
# Per-GPU tokens, not sequence length, set the memory: without context parallelism (V4.1 attention has none yet) a
# whole sequence sits on one GPU, so a CP4 setup's ~200k-token rollouts are modelled as ~50k-token sequences
# (RESP=50000 MIN_RESP=46000 TURNS=17: the same 1065 generated / 1870 observation tokens per turn).
set -x
NAME=$1
RESP=$2
shift 2
SKYRL_DIR="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$SKYRL_DIR"
export SKYRL_MEGATRON_RANDOM_INIT=0 SKYRL_DSV41_INDEXER_IMPL=fused
# OFFLOAD_HOST_GB is a per-process (per-GPU) limit on pinned host memory for offloaded layer inputs: 8 x 32 GB plus
# the ~25 GB/rank pinned Engram shards keeps pinned memory under half of a 1 TB B300 node (320 per rank took one down).
export SKYRL_DSV41_RECOMPUTE_OFFLOAD=1 SKYRL_DSV41_RECOMPUTE_OFFLOAD_HOST_GB="${OFFLOAD_HOST_GB:-32}"
export MBRIDGE_STATE_REUSE_HANDLES=0 MBRIDGE_STATE_CACHE_SMALL_TENSORS=0
MAX_LEN=$(( 512 + RESP ))
${UV_RUN:-uv run --isolated} --extra megatron python -m examples.train_scripts.full_context.main_full_ctx \
  data.train_data="['${DATA_DIR:-/root/data/gsm8k_multi_turn}/train.parquet']" \
  data.val_data="['${DATA_DIR:-/root/data/gsm8k_multi_turn}/validation.parquet']" \
  trainer.num_dummy_steps="${STEPS:-3}" trainer.skip_inference_engines=true \
  trainer.dummy_length_distribution=mixture trainer.dummy_min_prompt_length=128 trainer.dummy_max_prompt_length=512 \
  trainer.dummy_min_response_length="${MIN_RESP:-$(( RESP / 2 ))}" trainer.dummy_truncated_fraction=0.25 \
  trainer.dummy_token_source=dataset trainer.dummy_num_turns="${TURNS:-69}" \
  trainer.dummy_observation_length="${OBS:-1870}" trainer.dummy_routes=true \
  trainer.algorithm.advantage_estimator=grpo trainer.algorithm.use_kl_loss=false trainer.algorithm.loss_reduction=token_mean \
  trainer.policy.model.path="$MODEL_PATH" trainer.strategy=megatron \
  trainer.placement.colocate_all=false trainer.placement.policy_num_gpus_per_node=8 \
  trainer.policy.megatron_config.tensor_model_parallel_size=1 trainer.policy.megatron_config.expert_model_parallel_size=8 \
  trainer.policy.megatron_config.mtp_num_layers=0 \
  trainer.policy.megatron_config.moe_enable_routing_replay=true generator.inference_engine.enable_return_routed_experts=true \
  generator.max_input_length=512 generator.sampling_params.max_generate_length="$RESP" \
  generator.n_samples_per_prompt="${SAMPLES:-8}" \
  trainer.max_prompt_length=512 trainer.algorithm.max_seq_len=$MAX_LEN \
  trainer.train_batch_size="${PROMPTS:-2}" trainer.policy_mini_batch_size="${PROMPTS:-2}" \
  trainer.micro_forward_batch_size_per_gpu=1 trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_tokens_per_microbatch=-1 trainer.remove_microbatch_padding=false \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity=full \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_method=uniform \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=1 \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_modules=null \
  trainer.policy.model.lora.rank=32 trainer.policy.model.lora.alpha=32 \
  'trainer.policy.model.lora.target_modules=[linear_q_down_proj,linear_q_up_proj,linear_kv_proj,linear_proj,linear_fc1,linear_fc2]' \
  trainer.policy.megatron_config.lora_config.merge_lora=false \
  trainer.old_logprobs_from_first_minibatch=true trainer.balance_dp_tokens=true \
  trainer.fused_lm_head_logprob=true trainer.fused_lm_head_logprob_backend=triton trainer.use_expandable_segments=true \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.logger=console trainer.project_name=dsv41_train_bench trainer.run_name="$NAME" \
  trainer.log_path="${OUT:-/tmp/skyrl-logs}/$NAME" "$@"
