set -x

# Colocated GRPO multi-turn SearchR1 training+generation for XiaomiMiMo/MiMo-7B-RL (dense) with
# Megatron, with native Multi-Token Prediction (MTP) speculative decoding (k=3). Runs on 1x8 H100.
#
# Setup (see examples/train/search/README.md):
#   1. Dataset parquet:
#        uv run --isolated examples/train/search/searchr1_dataset.py --local_dir $HOME/data/searchR1
#   2. Download the wiki-18 e5 index + corpus into the same dir, assemble e5_Flat.index, gunzip corpus.
#   3. Start the local e5 retrieval server (separate faiss-gpu conda env) on :8000.
#   export WANDB_API_KEY=<your_key_here>
#   bash examples/train/spec_decode/run_search_megatron_mimo_7b_specdecode.sh

MODEL_NAME="XiaomiMiMo/MiMo-7B-RL"
# Dataset on the fast, non-persistent local disk (not the ~/default quota).
DATA_DIR="$HOME/data/searchR1"

NUM_NODES=1
NUM_GPUS_PER_NODE=8

# megatron config -- MiMo-7B-RL is a dense model, so no expert parallelism.
# TP=4, PP=1, CP=1 => DP=2; TP stays within the single-node NVLink domain.
MEGATRON_TP=4
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=1
MEGATRON_ETP=null

# One vLLM engine per GPU (TP=1). A single MiMo-7B copy is ~14GB bf16, fits per engine alongside
# KV cache + the small MTP drafter at gpu_memory_utilization=0.5 (policy offloaded during gen under
# colocate_all). gmu=0.5 also leaves headroom for the faiss-gpu retriever server (~6GB/GPU).
NUM_INFERENCE_ENGINES=8
INFERENCE_ENGINE_TP=1

MICRO_TRAIN_BATCH_SIZE_PER_GPU=1
MICRO_FORWARD_BATCH_SIZE_PER_GPU=2

# TIS parameters (match run_search.sh)
TIS_TYPE=token
TIS_IMP_RATIO_CAP=2.0

# Multi-Token Prediction (MTP) speculative decoding -- k=3.
# trainer.mtp is the single high-level knob; validate_cfg propagates it to the training side
# (policy.megatron_config.mtp_*) and the inference side (speculative_config). mtp_num_layers is left
# at default => inferred from MiMo's HF config (1 native head). k>1 reuses that head autoregressively.
MTP_ENABLED=true
MTP_NUM_SPECULATIVE_TOKENS=3
MTP_LOSS_WEIGHT=0.5
MTP_LOSS_TOPK=256         # top-k draft loss: O(seq*k) memory vs O(seq*vocab)

# MiMo flags -- plain Qwen2-style dense attention (no GDN): sample packing supported, no special
# vLLM prefill backend needed.
REMOVE_MICROBATCH_PADDING=true
DISTRIBUTED_EXECUTOR_BACKEND="mp"
ENFORCE_EAGER=true  # cuda graphs can cause instability with weight sync
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800

RUN_NAME="sd_search_mimo_7b_rl_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}_k${MTP_NUM_SPECULATIVE_TOKENS}"

uv run --isolated --extra megatron -m skyrl.train.entrypoints.main_base \
  data.train_data="['${DATA_DIR}/train.parquet']" \
  data.val_data="['${DATA_DIR}/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.policy.optimizer_config.num_warmup_steps=94 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.algorithm.off_policy_correction.tis_ratio_type=$TIS_TYPE \
  trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high=$TIS_IMP_RATIO_CAP \
  trainer.policy.model.path="$MODEL_NAME" \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.ref_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.ref.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.ref.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.ref.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.ref.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.ref.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TP \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.distributed_executor_backend="$DISTRIBUTED_EXECUTOR_BACKEND" \
  generator.inference_engine.enforce_eager=$ENFORCE_EAGER \
  generator.inference_engine.gpu_memory_utilization=0.5 \
  trainer.remove_microbatch_padding=$REMOVE_MICROBATCH_PADDING \
  trainer.epochs=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=512 \
  trainer.policy_mini_batch_size=256 \
  trainer.micro_forward_batch_size_per_gpu=$MICRO_FORWARD_BATCH_SIZE_PER_GPU \
  trainer.micro_train_batch_size_per_gpu=$MICRO_TRAIN_BATCH_SIZE_PER_GPU \
  trainer.max_prompt_length=2048 \
  generator.max_input_length=4096 \
  generator.sampling_params.max_generate_length=500 \
  generator.inference_engine.async_engine=true \
  generator.batched=false \
  generator.use_conversation_multi_turn=false \
  generator.n_samples_per_prompt=5 \
  generator.max_turns=4 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.stop='["</search>", "</answer>"]' \
  environment.env_class="search" \
  environment.skyrl_gym.max_env_workers=16 \
  environment.skyrl_gym.search.log_requests=false \
  environment.skyrl_gym.search.search_url="http://127.0.0.1:8000/retrieve" \
  environment.skyrl_gym.search.topk=3 \
  trainer.mtp.enabled=$MTP_ENABLED \
  trainer.mtp.num_speculative_tokens=$MTP_NUM_SPECULATIVE_TOKENS \
  trainer.mtp.loss_weight=$MTP_LOSS_WEIGHT \
  trainer.policy.megatron_config.mtp_loss_topk=$MTP_LOSS_TOPK \
  trainer.logger="wandb" \
  trainer.project_name="mimo_7b_rl_searchr1" \
  trainer.run_name="${RUN_NAME}" \
  trainer.ckpt_interval=20 \
  trainer.hf_save_interval=100 \
  trainer.max_ckpts_to_keep=3 \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$HOME/ckpts/${RUN_NAME}" \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=false \
  trainer.eval_interval=50 \
  generator.eval_sampling_params.temperature=0 \
  generator.eval_sampling_params.stop='["</search>", "</answer>"]' \
  generator.eval_sampling_params.max_generate_length=500 \
  trainer.export_path="$HOME/exports/${RUN_NAME}" \
  $@
