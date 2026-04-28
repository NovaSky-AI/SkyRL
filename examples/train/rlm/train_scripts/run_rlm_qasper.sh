set -x

# RLM training on QASPER with alphaXiv/rlm-sft-Qwen3.5-9B-v1.
# One step = 8 prompts × 8 samples = 64 rollouts, then one optimizer step.
#
# 1. Create data: uv run -- python examples/train/rlm/datasets/rlm_dataset_qasper.py --output_dir $DATA_DIR
# 2. Run: bash examples/train/rlm/train_scripts/run_rlm_qasper.sh

: "${DATA_DIR:=$HOME/data/rlm}"
: "${ROLLOUT_OUTPUT_DIR:=$(pwd)/.neer/artifacts/rollouts}"
: "${NUM_ENGINES:=1}"
: "${TP_SIZE:=4}"
: "${LOGGER:=wandb}"
: "${INFERENCE_BACKEND:=vllm}"

# Increase Ray compiled-graph channel timeout (default 300s) to avoid false
# timeouts when a rollout batch with max_turns=10 takes >5 minutes to generate.
export RAY_CGRAPH_get_timeout="${RAY_CGRAPH_get_timeout:-900}"

uv run --isolated --extra fsdp -m examples.train.rlm.main_rlm \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  environment.env_class=multipaper_evidence_rlm \
  generator.step_wise_trajectories=true \
  generator.max_turns=10 \
  generator.batched=false \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="alphaXiv/rlm-sft-Qwen3.5-9B-step-250" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$TP_SIZE \
  trainer.placement.ref_num_gpus_per_node=$TP_SIZE \
  generator.inference_engine.num_engines=$NUM_ENGINES \
  generator.inference_engine.tensor_parallel_size=$TP_SIZE \
  trainer.policy.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3_5DecoderLayer']" \
  trainer.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3_5DecoderLayer']" \
  trainer.epochs=1 \
  trainer.eval_before_train=true \
  trainer.eval_interval=10 \
  trainer.update_epochs_per_batch=1 \
  trainer.eval_batch_size=8 \
  trainer.train_batch_size=2 \
  trainer.policy_mini_batch_size=2 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=20 \
  trainer.use_sample_packing=false \
  trainer.max_prompt_length=32768 \
  generator.sampling_params.max_generate_length=4096 \
  generator.eval_sampling_params.max_generate_length=4096 \
  generator.sampling_params.temperature=0.7 \
  generator.sampling_params.top_p=0.8 \
  generator.sampling_params.top_k=20 \
  generator.sampling_params.min_p=0.0 \
  generator.sampling_params.repetition_penalty=1.0 \
  generator.sampling_params.additional_kwargs.presence_penalty=1.5 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.gpu_memory_utilization=0.6 \
  generator.max_input_length=32768 \
  generator.inference_engine.engine_init_kwargs.language_model_only=true \
  generator.chat_template_kwargs.enable_thinking=false \
  generator.n_samples_per_prompt=8 \
  trainer.logger="['console','wandb']" \
  trainer.project_name="rlm" \
  trainer.run_name="rlm_qasper_grpo" \
  trainer.log_path="$(pwd)/.neer/artifacts/skyrl-logs" \
  trainer.ckpt_path="$(pwd)/.neer/artifacts/ckpts/rlm_ckpt" \
  trainer.export_path="$(pwd)/.neer/artifacts/rlm_exports" \
  trainer.dump_eval_results=true \
  "$@"
