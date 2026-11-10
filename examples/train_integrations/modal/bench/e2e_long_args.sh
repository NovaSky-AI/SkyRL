# Long-horizon (~200k-token) multi-turn LoRA RL for DeepSeek-V4.1-Flash on the async recipe: CP=4 trainer with packed
# sequences, stateful full recompute + frozen-weight compression + derived MLP inputs, vLLM with the TRT-LLM MXFP4
# LoRA MoE, 32k prefill budget and mixed-step CUDA graphs. Sourced by the B200 smoke and the Modal launcher.
# Caller sets: MODEL_PATH, DATA_DIR, MAX_TURNS, TRAIN_BATCH_SIZE, N_SAMPLES_PER_PROMPT, MAX_TRAINING_STEPS, RUN_NAME,
#              CKPT_PATH, OFFLOAD_GB, MAX_MODEL_LEN.
export ENV_CLASS=gsm8k_long_horizon MAX_GENERATE_LENGTH=1065 MAX_PROMPT_LENGTH=2048 MAX_STALENESS_STEPS=1
export NUM_TRAIN_NODES=1 NUM_ENGINES="${ENGINES_PER_NODE:-1}" LOGGER="${LOGGER:-wandb}"  # one sampler node
export SKYRL_DSV41_COMPRESS_FROZEN_WEIGHTS=1 SKYRL_DSV41_RECOMPUTE_DERIVE_MLP_INPUT=1 SKYRL_DSV41_RECOMPUTE_OFFLOAD=1
export SKYRL_DSV41_RECOMPUTE_OFFLOAD_HOST_GB="${OFFLOAD_GB:-24}" SKYRL_DSV41_FAKE_QUANT=reference
export MBRIDGE_STATE_REUSE_HANDLES=0 MBRIDGE_STATE_CACHE_SMALL_TENSORS=0 SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1
LONG_ARGS=(
  # trainer: context parallelism (packed THD, contiguous layout inside the V4.1 stack)
  "trainer.policy.megatron_config.context_parallel_size=${CP:-4}"
  trainer.remove_microbatch_padding=true
  trainer.policy.megatron_config.transformer_config_kwargs.attention_cp_layout=contiguous
  trainer.policy.megatron_config.transformer_config_kwargs.linear_cp_layout=contiguous
  trainer.policy.megatron_config.transformer_config_kwargs.dsa_kernel_backend=cudnn
  # trainer: stateful full recompute (needed by the frozen-weight compression and the derived MLP inputs)
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity=full
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_method=uniform
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=1
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_modules=null
  trainer.balance_dp_tokens=true
  # generator: long multi-turn rollouts (natural-length turns, ~2,600-token observations)
  "generator.max_input_length=${MAX_MODEL_LEN}"
  trainer.ckpt_interval=-1
  trainer.project_name=dsv41_long_horizon
)
