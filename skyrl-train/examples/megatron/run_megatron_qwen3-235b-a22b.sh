set -x

# Colocated GRPO training+generation for Qwen3-235B-A22B on GSM8K with Megatron.
# Runs on 8 nodes of 8xH100s

# uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/megatron/run_megatron_qwen3-235b-a22b.sh

LOGGER="wandb"  # change to "console" to print to stdout

# Make sure these paths are accessible by or present on all nodes
DATA_DIR="$HOME/data/gsm8k"
# download Qwen/Qwen3-235B-A22B-Instruct-2507 from huggingface
# `pip install huggingface_hub hf_transfer`
# `HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Qwen/Qwen3-235B-A22B-Instruct-2507 --local-dir ~/qwen235b`
MODEL_NAME="$HOME/qwen235b"

NUM_NODES=8
NUM_GPUS=8

### Megatron configuration
# the max TP that can be used is 4, since Qwen3-235B-A22B uses Grouped Query Attention with 4 groups
MEGATRON_TP=4
MEGATRON_PP=16
MEGATRON_CP=1
MEGATRON_EP=4
MEGATRON_ETP=1
# Qwen3-235B-A22B has 94 blocks, so we need to set the last pipeline stage layer to use 4 blocks
MEGATRON_LAST_PIPELINE_STAGE_LAYER=4
FLASH_ATTN=true
# configure optimizer offloading
OPTIMIZER_OFFLOAD=true
OPTIMIZER_OFFLOAD_FRACTION=1.0

### Inference engine configuration
INFERENCE_BACKEND="vllm" # currently only vllm is supported for megatron
NUM_INFERENCE_ENGINES=4
# this is not ideal at the moment - enable inference engine pp in order to avoid this
# https://github.com/NovaSky-AI/SkyRL/issues/353
INFERENCE_ENGINE_TP=16
# the default max model len for Qwen3-235B-A22B-Instruct-2507 is 262K, and VLLM checks that 
# the KV cache memory allocated is enough to serve 1 request with max model len. Lowering to the actual
# max model len value for this script.
INFERENCE_ENGINE_MAX_MODEL_LEN=2048

# no kl loss, so just use the policy model
USE_KL_LOSS=false

export SKYRL_PYTHONPATH_EXPORT=1
# make sure PYTHONPATH is set to the location of TransformerEngine installation
export PYTHONPATH="$HOME/anaconda3/lib/python3.12/site-packages"

uv run --isolated --extra $INFERENCE_BACKEND --extra mcore -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$INFERENCE_ENGINE_TP \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=$OPTIMIZER_OFFLOAD \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=$OPTIMIZER_OFFLOAD_FRACTION \
  +trainer.policy.megatron_config.transformer_config_kwargs.num_layers_in_last_pipeline_stage=$MEGATRON_LAST_PIPELINE_STAGE_LAYER \
  +generator.engine_init_kwargs.max_model_len=$INFERENCE_ENGINE_MAX_MODEL_LEN \
  trainer.use_sample_packing=true \
  trainer.flash_attn=$FLASH_ATTN \
  trainer.epochs=20 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=128 \
  trainer.policy_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.80 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k_megatron" \
  trainer.run_name="gsm8k_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}_ep${MEGATRON_EP}_etp${MEGATRON_ETP}_qwen3-235b-a22b" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/gsm8k_megatron_ckpt" \
  $@