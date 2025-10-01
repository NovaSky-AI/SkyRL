set -x

# Colocated GRPO training+generation for Moonlight-16B-A3B-Instruct on GSM8K with Megatron.
# Runs on 2 nodes of 8xH100s

# uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/megatron/run_megatron_moonlight.sh

# running moonlight16b
# huggingface-cli download moonshotai/Moonlight-16B-A3B-Instruct --local-dir ~/moonlight16b

DATA_DIR="$HOME/data/gsm8k"
LOGGER="wandb"  # change to "console" to print to stdout
MODEL_NAME="$HOME/moonlight16b"

INFERENCE_BACKEND="vllm" # currently only vllm is supported for megatron

NUM_NODES=2
NUM_GPUS=8

MEGATRON_TP=2
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=8
MEGATRON_ETP=1

NUM_INFERENCE_ENGINES=2
INFERENCE_ENGINE_TP=8

# flash attn is not supported for moonlight16b since it is a DeepSeekV3 like model, and uses Multi-Head Latent Attention (MLA)
# https://github.com/NVIDIA/TransformerEngine/blob/483d9594fb070f62966f6a12ed6c90942310b48e/transformer_engine/pytorch/attention/dot_product_attention/utils.py#L483
FLASH_ATTN=false

export SKYRL_PYTHONPATH_EXPORT=1
# make sure PYTHONPATH is set to the location of TransformerEngine installation
export PYTHONPATH="$HOME/anaconda3/lib/python3.12/site-packages"

uv run --isolated --extra $INFERENCE_BACKEND --extra mcore --with blobfile -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.ref_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$INFERENCE_ENGINE_TP \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.ref.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.ref.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.ref.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.ref.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.ref.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  +trainer.policy.megatron_config.transformer_config_kwargs.num_layers_in_last_pipeline_stage=13 \
  +trainer.ref.megatron_config.transformer_config_kwargs.num_layers_in_last_pipeline_stage=13 \
  trainer.use_sample_packing=true \
  trainer.flash_attn=$FLASH_ATTN \
  trainer.epochs=20 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=128 \
  trainer.policy_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.6 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k_megatron" \
  trainer.run_name="gsm8k_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}_ep${MEGATRON_EP}_etp${MEGATRON_ETP}_moonlight16b-a3b" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/gsm8k_megatron_ckpt" \
  $@