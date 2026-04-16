set -x

# Colocated GRPO training+generation for Qwen3.5-0.8B on GSM8K.

# uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/train/models/run_qwen3.5_0.8b.sh

: "${DATA_DIR:="$HOME/data/gsm8k"}"
: "${NUM_GPUS:=4}"
: "${LOGGER:=wandb}" # change to "console" to print to stdout

: "${INFERENCE_BACKEND:=vllm}"

# Qwen 3.5 flags
MODEL_NAME="Qwen/Qwen3.5-0.8B"
LANGUAGE_MODEL_ONLY=true
# there are issues with sample packing for GDN layers + flash attention
# https://github.com/huggingface/transformers/issues/44910 
# https://github.com/QwenLM/Qwen3.5/issues/104 
# disabling for now
USE_SAMPLE_PACKING=false # sample packing 

uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.policy.language_model_only=$LANGUAGE_MODEL_ONLY \
  trainer.ref.language_model_only=$LANGUAGE_MODEL_ONLY \
  generator.inference_engine.language_model_only=$LANGUAGE_MODEL_ONLY \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  trainer.epochs=20 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.use_sample_packing=$USE_SAMPLE_PACKING \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=1024 \
  trainer.policy_mini_batch_size=256 \
  trainer.micro_forward_batch_size_per_gpu=16 \
  trainer.micro_train_batch_size_per_gpu=16 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="qwen3.5-0.8b" \
  trainer.run_name="qwen3.5-0.8b_fsdp" \
  trainer.resume_mode=null \
  trainer.log_path="/tmp/skyrl-logs" \
  trainer.ckpt_path="$HOME/ckpts/qwen3.5-0.8b_ckpt" \
  $@
