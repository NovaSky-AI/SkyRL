set -exo pipefail

# Colocated GRPO training+generation for GPT-OSS-20B on GSM8K.
# NOTE (sumanthrh): Currently, gpt-oss requires flash attention to be disabled since attention sinks are not supported: https://github.com/Dao-AILab/flash-attention/issues/1797
# We thus disable flash attention as well as sample packing
# We only support GPT-OSS training in BF16 precision and single-turn tasks at the moment, and are actively working on multi-turn support.

# uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/gptoss/run_gsm8k_gptoss.sh

# NOTE (sumanthrh): `micro_train_batch_size_per_gpu` and `micro_forward_batch_size_per_gpu` can be tuned

DATA_DIR="$HOME/data/gsm8k"
NUM_GPUS=8
LOGGER="wandb"  # change to "console" to print to stdout

INFERENCE_BACKEND="vllm"
# INFERENCE_BACKEND="sglang"

uv run --isolated --extra $INFERENCE_BACKEND -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="unsloth/gpt-oss-20b-BF16" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=2 \
  trainer.flash_attn=false \
  trainer.use_sample_packing=false \
  generator.inference_engine_tensor_parallel_size=4 \
  generator.enforce_eager=true \
  trainer.epochs=20 \
  trainer.eval_batch_size=32 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=32 \
  trainer.policy_mini_batch_size=32 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=5 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=4096 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=4 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k_gptoss" \
  trainer.run_name="gsm8k_test_gptoss_low" \
  trainer.resume_mode=latest \
  trainer.ckpt_path="$HOME/ckpts/gsm8k_1.5B_ckpt_gptoss" \
  +generator.chat_template_kwargs={reasoning_effort:'low'} \
  trainer.dump_data_batch=true \
  $@
