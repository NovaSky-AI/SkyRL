#!/bin/bash
# Multi-core (4 NeuronCore) FSDP GRPO smoke on the `multiply` text env.
#
# The VLM smoke (run_multicore_smoke.sh) needs the cholec-sft-qwen3.5-4b checkpoint,
# which is S3-blocked.  This smoke exercises the same multi-core FSDP2 + in-process
# neuron engine + LoRA weight-sync path on a tiny HF text model, so the infrastructure
# (worker core pinning, neuron collectives, LoRA reload across the step boundary) is
# verified without the VLM vision tower.
#
# Ray's raylet misses the 64 logical-core topology on trn2.48xlarge, so start a head
# with the true `neuron_cores` count; `initialize_ray` attaches via bare `ray.init()`.
set -euo pipefail
ulimit -c unlimited
source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate

ray stop --force >/dev/null 2>&1 || true
ray start --head --disable-usage-stats --num-cpus=192 --resources='{"neuron_cores": 64}' >/dev/null 2>&1

# The driver's in-process engine pins the last logical NeuronCore (63), and each FSDP
# worker pins core == rank in its own process, so nothing is exported from this shell.
export NKI_ENABLE_TRACE_CACHE=0 USE_HUB_KERNELS=NO
export PYTHONFAULTHANDLER=1
cd ~/SkyRL
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}
DATA_DIR=${DATA_DIR:-$HOME/data/multiply}

python examples/train/multiply/multiply_neuron_entrypoint.py \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.use_kl_loss=false \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.policy.model.lora.rank=16 \
  trainer.policy.model.lora.alpha=32 \
  trainer.policy.model.lora.target_modules="all-linear" \
  trainer.strategy=fsdp \
  trainer.placement.colocate_all=false \
  trainer.placement.policy_num_gpus_per_node=4 \
  trainer.epochs=1 \
  trainer.max_training_steps=3 \
  trainer.train_batch_size=4 \
  trainer.policy_mini_batch_size=2 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=0 \
  trainer.hf_save_interval=0 \
  trainer.max_prompt_length=512 \
  trainer.eval_batch_size=8 \
  trainer.eval_before_train=false \
  trainer.eval_interval=0 \
  trainer.remove_microbatch_padding=false \
  trainer.gradient_checkpointing=true \
  generator.sampling_params.max_generate_length=64 \
  generator.sampling_params.temperature=0.7 \
  generator.sampling_params.top_p=1.0 \
  generator.max_turns=1 \
  trainer.policy.optimizer_config.lr=1e-5 \
  generator.inference_engine.backend=neuron \
  generator.inference_engine.enable_prefix_caching=false \
  environment.env_class=multiply \
  generator.n_samples_per_prompt=2 \
  trainer.bf16=true \
  trainer.logger=console \
  trainer.project_name=multiply \
  trainer.run_name=multiply_neuron_smoke \
  trainer.resume_mode=none \
  trainer.log_path="/tmp/skyrl-logs" \
  trainer.dump_eval_results=true \
  trainer.algorithm.loss_reduction=token_mean_legacy \
  trainer.flash_attn=false \
  "$@"
echo "MULTIPLY_NEURON_SMOKE_DONE"
