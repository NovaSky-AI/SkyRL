#!/usr/bin/env bash
# CholecT50 GRPO (RLVR) on SkyRL for AWS Trainium2 (TorchNeuron) -- single core.
#
# Mirrors the TRL trainium2 leg one-to-one where the two frameworks allow:
#   Qwen3.5-4B VLM (SFT checkpoint), 256/32 CholecT50 partition, LoRA r=16,
#   GRPO: 4 prompts/batch, 8 rollouts/prompt, lr 1e-5, beta 0.0 (no ref model),
#   temp 0.7, max_tokens 256, 1 epoch, bf16 eager, single NeuronCore.
#
# Differences from the A10G full-model SkyRL run (run_cholec.sh):
#   * LoRA r16 + beta=0.0 (the full-model run used full fine-tune + KL 0.04).
#   * backend=neuron (in-process transformers model.generate) instead of vLLM.
#   * colocate_all=false, single core (the A10G run used 4xA10G colocated).
#
# Smoke test: pass `trainer.max_training_steps=1` as an extra arg.
set -euo pipefail

# The torch-neuronx DLAMI ships the patched torch + Neuron runtime in a managed
# venv. torch-neuronx is not on PyPI (and the `neuron` extra deliberately does
# not declare `torch`), so the repo's usual `uv run --isolated --extra neuron`
# cannot resolve it; run straight from the DLAMI venv instead (the same path the
# TRL trainium2 leg uses). `setup_skyrl.sh` has already pinned the venv's
# torch-neuronx/transformers/peft and installed skyrl+skyrl-gym editable.
# shellcheck disable=SC1091
source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate

# Single NeuronCore of a trn2.48xlarge (0 of 48). Pin the visible core + disable
# the NKI trace cache (its ~10GB HBM footprint otherwise OOMs a 25.8GB core).
export NEURON_RT_VISIBLE_CORES=0
export NKI_ENABLE_TRACE_CACHE=0

# transformers "hub kernels" (kernels-community fla + mamba-ssm) for the Qwen3.5
# gated-delta-net layers are CUDA-only; on neuron they must fall back to reference
# PyTorch (the same path the TRL trainium2 leg uses).
export USE_HUB_KERNELS=NO

# Resolve the SkyRL repo root (this script lives at examples/train/cholec/,
# three levels below the repo root).
SKYRL_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." &>/dev/null && pwd)"
cd "$SKYRL_ROOT"

: "${DATA_DIR:="$HOME/cholec/skyrl_parquet"}"
: "${MODEL_PATH:="$HOME/cholec/merged"}"
: "${EXPORT_PATH:="$HOME/exports/cholec_skyrl_neuron"}"
: "${CKPT_PATH:="$HOME/ckpts/cholec_skyrl_neuron"}"

# The TRL leg's exact LoRA target list (Qwen3.5-4B attention + MLP + gated-delta-net
# projections). If a module name does not resolve under a given transformers/PEFT
# release, swap this for `trainer.policy.model.lora.target_modules="all-linear"`.
LORA_TARGETS="['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj']"

python examples/train/cholec/cholec_entrypoint.py \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/val.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.use_kl_loss=false \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.policy.model.lora.rank=16 \
  trainer.policy.model.lora.alpha=32 \
  trainer.policy.model.lora.target_modules="$LORA_TARGETS" \
  trainer.strategy=fsdp \
  trainer.placement.colocate_all=false \
  trainer.placement.policy_num_gpus_per_node=1 \
  trainer.epochs=1 \
  trainer.train_batch_size=4 \
  trainer.policy_mini_batch_size=2 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=0 \
  trainer.hf_save_interval=0 \
  trainer.max_prompt_length=4096 \
  trainer.eval_batch_size=32 \
  trainer.eval_before_train=true \
  trainer.eval_interval=16 \
  trainer.remove_microbatch_padding=false \
  trainer.gradient_checkpointing=true \
  generator.sampling_params.max_generate_length=256 \
  generator.sampling_params.temperature=0.7 \
  generator.sampling_params.top_p=1.0 \
  generator.max_turns=1 \
  trainer.policy.optimizer_config.lr=1e-5 \
  generator.inference_engine.backend=neuron \
  generator.inference_engine.enable_prefix_caching=false \
  generator.vision_language_generator=true \
  environment.env_class=cholec \
  generator.n_samples_per_prompt=8 \
  trainer.bf16=true \
  trainer.logger=console \
  trainer.project_name=cholec \
  trainer.run_name=cholec_rl_qwen3_5_4b_neuron \
  trainer.resume_mode=none \
  trainer.log_path="/tmp/skyrl-logs" \
  trainer.export_path="$EXPORT_PATH" \
  trainer.dump_eval_results=true \
  trainer.ckpt_path="$CKPT_PATH" \
  trainer.algorithm.loss_reduction=token_mean_legacy \
  trainer.flash_attn=false \
  "$@"
