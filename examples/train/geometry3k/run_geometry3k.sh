set -x

# Multi-turn GRPO training for Geometry-3K (VLM).

# uv run examples/train/geometry3k/geometry_3k_dataset.py --output_dir $HOME/data/geometry_3k
# export WANDB_API_KEY=<your_key_here>
# bash examples/train/geometry3k/run_geometry3k.sh

: "${DATA_DIR:="$HOME/data/geometry_3k"}"
: "${NUM_GPUS:=8}"

if [ ! -f "$DATA_DIR/train.parquet" ]; then
  echo "=== Generating Geometry-3K dataset ==="
  uv run examples/train/geometry3k/geometry_3k_dataset.py --output_dir "$DATA_DIR"
fi
: "${LOGGER:=wandb}"
: "${INFERENCE_BACKEND:=vllm}"
: "${EXPORT_PATH:="$HOME/exports/geometry3k_vlm"}"
: "${DUMP_EVAL_RESULTS:=true}"

# Requires a specific vLLM commit for multi-modal generation. We install into a
# dedicated venv so the pyproject-managed `.venv` (which pins stable vllm +
# torch 2.10) stays usable for other examples. `UV_OVERRIDE` forces torch 2.11
# and a torch-2.11-compatible flash-attn in a single resolve.
: "${G3K_VENV:="$HOME/.venvs/skyrl-g3k"}"
OVERRIDES="$(dirname "$0")/custom_vllm_overrides.txt"

if [ ! -x "$G3K_VENV/bin/python" ]; then
  uv venv "$G3K_VENV" --python 3.12
fi

VLLM_USE_PRECOMPILED=1 \
UV_OVERRIDE="$OVERRIDES" \
  uv pip install \
    --python "$G3K_VENV/bin/python" \
    --index-strategy unsafe-best-match \
    --extra-index-url https://flashinfer.ai/whl/cu128 \
    -e "$PWD[fsdp]" \
    pylatexenc

PY="$G3K_VENV/bin/python"
# Ray workers on the existing (anaconda3-based) cluster must use this venv's
# interpreter so they can import omegaconf, the custom vllm fork, torch 2.11, etc.
# pyproject constraint-dependencies pins flashinfer-jit-cache==0.6.6 which
# clashes with the fork vllm's flashinfer-python==0.6.7 at flashinfer import
# time. vllm picks FLASH_ATTN for both text and vit anyway; flashinfer only
# needs to import during backend enumeration, so bypass the defensive check.
export FLASHINFER_DISABLE_VERSION_CHECK=1
export RAY_JOB_CONFIG_JSON_ENV_VAR='{"runtime_env":{"py_executable":"'"$PY"'","env_vars":{"FLASHINFER_DISABLE_VERSION_CHECK":"1"}}}'

_SKYRL_USE_NEW_INFERENCE=1 "$PY" examples/train/geometry3k/geometry3k_entrypoint.py \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/test.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen3-VL-8B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  trainer.epochs=6 \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=256 \
  trainer.policy_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.ckpt_interval=5 \
  trainer.use_sample_packing=false \
  trainer.max_prompt_length=1024 \
  generator.sampling_params.max_generate_length=2048 \
  generator.max_turns=3 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=false \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=false \
  generator.vision_language_generator=true \
  environment.env_class=geometry3k \
  generator.n_samples_per_prompt=8 \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="geometry3k" \
  trainer.run_name="geometry3k_vlm" \
  trainer.resume_mode=null \
  trainer.log_path="/tmp/skyrl-logs" \
  trainer.export_path="$EXPORT_PATH" \
  trainer.dump_eval_results="$DUMP_EVAL_RESULTS" \
  trainer.ckpt_path="$HOME/ckpts/geometry3k_vlm_ckpt" \
  $@
