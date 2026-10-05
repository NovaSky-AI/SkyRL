set -euo pipefail

# Reproduction for https://github.com/NovaSky-AI/SkyRL/issues/2246: prefix-cache reuse across weight syncs
# when LLM calls reach SkyRL's inference endpoint through the OpenAI-compatible route without a cache_salt.
#
# Fully-async GRPO on multi-turn GSM8K with Qwen3-0.6B: 1 trainer GPU + 1 vLLM GPU (non-colocated), prefix
# caching on, NUM_STEPS training steps. Run it once on main and once on the fix branch, then compare:
#
#   uv run --isolated examples/train/turn_level_rewards/gsm8k_multi_turn_dataset.py --output_dir $HOME/data/gsm8k_multi_turn
#   OUT_DIR=$HOME/prefix_cache_repro/main   bash examples/train/fully_async/prefix_cache_salt/run_prefix_cache_salt_repro.sh
#   OUT_DIR=$HOME/prefix_cache_repro/branch bash examples/train/fully_async/prefix_cache_salt/run_prefix_cache_salt_repro.sh
#   uv run --isolated examples/train/fully_async/prefix_cache_salt/summarize.py $HOME/prefix_cache_repro/main $HOME/prefix_cache_repro/branch

: "${DATA_DIR:=$HOME/data/gsm8k_multi_turn}"
: "${OUT_DIR:=$HOME/prefix_cache_repro/run}"
: "${NUM_STEPS:=5}"
: "${TRAIN_BATCH_SIZE:=16}"
: "${N_SAMPLES:=4}"
: "${LR:=1.0e-5}"

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR/requests.jsonl" "$OUT_DIR/metrics.jsonl"

# One epoch of exactly NUM_STEPS steps.
uv run --isolated --no-project --with pandas --with pyarrow python - <<EOF
import pandas as pd
pd.read_parquet("$DATA_DIR/train.parquet").head($NUM_STEPS * $TRAIN_BATCH_SIZE).to_parquet("$OUT_DIR/train.parquet")
EOF

SKYRL_PREFIX_CACHE_REPRO_DIR="$OUT_DIR" uv run --isolated --extra fsdp -m examples.train.fully_async.prefix_cache_salt.main_openai_path \
  data.train_data="['$OUT_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.policy.model.path="Qwen/Qwen3-0.6B" \
  trainer.strategy=fsdp \
  trainer.placement.colocate_all=false \
  trainer.placement.policy_num_gpus_per_node=1 \
  generator.inference_engine.num_engines=1 \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.enable_prefix_caching=true \
  generator.inference_engine.gpu_memory_utilization=0.6 \
  generator.inference_engine.engine_init_kwargs.enable_prompt_tokens_details=true \
  trainer.fully_async.enabled=true \
  trainer.fully_async.max_staleness_steps=1 \
  trainer.fully_async.num_parallel_generation_workers=$(( TRAIN_BATCH_SIZE * 2 )) \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.algorithm.use_kl_loss=false \
  trainer.policy.optimizer_config.lr=$LR \
  trainer.epochs=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$TRAIN_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.max_prompt_length=1024 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.ckpt_interval=-1 \
  trainer.resume_mode=null \
  trainer.logger=console \
  generator.batched=false \
  generator.max_turns=3 \
  generator.n_samples_per_prompt=$N_SAMPLES \
  generator.sampling_params.logprobs=0 \
  generator.sampling_params.max_generate_length=256 \
  environment.env_class=gsm8k_multi_turn \
  "$@"
