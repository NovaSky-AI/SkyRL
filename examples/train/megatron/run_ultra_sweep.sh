#!/usr/bin/env bash
# Throughput / memory sweep harness for Nemotron-3-Ultra-550B (Megatron).
#
# Drives examples.train_scripts.full_context.main_ultra_sweep, which runs the real
# fwd+bwd training path on fabricated rollouts (no vLLM generation) and logs per-step
# peak CUDA memory + step time to $SWEEP_RESULTS_FILE.
#
# All knobs are env vars (with defaults). Example:
#   TP=8 PP=4 CP=1 EP=16 ETP=1 MTPM=131072 MODE=uniform SEQ_LEN=10240 NUM_SEQ=64 \
#   TAG=tp8pp4ep16_mtpm128k SWEEP_RESULTS_FILE=/path/results.jsonl \
#   bash examples/train/megatron/run_ultra_sweep.sh
set -x

# ---- Megatron parallelism (world = TP*PP*CP*DP = 64) ----
TP=${TP:-8}
PP=${PP:-4}
CP=${CP:-1}
EP=${EP:-16}
ETP=${ETP:-1}

# ---- Microbatch token budget (per DP rank) ----
MTPM=${MTPM:-131072}

# ---- Workload ----
MODE=${MODE:-uniform}          # uniform | varlen
SEQ_LEN=${SEQ_LEN:-10240}      # uniform: total tokens/seq
PROMPT_LEN=${PROMPT_LEN:-512}
AVG_LEN=${AVG_LEN:-60000}      # varlen
STD_LEN=${STD_LEN:-30000}
MIN_LEN=${MIN_LEN:-1024}
MAX_LEN=${MAX_LEN:-131072}
NUM_STEPS=${NUM_STEPS:-3}

NUM_NODES=8
NUM_GPUS=8
WORLD=$((NUM_NODES * NUM_GPUS))
DP=$((WORLD / (TP * PP * CP)))

# ---- sequence count & batch sizes ----
# Default: enough sequences so each DP rank forms ~2 full microbatches (uniform),
# or exactly 256 for varlen (the stage-3 target distribution).
if [ -z "${NUM_SEQ:-}" ]; then
  if [ "$MODE" = "varlen" ]; then
    NUM_SEQ=256
  else
    # ceil(2 * MTPM * DP / SEQ_LEN), rounded up to a multiple of 8
    NUM_SEQ=$(( (2 * MTPM * DP + SEQ_LEN - 1) / SEQ_LEN ))
    NUM_SEQ=$(( ((NUM_SEQ + 7) / 8) * 8 ))
    if [ "$NUM_SEQ" -lt 8 ]; then NUM_SEQ=8; fi
  fi
fi
# Round NUM_SEQ up to a multiple of (2*DP) so n_samples=2 and DP divisibility both hold.
LCM=$((2 * DP))
NUM_SEQ=$(( ((NUM_SEQ + LCM - 1) / LCM) * LCM ))
N_SAMPLES=2
TBS=$(( NUM_SEQ / N_SAMPLES ))
MINI=$TBS

TAG=${TAG:-tp${TP}pp${PP}cp${CP}ep${EP}_mtpm${MTPM}_${MODE}}
SWEEP_RESULTS_FILE=${SWEEP_RESULTS_FILE:-/home/ray/ultra_sweep/results.jsonl}
mkdir -p "$(dirname "$SWEEP_RESULTS_FILE")"

# ---- Environment (mirror the validated nemotron recipe) ----
export HF_HOME=${HF_HOME:-/mnt/local_storage/hf_cache}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/mnt/local_storage/.cache}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/mnt/local_storage/.cache/uv}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/mnt/local_storage/.cache/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/mnt/local_storage/.cache/inductor}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/mnt/local_storage/.cache/vllm}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:${LD_LIBRARY_PATH:-}
export SKYRL_LD_LIBRARY_PATH_EXPORT=1
export VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1
export NVTE_FLASH_ATTN=0
export SKYRL_RAY_PG_TIMEOUT_IN_S=${SKYRL_RAY_PG_TIMEOUT_IN_S:-1800}
export SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=${SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S:-2400}
export SKYRL_DUMP_INFRA_LOG_TO_STDOUT=${SKYRL_DUMP_INFRA_LOG_TO_STDOUT:-1}

# ---- sweep trainer params (read by trainer_ultra_sweep.py) ----
export SWEEP_RESULTS_FILE SWEEP_TAG="$TAG" SWEEP_MODE="$MODE"
export SWEEP_NUM_STEPS="$NUM_STEPS" SWEEP_NUM_SEQ="$NUM_SEQ" SWEEP_PROMPT_LEN="$PROMPT_LEN"
export SWEEP_SEQ_LEN="$SEQ_LEN" SWEEP_AVG_LEN="$AVG_LEN" SWEEP_STD_LEN="$STD_LEN"
export SWEEP_MIN_LEN="$MIN_LEN" SWEEP_MAX_LEN="$MAX_LEN"

DATA_DIR="/mnt/local_storage/data/gsm8k"
MODEL_NAME="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"

# vLLM colocated config (matches the recipe; vLLM sleeps during the training step).
NUM_INFERENCE_ENGINES=2
INFERENCE_ENGINE_TP=8
INFERENCE_ENGINE_PP=4

echo "[run_ultra_sweep] TAG=$TAG TP=$TP PP=$PP CP=$CP EP=$EP ETP=$ETP DP=$DP MTPM=$MTPM MODE=$MODE NUM_SEQ=$NUM_SEQ TBS=$TBS MINI=$MINI"

uv run --isolated --extra megatron -m examples.train_scripts.full_context.main_ultra_sweep \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine.tensor_parallel_size=$INFERENCE_ENGINE_TP \
  generator.inference_engine.pipeline_parallel_size=$INFERENCE_ENGINE_PP \
  generator.inference_engine.distributed_executor_backend=ray \
  generator.inference_engine.use_expandable_segments=true \
  trainer.policy.megatron_config.tensor_model_parallel_size=$TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$PP \
  trainer.policy.megatron_config.context_parallel_size=$CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$ETP \
  trainer.policy.megatron_config.transformer_config_kwargs.mtp_num_layers=0 \
  trainer.policy.megatron_config.transformer_config_kwargs.mtp_hybrid_override_pattern=null \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_granularity=full \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_method=uniform \
  trainer.policy.megatron_config.transformer_config_kwargs.recompute_num_layers=1 \
  trainer.policy.megatron_config.optimizer_config_kwargs.overlap_cpu_optimizer_d2h_h2d=true \
  trainer.policy.megatron_config.optimizer_config_kwargs.use_precision_aware_optimizer=true \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_cpu_offload=true \
  trainer.policy.megatron_config.optimizer_config_kwargs.optimizer_offload_fraction=1.0 \
  trainer.remove_microbatch_padding=true \
  trainer.max_tokens_per_microbatch=$MTPM \
  trainer.epochs=1 \
  trainer.eval_before_train=false \
  trainer.eval_interval=100000 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TBS \
  trainer.policy_mini_batch_size=$MINI \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=$PROMPT_LEN \
  generator.max_input_length=$PROMPT_LEN \
  generator.sampling_params.max_generate_length=$((SEQ_LEN - PROMPT_LEN)) \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=false \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.engine_init_kwargs.max_model_len=4096 \
  generator.inference_engine.gpu_memory_utilization=0.6 \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=$N_SAMPLES \
  trainer.num_dummy_steps=$NUM_STEPS \
  trainer.sweep_results_file="$SWEEP_RESULTS_FILE" \
  trainer.sweep_tag="$TAG" \
  trainer.sweep_mode="$MODE" \
  trainer.sweep_num_seq=$NUM_SEQ \
  trainer.sweep_prompt_len=$PROMPT_LEN \
  trainer.sweep_seq_len=$SEQ_LEN \
  trainer.sweep_avg_len=$AVG_LEN \
  trainer.sweep_std_len=$STD_LEN \
  trainer.sweep_min_len=$MIN_LEN \
  trainer.sweep_max_len=$MAX_LEN \
  trainer.logger="console" \
  trainer.project_name="ultra_sweep" \
  trainer.run_name="$TAG" \
  trainer.resume_mode=none \
  trainer.ckpt_interval=100000 \
  trainer.ckpt_path="/mnt/local_storage/ultra_sweep_ckpt" \
  "$@"
