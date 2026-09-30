#!/usr/bin/env bash
# Evaluate CodeContests on two Qwen3-8B engines with a constrained GPU KV cache.
# Usage: uv run --isolated --no-project bash examples/train_integrations/harbor/kv_offload/run.sh baseline
set -euo pipefail

BACKEND=${1:?Expected baseline or cpu}
shift
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/../../../.."
DATA_DIR=${DATA_DIR:-$HOME/data/harbor/CodeContests}
ENV_FILE=${ENV_FILE:-$HOME/hackskyrl/.env.srh}
RUN_DIR=${RUN_DIR:-/tmp/skyrl-kv-offload/$BACKEND-$(date -u +%Y%m%d-%H%M%S)}
if [[ -e "$RUN_DIR/driver.log" || -e "$RUN_DIR/metrics.jsonl" ]]; then
  echo "Run directory already contains an evaluation: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"
RUN_DIR=$(realpath "$RUN_DIR")
NUM_SAMPLES=${NUM_SAMPLES:-500}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-200}
GPU_BLOCKS=${GPU_BLOCKS:-50000}
EXTRAS=(--extra fsdp --extra harbor --with ray==2.58.0)
CONNECTOR=()

case "$BACKEND" in
  baseline) ;;
  cpu)
    CONNECTOR=('generator.inference_engine.engine_init_kwargs.kv_transfer_config={"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"block_size":64,"cpu_bytes_to_use":128000000000}}')
    ;;
  *) echo "Unknown backend: $BACKEND" >&2; exit 2 ;;
esac

# uv reuses both env files when Ray starts worker processes.
cat > "$RUN_DIR/backend.env" <<EOF
SKYRL_HARBOR_NUM_SAMPLES=$NUM_SAMPLES
SKYRL_HARBOR_ARTIFACT_DIR=$RUN_DIR
SKYRL_VLLM_START_PORT=${START_PORT:-8000}
PYTHONHASHSEED=0
EOF

uv run --isolated --frozen "${EXTRAS[@]}" \
  --env-file "$ENV_FILE" --env-file "$RUN_DIR/backend.env" \
  -m examples.train_integrations.harbor.kv_offload.main_generate \
  "data.train_data=['$DATA_DIR']" "data.val_data=['$DATA_DIR']" \
  "harbor_trial_config.trials_dir=$RUN_DIR/trials" \
  "trainer.log_path=$RUN_DIR/infra" \
  trainer.policy.model.path=Qwen/Qwen3-8B \
  generator.inference_engine.served_model_name=Qwen3-8B \
  generator.inference_engine.num_engines=2 \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.pipeline_parallel_size=1 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.enable_ray_prometheus_stats=false \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.50 \
  generator.inference_engine.max_num_seqs=256 \
  generator.inference_engine.engine_init_kwargs.max_model_len=32768 \
  generator.inference_engine.engine_init_kwargs.kv_cache_dtype=fp8 \
  generator.inference_engine.engine_init_kwargs.block_size=16 \
  "generator.inference_engine.engine_init_kwargs.num_gpu_blocks_override=$GPU_BLOCKS" \
  "generator.inference_engine.engine_init_kwargs.chat_template=$PWD/skyrl/train/utils/templates/qwen3_acc_thinking.jinja2" \
  generator.sampling_params.max_generate_length=12288 \
  trainer.algorithm.max_seq_len=32768 \
  generator.step_wise_trajectories=true generator.merge_stepwise_output=true \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.placement.colocate_all=false \
  trainer.placement.policy_num_gpus_per_node=2 \
  trainer.placement.ref_num_gpus_per_node=2 \
  trainer.train_batch_size=2 trainer.policy_mini_batch_size=2 \
  trainer.logger=console \
  generator.rate_limit.enabled=true \
  generator.rate_limit.trajectories_per_second=5 \
  "generator.rate_limit.max_concurrency=$MAX_CONCURRENCY" \
  harbor_trial_config.agent.kwargs.model_info.max_input_tokens=32768 \
  harbor_trial_config.agent.kwargs.model_info.max_output_tokens=12288 \
  harbor_trial_config.agent.kwargs.llm_kwargs.max_tokens=12288 \
  harbor_trial_config.environment.kwargs.connection_pool_maxsize=null \
  "${CONNECTOR[@]}" "$@" > "$RUN_DIR/driver.log" 2>&1
