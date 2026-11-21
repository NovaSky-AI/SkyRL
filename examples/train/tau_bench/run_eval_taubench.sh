set -x

# NO-TRAINING baseline eval on tau-bench retail (test split, 115 tasks).
# Default policy is Qwen/Qwen3.6-35B-A3B; override MODEL_NAME for any HF model.
# Single 8x H100 node, partitioned:
#   GPUs 6,7 -> user-simulator vLLM OpenAI server (fixed model, talks to the env over HTTP)
#   GPUs 0-5 -> SkyRL eval engines serving the policy (agent) under test
# Uses SkyRL's eval-only entrypoint (main_generate) with async inference so the many
# concurrent multi-turn conversations + user-sim HTTP calls overlap. Reports eval success rate.

export HF_HOME=${HF_HOME:-/mnt/cluster_storage/hf_cache}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}

# --- models (override MODEL_NAME to the exact HF id you want to eval) ---
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3.6-35B-A3B}
USER_SIM_MODEL=${USER_SIM_MODEL:-Qwen/Qwen2.5-7B-Instruct}

DATA_DIR=${DATA_DIR:-/mnt/cluster_storage/data/tau_bench}
VAL_FILE="$DATA_DIR/retail_test.parquet"
# Persist eval result dumps on shared cluster storage (survives job termination).
# Dumps land under $EXPORT_PATH/dumped_evals/eval_only/.
EXPORT_PATH=${EXPORT_PATH:-/mnt/cluster_storage/exports/taubench_eval}
LOGGER=${LOGGER:-console}
EVAL_N=${EVAL_N:-1}              # pass@1; raise for pass^k
MAX_TURNS=${MAX_TURNS:-30}

# --- GPU partition ---
USER_SIM_GPUS=${USER_SIM_GPUS:-6,7}
USER_SIM_TP=${USER_SIM_TP:-2}
USER_SIM_PORT=${USER_SIM_PORT:-8001}
POLICY_GPUS=${POLICY_GPUS:-0,1,2,3,4,5}
NUM_ENGINES=${NUM_ENGINES:-3}
ENGINE_TP=${ENGINE_TP:-2}        # NUM_ENGINES * ENGINE_TP must equal the #policy GPUs

USER_SIM_ENDPOINT="http://127.0.0.1:${USER_SIM_PORT}/v1"

# 1. Build the retail eval parquet (always rebuild so dataset-schema changes take effect;
#    cluster_storage persists across jobs, so a stale parquet would otherwise be reused).
mkdir -p "$DATA_DIR"
uv run --isolated --extra fsdp python examples/train/tau_bench/tau_bench_dataset.py --output_dir "$DATA_DIR"

# 2. Launch the user-simulator vLLM OpenAI server (background) and wait until healthy.
CUDA_VISIBLE_DEVICES=$USER_SIM_GPUS uv run --isolated --extra fsdp \
  vllm serve "$USER_SIM_MODEL" \
  --tensor-parallel-size "$USER_SIM_TP" \
  --host 127.0.0.1 --port "$USER_SIM_PORT" \
  --dtype bfloat16 --gpu-memory-utilization 0.85 \
  --max-model-len 32768 --enable-prefix-caching --trust-remote-code \
  > /tmp/tau_user_sim.log 2>&1 &
USER_SIM_PID=$!
trap 'echo "[script] stopping user-sim pid=$USER_SIM_PID"; kill -TERM "$USER_SIM_PID" 2>/dev/null' EXIT INT TERM

echo "[script] waiting for user-sim server on :$USER_SIM_PORT ..."
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${USER_SIM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "[script] user-sim server is up."
    break
  fi
  if ! kill -0 "$USER_SIM_PID" 2>/dev/null; then
    echo "[script] user-sim server died; see /tmp/tau_user_sim.log"; tail -50 /tmp/tau_user_sim.log; exit 1
  fi
  sleep 10
done

# 3. Eval-only run (no training). Policy engines on the remaining GPUs.
CUDA_VISIBLE_DEVICES=$POLICY_GPUS uv run --isolated --extra fsdp -m skyrl.train.entrypoints.main_generate \
  data.val_data="['$VAL_FILE']" \
  environment.env_class=tau_bench \
  environment.skyrl_gym.tau_bench.user_simulator_endpoint="$USER_SIM_ENDPOINT" \
  environment.skyrl_gym.tau_bench.user_simulator_model="$USER_SIM_MODEL" \
  trainer.policy.model.path="$MODEL_NAME" \
  trainer.placement.colocate_all=false \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.num_engines=$NUM_ENGINES \
  generator.inference_engine.tensor_parallel_size=$ENGINE_TP \
  generator.inference_engine.gpu_memory_utilization=0.85 \
  generator.max_turns=$MAX_TURNS \
  generator.use_conversation_multi_turn=true \
  generator.max_input_length=16384 \
  generator.eval_n_samples_per_prompt=$EVAL_N \
  generator.eval_sampling_params.temperature=0.0 \
  generator.eval_sampling_params.max_generate_length=2048 \
  trainer.max_prompt_length=8192 \
  trainer.dump_eval_results=true \
  trainer.export_path="$EXPORT_PATH" \
  trainer.logger="$LOGGER" \
  trainer.project_name="taubench-eval" \
  trainer.run_name="qwen3.6_35b_retail_baseline" \
  "$@"
