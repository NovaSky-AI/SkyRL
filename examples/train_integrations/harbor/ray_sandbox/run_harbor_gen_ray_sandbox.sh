#!/bin/bash
# run_harbor_gen.sh with Harbor's `modal` environment served by Ray Sandbox's
# Modal-compatible API instead of Modal's cloud. See README.md.
#
#   MODE=colocated  sandboxes run in this Ray cluster, through a facade on every
#                   node's loopback (node_facades.py). Set EXPECTED_NODES on
#                   multi-node clusters so every node gets one.
#   MODE=service    sandboxes run in a Ray Sandbox service: set MODAL_SERVER_URL
#                   (https://...) and MODAL_TOKEN_SECRET (its RAY_SANDBOX_API_TOKEN).
#
# Run from the SkyRL checkout root after prepare_tasks.py has written tasks/.
# Extra arguments are passed to the generation entrypoint as config overrides.
set -euo pipefail

HERE=examples/train_integrations/harbor/ray_sandbox
DATA="['$HERE/tasks']"
TRIALS_DIR="$HOME/trials_run"

case "${MODE:?set MODE=colocated or service}" in
  colocated)
    # Modal's SDK takes a plain-http command router only from a localhost server
    # URL. A loopback facade needs no token: sandboxes can't reach it.
    export MODAL_SERVER_URL=http://127.0.0.1:50051
    export MODAL_TOKEN_ID=ak-ray MODAL_TOKEN_SECRET=as-ray
    python "$HERE/node_facades.py" --port 50051 --expected-nodes "${EXPECTED_NODES:-1}"
    ;;
  service)
    : "${MODAL_SERVER_URL:?set MODAL_SERVER_URL to the service URL}"
    : "${MODAL_TOKEN_SECRET:?set MODAL_TOKEN_SECRET to the service token}"
    export MODAL_TOKEN_ID=ak-unused
    ;;
  *)
    echo "unknown MODE=$MODE" >&2; exit 2
    ;;
esac

NUM_GPUS=4
# run_harbor_gen.sh uses 8192, which ends most terminus-2 trials with
# ContextLengthExceededError; at 32768, 127 of 128 trials ran to the verifier.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}

uv run --isolated --extra fsdp --extra harbor -m examples.train_integrations.harbor.entrypoints.main_harbor_generate \
  data.train_data="$DATA" \
  data.val_data="$DATA" \
  harbor_trial_config.trials_dir="$TRIALS_DIR" \
  harbor_trial_config.environment.type=modal \
  harbor_trial_config.environment.kwargs.sandbox_timeout_secs=3600 \
  trainer.policy.model.path="Qwen/Qwen3-8B" \
  generator.inference_engine.served_model_name="Qwen3-8B" \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.sampling_params.max_generate_length=4096 \
  trainer.algorithm.max_seq_len=$MAX_MODEL_LEN \
  generator.inference_engine.engine_init_kwargs.max_model_len=$MAX_MODEL_LEN \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  generator.inference_engine.engine_init_kwargs.chat_template=skyrl/train/utils/templates/qwen3_acc_thinking.jinja2 \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.placement.colocate_all=false \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.train_batch_size=$NUM_GPUS \
  trainer.policy_mini_batch_size=$NUM_GPUS \
  trainer.logger=console \
  generator.step_wise_trajectories=true \
  generator.merge_stepwise_output=true \
  generator.rate_limit.enabled=true \
  generator.rate_limit.trajectories_per_second=5 \
  generator.rate_limit.max_concurrency=512 \
  "$@"
