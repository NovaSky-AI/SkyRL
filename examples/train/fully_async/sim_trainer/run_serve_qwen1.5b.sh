set -x

# Inference-only serving for Qwen2.5-1.5B-Instruct with 4 engines.
#
# Spins up the vLLM server groups + vllm-router and keeps them alive so that
# client code / generation configs can be iterated against a fixed deployment.
# Point your OpenAI-compatible client at the printed proxy_url.
#
# bash examples/fully_async/sim_trainer/serve.5b.sh
#
# Requires 4 GPUs (4 engines x tensor_parallel_size=1). Serving must be
# non-colocated (there is no trainer to share GPUs with).
#
# Override on the command line, e.g. enable prefill-decode (2 prefill + 2 decode):
#   bash examples/serve/run_serve_qwen1.5b.sh \
#     generator.inference_engine.enable_pd=true \
#     generator.inference_engine.num_prefill=2 \
#     generator.inference_engine.engine_init_kwargs.kv_transfer_config.kv_connector=NixlConnector

: "${MODEL:=Qwen/Qwen2.5-1.5B-Instruct}"
: "${NUM_ENGINES:=2}"
: "${TP_SIZE:=1}"
: "${INFERENCE_BACKEND:=vllm}"

uv run --isolated --extra fsdp -m skyrl.train.entrypoints.serve \
  trainer.policy.model.path="$MODEL" \
  trainer.placement.colocate_all=false \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.num_engines=$NUM_ENGINES \
  generator.inference_engine.tensor_parallel_size=$TP_SIZE \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.log_path="/tmp/skyrl-serve-logs" \
  "$@"
