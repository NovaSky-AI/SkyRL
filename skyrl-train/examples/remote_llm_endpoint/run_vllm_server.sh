# Launches vllm server for Qwen2.5-1.5B-Instruct on 4 GPUs.
# bash examples/remote_llm_endpoint/run_vllm_server.sh
set -x

CUDA_VISIBLE_DEVICES=4,5,6,7 uv run --isolated --extra vllm \
    -m skyrl_train.inference_engines.vllm.vllm_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --tensor-parallel-size 4 \
    --seed 42 \
    --distributed-executor-backend ray \
    --max-model-len 4096 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --dtype bfloat16 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --enable-sleep-mode \
    --max-num_batched_tokens 8192 \
    --max-num-seqs 1024 \
    --host 127.0.0.1 --port 8001 --worker-extension-cls skyrl_train.inference_engines.vllm.vllm_engine.WorkerWrap