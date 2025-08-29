# Launches vllm server for Qwen2.5-1.5B-Instruct on 4 GPUs.
# bash examples/remote_inference_engine/run_vllm_server.sh
set -x

# NOTE (sumanthrh): Currently, there's an issue with distributed executor backend ray for vllm 0.9.2.
# For standalone server, we use mp for now. 
CUDA_VISIBLE_DEVICES=3 uv run --isolated --extra vllm --env-file examples/mini_swe_agent/.env.miniswe -m skyrl_train.inference_engines.vllm.vllm_server \
    --model Qwen/Qwen3-4B \
    --tensor-parallel-size 1 \
    --host 127.0.0.1 \
    --port 8001 \
    --seed 42 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --enable-sleep-mode \
    --max-num_batched_tokens 8192 \
    --max-num-seqs 1024 \
    --trust-remote-code \
    --distributed-executor-backend ray \
    --worker-extension-cls skyrl_train.inference_engines.vllm.vllm_engine.WorkerWrap