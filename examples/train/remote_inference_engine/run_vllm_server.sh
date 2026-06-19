# Launches a standalone SkyRL inference deployment for Qwen2.5-1.5B-Instruct on 4 GPUs.
# bash examples/train/remote_inference_engine/run_vllm_server.sh
#
# This uses the `serve` entrypoint, which brings up the vLLM server group + router
# and keeps them alive. On startup it logs the `proxy_url` (data plane) and
# `server_urls` (control plane); plug those into the training run (see run_remote.sh).
set -x

CUDA_VISIBLE_DEVICES=4,5,6,7 uv run --isolated --extra fsdp -m skyrl.train.entrypoints.serve \
    trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
    trainer.placement.colocate_all=false \
    generator.inference_engine.num_engines=1 \
    generator.inference_engine.tensor_parallel_size=4 \
    generator.inference_engine.max_num_batched_tokens=8192 \
    generator.inference_engine.max_num_seqs=1024 \
    generator.inference_engine.gpu_memory_utilization=0.9
