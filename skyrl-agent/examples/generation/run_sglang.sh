set -x
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sglang-verl
data_path=/verl/data/gsm8k/test.parquet
save_path=/verl/data/math/deepseek_v2_lite_gen_test.parquet
model_path=Qwen/Qwen2.5-Coder-7B-Instruct
export HOST_IP="127.0.0.1"
export SGLANG_HOST_IP="127.0.0.1"

# port=6379
# ray start --head \
#     --port=$port \
#     --num-gpus=8 \
#     --include-dashboard=false \
#     --block &

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=$save_path \
    model.path=$model_path \
    +model.trust_remote_code=True \
    rollout.name=async \
    rollout.temperature=1.0 \
    rollout.top_k=50 \
    rollout.top_p=0.7 \
    rollout.prompt_length=2048 \
    rollout.response_length=1024 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.8 \
    +rollout.enable_memory_saver=True \
    +trainer.hybrid_engine=True
