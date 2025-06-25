set -x


uv run --isolated --extra vllm -m skyrl_train.entrypoints.main_base \
    trainer.algorithm.advantage_estimator="grpo" \
    generator.sampling_params.temperature=0.6 \
    generator.sampling_params.top_p=0.95 \
    trainer.logger="wandb" \
    trainer.project_name="skyrl" \
    trainer.run_name="skyrl-remote" \
    trainer.strategy=fsdp2 \
    trainer.train_batch_size=16 \
    trainer.micro_forward_batch_size_per_gpu=20 \
    trainer.micro_train_batch_size_per_gpu=20 \
    trainer.placement.policy_num_gpus_per_node=4 \
    trainer.placement.ref_num_gpus_per_node=4 \
    trainer.policy.sequence_parallel_size=1 \
    trainer.ref.sequence_parallel_size=1 \
    generator.run_engines_locally=False \
    trainer.placement.colocate_all=False \
    generator.remote_inference_engine_urls="['127.0.0.1:8001']" \
    generator.override_existing_update_group=True \
    trainer.resume_mode=null \
    trainer.ckpt_path="$HOME/ckpts/remote_ckpt"
