# My key
export DAYTONA_API_KEY=YOUR_KEY_HERE
export WANDB_API_KEY=YOUR_KEY_HERE

# Got after hf download DCAgent/code-contests-sandboxes-with-tests --repo-type=dataset
# cd into the downloaded folder, and do:
# python extract_parquet_tasks.py tasks_new.parquet ./extracted_tasks
TRAIN_DATA="['/home/ray/.cache/huggingface/hub/datasets--DCAgent--code-contests-sandboxes-with-tests/snapshots/23155a8cc2da4e0cbeea3b99fe78f8fc80c1aed4/extracted_tasks']"
# Got after hf download DCAgent/code-contests-sandboxes-with-tests-dev --repo-type=dataset
EVAL_DATA="['/home/ray/.cache/huggingface/hub/datasets--DCAgent--code-contests-sandboxes-with-tests-dev/snapshots/23155a8cc2da4e0cbeea3b99fe78f8fc80c1aed4/extracted_tasks']"

TRIALS_DIR="/home/ray/trials_run"
CKPTS_DIR="/home/ray/otagent/ckpts"
EXPORTS_DIR="/home/ray/otagent/exports"
CHAT_TEMPLATE_PATH="/home/ray/default/SkyRLHarbor3/skyrl-train/skyrl_train/utils/templates/qwen3_acc_thinking.jinja2"

NUM_GPUS=4

# Run SkyRL command
uv run --isolated --extra vllm --extra harbor -m examples.terminal_bench.entrypoints.main_tbench \
  data.train_data=$TRAIN_DATA \
  data.val_data=$EVAL_DATA \
  trainer.policy.model.path=Qwen/Qwen3-8B \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.use_kl_loss=true \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.ref_num_nodes=1 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.epochs=3 \
  trainer.eval_batch_size=128 \
  trainer.eval_before_train=true \
  trainer.eval_interval=20 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=64 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=5 \
  trainer.hf_save_interval=5 \
  trainer.max_prompt_length=2048 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.logger=wandb \
  trainer.project_name=harbor \
  trainer.run_name=codecontest \
  trainer.resume_mode=latest \
  trainer.export_path=$EXPORTS_DIR \
  trainer.ckpt_path=$CKPTS_DIR \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  generator.inference_engine.served_model_name=Qwen3-8B \
  generator.inference_engine.enable_http_endpoint=true \
  generator.inference_engine.http_endpoint_host=127.0.0.1 \
  generator.inference_engine.http_endpoint_port=8000 \
  generator.inference_engine.engine_init_kwargs.chat_template=$CHAT_TEMPLATE_PATH \
  generator.sampling_params.max_generate_length=30720 \
  generator.n_samples_per_prompt=8 \
  generator.eval_n_samples_per_prompt=4 \
  generator.batched=false \
  terminal_bench_config.trials_dir=$TRIALS_DIR \
  $@
