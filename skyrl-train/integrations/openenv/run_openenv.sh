set -x

# Colocated GRPO training+generation for Qwen2.5-0.5B-Instruct on OpenEnv.

# uv run examples/openenv/dummy_openenv_dataset.py --output_dir $HOME/data/openenv --env_name echo_env
# Env name: echo_env, coding_env, openspiel-env, atari-env, sumo-rl-env, finrl-env

# Prestart the docker environment with 
# cd /skyrl-gym
# uv run envs/openenv/install_environment.py

# export WANDB_API_KEY=<your_key_here>
# bash examples/openenv/run_dummy_openenv.sh

# You can override the default values with e.g.: `NUM_GPUS=1 bash examples/openenv/run_dummy_openenv.sh`.

: "${ENV_NAME:="coding_env"}"
: "${DATA_DIR:="$HOME/data/openenv/$ENV_NAME"}"
: "${NUM_GPUS:=4}"
: "${LOGGER:=wandb}" # change to "console" to print to stdout

: "${INFERENCE_BACKEND:=vllm}"
# : "${INFERENCE_BACKEND:=sglang}"
: "${MAX_TURNS:=4}"

uv run --isolated --extra $INFERENCE_BACKEND --with "openenv@git+https://github.com/meta-pytorch/OpenEnv.git" --with "litellm>=1.75.5" -m integrations.openenv.entrypoints.main_openenv \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen3-4B" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=20 \
  trainer.eval_batch_size=16 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=16 \
  trainer.policy_mini_batch_size=16 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.max_turns=$MAX_TURNS \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=openenv \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="openenv" \
  trainer.run_name="openenv_test" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/openenv_0.5B_ckpt" \
  trainer.dump_data_batch=true \
  
  $@  