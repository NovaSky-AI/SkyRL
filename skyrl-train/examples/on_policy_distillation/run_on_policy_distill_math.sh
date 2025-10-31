set -x

# Running on policy distillation for Math on GSM8K
# Uses Qwen-3-4B-Base as the student model and Qwen-3-8B as the teacher model
# uv run --isolated examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# bash examples/on_policy_distillation/run_on_policy_distill_math.sh

DATA_DIR="/mnt/cluster_storage/data/dapo"
TRAIN_FILE="$DATA_DIR/dapo-math-17k-cleaned.parquet"
TEST_FILE="$DATA_DIR/aime-2024-cleaned.parquet"
LOGGER=wandb

# On Policy Distillation args
TEACHER_MODEL="Qwen/Qwen3-8B"
# huggingface-cli download Qwen/Qwen2.5-Math-7B
# then modify config.json to set max_position_embeddings to 32768
STUDENT_MODEL="/mnt/local_storage/qwen2.5-math"
ADVANTAGE_ESTIMATOR="no_op"
POLICY_LOSS="importance_sampling"
USE_KL_IN_REWARD=true # this adds the kl penalty to the adva
USE_KL_LOSS=false # turns off kl loss in the loss since we are using it directly in the reward

# Placement args
POLICY_NUM_NODES=2
REF_NUM_NODES=2
NUM_GPUS_PER_NODE=8
NUM_INFERENCE_ENGINES=4
INFERENCE_ENGINE_TP_SIZE=2

# enable efa
export SKYRL_LD_LIBRARY_PATH_EXPORT=true
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH

uv run --isolated --extra vllm -m examples.on_policy_distillation.main_on_policy_distill \
  data.train_data="['$TRAIN_FILE']" \
  data.val_data="['$TEST_FILE']" \
  trainer.algorithm.advantage_estimator=$ADVANTAGE_ESTIMATOR \
  trainer.algorithm.policy_loss_type=$POLICY_LOSS \
  trainer.policy.model.path=$STUDENT_MODEL \
  trainer.ref.model.path=$TEACHER_MODEL \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_nodes=$POLICY_NUM_NODES \
  trainer.placement.ref_num_nodes=$REF_NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.policy.fsdp_config.fsdp_size=$NUM_GPUS_PER_NODE \
  trainer.ref.fsdp_config.fsdp_size=$NUM_GPUS_PER_NODE \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$INFERENCE_ENGINE_TP_SIZE \
  trainer.epochs=20 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=1024 \
  trainer.policy_mini_batch_size=256 \
  trainer.micro_forward_batch_size_per_gpu=2 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=2048 \
  generator.sampling_params.max_generate_length=8192 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=$USE_KL_LOSS \
  trainer.algorithm.use_kl_in_reward=$USE_KL_IN_REWARD \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.async_engine=false \
  generator.batched=true \
  environment.env_class=aime \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="aime_on_policy_distillation" \
  trainer.run_name="on_policy_distillation_aime_qwen2.5_7b_base_from_8b" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/aime_7B_on_policy_distillation" \
  $@
