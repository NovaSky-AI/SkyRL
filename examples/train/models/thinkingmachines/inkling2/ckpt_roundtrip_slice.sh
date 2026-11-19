# Dev check (not a recipe): checkpoint save + resume of the colocated full fine-tuning recipe
# (run_dapo_inkling2_fullft_colocated.sh) on the 8-layer / 64-expert slice, 2 nodes / 16 GPUs,
# with the checkpoint on node-local NVMe. PHASE=save runs steps 1-2 (checkpoint each step), PHASE=resume
# resumes the latest checkpoint and runs step 3. SKYRL_CKPT_CHECKSUMS=1 logs per-rank model and optimizer
# checksums at save and load. The slice's rollouts all get the same reward (zero GRPO advantage), so an
# entropy bonus and a large LR make every step change the weights and the optimizer state.
#   MODEL_PATH=<slice> DATA_DIR=<dapo> CKPT_PATH=/mnt/nvme/<dir> PHASE=save bash examples/train/models/thinkingmachines/inkling2/ckpt_roundtrip_slice.sh
set -x
PHASE=${PHASE:-save}
if [ "$PHASE" = save ]; then STEPS=2; RESUME=null; else STEPS=3; RESUME=latest; fi
export SKYRL_CKPT_CHECKSUMS=${SKYRL_CKPT_CHECKSUMS:-1}
MAX_RESPONSE_LENGTH=1024 TRAIN_BATCH_SIZE=16 MINI_BATCH_SIZE=8 N_SAMPLES_PER_PROMPT=4 \
EVAL_BEFORE_TRAIN=false EVAL_INTERVAL=-1 CKPT_INTERVAL=1 MAX_TRAINING_STEPS=$STEPS LOGGER=console \
INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION=0.6 LR=${LR:-1e-4} \
bash examples/train/models/thinkingmachines/inkling2/run_dapo_inkling2_fullft_colocated.sh \
  trainer.placement.policy_num_nodes=2 \
  trainer.policy.megatron_config.expert_model_parallel_size=16 \
  generator.inference_engine.num_engines=2 \
  trainer.max_ckpts_to_keep=1 \
  trainer.algorithm.use_entropy_loss=true \
  trainer.resume_mode=$RESUME \
  trainer.run_name=${RUN_NAME:-inkling2_ckpt_roundtrip_slice} "$@"
