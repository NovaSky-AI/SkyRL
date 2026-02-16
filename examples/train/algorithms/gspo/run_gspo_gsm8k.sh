# Colocated GSPO training+generation for Qwen2.5-1.5B-Instruct on GSM8K.

# uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# bash examples/algorithms/gspo/run_gspo_gsm8k.sh

bash examples/gsm8k/run_gsm8k.sh \
  trainer.algorithm.policy_loss_type="gspo" \
  trainer.algorithm.loss_reduction="sequence_mean" \
  trainer.micro_forward_batch_size_per_gpu=16 \
  trainer.micro_train_batch_size_per_gpu=16 \
  trainer.run_name="gspo_gsm8k"
