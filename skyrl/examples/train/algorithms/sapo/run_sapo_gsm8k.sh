# Colocated SAPO training+generation for Qwen2.5-1.5B-Instruct on GSM8K.

# uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# bash examples/algorithms/sapo/run_sapo_gsm8k.sh

bash examples/gsm8k/run_gsm8k.sh \
  trainer.algorithm.policy_loss_type="sapo" \
  trainer.algorithm.sapo.tau_pos=1.0 \
  trainer.algorithm.sapo.tau_neg=1.05 \
  trainer.algorithm.loss_reduction="sequence_mean" \
  trainer.run_name="sapo_gsm8k"