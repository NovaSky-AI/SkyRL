# GOLD: Cross-Tokenizer On-Policy Distillation

This folder contains scripts for running **GOLD (General On-policy Logit Distillation)** on SkyRL, enabling distillation between models with different tokenizers (e.g., Qwen and Llama).

GOLD extends standard on-policy distillation to support student-teacher pairs with **different tokenizers** using ULD (Universal Logit Distillation), which aligns token spans and compares sorted probability distributions.

## Cross-Tokenizer Distillation
When tokenizers differ, comparing log probabilities directly is not meaningful. GOLD uses the ULD methodology:

1. **Re-tokenization**: Decode student-generated text and re-tokenize with teacher's tokenizer
2. **Token Alignment**: Use greedy text matching to align token spans between student and teacher sequences
3. **Probability Merging**: Merge probabilities for aligned spans using geometric mean
4. **Sorted Comparison**: Sort probability distributions and compute L1 distance (ULD approach)

This allows meaningful comparison of probability distributions even when vocabularies differ.

## Files

- `main_gold_distill.py`: Main trainer with `GOLDDistillationTrainer` class
- `gold_utils.py`: GOLD/ULD utilities (alignment, probability merging, loss computation)
- `gold_workers.py`: Custom workers that return logits for GOLD computation
- `run_gold_distill_qwen_to_llama.sh`: Example script for Qwen-to-Llama distillation
- `run_gold_distill_llama_to_qwen.sh`: Example script for Llama-to-Qwen distillation
- `run_gold_test_modal.sh`: Integration test script for Modal

## Implementation Details

### Why Custom Workers?

GOLD distillation requires **full logits** (shape `[batch, seq_len, vocab_size]`) to compute sorted probability distributions, but the standard RL training pipeline uses **log probabilities** (shape `[batch, seq_len]`). To support both use cases:

- **`GOLDFSDPPolicyWorkerBase`**: Inherits standard `_forward_micro_batch` for RL training (returns log probs), but adds a `forward_logits` method specifically for GOLD reward computation.
- **`GOLDFSDPRefWorkerBase`**: Overrides `_forward_micro_batch` to always return logits, since the teacher model is only used for GOLD reward computation.

This design allows the policy model to participate in both standard RL training (which needs log probs for importance sampling) and GOLD distillation (which needs logits for ULD loss).

### Reward Computation Flow

1. **`fwd_logprobs_values_reward`**: Runs standard forward pass for log probs, then re-tokenizes student output with teacher tokenizer and stores in metadata.
2. **`apply_reward_kl_penalty`**: Calls `forward_logits` on both policy and ref models to get full logits, then computes GOLD rewards via `_compute_gold_rewards`.
3. **`_compute_gold_rewards`**: For each sample, aligns student/teacher token spans using greedy text matching, merges probabilities, sorts distributions, and computes L1 distance as the reward signal.

## Quickstart

To get started, first set up the dataset from the DAPO example:

```bash
# Run from the `skyrl-train` directory
bash examples/algorithms/dapo/prepare_dapo_data.sh
```

Then, just make sure to set the path to your desired teacher model, and you're ready to kick off training!

```bash
TEACHER_MODEL=<YOUR_MODEL_HERE>
bash examples/gold_distillation/run_gold_distill_qwen_to_llama.sh trainer.ref.model.path=$TEACHER_MODEL
```

## References

- [HuggingFace GOLD Blog](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation)
- [TRL GOLDTrainer](https://github.com/huggingface/trl/blob/v0.25.1/trl/experimental/gold/gold_trainer.py)
- [Universal Logit Distillation (ULD)](https://arxiv.org/abs/2402.12030)
