# GOLD: Cross-Tokenizer On-Policy Distillation

This folder contains scripts for running **GOLD (General On-policy Logit Distillation)** on SkyRL, enabling distillation between models with different tokenizers (e.g., Qwen and Llama).

GOLD extends standard on-policy distillation to support student-teacher pairs with **different tokenizers** using ULD (Universal Logit Distillation), which aligns token spans and compares sorted probability distributions.

## Cross-Tokenizer Distillation (GOLD/ULD)
When tokenizers differ, comparing log probabilities directly is meaningless. GOLD uses the ULD methodology:

1. **Re-tokenization**: Decode student-generated text and re-tokenize with teacher's tokenizer
2. **Token Alignment**: Use greedy text matching to align token spans between student and teacher sequences
3. **Probability Merging**: Merge probabilities for aligned spans using geometric mean
4. **Sorted Comparison**: Sort probability distributions and compute L1 distance (ULD approach)

This allows meaningful comparison of probability distributions even when vocabularies differ.

## Files

- `main_gold_distill.py`: Main trainer with `GOLDDistillationTrainer` class
- `gold_utils.py`: GOLD/ULD utilities (alignment, probability merging, loss computation)
- `gold_ref_worker.py`: Custom ref worker that returns logits instead of log_probs
- `run_gold_distill_qwen_to_llama.sh`: Example script for Qwen-to-Llama distillation
- `run_gold_distill_llama_to_qwen.sh`: Example script for Llama-to-Qwen distillation

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
