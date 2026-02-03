# GOLD: Cross-Tokenizer On-Policy Distillation

This folder contains scripts for running **GOLD (General On-policy Logit Distillation)** on SkyRL, enabling distillation between models with different tokenizers (e.g., Qwen and Llama).

GOLD extends standard on-policy distillation to support student-teacher pairs with **different tokenizers** using ULD (Universal Logit Distillation), which aligns token spans and compares sorted probability distributions.

In `main_gold_distill.py` we provide a simple example for modifying SkyRL to implement cross-tokenizer distillation by loading a separate teacher tokenizer and using ULD loss for reward computation.

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
