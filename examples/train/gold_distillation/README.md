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
- `gold_config.py`: GOLD-specific config extensions (subclasses core SkyRL config)
- `gold_utils.py`: GOLD/ULD utilities (alignment, probability merging, loss computation)
- `gold_workers.py`: Custom workers that load the teacher model and compute GOLD loss inside the policy worker
- `run_gold_distill_qwen_to_llama.sh`: Example script for Qwen-to-Llama distillation
- `run_gold_distill_llama_to_qwen.sh`: Example script for Llama-to-Qwen distillation
- `run_gold_smoke_test.sh`: Minimal smoke test to verify the training loop end-to-end

## Implementation Details

### Config

GOLD adds distillation-specific fields (temperatures, loss weights, beta) to the algorithm config. These are defined in `gold_config.py` as subclasses of the core SkyRL config, keeping the main `config.py` clean:

- `GOLDAlgorithmConfig` extends `AlgorithmConfig` with GOLD knobs
- `GOLDTrainerConfig` and `GOLDSkyRLTrainConfig` wire it up

### Why Custom Workers?

GOLD computes its loss (JSD on matched tokens + L1 on unmatched tokens) directly inside the policy worker with gradients flowing through the student model. The teacher model is loaded inside the policy worker to avoid passing large teacher logits through the data pipeline. The trainer only prepares teacher tokenization and alignment data, which are small tensors.

- **`GOLDFSDPPolicyWorkerBase`**: Loads the teacher model and implements `_gold_forward_backward_micro` which runs both student and teacher forward passes, aligns tokens, and computes the GOLD loss.
- **`GOLDRefWorker`**: Lightweight ref worker for the teacher model.

### Training Flow

1. **Generation**: Student generates responses via vLLM
2. **Teacher tokenization**: Trainer re-tokenizes student output with teacher tokenizer and prepares alignment data
3. **Forward-backward**: Policy worker runs student forward pass, teacher forward pass, aligns token spans, and computes JSD + L1 loss
4. **Weight sync**: Updated student weights are synced back to vLLM engines

## Quickstart

### Smoke Test (Modal)

```bash
cd examples/train_integrations/modal && \
MODAL_GPU=A100:1 modal run main.py \
  --command "cd /root/SkyRL && \
    uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir /root/data/gsm8k && \
    bash examples/train/gold_distillation/run_gold_smoke_test.sh"
```

### Full Run

First prepare the dataset:

```bash
bash examples/train/algorithms/dapo/prepare_dapo_data.sh
```

Then run distillation:

```bash
# Qwen teacher -> Llama student
bash examples/train/gold_distillation/run_gold_distill_qwen_to_llama.sh

# Llama teacher -> Qwen student
bash examples/train/gold_distillation/run_gold_distill_llama_to_qwen.sh

# Override models or other config via command-line
bash examples/train/gold_distillation/run_gold_distill_qwen_to_llama.sh \
  trainer.ref.model.path=<YOUR_TEACHER> \
  trainer.policy.model.path=<YOUR_STUDENT>
```

## References

- [HuggingFace GOLD Blog](https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation)
- [TRL GOLDTrainer](https://github.com/huggingface/trl/blob/v0.25.1/trl/experimental/gold/gold_trainer.py)
- [Universal Logit Distillation (ULD)](https://arxiv.org/abs/2402.12030)
