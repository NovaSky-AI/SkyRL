# LiveCodeBench Code Generation

This directory contains code for training and evaluating code generation models on the LiveCodeBench dataset. The project implements various approaches for generating code solutions and test cases using language models.

## Overview

This project focuses on:
- **Code Generation**: Generating Python code solutions for programming problems
- **Test Generation**: Generating test cases for code validation
- **Model Training**: Fine-tuning language models using DSPy and Arbor
- **Evaluation**: Comprehensive evaluation of model performance on train and test splits

## Workflow

The typical workflow consists of three main steps:

1. **Generate Golden Solutions** → 2. **Train Models** → 3. **Evaluate Models**

---

## Step 1: Generate Golden Solutions

Before training, you need to generate reference ("golden") solutions for the dataset.

### Usage

```bash
python gen_golden_soln.py --mode train  # Generate for training set
python gen_golden_soln.py --mode test   # Generate for test set
```

### What it does

- Uses GPT-4o to generate correct solutions for each problem in the dataset
- Handles both stdin-based and functional programming problems
- Implements retry logic with error feedback for failed attempts
- Saves results to `gold_preds_train.pkl` or `gold_preds_test.pkl`

### Requirements

- Must be run from the `livecodebench` directory
- Requires OpenAI API key (configured via `.env` file)
- Uses `openai/gpt-4o` model

### Output

- `gold_preds_train.pkl`: Dictionary mapping `task_id` to golden solution predictions
- `gold_preds_test.pkl`: Dictionary mapping `task_id` to golden solution predictions

---

## Step 2: Training Models

After generating golden solutions, you can train various models using the training scripts.

### Training Scripts

#### Test Generation Training

- **`train_test_gen_dspy.py`**: Train test generation model using DSPy with GRPO (Group Relative Policy Optimization)
  - Uses Arbor for training
  - Supports multiple model sizes (1.5B, 3B, 7B)
  - Trains `CodeGeneratorWithRankerAssertTestGen_single_dspy` program

- **`train_test_gen_dspy_3b.py`**: Specialized version for 3B model training
  - Similar to above but configured for 3B models

- **`train_test_gen_single.py`**: Single-model test generation training
  - Simpler training setup without ranker

- **`train_test_gen.py`**: Basic test generation training script

#### Program Generation Training

- **`train_prog_gen_dspy.py`**: Train program generation model using DSPy
  - Uses dual-model architecture with generator and ranker
  - Supports configurable batch size, number of tests/programs, and epochs

### Training Requirements

- **Arbor**: Required for GRPO training (starts server automatically)
- **vLLM**: For serving base models during training
- **Weights & Biases**: For experiment tracking (configure `WANDB_API_KEY`)
- **GPU Resources**: Sufficient GPU memory for model training

### Training Configuration

Most training scripts support:
- Model selection (Qwen models: 1.5B, 3B, 7B, etc.)
- Batch size configuration
- Number of test cases and program candidates
- Number of training epochs
- Arbor server port configuration

---

## Step 3: Evaluation

After training, evaluate your models using the evaluation scripts in the `evals/` directory.

### Quick Start

See the comprehensive evaluation guide:
```bash
cd evals/
cat README.md
```

### Evaluation Systems

1. **`evals/ranker_original/`**: Dual-model evaluation with generator and ranker
   - Baseline and trained model evaluations
   - Comprehensive analysis utilities
   - Visualization tools

2. **`evals/single_module/`**: Single-model evaluations
   - Program generation evaluation
   - Test generation evaluation
   - Baseline and trained model support

### Evaluation Workflow

1. **Start vLLM servers** for model serving
2. **Run evaluation scripts** for your trained models
3. **Analyze results** using provided analysis utilities

For detailed evaluation instructions, see `evals/README.md`.

---

## Directory Structure

```
livecodebench/
├── gen_golden_soln.py          # Generate golden solutions
├── train_*.py                   # Training scripts
├── data.py                      # Dataset loading utilities
├── program.py                   # DSPy program definitions
├── lcb_utils.py                 # LiveCodeBench utilities
├── live_code_bench_execute.py   # Code execution and testing
├── utils.py                     # General utilities
├── evals/                       # Evaluation scripts
│   ├── README.md               # Evaluation documentation
│   ├── ranker_original/        # Dual-model evaluations
│   └── single_module/          # Single-model evaluations
└── README.md                    # This file
```

---

## Prerequisites

### Environment Setup

1. **Conda Environment**: Activate the `assertion` environment
   ```bash
   conda activate assertion
   ```

2. **Required Packages**:
   - `dspy`: DSPy framework for language model programming
   - `arbor`: For GRPO training
   - `vllm`: For model serving
   - `wandb`: For experiment tracking
   - `datasets`: For loading LiveCodeBench dataset
   - Standard ML libraries (torch, numpy, etc.)

3. **API Keys**:
   - OpenAI API key (for `gen_golden_soln.py`) - set in `.env` file
   - Weights & Biases API key (for training) - set as `WANDB_API_KEY` environment variable

### Data Requirements

- LiveCodeBench dataset is automatically downloaded via HuggingFace `datasets`
- Dataset is cached locally after first download
- Golden solutions must be generated before training

---

## Common Tasks

### Generate Golden Solutions for Both Splits

```bash
python gen_golden_soln.py --mode train
python gen_golden_soln.py --mode test
```

### Train a Test Generation Model

```bash
# Using DSPy with GRPO
python train_test_gen_dspy.py

# Or single model approach
python train_test_gen_single.py
```

### Train a Program Generation Model

```bash
# Using DSPy with ranker
python train_prog_gen_dspy.py

# Or single model approach
python train_prog_gen_single.py --model "Qwen/Qwen2.5-Coder-7B-Instruct" --epoch 5
```

### Evaluate Trained Models

```bash
cd evals/
# Follow instructions in evals/README.md
```

---

## Notes

- **Working Directory**: Most scripts must be run from the `livecodebench` directory
- **Model Checkpoints**: Trained models are typically saved to HuggingFace or local paths
- **Results**: Evaluation results are saved to `/work/jiashu/assertion-data/livecodebench/evals/results/`
- **Logging**: Training and evaluation logs are saved to `wandb/` and log directories
- **GPU Memory**: Ensure sufficient GPU memory for model training and serving

---

## Troubleshooting

### Golden Solution Generation Fails

- Check OpenAI API key is set in `.env` file
- Verify you have API credits/quota
- Check network connectivity

### Training Fails to Start

- Verify Arbor server starts correctly (check `arbor_log.txt`)
- Ensure vLLM servers are running if required
- Check GPU memory availability
- Verify `WANDB_API_KEY` is set

### Evaluation Issues

- See `evals/README.md` for detailed troubleshooting
- Verify vLLM servers are running and accessible
- Check port numbers match between servers and evaluation scripts

---

## Additional Resources

- **Evaluation Guide**: See `evals/README.md` for comprehensive evaluation documentation
- **Analysis Tools**: See `evals/ranker_original/analysis.ipynb` for result analysis examples
- **Dataset**: LiveCodeBench dataset documentation at [HuggingFace](https://huggingface.co/datasets/livecodebench/code_generation_lite)
