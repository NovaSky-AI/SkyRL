# LiveCodeBench Evaluations

This directory contains evaluation scripts for different code generation systems on the LiveCodeBench dataset.

## Overview

This directory includes two main evaluation systems:

1. **`ranker_original/`**: Dual-model architecture with generator and ranker models
2. **`single_module/`**: Single-model evaluations for program generation and test generation

Both systems evaluate code generation performance by running generated code against test cases and measuring pass rates.

---

## Ranker Original Evaluation

The `ranker_original` system uses a dual-model architecture:
- **Generator Model**: Generates code solutions
- **Program Model**: Used by the ranker to evaluate and rank generated solutions

### Directory Structure

- **`baseline/`**: Baseline evaluation using standard Qwen models
- **`trained/`**: Evaluation using fine-tuned models
- **`eval_ranker_original_cli.py`**: Main evaluation script
- **`analysis_utils.py`**: Utility functions for analyzing results and creating visualizations
- **`analysis.ipynb`**: Jupyter notebook demonstrating result analysis workflows

### Quick Start

1. **Start vLLM servers**:
   ```bash
   ./evals/ranker_original/baseline/start_vllms.sh  # or trained/start_vllms.sh
   ```

2. **Run evaluations**:
   ```bash
   ./evals/ranker_original/baseline/start_eval.sh  # or trained/start_eval.sh
   ```

3. **Results location**: `/work/jiashu/assertion-data/livecodebench/evals/results/ranker-original/`

### Analysis

The `ranker_original` directory includes comprehensive analysis utilities:

- **Data loading**: `load_result_files()`, `extract_scores_from_results()`
- **Statistics**: `compute_mean_scores_per_run()`, `compute_mean_scores_across_runs()`
- **Visualizations**: 
  - `plot_mean_scores_barplot()` - Barplots with customizable `model_order` and `split_order`
  - `plot_mean_scores_barplot_with_ci()` - Barplots with confidence intervals
  - `plot_comparison_barplot()` - Compare baseline vs. trained models
  - `plot_horizontal_stacked_barplot()` - Horizontal stacked barplots

See `ranker_original/analysis.ipynb` for usage examples.

---

## Single Module Evaluation

The `single_module` system evaluates single-model approaches with two evaluation types:

1. **Program Generation** (`eval_single_module_prog_cli.py`): Generates code solutions using `NaiveCodeGenerator`
2. **Test Generation** (`eval_single_module_test_cli.py`): Generates test cases using `NaiveTestGenerator`

### Directory Structure

- **`baseline/`**: Baseline evaluation using standard Qwen models
- **`trained/`**: Evaluation using fine-tuned models
- **`eval_single_module_prog_cli.py`**: Program generation evaluation script
- **`eval_single_module_test_cli.py`**: Test generation evaluation script

### Quick Start

1. **Start vLLM servers**:
   ```bash
   ./evals/single_module/baseline/start_vllms.sh  # or trained/start_vllms.sh
   ```

2. **Run program generation evaluations**:
   ```bash
   ./evals/single_module/baseline/start_eval_prog_gen.sh  # or trained/start_eval_prog_gen.sh
   ```

3. **Run test generation evaluations**:
   ```bash
   ./evals/single_module/baseline/start_eval_test_gen.sh  # or trained/start_eval_test_gen.sh
   ```

4. **Results location**: `/work/jiashu/assertion-data/livecodebench/evals/results/single_module/`

---

## Prerequisites

- Python environment with required dependencies (dspy, wandb, etc.)
- vLLM installed and configured
- Access to model checkpoints:
  - **Baseline**: Standard Qwen models (`Qwen/Qwen2.5-Coder-7B-Instruct`, `Qwen/Qwen2.5-Coder-1.5B-Instruct`, `Qwen/Qwen2.5-3B`)
  - **Trained**: Fine-tuned models (e.g., `Harryllh/lcb_test_generator_1.5b_400steps`)
- Sufficient GPU resources (typically 6 GPUs across model servers)
- tmux for managing multiple server processes

## Setup

1. Navigate to the project directory:
   ```bash
   cd assertion/assertion/livecodebench
   ```

2. Ensure your conda environment is set up with the `assertion` environment activated.

## Manual Evaluation

### Ranker Original

```bash
python evals/ranker_original/eval_ranker_original_cli.py \
  --dataset test \
  --gen_model "Qwen/Qwen2.5-Coder-7B-Instruct" \
  --prog_model "Qwen/Qwen2.5-Coder-7B-Instruct" \
  --gen_port 7001 \
  --prog_port 7000 \
  --num_workers 20 \
  --base_path /path/to/results
```

### Single Module - Program Generation

```bash
python evals/single_module/eval_single_module_prog_cli.py \
  --dataset test \
  --prog_model "Qwen/Qwen2.5-Coder-7B-Instruct" \
  --prog_port 7000 \
  --num_workers 20 \
  --num_runs 3 \
  --base_path /path/to/results
```

### Single Module - Test Generation

```bash
python evals/single_module/eval_single_module_test_cli.py \
  --dataset test \
  --gen_model "Qwen/Qwen2.5-Coder-1.5B-Instruct" \
  --gen_port 7001 \
  --num_workers 20 \
  --num_runs 3 \
  --base_path /path/to/results
```

## Output Format

Results are saved as JSON files with the following naming convention:
```
gen_{gen_model}_prog_{prog_model}_{dataset}.json  # ranker_original
{model_type}_{model}_{dataset}.json                # single_module
```

Each JSON file contains:
- `time_stamp`: When the evaluation was run (ISO format)
- `results`: Array of evaluation results, each containing:
  - `example`: The input example (prompt, test cases, task_id, etc.)
  - `pred`: The generated prediction (code or tests)
  - `score`: Pass/fail score (1 for passed, 0 for failed)

The scripts also print the final average metric score to the console.

## Configuration

### vLLM Server Settings

Edit the `start_vllms.sh` scripts in `baseline/` or `trained/` directories to modify:
- GPU assignments (`GPU_GROUPS`)
- Port numbers (`PORTS`)
- Model configurations (`MAX_MODEL_LEN`, `MAX_NUM_SEQS`, etc.)
- Conda environment path (`CONDA_SH`)
- Model names (`MODELS`)

### Evaluation Settings

Edit the `start_eval*.sh` scripts to modify:
- Model selection and ports
- Dataset splits to evaluate (`DATASETS`)
- Number of workers (`--num_workers`)
- Number of runs (`--num_runs`)
- Results output path (`BASE_PATH`)

## Monitoring

- Use `tmux attach -t vllm` to monitor vLLM servers
- Use `tmux attach -t evals` (or `evals-baseline-*`, `evals-trained-*`) to monitor evaluation progress
- Each evaluation window shows progress and running average scores
- Logs are saved to `{BASE_PATH}/logs/` for each run

## Notes

- The evaluation uses a timeout of 50-100 seconds per test case (varies by system)
- Fast check mode is enabled (stops early if a test fails)
- Results are saved incrementally, so partial results are preserved if interrupted
- Error handling has been improved to catch and report test case parsing errors gracefully
- The evaluation scripts handle malformed test cases by logging errors and continuing with the next test case
- Single module evaluations support multiple runs (`--num_runs`) for statistical significance

## Troubleshooting

### KeyError: 'output'

If you encounter a `KeyError: 'output'` error, it means some test cases are missing the required "output" field. The code has been updated to handle this gracefully:
- The exception is caught and logged
- The evaluation continues with the next test case
- Error messages indicate which test case failed to parse

### Server Connection Issues

If evaluations fail to connect to vLLM servers:
- Verify servers are running: `tmux attach -t vllm`
- Check port numbers match between server and evaluation scripts
- Ensure servers have fully initialized before starting evaluations

### Port Conflicts

If you see port conflicts:
- Check which ports are in use: `lsof -i :7000` (replace with your port)
- Modify port assignments in the `start_vllms.sh` scripts
- Ensure no other processes are using the same ports
