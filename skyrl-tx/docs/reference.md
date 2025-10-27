# `tx`

**Usage**:

```console
$ tx [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `train`: Train a model
* `version`

## `tx train`

Train a model

**Usage**:

```console
$ tx train [OPTIONS]
```

**Options**:

* `--model TEXT`: HuggingFace model ID or local model path  [required]
* `--dataset TEXT`: HuggingFace dataset to use for training  [required]
* `--loader TEXT`: Loader used for loading the dataset  [default: tx.loaders.text]
* `--split TEXT`: The dataset split to use  [default: train]
* `--output-dir PATH`: The output directory where the model predictions and checkpoints will be written  [required]
* `--load-checkpoint-path PATH`: If specified, resume training from this checkpoint
* `--save-steps INTEGER`: Number of steps between checkpoints  [default: 500]
* `--max-steps INTEGER`: The maximum number of training steps
* `--batch-size INTEGER`: Batch size of each training batch  [required]
* `--optimizer [adamw]`: Which optax optimizer to use  [default: adamw]
* `--optimizer-args LOADS`: Arguments for the optax optimizer (in JSON format)  [default: {&quot;learning_rate&quot;: 1e-5, &quot;weight_decay&quot;: 0.1}]
* `--tp-size INTEGER`: Tensor parallelism degree to use for the model  [default: 1]
* `--tracker [wandb]`: Experiment tracker to report results to
* `--tracker-args LOADS`: Arguments that will be passed to the experiment tracker (in JSON format)  [default: {}]
* `--help`: Show this message and exit.

## `tx version`

**Usage**:

```console
$ tx version [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

## Tinker API

The Tinker API provides REST endpoints for model training and inference. The API supports both base model and LoRA adapter sampling with advanced text generation parameters.

### Sampling Parameters

The sampling API supports the following parameters for controlling text generation:

#### Basic Parameters

* **`temperature`** (float, default: 1.0): Controls randomness in generation. Lower values make output more deterministic.
  - `temperature=0.0`: Deterministic (always selects highest probability token)
  - `temperature=1.0`: Standard sampling
  - `temperature>1.0`: More random/creative output

* **`max_tokens`** (int, required): Maximum number of tokens to generate.

* **`seed`** (int, optional): Random seed for reproducible generation. If not provided, a random seed is generated.

* **`stop`** (list of int, optional): Stop generation when encountering these token IDs.
  - Example: `stop=[2, 13]` - Stop at tokens 2 and 13
  - Note: Requires token IDs (integers). Use your tokenizer to convert strings to token IDs.

#### Parameter Validation

* `temperature` must be >= 0

### Example API Request

```json
{
  "base_model": "microsoft/DialoGPT-medium",
  "prompt": {
    "chunks": [
      {
        "tokens": [1, 2, 3, 4, 5]
      }
    ]
  },
  "sampling_params": {
    "temperature": 0.8,
    "max_tokens": 50,
    "stop": [2],
    "seed": 42
  },
  "num_samples": 1
}
```

### Example API Response

```json
{
  "sequences": [
    {
      "stop_reason": "stop",
      "tokens": [6, 7, 8, 2],
      "logprobs": [-0.5, -0.3, -0.8, -0.2]
    }
  ],
  "prompt_logprobs": []
}
```

### Stop Reasons

* **`"length"`**: Generation stopped because `max_tokens` was reached
* **`"stop"`**: Generation stopped because a stop token was encountered