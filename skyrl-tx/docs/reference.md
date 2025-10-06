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
