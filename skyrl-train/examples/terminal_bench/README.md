### Terminal-Bench integration (WIP)

Integration with Terminal-Bench is a work in progress.

This integration requires the `sandboxes` repo (ie, the new and improved terminal bench):
```bash
cd SkyRL/skyrl-train
git clone https://github.com/laude-institute/sandboxes.git
```

There is an existing package conflict between `skyrl-train` and `sandboxes`. Resolve it by modifying `sandboxes/pyproject.toml` with the following:
* `rich==13.7.1`
* `requires-python = ">=3.12"`

### Dataset Generation
First, generate the training (and, optionally, validation) dataset. First, download `sandboxes` tasks to a local directory. Then:
```bash
uv run examples/terminal_bench/prepare_dataset.py \
  --task_dir "path/to/sandbox/tasks" \
  --output_dir $HOME/data/terminal_bench \
  --output_name train
```

### Training
Run the GRPO training pipeline:
```bash
bash examples/terminal_bench/run_tbench.sh
```
### Generation Only
Launch the generation process without training. This entrypoint is primarily for rapid debugging to avoid the trainer setup overhead.
```bash
bash examples/terminal_bench/run_tbench_gen.sh
```