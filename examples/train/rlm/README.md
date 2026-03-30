# Training with RLMEnv

RLMEnv is a multi-turn environment where the model interacts with a Python REPL. Each episode has two text inputs:

- **Prompt** – direct instructions / tips the model sees in full in its context window (like a system prompt).
- **Context** – a large text stored externally in the REPL as the `context` variable. The model only sees a short preview and must run code to read more.

The model writes Python code (in ` ```repl ... ``` ` blocks) to inspect `context` using `search()` and `extract_section()` tools, then returns a list of retrieved text spans via `FINAL_VAR(var)`. Reward is F1 over retrieved text intervals vs. ground-truth evidence spans.

## Simplest way to start a training loop

### 1. Create an RLM dataset

The default dataset is **QASPER** (academic paper evidence retrieval), loaded from `data/qasper-train-cleaned.json`.

Each row has:

- **`prompt`**: list of messages — the question the model must answer.
- **`env_class`**: `"rlm"`.
- **`reward_spec`**: dict with **`evidence`** (list of ground-truth text spans). Reward is F1 over retrieved intervals. Alternatively, set **`ground_truth`** for exact-match reward.
- **`max_turns`** (default 10).
- **`extra_info`**: dict with **`context_text`** — the full paper text loaded into the REPL as `context`.

The generator automatically builds per-example `search()` / `extract_section()` REPL tools and the F1 `reward_fn` from the serializable dataset fields at runtime.

Generate the dataset:

```bash
uv run -- python examples/train/rlm/rlm_dataset.py --output_dir $HOME/data/rlm
uv run -- python examples/train/rlm/rlm_dataset.py --output_dir $HOME/data/rlm --n_val 100  # larger eval set
```

### 2. Launch training

**Recommended:** use the run script (same pattern as other examples like `gsm8k/run_gsm8k.sh`):

```bash
# Optional: prepare data first
uv run -- python examples/train/rlm/rlm_dataset.py --output_dir $HOME/data/rlm

# Run training (override defaults via env: DATA_DIR, NUM_GPUS, LOGGER)
bash examples/train/rlm/run_rlm.sh
```

The script lives at **`examples/train/rlm/run_rlm.sh`** and is the right place to commit and tweak the training command (batch sizes, model path, epochs, etc.).

Required for RLM:

- **`environment.env_class=rlm`** – use RLMEnv.
- **`generator.step_wise_trajectories=true`** – multi-turn rollouts (RLM is multi-turn).
- **`generator.batched=false`** – step-wise generator does not support batched mode.
- **`generator.max_turns`** – e.g. `10` (must be > 1 for multi-turn).

Optional RLM env config (under `environment.skyrl_gym.rlm`):

- `repl_timeout` (default 15.0) – timeout per REPL execution (seconds).
- `custom_system_prompt` – fully replace the default RLM system prompt with a custom one.
