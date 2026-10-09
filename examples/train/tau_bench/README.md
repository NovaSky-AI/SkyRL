# tau-bench (retail) evaluation

Baseline evaluation of a policy model on the [tau-bench](https://github.com/sierra-research/tau-bench)
retail domain, using SkyRL's eval-only entrypoint (`skyrl.train.entrypoints.main_generate`)
and the `tau_bench` SkyRL-Gym environment.

The environment is multi-turn and tool-using: the agent either calls a retail tool
(`<tool_call>{"name": ..., "arguments": {...}}</tool_call>`) or sends a message to a
simulated user. The user is an LLM served separately over an OpenAI-compatible
endpoint. Reward is the upstream tau-bench retail reward (final DB-state match +
required outputs), so a run scores `pass@1` over the 115-task test split.

## Files

- `tau_bench_dataset.py` — writes `retail_test.parquet` / `retail_train.parquet`
  (one row per task; the system prompt + opening user message are built at rollout
  time in `TauBenchEnv.init`).
- `run_eval_taubench.sh` — launches a user-simulator vLLM OpenAI server, then runs
  the eval-only generation. Single 8×H100 node: user-sim on GPUs 6,7, policy engines
  on GPUs 0-5. Override `MODEL_NAME` for the policy under test.
- `anyscale_taubench_eval.yaml` — Anyscale job wrapper for the above.

## Run

```bash
# Local (single node):
bash examples/train/tau_bench/run_eval_taubench.sh MODEL_NAME=<hf-model-id>

# Anyscale:
anyscale job submit -f examples/train/tau_bench/anyscale_taubench_eval.yaml --env HF_TOKEN=$HF_TOKEN
```

Results (incl. `pass@1`) are logged and dumped under `trainer.export_path`.
