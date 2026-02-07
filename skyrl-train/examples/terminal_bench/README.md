## Terminal-Bench integration (WIP)

Integration with Terminal-Bench is a work in progress.

We specify a specific harbor commit in our `pyproject.toml`, which you can easily substitute or even use a local copy of Harbor.

```toml
harbor = { git = "https://github.com/laude-institute/harbor", rev = "5921cc40e8f6b5b0df0a3dc6354e0e679447eb54" }
```

Tracked here: https://github.com/NovaSky-AI/SkyRL/issues/866

But you can already run:

OpenThoughts-Agent first release's RL job with:

```bash
cd SkyRL/skyrl-train
bash examples/terminal_bench/run_otagent.sh
```

Training on code-contest with Qwen3-8B as the base model:

```bash
cd SkyRL/skyrl-train
bash examples/terminal_bench/run_codecontest.sh
```

Generation-only for debugging
```bash
cd SkyRL/skyrl-train
bash examples/terminal_bench/run_tbench_gen.sh
```

Currently, you'd have to have [Daytona](https://app.daytona.io/) access to host the containers.

### Configuration

To configure the Harbor-specific parameters (e.g. the maximum turns a rollout can take), we offer the base yaml in `terminal_bench_config/default.yaml`. Then in the launch script, specifying the following feeds that yaml to `TerminalBenchGenerator`. 

```sh
  hydra.searchpath=['file://examples/terminal_bench'] \
  +terminal_bench_config=default \
  ++terminal_bench_config.trials_dir=$TRIALS_DIR \
```

You can override any config supported by Harbor's `TrialConfig` in the script with `++`, just like what we do for `trials_dir` here.

For all the configurations, see [Harbor's documentation](https://harborframework.com/docs), and the `TrialConfig` definition: https://github.com/laude-institute/harbor/blob/5921cc40e8f6b5b0df0a3dc6354e0e679447eb54/src/harbor/models/trial/config.py

### Main configuration knobs

- `++terminal_bench_config.environment.type=daytona` or `=modal`
  - Harbor supports various ways of [hosting the sandboxes](https://harborframework.com/docs/core-concepts#container-environment) for the agent to run the task (Daytona, Modal, E2B, GKE (Google Kubernetes Engine))
  - SkyRL + Harbor integration has tested with Daytona and Modal, but the other providers should work out of the box
- More documentations to come
