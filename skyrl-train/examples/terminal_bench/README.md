### Terminal-Bench integration (WIP)

Integration with Terminal-Bench is a work in progress.

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
