### Terminal-Bench integration (WIP)

Integration with Terminal-Bench is a work in progress. For now, training tasks are hard-coded as "hello-world" in the prototype. The next TODO is to support specifying a training set of Terminal-Bench tasks.

This integration requires the `harbor` repo (ie, the new and improved terminal bench):
```bash
cd SkyRL/skyrl-train
git clone https://github.com/laude-institute/harbor.git
```


- **Training**:
```bash
bash examples/terminal_bench/run_tbench.sh
```

- **Generation only**: launch the generator/serving process. This entrypoint is primarily for rapid debugging to avoid the trainer setup overhead.
```bash
bash examples/terminal_bench/run_tbench_gen.sh
```