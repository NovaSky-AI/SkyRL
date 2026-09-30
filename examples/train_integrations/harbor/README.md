## Harbor Integration

RL training with [Harbor](https://github.com/laude-institute/harbor) as the environment and reward source. See the [full documentation](https://docs.skyrl.ai/docs/harbor) for details.

For generation-only evaluations with native CPU KV cache offloading, use
[kv_offload/run.sh](kv_offload/run.sh):

```bash
export DATA_DIR="$HOME/data/harbor/CodeContests"
export ENV_FILE="$HOME/hackskyrl/.env.srh"
export RAY_ADDRESS=<cluster-address>
uv run --isolated --no-project bash examples/train_integrations/harbor/kv_offload/run.sh baseline > /tmp/kv-baseline.log 2>&1
uv run --isolated --no-project bash examples/train_integrations/harbor/kv_offload/run.sh cpu > /tmp/kv-cpu.log 2>&1
```

The script uses two Qwen3-8B engines and 500 seed-42 tasks, with `GPU_BLOCKS=50000`
(800,000 FP8 KV tokens per engine) and `MAX_CONCURRENCY=200`. CPU mode adds a
128 GB cache per engine. Set `NUM_SAMPLES=4 MAX_CONCURRENCY=4` for a smoke run.
Credentials and settings are passed through `uv --env-file` for Ray workers;
the Ray version override is 2.58.0. Use fresh `RUN_DIR` values and run comparisons
sequentially. Logs and trial results are saved there. Extra CLI overrides can
be appended to the script.

On one eight-B200 node, the 500-task baseline recorded 5 preemptions in 751.40 s;
CPU offloading recorded 1 in 681.47 s, with 56.61 GB of cache reads. Neither run
had masked tasks or timeouts. Timing excludes engine startup; overlapping
cluster activity and single-run variability limit attribution of the 9.3%
improvement. Raw experiment artifacts remain in `/tmp/skyrl-kv-offload/`.

### Structure

```
examples/train_integrations/harbor/
  harbor_generator.py              # HarborGenerator: bridges SkyRL <-> Harbor
  dataset.py                       # HarborTaskDataset: loads task directory paths
  prepare_harbor_dataset.py        # Downloads + extracts datasets from HuggingFace
  harbor_trial_config/
    default.yaml                   # Harbor TrialConfig template
  entrypoints/
    main_harbor.py                 # Full training entrypoint
    main_harbor_generate.py        # Generation-only debug entrypoint
  run_codecontest.sh               # Code contest training (Qwen3-8B)
  run_harbor_gen.sh                # Debug generation-only
```

### Quick Start

```bash
cd SkyRL

# 1. Set credentials
export WANDB_API_KEY=your_wandb_api_key
# Pick your sandbox provider:
export DAYTONA_API_KEY=your_daytona_api_key
# export MODAL_TOKEN_ID=your_modal_token_id
# export MODAL_TOKEN_SECRET=your_modal_token_secret

# 2. Prepare dataset
uv run examples/train_integrations/harbor/prepare_harbor_dataset.py \
    --dataset open-thoughts/CodeContests
uv run examples/train_integrations/harbor/prepare_harbor_dataset.py \
    --dataset open-thoughts/OpenThoughts-TB-dev

# 3. Launch training
bash examples/train_integrations/harbor/run_codecontest.sh
```
