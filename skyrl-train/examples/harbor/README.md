## Harbor Integration

RL training with [Harbor](https://github.com/laude-institute/harbor) as the environment and reward source. See the [full documentation](https://docs.skyrl.ai/docs/harbor) for details.

### Structure

```
examples/harbor/
  harbor_generator.py              # HarborGenerator: bridges SkyRL <-> Harbor
  dataset.py                       # HarborTaskDataset: loads task directory paths
  prepare_harbor_dataset.py        # Downloads + extracts datasets from HuggingFace
  harbor_trial_config/
    default.yaml                   # Harbor TrialConfig template
  entrypoints/
    main_harbor.py                 # Full training entrypoint
    main_harbor_generate.py        # Generation-only debug entrypoint
  run_codecontest.sh               # Code contest training (Qwen3-8B)
  run_otagent.sh                   # OpenThoughts-Agent training
  run_harbor_gen.sh                # Debug generation-only
```

### Quick Start

```bash
cd SkyRL/skyrl-train

# 1. Set credentials
export WANDB_API_KEY=your_wandb_api_key
# Pick your sandbox provider:
export DAYTONA_API_KEY=your_daytona_api_key
# export MODAL_TOKEN_ID=your_modal_token_id
# export MODAL_TOKEN_SECRET=your_modal_token_secret

# 2. Prepare dataset
python examples/harbor/prepare_harbor_dataset.py \
    --dataset DCAgent/code-contests-sandboxes-with-tests
python examples/harbor/prepare_harbor_dataset.py \
    --dataset open-thoughts/OpenThoughts-TB-dev

# 3. Launch training
bash examples/harbor/run_codecontest.sh
```
