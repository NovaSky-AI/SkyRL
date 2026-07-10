## Harbor Integration

RL training with [Harbor](https://github.com/laude-institute/harbor) as the environment and reward source. See the [full documentation](https://docs.skyrl.ai/docs/harbor) for details.

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
  run_codecontest.sh               # Code contest training (Qwen3-8B, FSDP baseline)
  run_codecontest_arctic.sh        # Same recipe, routed through the Arctic RL backend
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

### Arctic RL backend

Any Harbor recipe can be routed through the Arctic RL server (ZoRRo / FCA / Arctic speculative decoding) by adding one CLI flag to `main_harbor`:

```
trainer.override_entrypoint=integrations.arctic_rl.harbor_entrypoint
```

A ready-to-run companion to `run_codecontest.sh` lives next to it — same Qwen3-8B recipe, Arctic backend:

```bash
export WANDB_API_KEY=your_wandb_api_key
export DAYTONA_API_KEY=your_daytona_api_key
bash examples/train_integrations/harbor/run_codecontest_arctic.sh
```

Override any knob on the CLI (e.g. `NUM_POLICY_GPUS=4 bash examples/train_integrations/harbor/run_codecontest_arctic.sh` for a 4-GPU smoke, or append `trainer.train_batch_size=64 generator.n_samples_per_prompt=16`). The launcher validates credentials up front, exports `ARCTIC_HARBOR_SHIM_HOST/_PORT` for the OpenAI shim, and layers Arctic's pinned stack (vLLM 0.18 + torch 2.10 + FA2) on top of `--extra harbor` without touching `pyproject.toml`.

Batteries-included variant (env-var driven, includes a smaller-model smoke path): [`integrations/arctic_rl/examples/run_codecontest_arctic_harbor.sh`](../../../integrations/arctic_rl/examples/run_codecontest_arctic_harbor.sh). Setup, tier caveats, and the design overview live in [`integrations/arctic_rl/README.md`](../../../integrations/arctic_rl/README.md) and [`integrations/arctic_rl/docs/HARBOR_DESIGN.md`](../../../integrations/arctic_rl/docs/HARBOR_DESIGN.md).
