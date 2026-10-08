# Model checks: trainer vs inference logprobs

`run_nemotron_logprobs.py` checks that the Megatron trainer and the vLLM inference engines
score the same tokens the same way, and that a weight sync carries a trainer update to the
engines. It runs on a large model without creating an optimizer, so it fits where a training
run would not. The defaults target Nemotron-3-Super-120B on 8 trainer GPUs (TP8 EP8) plus
8 inference GPUs (vLLM TP8), non-colocated across two nodes.

## What it checks

Both backends score two fixed probe sequences (192 token positions in total). Each phase
reports the mean, p99 and max absolute logprob error.

```text
base     trainer vs inference on the loaded weights           must agree
stale    perturbed trainer vs the not-yet-updated inference   must disagree by more than --mean-atol
updated  perturbed trainer vs inference after the sync        must agree
repeat   inference scored twice after the sync                must match within --repeat-atol
```

The `stale` phase is what makes `updated` meaningful: if the perturbation were too small
to fail the agreement check on its own, a sync that silently did nothing would still pass.

With LoRA (the default) the trainer holds a rank-8 adapter on `linear_proj`, `linear_fc1`
and `linear_fc2`. The perturbation adds name-seeded noise to every LoRA B tensor (which
start at zero), and the sync exports the adapter to `--lora-sync-path` for the engines to
hot-load. With `--full-ft` there is no adapter: the perturbation adds relative noise to
every weight and the sync broadcasts full weights.

## Run

Start Ray across both nodes first. With LoRA, `--lora-sync-path` must be on a filesystem
that every node's inference engines can read.

```bash
# LoRA adapter sync
uv run --isolated --extra megatron -m examples.model_checks.run_nemotron_logprobs \
  --lora-sync-path /shared/skyrl-logprob-adapter

# Full fine-tuning weight sync
uv run --isolated --extra megatron -m examples.model_checks.run_nemotron_logprobs \
  --full-ft
```

Other models and meshes: `--model`, `--trainer-gpus`, `--tp`, `--ep`, `--etp`,
`--inference-tp`, `--max-model-len`. Tolerances: `--mean-atol` (0.05), `--max-atol` (0.5),
`--repeat-atol` (1e-6). `--perturb-multiplier` (10) scales the perturbation; raise it if the
`stale` phase reports too small a difference.

## Output

Each phase prints its statistics as it completes, and the first failing check raises with the
offending number. Redirect stdout to a file to keep a record of a run.

`logprob_checks.py` holds the pure helpers (probe sequences, error statistics, the two
perturbations) and has CPU unit tests under `tests/backends/skyrl_train/`.
