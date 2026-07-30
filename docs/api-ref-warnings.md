# API reference build warnings — before / after

Reference for the warnings emitted by `docs/generate-api-docs.py` (griffe / griffe2md)
and what was changed to clear them.

## Reproducing

```bash
# from the repo root
uv run --extra dev python docs/generate-api-docs.py 2>&1 | grep -E '^skyrl.*: '
```

Warnings come from [griffe](https://mkdocstrings.github.io/griffe/)'s Google-style
docstring parser. They are only emitted for objects that are actually rendered, i.e.
the objects listed in `docs/api-pages.yaml`. Private members (leading underscore) are
filtered out by griffe2md and therefore never warn.

## Summary

| | Warnings | Pages generated |
|---|---|---|
| Before | **25** | 18 |
| After | **0** | 18 |

The full baseline list is preserved verbatim in
[`api-ref-warnings-before.txt`](./api-ref-warnings-before.txt) — 25 emitted lines
covering 21 distinct sites (four `main_base.py` entries are emitted twice).

Warning categories in the baseline:

| Count | Category |
|---|---|
| 12 | `No type or annotation for parameter 'X'` |
| 9 | `No type or annotation for returned value 'X'` |
| 2 | `No type or annotation for returned value N` |
| 2 | `Failed to get 'name: description' pair from '...'` |

## Why each category happens

- **`No type or annotation for parameter 'X'`** — the docstring's `Args:` section
  documents `X`, but the signature has no annotation for it. griffe pulls parameter
  types from the signature, not the docstring, so it has nothing to show.
- **`No type or annotation for returned value 'X'`** — a `Returns:` entry written as
  `SomeType: description`. griffe reads the part before the colon as a *name*, not a
  type, then finds no return annotation on the function. Adding `-> SomeType` fixes it;
  the `SomeType:` prefix in the docstring then becomes redundant and was dropped so the
  rendered table shows a clean `Type | Description` pair.
- **`No type or annotation for returned value N`** — a `Returns:` entry with no colon
  and no return annotation.
- **`Failed to get 'name: description' pair from '...'`** — a continuation line inside
  `Args:` that was not indented under its parameter, so griffe treated it as a new
  parameter entry.

## Changes made

| File | Change |
|---|---|
| `skyrl/backends/backend.py` | Annotated `output_path` / `checkpoint_path` as `AnyPath` on `save_checkpoint`, `load_checkpoint`, `save_sampler_checkpoint`, matching the concrete `jax.py` / `ray_jax.py` overrides and the `AnyPath`-derived paths the engine passes in. |
| `skyrl/train/entrypoints/main_base.py` | Added return annotations to `get_train_dataset`, `get_eval_dataset`, `get_generator`, `get_trainer`, `get_tracker` (plus private `_get_new_inference_client`, `_setup_trainer`) and dropped the now-redundant `SomeType:` prefixes from `Returns:` blocks. |
| `skyrl/backends/skyrl_train/workers/worker.py` | Annotated the `bool` flags on `Worker.offload_to_cpu` / `backload_to_gpu` and the actor-group variants; annotated `*args: Any, **kwargs: Any` on `async_run_ray_method`; fixed the unindented `nonblocking` continuation lines and documented the previously undocumented offload/backload flags. |
| `skyrl/backends/skyrl_train/distributed/dispatch.py` | Annotated `**kwargs: Any` on `dispatch_from_staged`. |
| `skyrl/backends/skyrl_train/training_batch.py` | Added `-> "TensorBatch[DictType]"` to `repeat` and `repeat_interleave`, consistent with the already-annotated `select` / `slice` / `cat`. |

`get_eval_dataset` is annotated `Optional[PromptDataset]` (it returns `None` when
`eval_interval <= 0` or no val data is configured) and its description was updated to
say so.

One unrelated hygiene fix is included: `docs/.gitignore` had no rule for the
`content/docs/api-ref/skyrl/sft/` directory added by #1957, so the three generated
`sft/*.mdx` pages showed up as untracked files after every build. Added a matching
rule alongside the existing ones.

## Verification

- `uv run --extra dev python docs/generate-api-docs.py` → 0 warnings, 18/18 pages, no
  `Could not load` / `Error rendering` markers in the generated MDX.
- `cd docs && npm install && npm run build` → succeeds, 85 static pages, no warnings.
- `uv run --with pre-commit pre-commit run --all-files` → ruff, black, secret scan pass.
- CPU tests (`tests/train/`, `tests/backends/skyrl_train/` excluding `gpu/`):
  617 passed / 19 skipped, identical to the pre-change baseline on the same machine.
  (37 failures + 35 collection errors are pre-existing on macOS and reproduce
  unchanged on `origin/main` — see item 5 in the follow-ups doc.)

## Open items

Items that need a decision before going further are collected in
[`api-ref-warnings-followups.md`](./api-ref-warnings-followups.md). Notably, there is a
**griffe2md rendering bug that publishes incorrect parameter types without emitting any
warning** — worth reading before adding new pages to `api-pages.yaml`.
