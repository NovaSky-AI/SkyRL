# API reference docs — follow-ups needing a decision

The 25 warnings the API reference build actually emitted are fixed (see
[`api-ref-warnings.md`](./api-ref-warnings.md)); that build is now at **0 warnings**.

The items below came up while fixing them. None are required to get the build clean, and
each involves a judgement call I did not want to make unilaterally.

---

## 1. griffe2md silently publishes **incorrect** parameter types (no warning emitted)

**This is the one worth looking at first.** It is a correctness bug in the rendered docs,
not a warning.

When a function is *partially* annotated, griffe2md's signature renderer carries the most
recent annotation forward onto every following unannotated parameter. griffe itself parses
these correctly (`annotation=None`); the defect is in griffe2md's rendering.

Minimal reproduction:

```python
def leaky(self, first: int, second=None, third: str = "x", fourth=3): ...
```

renders in the API reference as:

```
leaky(first: int, second: int = None, third: str = 'x', fourth: str = 3)
```

`second` and `fourth` have no annotation at all, but the docs assert they are `int` and
`str`. Nothing is logged, so this does not show up in a warning sweep.

### Currently affected: 15 parameters across 10 documented functions

| Location | Function | Misrendered |
|---|---|---|
| `skyrl/train/entrypoints/main_base.py:155` | `get_trainer()` | `colocate_pg` shown as `GeneratorInterface` |
| `skyrl/backends/skyrl_train/workers/worker.py:473` | `save_checkpoint()` | `tokenizer` shown as `str` |
| `skyrl/backends/skyrl_train/workers/worker.py:500` | `save_hf_model()` | `tokenizer` shown as `str` |
| `skyrl/backends/skyrl_train/workers/worker.py:232` | `__init__()` | `args`, `kwargs` shown as `TrainerConfig` |
| `skyrl/backends/skyrl_train/workers/worker.py:548` | `__init__()` | `num_nodes`, `num_gpus_per_node` shown as `TrainerConfig` |
| `skyrl/backends/skyrl_train/distributed/dispatch.py:70` | `dispatch()` | `args`, `kwargs` shown as `str` |
| `skyrl/backends/skyrl_train/distributed/dispatch.py:104` | `dispatch()` | `kwargs` shown as `TrainingInputBatch` |
| `skyrl/backends/skyrl_train/distributed/dispatch.py:221` | `dispatch()` | `args`, `kwargs` shown as `str` |
| `skyrl-gym/skyrl_gym/tools/core.py:51` | `execute_tool()` | `args`, `kwargs` shown as `str` |
| `skyrl/tx/models/types.py:30` | `is_lora_param()` | `_value` shown as `tuple` |

`get_trainer(): colocate_pg` shown as `GeneratorInterface` is the most misleading of these —
it is a placement group, and the type is confidently wrong on a public documented hook.

### Options

1. **Annotate the affected parameters** (my recommendation). Fully annotating a signature
   makes the leak impossible. Small, contained diff over the 10 functions above; the risk
   is picking the wrong type for things like `tokenizer` (see item 3).
2. **Fix or patch griffe2md.** The right long-term fix and it protects every future page,
   but it means an upstream PR or a local patch/vendored template to maintain.
3. **Both** — annotate now, upstream the renderer fix separately.

**Question:** which do you want? If (1), should I annotate `tokenizer` as
`Optional[PreTrainedTokenizerBase]`, or use the broader
`PreTrainedTokenizerBase | ProcessorMixin` union where processors are also accepted?

---

## 2. 67 latent docstring warnings outside the current page config

griffe only parses docstrings for objects that are actually rendered, so these are
invisible today. They will start firing the moment the corresponding objects are added to
`docs/api-pages.yaml`.

Breakdown:

| Count | Category |
|---|---|
| 44 | `No type or annotation for parameter 'X'` |
| 10 | `Parameter 'X' does not appear in the function signature` |
| 9 | `No type or annotation for returned value N` |
| 2 | `No type or annotation for returned value 'X'` |
| 2 | `Confusing indentation for continuation line` |

I did **not** fix these, for two reasons: they are not part of "warnings while building the
API reference", and the 44 annotation ones need per-case judgement (`tokenizer`,
`processor`, `kv_cache`, `func`, `model_runner`, `placement_group`, …) where a guessed type
is worse than no type. Full list at the bottom of this file.

The 10 `does not appear in the function signature` ones are **genuine documentation bugs** —
the docstring documents a parameter the function does not have:

- `skyrl/backends/skyrl_train/utils/ppo_utils.py:1234-1274` — six of these come from two
  docstrings that use a Markdown bullet list inside `Args:`, so griffe reads
  `- token_level_rewards`, `- response_mask`, `- index` as parameter *names*. Cosmetic
  formatting fix, no ambiguity.
- `skyrl/backends/skyrl_train/utils/ppo_utils.py:100` — documents `action_mask`, which is
  not in the signature. Looks like a rename that the docstring missed; needs someone who
  knows whether it became `response_mask`.
- `skyrl/backends/skyrl_train/workers/model_wrapper.py:53` — documents `packing_samples`,
  not in the signature.
- `skyrl/backends/skyrl_train_backend.py:75` — documents `config_container`, not in the
  signature.
- `skyrl-gym/skyrl_gym/envs/registration.py:123` — documents `print_all`, not in the
  signature.

**Question:** want a follow-up PR for these? I'd suggest splitting it: (a) the ppo_utils
bullet-list formatting and the two indentation warnings (mechanical, safe), then (b) the
stale-parameter-name ones, which need a reviewer who knows the intent, then (c) the bulk
annotation pass.

---

## 3. `tokenizer` has no consistent annotation anywhere in the codebase

Nine of the latent warnings, plus two entries in item 1, are an unannotated `tokenizer`
parameter. Related parameters (`processor` ×1, `**tokenizer_kwargs` ×3) are also bare.
Deciding this once would let it be applied everywhere mechanically.

Candidates seen in the codebase: `PreTrainedTokenizerBase` (used in
`skyrl/train/dataset/dataset.py`), or a union with `ProcessorMixin` for the VLM paths.

**Question:** what is the house style here?

---

## 4. `uv` emits a warning on every doc build (cosmetic)

Every `uv run` in the repo, including the documented doc-build command, prints:

```
warning: The `extra-build-dependencies` option is experimental and may change without
warning. Pass `--preview-features extra-build-dependencies` to disable this warning.
```

This comes from `[tool.uv] extra-build-dependencies` in `pyproject.toml`, not from the docs
tooling. Silencing it means either passing `--preview-features extra-build-dependencies` or
setting `UV_PREVIEW_FEATURES` — a repo-wide change affecting every documented command, so I
left it alone.

**Question:** worth silencing repo-wide, or leave it?

---

## 5. Aside: `transformers` is never installed on macOS

Not a docs issue, but it limited how much I could verify locally, so flagging it.

`[tool.uv] override-dependencies` in `pyproject.toml` contains:

```
"transformers>=5.6.1,<=5.8.0; sys_platform == 'linux'",
```

A `uv` override *replaces* the original requirement, so on macOS the marker is false and
`transformers` is dropped from the resolution entirely — even though it is declared as a
core dependency. Consequence: `import transformers` fails on darwin, so 35 test modules
fail to collect and another 37 tests fail at import on macOS. This reproduces unchanged on
`origin/main`, so nothing here is a regression from this PR. The doc build is unaffected
because griffe is a static analyser and never imports the code.

If a darwin-compatible override was intended, adding a second entry with
`sys_platform == 'darwin'` would restore it. Not touched here.

---

## Appendix: full list of the 67 latent warnings

Regenerate with the doc build after adding the relevant objects to `api-pages.yaml`, or
parse every docstring in the package directly.

```
skyrl-gym/skyrl_gym/core.py:84: No type or annotation for returned value 1
skyrl-gym/skyrl_gym/envs/registration.py:123: No type or annotation for parameter 'print_all'
skyrl-gym/skyrl_gym/envs/registration.py:123: Parameter 'print_all' does not appear in the function signature
skyrl/backends/skyrl_train/distributed/dispatch.py:343: No type or annotation for parameter 'dtype'
skyrl/backends/skyrl_train/distributed/dispatch.py:344: No type or annotation for parameter 'device'
skyrl/backends/skyrl_train/distributed/fsdp_utils.py:185: No type or annotation for returned value 'dict'
skyrl/backends/skyrl_train/distributed/ulysses/monkey_patch.py:72: No type or annotation for returned value 1
skyrl/backends/skyrl_train/distributed/ulysses/utils.py:284: No type or annotation for returned value 1
skyrl/backends/skyrl_train/distributed/ulysses/utils.py:285: No type or annotation for returned value 2
skyrl/backends/skyrl_train/distributed/ulysses/utils.py:286: No type or annotation for returned value 'int'
skyrl/backends/skyrl_train/inference_servers/setup.py:238: No type or annotation for parameter 'tokenizer'
skyrl/backends/skyrl_train/inference_servers/setup.py:67: No type or annotation for parameter 'placement_group'
skyrl/backends/skyrl_train/inference_servers/spec_decode_utils.py:35: No type or annotation for parameter 'model_runner'
skyrl/backends/skyrl_train/inference_servers/utils.py:228: No type or annotation for returned value 1
skyrl/backends/skyrl_train/utils/off_policy_correction_utils.py:165: Confusing indentation for continuation line 23 in docstring, should be 4 * 2 = 8 spaces, not 6
skyrl/backends/skyrl_train/utils/off_policy_correction_utils.py:167: Confusing indentation for continuation line 25 in docstring, should be 4 * 2 = 8 spaces, not 6
skyrl/backends/skyrl_train/utils/ppo_utils.py:100: No type or annotation for parameter 'action_mask'
skyrl/backends/skyrl_train/utils/ppo_utils.py:100: Parameter 'action_mask' does not appear in the function signature
skyrl/backends/skyrl_train/utils/ppo_utils.py:1234: No type or annotation for parameter '- token_level_rewards'
skyrl/backends/skyrl_train/utils/ppo_utils.py:1234: Parameter '- token_level_rewards' does not appear in the function signature
skyrl/backends/skyrl_train/utils/ppo_utils.py:1235: No type or annotation for parameter '- response_mask'
skyrl/backends/skyrl_train/utils/ppo_utils.py:1235: Parameter '- response_mask' does not appear in the function signature
skyrl/backends/skyrl_train/utils/ppo_utils.py:1272: No type or annotation for parameter '- token_level_rewards'
skyrl/backends/skyrl_train/utils/ppo_utils.py:1272: Parameter '- token_level_rewards' does not appear in the function signature
skyrl/backends/skyrl_train/utils/ppo_utils.py:1273: No type or annotation for parameter '- response_mask'
skyrl/backends/skyrl_train/utils/ppo_utils.py:1273: Parameter '- response_mask' does not appear in the function signature
skyrl/backends/skyrl_train/utils/ppo_utils.py:1274: No type or annotation for parameter '- index'
skyrl/backends/skyrl_train/utils/ppo_utils.py:1274: Parameter '- index' does not appear in the function signature
skyrl/backends/skyrl_train/workers/model_wrapper.py:53: Parameter 'packing_samples' does not appear in the function signature
skyrl/backends/skyrl_train/workers/worker_dispatch.py:235: No type or annotation for parameter 'worker_kwargs'
skyrl/backends/skyrl_train/workers/worker_dispatch.py:277: No type or annotation for parameter 'worker_kwargs'
skyrl/backends/skyrl_train/workers/worker_dispatch.py:345: No type or annotation for parameter 'worker_kwargs'
skyrl/backends/skyrl_train/workers/worker_dispatch.py:387: No type or annotation for parameter 'worker_kwargs'
skyrl/backends/skyrl_train/workers/worker_utils.py:109: No type or annotation for parameter 'group'
skyrl/backends/skyrl_train_backend.py:75: No type or annotation for parameter 'config_container'
skyrl/backends/skyrl_train_backend.py:75: Parameter 'config_container' does not appear in the function signature
skyrl/backends/utils.py:33: No type or annotation for parameter 'dtype'
skyrl/train/generators/skyrl_gym_generator.py:151: No type or annotation for parameter 'tokenizer'
skyrl/train/generators/skyrl_gym_generator.py:731: No type or annotation for parameter 'max_input_length'
skyrl/train/generators/skyrl_gym_generator.py:731: Parameter 'max_input_length' does not appear in the function signature
skyrl/train/generators/utils.py:434: No type or annotation for returned value 1
skyrl/train/generators/utils.py:589: No type or annotation for parameter 'tokenizer'
skyrl/train/generators/utils.py:594: No type or annotation for returned value 1
skyrl/train/generators/utils.py:673: No type or annotation for parameter 'tokenizer'
skyrl/train/generators/utils.py:674: No type or annotation for parameter 'assistant_logprobs'
skyrl/train/generators/utils.py:680: No type or annotation for returned value 1
skyrl/train/sft_trainer.py:465: No type or annotation for parameter 'tokenizer'
skyrl/train/sft_trainer.py:473: No type or annotation for parameter '**tokenizer_kwargs'
skyrl/train/sft_trainer.py:551: No type or annotation for parameter 'tokenizer'
skyrl/train/sft_trainer.py:554: No type or annotation for parameter 'processor'
skyrl/train/sft_trainer.py:557: No type or annotation for parameter '**tokenizer_kwargs'
skyrl/train/sft_trainer.py:640: No type or annotation for parameter 'tokenizer'
skyrl/train/sft_trainer.py:642: No type or annotation for parameter '**tokenizer_kwargs'
skyrl/train/utils/trainer_utils.py:98: No type or annotation for parameter '*args'
skyrl/train/utils/trainer_utils.py:99: No type or annotation for parameter '**kwargs'
skyrl/train/utils/utils.py:1118: No type or annotation for parameter 'module_config'
skyrl/train/utils/utils.py:1119: No type or annotation for parameter 'override_config_kwargs'
skyrl/train/utils/utils.py:1122: No type or annotation for returned value 1
skyrl/tx/layers/stacked.py:214: No type or annotation for parameter 'decode_layers'
skyrl/tx/layers/util.py:95: No type or annotation for parameter 'func'
skyrl/tx/layers/util.py:96: No type or annotation for parameter '*args'
skyrl/tx/utils/generator.py:166: No type or annotation for parameter 'tokenizer'
skyrl/tx/utils/generator.py:335: No type or annotation for parameter 'tokenizer'
skyrl/tx/utils/generator.py:81: No type or annotation for parameter 'kv_cache'
skyrl/tx/utils/generator.py:82: No type or annotation for parameter 'k'
skyrl/tx/utils/generator.py:83: No type or annotation for parameter 'v'
skyrl/tx/utils/generator.py:84: No type or annotation for parameter 'positions'
```
