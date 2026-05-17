# Contributing

Start off any modification or debugging with SkyRL using this file as the primary instruction manual. Before coming up with a plan, ensure you have thoroughly gone through the instructions here as well as relevant documentation in `docs/`. (Ex: for understanding configuration, go through `docs/content/docs/configuration/config.mdx`)

**Google Style Guide**: Overall, you should follow Google's Python Style Guide while writing code for the project, unless specifically instructed by the user or by instructions here. 

**Follow existing patterns in the code**: Make sure to understand existing patterns in the codebase. Writing a new test? Study existing tests to understand common patterns for init/ teardown as well as helper functions used. For example: 
- For creating a tokenizer, use `skyrl.utils.tok.get_tokenizer` helper instead of manual init. 
- Use `InferenceEngineState` helper in tests for managing inference engine state instead of manual init of `VLLMRouter` and `VLLMServerGroup`
- Use the `ray_init_fixture` for cleaning up state between tests for GPU based tests. If possible, use `class_scoped_ray_init_fixture` or `module_scoped_ray_init_fixture` to avoid repeated init/ teardown.

**Small iteration cycles**: As much as possible, keep iteration cycles small. For new code or configurations, figure out the smallest possible test unit and iterate on that. For all tests, it's helpful to use the smallest possible model(s) or smallest model in the desired model family (Ex: `Qwen/Qwen3-0.6B` for Qwen 3) with the smallest number of GPUs.
  - Is this a configuration change? Ensure relevant existing CPU tests can pass. Add new tests if needed
  - Does this modify inference/ generation? Ensure relevant generation tests pass. Does this touch training <> inference boundary? Ensure relevant weight sync tests pass. Finally move to E2E tests only if needed. A single pass with `main_generate.py` is typically enough for most modifications. Add new tests if needed.
  - Does this add a new algorithm? Ensure configurations are updated. Add relevant CPU tests. Perform an E2E test for convergence. 
  - If changing environment code, test out the environment mocking LLM responses
  - If changing training code, test out a single training step with dummy inputs
  - Does this modify a training backend? Ensure relevant unit tests pass. Does this touch training <> inference boundary? Ensure weight sync tests pass. Finally move to E2E tests
  - For E2E training, run only 1 or few steps for a sanity check and ensure metrics and training is as expected first.

## Contribution Checklist

- Go through the development guide: `.claude/docs/development.md`.
- Ensure you've updated relevant example scripts and documentation for any changes
- Ensure you've updated `.claude/` files and `CLAUDE.md` for any changes in paths, naming, etc.

## New Model Support (Megatron)

### Checklist

1. **Check Megatron-Bridge support** — The model needs a provider in Megatron-Bridge. Check available branches/commits for the model's provider class.
2. **Check dependency compatibility** — New architectures may need additional deps (e.g., `mamba-ssm` for Mamba). Verify no conflicts with existing pins.
3. **Test inference first** — Add a test case to `test_engine_generation.py` (token-based generation).
4. **Test Megatron forward** — Add a test case to `test_megatron_worker.py` comparing HF vs Megatron logprobs.
5. **Create example script** — Add to `examples/train/<model>/` with README and training script.

## Tokenizer Quirks

- Some models have `pad_token_id=None` — use `eos_token_id` fallback.
- Some models need `trust_remote_code=True`.

## Anti-patterns

- Using Ray tasks/ actors with `fork` start method  - This leads to undefined behaviour. Use `spawn` start method instead.
- Passing the full `SkyRLTrainConfig` as an argument to a method or a class when only a sub-config is sufficient (example: `InferenceEngineConfig`)


## Comments

Comments should describe what the code is doing, not what instruction the user provided or what learnings-on-the-journey the agent stumbled into.

Do:

"Uses the start/update/finish lifecycle to enable chunked transfers. Per chunk, all tensors are packed into a single contiguous CUDA buffer (one dtype per chunk, guaranteed by the weight extractor) and one IPC handle is created for the packed buffer per rank."

Don't:

"Uses the start/update/finish lifecycle to enable chunked transfers. Per chunk, all tensors are packed into a single contiguous CUDA buffer (one dtype per chunk, guaranteed by the weight extractor) and one IPC handle is created for the packed buffer per rank. This mirrors the legacy path and avoids the one-handle-per-param ceiling of vLLM's default IPCWeightTransferEngine, which otherwise dominates latency for models with many small parameters."


Do:

```python
def test_freeze_moe_router_two_level_wrap():
    """
    Under Megatron's ``bf16=True`` path, chunks are wrapped as
    ``DDP(Float16Module(GPTModel))``. This tests whether `freeze_moe_router`
    can handle 2 levels of wrapping.
    """
    inner_model = _Model()
```

Don't:

```python
def test_freeze_moe_router_two_level_wrap():
    """Regression: recursive unwrap handles DDP(Float16Module(model)).

    Under Megatron's ``bf16=True`` path, chunks are wrapped as
    ``DDP(Float16Module(GPTModel))``. Single-level ``.module`` peel leaves
    ``Float16Module`` (which lacks ``.decoder``) and the helper raises. The
    worker uses a ``while hasattr(..., "module")`` loop — this test inlines
    the same loop to guard against regressions.
    """
    inner_model = _Model()
```


Do:

```bash
  # SFT training with Megatron backend for Qwen2.5-1.5B-Instruct on a
  # tool-calling dataset (Salesforce/APIGen-MT-5k).
  #
  # This script runs supervised fine-tuning using the Megatron backend with
  # pure data parallelism (DP=4) on 4 GPUs.
  #
  # Usage:
  #
  # export DATA_DIR=$HOME/data/apigen-mt-5k-openai
  # export WANDB_API_KEY=<your_key_here>
  # bash examples/train/sft/run_sft_megatron_apigen_mt.sh num_epochs=1 num_steps=<num_steps>
```

Don't:

```bash
  # SFT training with Megatron backend for Qwen2.5-1.5B-Instruct on a
  # tool-calling dataset (Salesforce/APIGen-MT-5k).
  #
  # This script runs supervised fine-tuning using the Megatron backend with
  # pure data parallelism (DP=4) on 4 GPUs. It exercises the tool-calling SFT
  # path: per-row ``tools`` schemas and ``system`` policy are threaded into
  # every ``apply_chat_template`` call, ``tool`` observation tokens are
  # masked to 0, and every assistant turn (including ``tool_calls``)
  # contributes to the loss via ``train_on_what=all_assistant_messages``.
  #
  # APIGen-MT-5k ships in ShareGPT format and contains 5000 rows. The
  # preprocessing step below converts it to OpenAI messages format and writes
  # a parquet shard the SFT trainer can load directly.
  #
  # Usage:
  #   bash examples/train/sft/run_sft_megatron_apigen_mt.sh [extra overrides...]
  #
  # Example (default: 10-step smoke run on 4 GPUs):
  #   bash examples/train/sft/run_sft_megatron_apigen_mt.sh
  #
  # Example (full epoch over the 5000 rows):
  #   bash examples/train/sft/run_sft_megatron_apigen_mt.sh num_epochs=1 num_steps=null

```