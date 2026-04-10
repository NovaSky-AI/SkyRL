# Contributing

**Overall guide**: Always test in components. For different features/ bug fixes, follow the following pattern:
    - Is this a configuration change? Ensure CPU tests can pass. 
    - Does this modify inference/ generation? Ensure relevant generation tests pass. Does this touch training <> inference boundary? Ensure weight sync tests pass. Finally move to E2E tests only if needed. A single pass with `main_generate.py` is typically enough for most modifications.
    - Does this add a new algorithm? Ensure configurations are updated. Add relevant CPU tests. Perform an E2E test for convergence. 
    - Does this modify a training backend? Ensure relevant unit tests pass. Does this touch training <> inference boundary? Ensure weight sync tests pass. Finally move to E2E tests.

**Follow existing patterns in the code**: Make sure to understand existing patterns in the codebase. For example: 
- For creating a tokenizer, use `skyrl.utils.tok.get_tokenizer` helper instead of manual init. 
- Use `InferenceEngineState` helper in tests for managing inference engine state instead of manual init of `VLLMRouter` and `VLLMServerGroup`

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
- Passing the full `SkyRLTrainConfig` when only a sub-config is sufficient (example: `InferenceEngineConfig`)