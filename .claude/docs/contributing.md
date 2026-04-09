# Contributing


## New Model Support

### Checklist

1. **Check Megatron-Bridge support** — The model needs a provider in Megatron-Bridge. Check available branches/commits for the model's provider class.
2. **Check dependency compatibility** — New architectures may need additional deps (e.g., `mamba-ssm` for Mamba). Verify no conflicts with existing pins.
3. **Test inference first** — Add a test case to `test_engine_generation.py` (token-based generation).
4. **Test Megatron forward** — Add a test case to `test_megatron_worker.py` comparing HF vs Megatron logprobs.
5. **Create example script** — Add to `examples/train/<model>/` with README and training script.

## Tokenizer Quirks

- Some models have `pad_token_id=None` — use `eos_token_id` fallback.
- Some models need `trust_remote_code=True`.
