# FlashRL Example Setup

This directory contains the example-local FlashRL integration used by SkyRL's
quantized rollout examples.

## Support Boundary

The current example path supports:

- local Ray-wrapped inference engines
- synchronous vLLM engines only
- `fsdp` and `fsdp2`
- single-turn generation

It does not currently support the new HTTP/server-group inference path or
`async_engine`.

## Installation

The `skyrl[flashrl]` extra installs the shared SkyRL training dependencies, but
it does not install the custom FlashRL-compatible vLLM wheel. You must provide
that wheel separately.

The example scripts in this directory all accept these optional environment
overrides:

- `FLASHRL_VLLM_WHEEL_URL`
- `FLASHRL_TRANSFORMERS_VERSION`

If those are unset, the scripts default to the legacy `skyrl_train-v0.1.0`
wheel asset and `transformers==4.53.3`. During the vLLM migration, override
`FLASHRL_VLLM_WHEEL_URL` to test a newer wheel without editing each script.

## Required Environment Variables

`FLASHRL_CONFIG` is required.

- For FP8 examples, set `FLASHRL_CONFIG=fp8_vllm`.
- For Int8 examples, set `FLASHRL_CONFIG` to the FlashRL YAML config path.

The bundled env files are:

- `examples/train/flash_rl/.env.fp8`
- `examples/train/flash_rl/.env.int8`
- `examples/train/flash_rl/.env.0.5b_int8`

## Patch Hook Contract

The example wrapper expects the custom vLLM wheel to expose a patch hook before
engine construction.

Default import target:

- `vllm.model_executor.layers.patch:apply_patch`

If your migrated FlashRL wheel exposes the hook at a different import path, set:

- `SKYRL_FLASHRL_PATCH_FN=<module>:<callable>`

This lets the example wrapper adopt the new hook location without another repo
edit.
