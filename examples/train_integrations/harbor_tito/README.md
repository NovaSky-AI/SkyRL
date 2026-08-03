# Harbor TITO Integration

This example keeps the baseline `train_integrations/harbor` integration
unchanged and routes Harbor's Chat Completions calls through SkyRL's TITO proxy.

Use these modules in place of the baseline Harbor entrypoints:

```bash
# Generation-only validation
uv run --isolated --extra fsdp --extra harbor-tito \
  -m examples.train_integrations.harbor_tito.entrypoints.main_harbor_generate \
  <overrides>

# Synchronous training
uv run --isolated --extra fsdp --extra harbor-tito \
  -m examples.train_integrations.harbor_tito.entrypoints.main_harbor \
  <overrides>

# Fully asynchronous training
uv run --isolated --extra fsdp --extra harbor-tito \
  -m examples.train_integrations.harbor_tito.entrypoints.main_harbor_fully_async \
  <overrides>
```

The example reuses Harbor's dataset adapter and base trial defaults, then adds
TITO-specific interleaved-thinking and trace-parity settings.
