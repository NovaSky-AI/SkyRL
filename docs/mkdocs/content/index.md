# SkyRL API Reference

## SkyRL

- **Backends** — [Abstract Backend](backends.md), [JAX Backend](jax_backend.md), [SkyRL-Train Backend](skyrl_train_backend.md)
- [Tinker Engine](tinker.md) — Orchestration engine for RL training
- [Types](types.md) — Request/response types for the Tinker API
- [TX Models](tx_models.md) — Model loading and configuration for the JAX backend
- [Data Interface](data.md) — TensorBatch, TrainingInput, GeneratorInput/Output
- [Generator](generator.md) — GeneratorInterface, InferenceEngineInterface
- [Trainer](trainer.md) — RayPPOTrainer, Dispatch, Worker APIs
- [Entrypoint](entrypoint.md) — BasePPOExp, EvalOnlyEntrypoint
- [Algorithm Registry](registry.md) — Advantage estimators, policy loss registries
- [Environment Variables](env_vars.md) — Configuration via environment variables

## SkyRL-Gym

- [Environment](env.md) — Env, BaseTextEnv, step outputs
- [Tools](tools.md) — ToolGroup, tool decorator
