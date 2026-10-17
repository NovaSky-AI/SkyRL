# Vendored tau-bench retail domain

The code under `tau_core/` is vendored from Sierra's **tau-bench**
(https://github.com/sierra-research/tau-bench, MIT License) — specifically the
`retail` domain: the tool implementations, database (`data/*.json`), task sets
(`tasks_*.py`), policy `wiki.md`, rules, and the `Env` reward machinery.

Two changes were made versus upstream:

1. Imports were rewritten from `tau_bench.*` to
   `skyrl_gym.envs.tau_bench.tau_core.*`.
2. The litellm-based user simulator was removed. `base.Env` now takes an injected
   `user` (a `BaseUserSimulationEnv`) instead of constructing one via litellm, so
   SkyRL supplies its own user simulator (see `../user_simulator.py`).

The SkyRL-Gym wrapper that drives this domain is `../env.py` (`TauBenchEnv`).

To refresh against upstream, re-vendor the `retail` domain and re-apply the two
changes above.
