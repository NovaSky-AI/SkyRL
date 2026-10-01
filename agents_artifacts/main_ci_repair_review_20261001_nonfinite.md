# CI repair amendment review (2026-10-01)

Manual review of the Tinker wire-format amendment against `AGENTS.md`, `.agents/docs/{contributing,testing,tinker,ci,development}.md`, and the `skyrl-ci-fix` skill. Review-agent delegation was unavailable.

## Findings

- No major source-code finding. `ForwardBackwardOutput` is returned to the SDK as protobuf, whose metrics are a `map<string, double>`. Preserving NaN and infinities keeps the metric key and invalid-value signal visible to clients. Already-stored JSON `null` cannot reveal which non-finite value was originally emitted; encoding NaN is an explicit lossy legacy fallback.
- Exact-SHA GPU E2E remains untested after this wire amendment. The prior successful run demonstrates the training path, while the 19 local SDK round-trip tests cover the changed serialization boundary. A refreshed GPU run is the remaining validation gap.
- The producer of the non-finite `importance_ratio` is still unproven. The possible zero-denominator path must not be asserted as the cause without a captured microbatch/loss mask.

## Scope and checks

- Removed the wandb-only summary invocation from both Tinker scripts. The observed wheel-download HTTP 500 was transient external infrastructure, not a demonstrated SkyRL defect; the original `--extra fsdp` invocation matches the other E2E scripts.
- `uv run --isolated --extra tinker --extra dev --with torch pytest -q tests/tinker/test_proto_serialization.py`: 19 passed. `torch` was supplied because the Tinker-only extra does not install it on macOS and `test_proto_serialization.py` imports the API module.
- Pinned Ruff 0.11.9, Black 24.10.0, shell syntax, and `git diff --check` passed. No other tracked files were changed for this amendment.
