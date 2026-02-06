# RL Loop Verification Plan - Running tinker-cookbook/rl_loop.py on SkyRL

**Date:** 2026-02-01
**Goal:** Run `~/tinker-cookbook/tinker_cookbook/recipes/rl_loop.py` with zero code changes on SkyRL backend
**Status:** ✅ Server Running - Ready for Component Tests

---

## ✅ PROGRESS UPDATE (Hack Approach - Completed!)

**What we did:** Copied entire Tinker codebase from skyrl-tx to skyrl-train (~45 min)

### Completed Steps:
1. ✅ Copied `tx/tinker/` → `skyrl_train/tinker/` + `tx/utils/` → `skyrl_train/tx_utils/`
2. ✅ Updated all imports: `tx.tinker` → `skyrl_train.tinker`
3. ✅ Deleted JAX code: `jax.py`, `loss_fns.py`, removed JAX references
4. ✅ Fixed engine subprocess: `--extra vllm -m skyrl_train.tinker.engine`
5. ✅ Added dependencies: fastapi, sqlmodel, sqlalchemy, aiosqlite, cloudpathlib, httpx
6. ✅ Server running on http://0.0.0.0:8000 with Qwen3-0.6B

**Result:** Zero tx dependencies, no JAX conflicts, API server fully operational!

---

## Executive Summary

All required functionality for rl_loop.py is already implemented on branch `tyler/tinker-sampling-main`:
- ✅ Sampling with response logprobs
- ✅ save_weights_for_sampler() for weight sync
- ✅ forward_backward(loss_fn="importance_sampling")
- ✅ Checkpoint save/load/resume

**Key Finding:** rl_loop.py does NOT require prompt logprobs (only response logprobs at line 188), removing one major TODO from critical path.

**Current Branch:** `tyler/tinker-sampling-main` ✅

---

## Phase 1: Verification (Est: 2-4 hours)

### Commands to Run

#### 1. Start SkyRL Tinker API Server ✅ DONE!

**Command used (from skyrl-train):**
```bash
cd ~/SkyRL/skyrl-train

uv run --extra vllm python -m skyrl_train.tinker.api \
    --base-model "Qwen/Qwen3-0.6B" \
    --backend skyrl_train
```

**What to look for in logs:**
- ✅ "Created 1 inference engines for sampling" (NOT "SFT-only mode")
- ✅ "Application startup complete"
- ✅ "Uvicorn running on http://0.0.0.0:8000"

**Health check (run in separate terminal):**
```bash
curl http://localhost:8000/health
```

#### 2. Component Tests (Claude will create and run these)

Once server is running, Claude will create and run:
- Test 1: Sampling with response logprobs (num_samples=2)
- Test 2: Importance sampling loss function
- Test 3: Checkpoint save/load

#### 3. Run rl_loop.py End-to-End

**Quick 2-batch smoke test:**
```bash
cd ~/tinker-cookbook

uv run --with tinker --with datasets --with torch python -m tinker_cookbook.recipes.rl_loop \
    base_url=http://localhost:8000 \
    model_name="Qwen/Qwen2.5-0.5B-Instruct" \
    batch_size=8 \
    group_size=4 \
    lora_rank=32 \
    max_tokens=128 \
    log_path="/tmp/tinker-rl-test"
```

**Success criteria:**
- Script completes 2+ batches without errors
- Checkpoints saved to /tmp/tinker-rl-test/
- Metrics logged with reward values
- No Python exceptions

---

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| `NotImplementedError: Sampling not supported` | Wrong branch or server config | Verify on tyler/tinker-sampling-main |
| `KeyError: 'logprobs'` at rl_loop.py:188 | Sampling not returning logprobs | Check skyrl_train.py:240-280 |
| `Unknown loss function: importance_sampling` | Loss not registered | Check tx/tinker/loss_fns.py:42 |
| OOM during sampling | Too many samples | Reduce batch_size=4, group_size=2 |
| Server shows "SFT-only mode" | num_inference_engines=0 | Check backend-config has num_inference_engines=1 |

---

## Critical Files

1. **~/SkyRL/skyrl-tx/tx/tinker/backends/skyrl_train.py** (509 lines)
   - Lines 207-300: sample() with logprobs
   - Lines 301-350: save_weights_for_sampler()
   - Lines 400-509: checkpoint methods

2. **~/SkyRL/skyrl-train/skyrl_train/workers/worker_dispatch.py**
   - Lines 157-202: forward_backward(loss_fn=...)
   - Lines 318-338: save_weights_for_sampler()

3. **~/SkyRL/skyrl-tx/tx/tinker/loss_fns.py**
   - Line 42: "importance_sampling" in LOSS_FUNCTION_MAP

4. **~/tinker-cookbook/tinker_cookbook/recipes/rl_loop.py**
   - Line 148-154: save_weights_for_sampler()
   - Line 168-172: sample(num_samples=group_size)
   - Line 188: Assert response logprobs not None
   - Line 235: forward_backward(loss_fn="importance_sampling")

---

## Next Steps After Verification

### If Successful ✅
1. Add round-trip checkpoint tests
2. Implement config management (backend_config → SkyRL config)
3. Document setup in quickstart guide
4. Clean up and merge to main

### If Failed ❌
1. Check server logs in /tmp/skyrl-tinker-server.log
2. Verify branch with `wc -l skyrl_train.py` (should be ~509, not 220)
3. Test components individually
4. Add debug logging with `export SKYRL_LOG_LEVEL=DEBUG`

---

## What rl_loop.py Actually Requires

From detailed code analysis:

**Required APIs:**
1. `ServiceClient.create_lora_training_client(base_model, rank)` ✅
2. `TrainingClient.save_weights_for_sampler(name, ttl_seconds)` ✅
3. `ServiceClient.create_sampling_client(model_path)` ✅
4. `SamplingClient.sample(prompt, num_samples, sampling_params)` ✅
5. `TrainingClient.forward_backward(datums, loss_fn="importance_sampling")` ✅
6. `TrainingClient.optim_step(adam_params)` ✅
7. `ServiceClient.create_training_client_from_state_with_optimizer(path)` ✅

**Required Data Types:**
- `Datum(model_input, loss_fn_inputs)` ✅
- `loss_fn_inputs: {target_tokens, logprobs, advantages}` ✅
- `SampleOutput.sequences[].logprobs` (response only, NOT prompt) ✅
- `TensorData.from_torch()` serialization ✅

**NOT Required:**
- ❌ Prompt logprobs (rl_loop.py line 188 only asserts response logprobs)
- ❌ Multi-checkpoint sampling (always uses latest from line 148)

---

## Fallback Options

If verification reveals insurmountable issues:

**Option 1: Use JAX Backend**
```bash
uv run --extra gpu --extra tinker -m tx.tinker.api \
    --base-model "Qwen/Qwen2.5-0.5B-Instruct" \
    --backend jax
```

**Option 2: Debug specific components** based on error messages

**Option 3: Use external Tinker API** (e.g., Thinking Machines)

---

## Timeline

- **Phase 1 (Verification):** 2-4 hours ← WE ARE HERE
- **Phase 2 (Debug, if needed):** 4-8 hours
- **Phase 3 (Hardening):** 8-12 hours
- **Total:** 10-24 hours depending on verification outcome

---

## References

- Project Summary: ~/claude-docs/skyrl/project-summary.md
- Full Plan: ~/.claude/plans/tidy-coalescing-otter.md
- RL Loop Source: ~/tinker-cookbook/tinker_cookbook/recipes/rl_loop.py
- SkyRL Backend: ~/SkyRL/skyrl-tx/tx/tinker/backends/skyrl_train.py