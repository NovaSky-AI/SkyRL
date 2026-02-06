# SkyRL Tinker Integration - Project Summary

**Last Updated:** 2026-02-06
**Branch:** `tyler/tinker-sampling-main` (PR #999)
**Status:** Ready for Merge

---

## Completed Work

### PR #999: Tinker SkyRL Backend Sampling Support

**Key commits:**
1. Initial sampling implementation with logprobs support
2. `save_weights_for_sampler()` with ephemeral mode (persist=False)
3. Checkpoint save/load functionality
4. `importance_sampling` loss added to PolicyLossRegistry
5. `init_weight_sync_state()` fix for Tinker API flow

### Verified Functionality

| Feature | Status | Notes |
|---------|--------|-------|
| Tinker API server startup | Done | With SkyRL backend on 4xL4 GPUs |
| Model creation (LoRA) | Done | `create_lora_training_client()` |
| Sampling with logprobs | Done | Response logprobs returned correctly |
| Weight sync to inference | Done | `save_weights_for_sampler()` works |
| `forward_backward()` | Done | With importance_sampling loss |
| `optim_step()` | Done | Learning rate applied |
| Checkpoint save | Done | `tinker://model_id/weights/N` format |
| rl_loop.py end-to-end | Done | 9 batches completed successfully |

### Test Results (2026-02-06)

```
rl_loop.py with Qwen/Qwen3-0.6B:
- Batches completed: 9 (stopped due to disk space, not code error)
- Checkpoint saved: tinker://model_5987ddb1/weights/000005
- Metrics logged: /tmp/tinker-rl-test/metrics.jsonl
- Average batch time: ~18 seconds
```

---

## Key Files Modified

1. **skyrl-train/skyrl_train/utils/ppo_utils.py**
   - Added `IMPORTANCE_SAMPLING` to `PolicyLossType` enum
   - Implemented `importance_sampling_loss()` function
   - Registered in `PolicyLossRegistry.repopulate_registry()`

2. **skyrl-tx/tx/tinker/backends/skyrl_train.py**
   - Added `init_weight_sync_state()` call after `build_models()`
   - This initializes `_weight_transfer_sender` required for weight sync

---

## Architecture Notes

### Tinker API Flow (SkyRL Backend)
```
ServiceClient.create_lora_training_client()
    -> SkyRLTrainBackend.create_model()
        -> RayPPOTrainer(...)
        -> trainer.build_models(PolicyWorker, ...)
        -> trainer.init_weight_sync_state()  <- CRITICAL: must be called!

TrainingClient.save_weights_for_sampler()
    -> backend.save_weights_for_sampler(persist=False)
        -> dispatch.broadcast_to_inference_engines()
            -> worker._weight_transfer_sender.send_chunks()
```

### Loss Function Implementation
```python
# importance_sampling matches Tinker docs:
# https://tinker-docs.thinkingmachines.ai/losses#policy-gradient-importance_sampling
prob_ratio = torch.exp(log_probs - old_log_probs)
loss = -(prob_ratio * advantages).sum()
```

---

## Known Issues / TODOs

### High Priority
- [ ] **Disk space management**: Checkpoints fill /tmp quickly on multi-batch runs
- [ ] Clean up tinker.db between test runs: `rm skyrl-tx/tx/tinker/tinker.db`

### Medium Priority
- [ ] **Config management**: `backend_config` params don't fully propagate to SkyRL config
- [ ] **Prompt logprobs**: Not yet implemented (warning logged, not blocking)
- [ ] Review pcmoritz feedback on hardcoded model workaround

### Low Priority
- [ ] Add explicit tests for importance_sampling loss in test suite
- [ ] Document Tinker + SkyRL setup in quickstart guide
- [ ] Consider adding PPO loss to PolicyLossRegistry (currently only in JAX backend)

---

## How to Test

### Start Server
```bash
cd ~/tgriggs/SkyRL/skyrl-tx
rm -f tx/tinker/tinker.db  # Clean database

uv run --extra skyrl_train --extra tinker -m tx.tinker.api \
    --base-model "Qwen/Qwen3-0.6B" \
    --backend skyrl_train
```

### Run RL Loop Test
```bash
cd ~/tinker-cookbook
TINKER_API_KEY=tml-test uv run --with tinker --with datasets --with torch \
    python -m tinker_cookbook.recipes.rl_loop \
    base_url=http://localhost:8000 \
    model_name="Qwen/Qwen3-0.6B" \
    batch_size=8 \
    group_size=4 \
    lora_rank=32 \
    max_tokens=128 \
    save_every=5 \
    log_path="/tmp/tinker-rl-test"
```

---

## References

- PR #999: https://github.com/NovaSky-AI/SkyRL/pull/999
- Tinker Loss Docs: https://tinker-docs.thinkingmachines.ai/losses
- RL Loop Recipe: ~/tinker-cookbook/tinker_cookbook/recipes/rl_loop.py
- Detailed Plan: ~/tgriggs/SkyRL/claude/rl-loop-verify.md
