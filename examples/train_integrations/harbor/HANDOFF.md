# Stepwise Training Handoff — Harbor + SkyRL

## Current State (2026-03-12 ~23:00 UTC)

- **Training is STOPPED** (killed manually)
- **Monitoring cron is CANCELLED**
- **Latest checkpoint**: `global_step_30` at `/home/ray/codecontest-stepwise/ckpts/global_step_30`
- **Checkpoints available**: global_step_24, global_step_27, global_step_30
- **Total effective training steps**: ~32 (rewards went from 0.32 to peak 0.58)
- **W&B run**: `codecontest-stepwise` in project `harbor` at `sky-posttraining-uc-berkeley`

## How to Resume Training

```bash
cd /home/ray/default/SkyRL
# 1. Kill any lingering sandboxes
uv run --isolated --extra fsdp --extra harbor examples/train_integrations/harbor/kill_daytona_sandboxes.py

# 2. Clean up stale Ray placement groups
python3 -c "
import ray
from ray._raylet import PlacementGroupID
from ray.util.placement_group import PlacementGroup
ray.init(address='auto')
for pg_id, pg_info in ray.util.placement_group_table().items():
    if pg_info.get('state') == 'CREATED':
        try:
            ray.util.remove_placement_group(PlacementGroup(PlacementGroupID.from_hex(pg_id)))
        except: pass
"

# 3. Free port 8000 if occupied
ss -tlnp | grep 8000 && fuser -k 8000/tcp

# 4. Launch training (resumes from latest checkpoint automatically)
nohup bash examples/train_integrations/harbor/run_codecontest_stepwise.sh > /tmp/skyrl-logs/codecontest-stepwise-launch.log 2>&1 &
```

## How to Monitor

Set up a 15-minute cron using Claude Code's `/loop` command:
```
/loop 15m Check the stepwise training job status: 1) Check if the process is still running (ps aux | grep main_harbor). 2) Check the last 30 lines of /tmp/skyrl-logs/codecontest-stepwise-launch.log for errors or progress. 3) Look for training step progress ("Training Batches Processed", "step", "ckpt", "generate"). 4) If the process has crashed: a) Check the error in the log. b) If it's a transient error, run `cd /home/ray/default/SkyRL && uv run --isolated --extra fsdp --extra harbor examples/train_integrations/harbor/kill_daytona_sandboxes.py` to clean up sandboxes. c) Clean up stale Ray placement groups. d) Free port 8000 if occupied. e) Relaunch with: `cd /home/ray/default/SkyRL && nohup bash examples/train_integrations/harbor/run_codecontest_stepwise.sh > /tmp/skyrl-logs/codecontest-stepwise-launch.log 2>&1 &`. 5) Report status summary.
```

## Key Files Modified (from main branch)

### SkyRL repo (`/home/ray/default/SkyRL`, branch `harbor-step-wise`)

1. **`pyproject.toml`** — Harbor dependency changed from git commit to local path:
   ```
   harbor = { path = "/home/ray/default/harbor" }
   ```

2. **`examples/train_integrations/harbor/run_codecontest_stepwise.sh`** — NEW file, launch script with:
   - 8 GPUs, step-wise enabled, dual_clip policy loss
   - `ckpt_interval=2`, `max_ckpts_to_keep=3`, `resume_mode=latest`
   - Export dir: `/mnt/local_storage/codecontest-stepwise/exports`
   - `eval_interval=900`, `eval_before_train=false` (skip eval)
   - `max_concurrency=500` rate limiting
   - `PYTORCH_ALLOC_CONF=expandable_segments:True`
   - `TORCH_NCCL_AVOID_RECORD_STREAMS=1`

3. **`examples/train_integrations/harbor/harbor_generator.py`** — Bug fixes:
   - `_worker()`: Added `except BaseException` to catch `CancelledError` and return zeroed output
   - Added safety fill loop after `TaskGroup` to handle `None` entries in `all_outputs`
   - Added `_make_error_output()` helper

4. **`skyrl/train/trainer.py`** — Added `gc.collect()` before all `empty_cache()` calls (lines ~959, 967, 1130)

5. **`skyrl/backends/skyrl_train/workers/fsdp/fsdp_worker.py`** — Added `gc.collect()` before both `empty_cache()` calls in `broadcast_to_inference_engines()`

6. **`examples/train_integrations/harbor/OOM_ANALYSIS.md`** — Comprehensive crash analysis doc
7. **`examples/train_integrations/harbor/HANDOFF.md`** — This file
8. **`test_vllm_sleep_wake.py`** — Minimal vLLM sleep/wake repro script

### vLLM patch (applied to uv cache)

**File**: `/home/ray/.cache/uv/archive-v0/qNofxNgHzcv1I1Qbx76wG/vllm/device_allocator/cumem.py`

Applied Python-side changes from [vLLM PR #36535](https://github.com/vllm-project/vllm/pull/36535):
- `AllocationData.mapped: bool = True` field
- `CuMemAllocator.sleeping = False` flag
- `_python_free_callback`: returns dummy handle `(0, 0, 0, 0)` during sleep
- `sleep()`: tracks `data.mapped = False`, sets `self.sleeping = True`
- `wake_up()`: checks `data.mapped`, sets `self.sleeping = False`

**NOTE**: This patch is in the uv archive and all build dirs. It helps prevent Python-level double-free but doesn't fix the C++ side. The full fix requires recompiling vLLM with PR #36535's C++ changes or upgrading to vLLM 0.17+ when the PR is merged.

## The Crash Bug

### Root Cause
**vLLM issue [#36651](https://github.com/vllm-project/vllm/issues/36651)**: cumem allocator double-free and stale error codes during sleep/wake cycles. The crash manifests as `cudaErrorInvalidValue` at `flash_attn.py:484` (`self.scheduler_metadata[:n] = scheduler_metadata`).

### Key Evidence
- **NOT OOM**: Zero OOM/SIGKILL messages in any infra log. Memory is stable at ~21 GiB per GPU across sleep cycles.
- **Probabilistic**: Crashes after 2-11 sleep/wake cycles (mode: 3-5)
- **Standalone repro**: Pure sleep/wake cycles (no weight updates) do NOT reproduce — 15 cycles passed. The crash requires the full training loop with NCCL weight broadcast + colocated FSDP.
- **Not step-wise specific**: Non-step-wise training was never run long enough to compare, but the sleep/wake cycle is identical in both modes.

### Fix Status
- [PR #36535](https://github.com/vllm-project/vllm/pull/36535) is OPEN, not merged (filed 2026-03-10)
- Python-side patch applied (insufficient alone)
- C++ side requires recompiling vLLM or upgrading to 0.17+

### What Was Tested

| Fix | Result |
|-----|--------|
| `PYTORCH_ALLOC_CONF=expandable_segments:True` | No effect (crash is not OOM) |
| `gc.collect()` before `empty_cache()` | Slight improvement (5 vs 3-4 steps) |
| `enforce_eager=true` | **Worse** — Xid 31 MMU Fault |
| `gpu_memory_utilization=0.7` | **Worse** — crashed after 1 step |
| cumem Python patch (PR #36535) | Insufficient — C++ bugs still trigger |
| `TORCH_NCCL_AVOID_RECORD_STREAMS=1` | Untested in isolation |
| `ckpt_interval=2` | **Key mitigation** — checkpoints before crash window |

### Current Mitigation Strategy
With `ckpt_interval=2`, the training checkpoints every 2 steps. Since crashes happen at 2-5 cycles from resume, we usually get 1 checkpoint per run. The monitoring cron auto-restarts, losing ~1 step per crash. Net efficiency: ~70-80%.

## Reward Curve

| Global Step | Avg Reward | Notes |
|------------|-----------|-------|
| 1 | 0.324 | Baseline |
| 5 | 0.258 | |
| 10 | 0.262 | |
| 13 | 0.398 | |
| 17 | 0.445 | |
| 22 | 0.430 | |
| 25 | 0.449 | |
| 28 | 0.508-0.582 | Peak range |
| 31 | 0.348-0.383 | |

## Saved Logs

Key infra/experiment logs preserved at `examples/train_integrations/harbor/logs/`:

| File | Run | Description |
|------|-----|-------------|
| `infra-260311_122019.log` | Run 4 | First `cudaErrorInvalidValue` crash (resumed from step 5) |
| `infra-260311_163209.log` | Run 6 | Longest pre-mitigation run (11 sleep/wake cycles, steps 5→15) |
| `infra-260312_092526.log` | Run 13 | `enforce_eager=true` test — Xid 31 MMU Fault |
| `infra-260312_100154.log` | Run 14 | Best run with `gc.collect()` fix (5 steps, step 24→29) |
| `infra-260312_204717.log` | Run 20 | Most recent run |
| `launch-last.log` | Run 20 | Main stdout/stderr from last training launch |
| `vllm_repro_test.log` | — | Standalone vLLM sleep/wake repro (15 cycles, all passed) |

To grep for crash patterns: `grep "cudaErrorInvalidValue\|scheduler_metadata\|still in use" logs/infra-*.log`

## Data Setup

- **Dataset**: CodeContests from HuggingFace (`open-thoughts/CodeContests`)
- **Location**: `/home/ray/data/harbor/CodeContests` (9644 tasks)
- **Prepared by**: `examples/train_integrations/harbor/prepare_harbor_dataset.py`

## Config Summary

```yaml
model: Qwen/Qwen3-8B
num_gpus: 8
colocate_all: true
strategy: fsdp2
train_batch_size: 32
n_samples_per_prompt: 8
max_model_len: 32768
ckpt_interval: 2
max_ckpts_to_keep: 3
resume_mode: latest
step_wise_trajectories: true
policy_loss_type: dual_clip
loss_reduction: seq_mean_token_sum_norm
environment: daytona
agent: terminus-2
max_turns: 32
max_concurrency: 500
```

## Next Steps

1. **Upgrade vLLM to 0.17+** when PR #36535 is merged — this is the real fix
2. **Or** rebuild vLLM 0.16.0 from source with PR #36535's C++ changes applied
3. **Or** continue with `ckpt_interval=2` + auto-restart as mitigation
4. **Consider**: Running a non-step-wise comparison to confirm the crash is not step-wise specific
5. **Consider**: Filing a vLLM issue with our specific repro (SkyRL + FSDP colocated + NCCL weight sync)
