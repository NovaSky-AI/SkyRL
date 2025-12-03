# Batching Support Matrix

## Current Implementation Analysis

### Transfer Backends
- **Broadcast**: Uses `torch.distributed.broadcast()` via process group
- **CUDA IPC**: Uses `torch.multiprocessing.reductions.reduce_tensor()` + IPC handles

### Training Backends
- **FSDP**: Fully Sharded Data Parallel
- **Megatron**: Megatron-LM
- **DeepSpeed**: DeepSpeed ZeRO

### Inference Backends
- **vLLM**: Local inference engine, supports CUDA IPC
- **SGLang**: Local inference engine, supports CUDA IPC
- **Remote**: OpenAI API compatible, does NOT support CUDA IPC

---

## Module Grouping Support

**Module Grouping**: Groups parameters by module name (e.g., all attention weights together)

| Training Backend | Transfer Backend | Module Grouping | Notes |
|-----------------|------------------|-----------------|-------|
| FSDP | Broadcast | ❌ No | Sends one param at a time (line 168) |
| FSDP | CUDA IPC | ✅ Yes | Groups by module (lines 201-215) |
| Megatron | Broadcast | ❌ No | Sends one param at a time (line 475) |
| Megatron | CUDA IPC | ❌ No | Uses threshold-based buckets, NOT module grouping (lines 321-329) |
| DeepSpeed | Broadcast | ❌ No | Sends one param at a time (line 131) |
| DeepSpeed | CUDA IPC | ✅ Yes | Groups by module (lines 164-178) |

**Key Finding**: 
- **FSDP + CUDA IPC**: Module grouping ✅ (for FlashRL)
- **DeepSpeed + CUDA IPC**: Module grouping ✅ (for FlashRL)
- **Megatron + CUDA IPC**: Uses threshold-based buckets instead of module grouping (Megatron doesn't use FlashRL)

---

## Threshold Batching Support

**Threshold Batching**: Accumulates multiple parameters/modules until size threshold is reached

| Training Backend | Transfer Backend | Threshold Batching | Threshold Config | Notes |
|-----------------|------------------|---------------------|------------------|-------|
| FSDP | Broadcast | ❌ No | N/A | Sends immediately, one param at a time |
| FSDP | CUDA IPC | ✅ Yes | `weight_transfer_threshold_cuda_ipc_GB` | Accumulates modules until threshold (line 246) |
| Megatron | Broadcast | ❌ No | N/A | Sends immediately, one param at a time |
| Megatron | CUDA IPC | ✅ Yes | `weight_transfer_threshold_cuda_ipc_GB` | Pre-buckets params by threshold (lines 321-329), then packs into single tensor |
| DeepSpeed | Broadcast | ❌ No | N/A | Sends immediately, one param at a time |
| DeepSpeed | CUDA IPC | ✅ Yes | `weight_transfer_threshold_cuda_ipc_GB` | Accumulates modules until threshold (line 212) |

**Key Finding**:
- **Broadcast**: No threshold batching (sends immediately)
- **CUDA IPC**: All training backends use threshold batching

---

## Inference Engine Support

| Inference Backend | CUDA IPC Support | Notes |
|-------------------|------------------|-------|
| vLLM | ✅ Yes | Has `update_weights_cuda_ipc()` method |
| SGLang | ✅ Yes | Has `update_weights_cuda_ipc()` function |
| Remote | ❌ No | Explicitly raises error: "Remote inference engines do not support CUDA IPC" |

**Key Finding**: Only local inference engines (vLLM, SGLang) support CUDA IPC. Remote engines use broadcast only.

---

## Complete Matrix: Module Grouping + Threshold Batching

| Training Backend | Transfer Backend | Inference Backend | Module Grouping | Threshold Batching | Notes |
|-----------------|------------------|-------------------|-----------------|-------------------|-------|
| FSDP | Broadcast | vLLM/SGLang/Remote | ❌ | ❌ | One param at a time |
| FSDP | CUDA IPC | vLLM/SGLang | ✅ | ✅ | Groups by module (FlashRL), batches by threshold |
| FSDP | CUDA IPC | Remote | ❌ | ❌ | Not supported (Remote doesn't support CUDA IPC) |
| Megatron | Broadcast | vLLM/SGLang/Remote | ❌ | ❌ | One param at a time |
| Megatron | CUDA IPC | vLLM/SGLang | ❌ | ✅ | Uses threshold buckets (not module grouping), packs into single tensor |
| Megatron | CUDA IPC | Remote | ❌ | ❌ | Not supported (Remote doesn't support CUDA IPC) |
| DeepSpeed | Broadcast | vLLM/SGLang/Remote | ❌ | ❌ | One param at a time |
| DeepSpeed | CUDA IPC | vLLM/SGLang | ✅ | ✅ | Groups by module (FlashRL), batches by threshold |
| DeepSpeed | CUDA IPC | Remote | ❌ | ❌ | Not supported (Remote doesn't support CUDA IPC) |

---

## Why Not Supported in Other Cases?

### 1. Broadcast doesn't use module grouping or threshold batching

**Reason**: Broadcast sends one parameter at a time immediately. No accumulation needed because:
- Broadcast is synchronous and efficient for individual tensors
- No IPC handle overhead (unlike CUDA IPC)
- Simpler implementation - just broadcast each param as it's processed

**Code Evidence**: All broadcast paths iterate `for name, param in params.items()` and send immediately (FSDP line 168, Megatron line 475, DeepSpeed line 131)

### 2. Remote inference engines don't support CUDA IPC

**Reason**: CUDA IPC requires:
- Shared GPU memory space (local only)
- Direct CUDA device access
- Same physical machine

Remote engines run on different machines, so CUDA IPC is impossible.

**Code Evidence**: `remote_inference_engine.py` line 194 explicitly raises error

### 3. Megatron CUDA IPC doesn't use module grouping

**Reason**: Megatron uses a different approach:
- Pre-buckets parameters by threshold during initialization (lines 321-329)
- Packs multiple tensors into a single contiguous buffer (lines 514-526)
- Sends packed tensor as one IPC handle

This is more efficient for Megatron's architecture but doesn't preserve module boundaries.

**Code Evidence**: Megatron uses `param_buckets` (threshold-based) instead of `module_to_params` (module-based)

### 4. Module grouping only for CUDA IPC (not broadcast)

**Reason**: Module grouping is specifically for **FlashRL** integration with vLLM:

**FlashRL Context**:
- FlashRL is a patched version of vLLM (applies `apply_flashrl_patch()`)
- FlashRL allocates **new storage for each parameter** during weight updates
- vLLM internally **fuses q, k, v weights** into a single QKV tensor (QKV fusion optimization)

**Why Module Grouping is Needed**:
- If you send `q_proj.weight`, then `k_proj.weight`, then `v_proj.weight` **separately**, FlashRL allocates storage for each individually
- But vLLM needs them **fused together** as a single QKV tensor
- By sending all attention weights **together** (module grouping), FlashRL can allocate the correct fused storage in one operation
- This ensures the fused QKV tensor gets properly allocated with the right layout

**Code Evidence**: 
- Comments in FSDP (lines 210-213) and DeepSpeed (lines 174-177): "For FlashRL integration, we allocate new storage for each param. Since q, k and v layer weights are fused internally by vllm, we need to pass the weights for all of these together."
- FlashRL engine: `examples/flash_rl/flash_rl_engine.py` applies `apply_flashrl_patch()` before vLLM initialization
- This is a **FlashRL-specific requirement**, not a general vLLM requirement

---

## Summary

**Module Grouping**:
- ✅ FSDP + CUDA IPC + vLLM/SGLang
- ✅ DeepSpeed + CUDA IPC + vLLM/SGLang
- ❌ All other combinations

**Threshold Batching**:
- ✅ All CUDA IPC combinations (FSDP/Megatron/DeepSpeed + CUDA IPC + vLLM/SGLang)
- ❌ All broadcast combinations

**Why**:
- Broadcast: No need for batching (sends immediately)
- CUDA IPC: Batching reduces IPC handle overhead
- Module grouping: Only needed for vLLM weight fusion optimization
- Remote: Can't use CUDA IPC (different machines)
