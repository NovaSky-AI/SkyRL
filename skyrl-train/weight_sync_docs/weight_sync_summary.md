# Weight Sync Flow Summary

## Overview

Weight sync transfers updated policy model weights from training workers to inference engines during RL training. It ensures inference engines use the latest policy weights for generation.

**Key Components:**
- **Training side**: Policy model workers (FSDP/Megatron/DeepSpeed)
- **Inference side**: Inference engines (vLLM/SGLang/Remote)
- **Transfer mechanisms**: NCCL Broadcast (standard) or CUDA IPC (optimized for colocated)

---

## Step 1: Initialization (`init_weight_sync_state`)

### Purpose
Set up a distributed process group connecting training rank 0 with all inference engine ranks.

### Key Functions
- **Training**: `trainer.py: init_weight_sync_state()` → `worker.py: init_weight_sync_state()` → `init_custom_process_group()`
- **Inference**: `inference_engine_client.init_weight_update_communicator()` → engine-specific init methods

### Flow

**Training Side (Rank 0 only):**
1. Gets master address (node IP) and binds to random port
   - Function: `ray._private.services.get_node_ip_address()`, `socket.socket().bind()`
   - Code: `skyrl_train/workers/worker.py:268-271`
2. Calculates world size: `1 (training) + num_inference_engines * tp_size * pp_size * dp_size`
   - Code: `skyrl_train/workers/worker.py:273-279`
3. Calls `inference_engine_client.init_weight_update_communicator()` (async)
   - Function: `inference_engine_client.py: init_weight_update_communicator()`
   - Code: `skyrl_train/workers/worker.py:288-298`
4. Calls `init_custom_process_group()` to join as rank 0
   - Function: `distributed/utils.py: init_custom_process_group()`
   - Code: `skyrl_train/workers/worker.py:300-309`
5. Stores `_model_update_group` for later broadcasts
   - Code: `skyrl_train/workers/worker.py:311`

**Inference Engine Side:**
- Each engine receives: `master_addr`, `master_port`, `rank_offset`, `world_size`, `group_name`, `backend`
  - Function: `inference_engine_client.py: init_weight_update_communicator()`
  - Code: `skyrl_train/inference_engines/inference_engine_client.py:355-381`
- Calculates rank: `torch.distributed.get_rank() + rank_offset`
  - Code: `skyrl_train/inference_engines/vllm/vllm_engine.py:98`
- Joins process group via `init_custom_process_group()` or engine-specific API
  - **vLLM**: `vllm_engine.py: init_weight_update_communicator()` → `init_custom_process_group()`
    - Code: `skyrl_train/inference_engines/vllm/vllm_engine.py:103-109`
  - **SGLang**: `sglang_engine.py: init_weight_update_communicator()` → `engine.tokenizer_manager.init_weights_update_group()`
    - Code: `skyrl_train/inference_engines/sglang/sglang_engine.py:260-271`
  - **Remote**: `remote_inference_engine.py: init_weight_update_communicator()` → HTTP POST
    - Code: `skyrl_train/inference_engines/remote_inference_engine.py:168-182`
- Stores `_model_update_group`
  - Code: `skyrl_train/inference_engines/vllm/vllm_engine.py:103`

### Differences

**Policy Backends (FSDP/Megatron/DeepSpeed):**
- ✅ All inherit same `init_weight_sync_state` from base `Worker` class
  - Code: `skyrl_train/workers/worker.py:256-311`
- ✅ Same implementation across all backends
- ⚠️ Some set `self.use_cuda_ipc = True` during `__init__` if:
  - `weight_sync_backend == "nccl"` AND `colocate_all == True`
  - FSDP: `skyrl_train/workers/fsdp/fsdp_worker.py:101-102`
  - Megatron: `skyrl_train/workers/megatron/megatron_worker.py:282-283`
  - DeepSpeed: `skyrl_train/workers/deepspeed/deepspeed_worker.py:99-100`
- ⚠️ Megatron also initializes weight conversion tasks/buckets for CUDA IPC
  - Code: `skyrl_train/workers/megatron/megatron_worker.py:285-329`

**Inference Engines:**
- **vLLM**: Direct `init_custom_process_group()` call
- **SGLang**: Uses SGLang's `tokenizer_manager.init_weights_update_group()` API
- **Remote**: HTTP POST to `/init_weight_update_communicator` or `/init_weights_update_group`

### Code References
- Training: `skyrl_train/workers/worker.py:256-311`
- vLLM: `skyrl_train/inference_engines/vllm/vllm_engine.py:88-113`
- SGLang: `skyrl_train/inference_engines/sglang/sglang_engine.py:256-272`
- Remote: `skyrl_train/inference_engines/remote_inference_engine.py:161-182`

---

## Step 2: Weight Gathering & Preparation

### Purpose
Extract weights from sharded models and prepare them for transfer, handling backend-specific sharding differences.

### Key Functions
- **Training**: `trainer.py: sync_policy_weights_to_inference_engines()` → `worker.py: broadcast_to_inference_engines()` (overridden by each backend)
- **FSDP**: `fsdp_worker.py: broadcast_to_inference_engines()` → `model.state_dict()` → `param.full_tensor()` (if DTensor)
- **Megatron**: `megatron_worker.py: broadcast_to_inference_engines()` → `bridge.export_hf_weights()` → packing logic
- **DeepSpeed**: `deepspeed_worker.py: broadcast_to_inference_engines()` → `model.named_parameters()` → `GatheredParameters()` (ZeRO-3)

### Common Setup (All Backends)
1. Prefix cache reset (if enabled): async task to reset prefix cache
   - Code: `skyrl_train/workers/fsdp/fsdp_worker.py:141-143`
   - Code: `skyrl_train/workers/megatron/megatron_worker.py:463-465`
   - Code: `skyrl_train/workers/deepspeed/deepspeed_worker.py:124-126`
2. CUDA cache clearing: `torch.cuda.empty_cache()`
   - Code: `skyrl_train/workers/fsdp/fsdp_worker.py:145`
   - Code: `skyrl_train/workers/megatron/megatron_worker.py:467`
   - Code: `skyrl_train/workers/deepspeed/deepspeed_worker.py:128`
3. Dtype conversion: convert to `generator.model_dtype`
   - Code: `skyrl_train/workers/fsdp/fsdp_worker.py:139`
   - Code: `skyrl_train/workers/megatron/megatron_worker.py:461`
   - Code: `skyrl_train/workers/deepspeed/deepspeed_worker.py:122`
4. LoRA check: if LoRA, handle separately (FSDP only)
   - Code: `skyrl_train/workers/fsdp/fsdp_worker.py:156-162`

### Weight Extraction Methods

**FSDP:**
- Uses `state_dict()` with `SHARDED_STATE_DICT` config
  - Function: `FSDP.set_state_dict_type()`, `model.model.state_dict()`
  - Code: `skyrl_train/workers/fsdp/fsdp_worker.py:146-151` (FSDP v1 setup)
  - Code: `skyrl_train/workers/fsdp/fsdp_worker.py:165` (state_dict call)
- Handles `DTensor` via `.full_tensor()` to gather shards
  - Function: `param.full_tensor()` (if isinstance(param, DTensor))
  - Code: `skyrl_train/workers/fsdp/fsdp_worker.py:186` (broadcast path)
  - Code: `skyrl_train/workers/fsdp/fsdp_worker.py:223` (IPC path)

**Megatron:**
- Uses `bridge.export_hf_weights()` to convert Megatron format → HuggingFace format
  - Function: `bridge.export_hf_weights(actor_module, ...)`
  - Code: `skyrl_train/workers/megatron/megatron_worker.py:474` (non-IPC)
  - Code: `skyrl_train/workers/megatron/megatron_worker.py:503-507` (IPC with buckets)
- Returns generator of (name, tensor) pairs
  - Code: `skyrl_train/workers/megatron/megatron_worker.py:475`
- Conversion handles TP/PP sharding internally
  - Code: `skyrl_train/workers/megatron/megatron_worker.py:474` (bridge handles conversion)

**DeepSpeed:**
- Uses `model.named_parameters()` directly
  - Function: `model.named_parameters()`
  - Code: `skyrl_train/workers/deepspeed/deepspeed_worker.py:131`
- For ZeRO-3: uses `GatheredParameters` context manager to allgather
  - Function: `deepspeed.zero.GatheredParameters([param], enabled=self.zero_stage == 3)`
  - Code: `skyrl_train/workers/deepspeed/deepspeed_worker.py:148` (broadcast)
  - Code: `skyrl_train/workers/deepspeed/deepspeed_worker.py:183` (IPC)
- Shape handling: uses `param.ds_shape` for ZeRO-3
  - Code: `skyrl_train/workers/deepspeed/deepspeed_worker.py:133`
  - Code: `skyrl_train/workers/deepspeed/deepspeed_worker.py:197`

### CUDA IPC Preparation

**FSDP/DeepSpeed:**
- Groups parameters by module name
  - Code: `skyrl_train/workers/fsdp/fsdp_worker.py:201-208`
  - Code: `skyrl_train/workers/deepspeed/deepspeed_worker.py:164-172`
- Creates one IPC handle per parameter
  - Function: `reduce_tensor(weight)` from `torch.multiprocessing.reductions`
  - Code: `skyrl_train/workers/fsdp/fsdp_worker.py:220-226`
  - Code: `skyrl_train/workers/deepspeed/deepspeed_worker.py:186`
- Batches handles until threshold reached
  - Function: `torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)`
  - Code: `skyrl_train/workers/fsdp/fsdp_worker.py:244-247`
  - Code: `skyrl_train/workers/deepspeed/deepspeed_worker.py:210-213`

**Megatron:**
- Uses pre-computed `param_buckets` from conversion tasks
  - Function: `bridge.get_conversion_tasks()` → `param_buckets` initialization
  - Code: `skyrl_train/workers/megatron/megatron_worker.py:285-329` (bucket initialization)
  - Code: `skyrl_train/workers/megatron/megatron_worker.py:502` (iterate buckets)
- **Packs multiple tensors** into one contiguous buffer
  - Function: `torch.empty(total_size, ...)` → `packed_tensor[offset:offset+size].copy_(tensor.view(-1))`
  - Code: `skyrl_train/workers/megatron/megatron_worker.py:513-530` (packing logic)
- Creates **one IPC handle per bucket** (more efficient)
  - Function: `reduce_tensor(packed_tensor)` (single handle for entire bucket)
  - Code: `skyrl_train/workers/megatron/megatron_worker.py:532` (single IPC handle)
  - Code: `skyrl_train/workers/megatron/megatron_worker.py:541-542` (packed=True)

**Key Insight:** Packing is generic PyTorch code - FSDP/DeepSpeed could adopt the same approach!

### Code References
- FSDP: `skyrl_train/workers/fsdp/fsdp_worker.py:137-273`
- Megatron: `skyrl_train/workers/megatron/megatron_worker.py:457-557`
- DeepSpeed: `skyrl_train/workers/deepspeed/deepspeed_worker.py:120-238`

---

## Step 3: Weight Transfer (Broadcast/IPC)

### Key Functions
- **Training**: `worker.py: broadcast_to_inference_engines()` → `torch.distributed.broadcast()` or `reduce_tensor()` + `all_gather_object()`
- **Inference Client**: `inference_engine_client.py: update_named_weights()` → `_run_on_all_engines()`
- **vLLM**: `vllm_engine.py: update_weights()` or `update_weights_cuda_ipc()` → `model.load_weights()`
- **SGLang**: `sglang_engine.py: update_named_weights()` → `tokenizer_manager.update_weights_from_distributed()` or `update_weights_from_tensor()` → custom loader
- **Remote**: `remote_inference_engine.py: update_named_weights()` → HTTP POST

### Two Transfer Mechanisms

#### Mechanism A: NCCL Broadcast (Standard Path)

**When Used:**
- `use_cuda_ipc == False` (non-colocated or Gloo backend)
- Remote inference engines (CUDA IPC not supported)

**Flow:**

1. **Training Side**: Broadcast via process group
   - Training rank 0 broadcasts: `torch.distributed.broadcast(param.data, 0, group=self._model_update_group)`
     - Function: `torch.distributed.broadcast()`
     - FSDP: `skyrl_train/workers/fsdp/fsdp_worker.py:189-190`
     - Megatron: `skyrl_train/workers/megatron/megatron_worker.py:494`
     - DeepSpeed: `skyrl_train/workers/deepspeed/deepspeed_worker.py:151`
   - All ranks participate (barrier after each param)
     - Function: `torch.distributed.barrier()`
     - FSDP: `skyrl_train/workers/fsdp/fsdp_worker.py:195`
     - Megatron: `skyrl_train/workers/megatron/megatron_worker.py:499`
     - DeepSpeed: `skyrl_train/workers/deepspeed/deepspeed_worker.py:156`

2. **Inference Engine Side**: Receive and load
   - **vLLM**: Creates empty tensor, receives via broadcast, loads weights
     - Function: `vllm_engine.py: update_weights()` → `torch.distributed.broadcast()` → `model.load_weights()`
     - Empty tensor creation: `skyrl_train/inference_engines/vllm/vllm_engine.py:121`
     - Broadcast receive: `skyrl_train/inference_engines/vllm/vllm_engine.py:122`
     - Load weights: `skyrl_train/inference_engines/vllm/vllm_engine.py:125`
   - **SGLang**: Uses SGLang's `update_weights_from_distributed` API
     - Function: `sglang_engine.py: update_named_weights()` → `tokenizer_manager.update_weights_from_distributed()`
     - Request creation: `skyrl_train/inference_engines/sglang/sglang_engine.py:318-320`
     - API call: `skyrl_train/inference_engines/sglang/sglang_engine.py:323`
   - **Remote**: HTTP POST to `/update_weights` or `/update_weights_from_distributed`
     - Function: `remote_inference_engine.py: update_named_weights()` → `aiohttp.ClientSession().post()`
     - Endpoint selection: `skyrl_train/inference_engines/remote_inference_engine.py:196-201`
     - HTTP request: `skyrl_train/inference_engines/remote_inference_engine.py:208-216`

#### Mechanism B: CUDA IPC (Optimized Path)

**When Used:**
- `use_cuda_ipc == True` (colocated + NCCL backend)
- Local inference engines only

**Flow:**

1. **Training Side**: Create IPC handles
   - **FSDP/DeepSpeed**: Per-parameter IPC handles
     - Function: `reduce_tensor(weight)` → `torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)`
     - FSDP: `skyrl_train/workers/fsdp/fsdp_worker.py:220-226`
     - DeepSpeed: `skyrl_train/workers/deepspeed/deepspeed_worker.py:186`
   - **Megatron**: Can pack multiple tensors into one IPC handle
     - Function: Pack into `packed_tensor` → `reduce_tensor(packed_tensor)`
     - Packing: `skyrl_train/workers/megatron/megatron_worker.py:513-530`
     - IPC handle: `skyrl_train/workers/megatron/megatron_worker.py:532`
   - Share handles via `all_gather_object`
     - Function: `torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)`
     - FSDP: `skyrl_train/workers/fsdp/fsdp_worker.py:229-230`
     - Megatron: `skyrl_train/workers/megatron/megatron_worker.py:534-535`
     - DeepSpeed: `skyrl_train/workers/deepspeed/deepspeed_worker.py:189-190`

2. **Training Side**: Send IPC handles via async request
   - Batched by size threshold: `weight_transfer_threshold_cuda_ipc_GB`
     - Function: `inference_engine_client.update_named_weights(weights_update_request)` [async]
     - FSDP: `skyrl_train/workers/fsdp/fsdp_worker.py:244-248`
     - Megatron: `skyrl_train/workers/megatron/megatron_worker.py:544-545`
     - DeepSpeed: `skyrl_train/workers/deepspeed/deepspeed_worker.py:210-214`
   - Request creation: `skyrl_train/workers/fsdp/fsdp_worker.py:172-179`
   - Async send: `skyrl_train/workers/fsdp/fsdp_worker.py:248`

3. **Inference Engine Side**: Open IPC handles and load
   - **vLLM**: Opens handles, reconstructs tensors, loads weights
     - Function: `vllm_engine.py: update_weights_cuda_ipc()` → IPC handle opening → `model.load_weights()`
     - Route to IPC method: `skyrl_train/inference_engines/vllm/vllm_engine.py:395-407`
     - Packed path: `skyrl_train/inference_engines/vllm/vllm_engine.py:140-163`
     - Unpacked path: `skyrl_train/inference_engines/vllm/vllm_engine.py:164-181`
     - Load weights: `skyrl_train/inference_engines/vllm/vllm_engine.py:183`
   - **SGLang**: Serializes request, uses custom weight loader
     - Function: `sglang_engine.py: update_named_weights()` → serialize → `tokenizer_manager.update_weights_from_tensor()` → custom loader
     - Serialization: `skyrl_train/inference_engines/sglang/sglang_engine.py:287-298`
     - Custom loader call: `skyrl_train/inference_engines/sglang/sglang_engine.py:300-311`
     - Loader implementation: `skyrl_train/inference_engines/sglang/sglang_engine.py:108-162`

### Key Differences

**Packed vs Unpacked:**
- **Megatron**: `packed=True` - multiple tensors in one IPC handle
- **FSDP/DeepSpeed**: `packed=False` - one IPC handle per parameter
- **Note**: FSDP/DeepSpeed could adopt packing (it's generic PyTorch code)

**Inference Engine APIs:**
- **vLLM**: Direct PyTorch calls + `collective_rpc`
- **SGLang**: SGLang-specific APIs (no direct per-worker access)
- **Remote**: HTTP endpoints

### Code References
- Training Broadcast: `skyrl_train/workers/fsdp/fsdp_worker.py:183-195`
- Training IPC: `skyrl_train/workers/fsdp/fsdp_worker.py:197-268`
- vLLM Broadcast: `skyrl_train/inference_engines/vllm/vllm_engine.py:115-127`
- vLLM IPC: `skyrl_train/inference_engines/vllm/vllm_engine.py:129-186`
- SGLang: `skyrl_train/inference_engines/sglang/sglang_engine.py:274-326`

---

## Step 4: Weight Loading & Cleanup

### Purpose
Load weights into inference models and clean up resources.

### Key Functions
- **vLLM**: `vllm_engine.py: update_weights()` or `update_weights_cuda_ipc()` → `model_runner.model.load_weights()`
- **SGLang**: Custom loader `update_weights_cuda_ipc(model, named_tensors)` → `model.load_weights()`
- **Cleanup**: `torch.cuda.ipc_collect()`, `torch.distributed.barrier()`, `torch.cuda.synchronize()`, `torch.cuda.empty_cache()`
- **Teardown**: `vllm_engine.py: destroy_weights_update_group()` → `destroy_process_group()`

### Weight Loading

**vLLM:**
- Direct `model.load_weights()` call with `(name, tensor)` pairs
  - Function: `model_runner.model.load_weights(weights=weight_list)`
  - Broadcast path: `skyrl_train/inference_engines/vllm/vllm_engine.py:125`
  - IPC path: `skyrl_train/inference_engines/vllm/vllm_engine.py:183`
- Delete temporary tensors after loading
  - Code: `skyrl_train/inference_engines/vllm/vllm_engine.py:126-127`, `185-186`

**SGLang:**
- Custom loader function called by SGLang's model runner
  - Function: `update_weights_cuda_ipc(model, named_tensors)` → `model.load_weights(weights_to_load)`
  - Loader function: `skyrl_train/inference_engines/sglang/sglang_engine.py:108-162`
  - Called via: `skyrl_train/inference_engines/sglang/sglang_engine.py:311`
- Deserializes request, opens IPC handles, loads weights
  - Function: `pickle.loads()`, IPC handle opening via `func(*list_args)`, `model.load_weights()`
  - Deserialization: `skyrl_train/inference_engines/sglang/sglang_engine.py:123-134`
  - IPC handle opening: `skyrl_train/inference_engines/sglang/sglang_engine.py:153-159`
  - Load weights: `skyrl_train/inference_engines/sglang/sglang_engine.py:162`

### Cleanup Operations

**Training Side:**
1. **CUDA IPC cleanup**: `torch.cuda.ipc_collect()` after each batch/module
   - Function: `torch.cuda.ipc_collect()`
   - FSDP: `skyrl_train/workers/fsdp/fsdp_worker.py:259`, `266`
   - Megatron: `skyrl_train/workers/megatron/megatron_worker.py:549`
   - DeepSpeed: `skyrl_train/workers/deepspeed/deepspeed_worker.py:224`, `232`
2. **Synchronization**: `torch.distributed.barrier()` and `torch.cuda.synchronize()`
   - Functions: `torch.distributed.barrier()`, `torch.cuda.synchronize()`
   - Barriers: `skyrl_train/workers/fsdp/fsdp_worker.py:260`, `267`
   - Synchronize: `skyrl_train/workers/fsdp/fsdp_worker.py:261`, `268`
   - Megatron: `skyrl_train/workers/megatron/megatron_worker.py:551-552`
   - DeepSpeed: `skyrl_train/workers/deepspeed/deepspeed_worker.py:226-227`, `233`
3. **Final cleanup**: `torch.cuda.empty_cache()`, wait for prefix cache reset
   - Functions: `torch.cuda.empty_cache()`, `await cache_reset_task`
   - Prefix cache: `skyrl_train/workers/fsdp/fsdp_worker.py:270-271`
   - Empty cache: `skyrl_train/workers/fsdp/fsdp_worker.py:272`
   - Final barrier: `skyrl_train/workers/fsdp/fsdp_worker.py:273`

**Inference Side:**
1. Delete temporary weight tensors after loading
   - Function: `del weight` (implicit cleanup)
   - vLLM: `skyrl_train/inference_engines/vllm/vllm_engine.py:126-127`, `185-186`
2. Process group cleanup: `destroy_weights_update_group()` (vLLM only)
   - Function: `vllm_engine.py: destroy_weights_update_group()` → `destroy_process_group()`
   - Code: `skyrl_train/inference_engines/vllm/vllm_engine.py:189-193`
   - Called during teardown: `skyrl_train/inference_engines/vllm/vllm_engine.py:416-424`

### Code References
- Training Cleanup:
  - FSDP: `skyrl_train/workers/fsdp/fsdp_worker.py:259-273`
  - Megatron: `skyrl_train/workers/megatron/megatron_worker.py:548-557`
  - DeepSpeed: `skyrl_train/workers/deepspeed/deepspeed_worker.py:224-238`
- vLLM Loading:
  - Broadcast: `skyrl_train/inference_engines/vllm/vllm_engine.py:115-127`
  - IPC: `skyrl_train/inference_engines/vllm/vllm_engine.py:129-186`
- SGLang Loading: `skyrl_train/inference_engines/sglang/sglang_engine.py:108-162`

---

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Initialization                                         │
│ ─────────────────────────────────────────────────────────────── │
│ Training Rank 0:                                               │
│   1. Get master_addr, master_port                              │
│   2. Calculate world_size                                       │
│   3. Call inference_engine_client.init_weight_update_...()     │
│   4. Call init_custom_process_group() → _model_update_group     │
│                                                                  │
│ Inference Engines:                                              │
│   1. Receive init params                                        │
│   2. Join process group (vLLM: direct, SGLang: API, Remote:HTTP)│
│   3. Store _model_update_group                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: Weight Gathering & Preparation                          │
│ ─────────────────────────────────────────────────────────────── │
│ All Backends:                                                   │
│   1. Reset prefix cache (if enabled)                           │
│   2. torch.cuda.empty_cache()                                   │
│   3. Convert to generator.model_dtype                           │
│                                                                  │
│ Weight Extraction:                                              │
│   • FSDP: state_dict() → handle DTensor                         │
│   • Megatron: bridge.export_hf_weights() → format conversion    │
│   • DeepSpeed: named_parameters() → ZeRO-3 gathering           │
│                                                                  │
│ CUDA IPC Prep:                                                  │
│   • FSDP/DeepSpeed: Group by module, create IPC handles        │
│   • Megatron: Pack tensors into buckets → one IPC handle/bucket │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: Weight Transfer                                         │
│ ─────────────────────────────────────────────────────────────── │
│ Path A: NCCL Broadcast                                          │
│   Training: torch.distributed.broadcast(param, src=0, ...)     │
│   Inference: Receive via broadcast → load_weights()             │
│                                                                  │
│ Path B: CUDA IPC                                                │
│   Training: Create IPC handles → send via async request        │
│   Inference: Open handles → reconstruct tensors → load_weights()│
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: Weight Loading & Cleanup                                │
│ ─────────────────────────────────────────────────────────────── │
│ Loading:                                                        │
│   • vLLM: model.load_weights(weights=[(name, tensor), ...])    │
│   • SGLang: Custom loader function called by SGLang            │
│                                                                  │
│ Cleanup:                                                        │
│   • torch.cuda.ipc_collect() (IPC path)                        │
│   • torch.distributed.barrier()                                 │
│   • torch.cuda.synchronize()                                    │
│   • torch.cuda.empty_cache()                                    │
│   • destroy_weights_update_group() (vLLM teardown)             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Weight Update Request Structures

The `NamedWeightsUpdateRequest` structure varies depending on the transfer mechanism. Here are the exact request formats for each case:

### Type Definition

```python
class NamedWeightsUpdateRequest(TypedDict):
    names: List[str]                    # Weight parameter names
    dtypes: List[str]                   # Data types (e.g., "bfloat16", "float32")
    shapes: List[List[int]]             # Tensor shapes
    sizes: NotRequired[List[int]]       # Element counts (only for packed=True)
    extras: Optional[List[Dict[str, Any]]]  # Additional data (IPC handles, disk paths, etc.)
    packed: NotRequired[bool]           # Whether tensors are packed (Megatron only)
```

### Case 1: Broadcast Path (Non-CUDA IPC)

**Structure:** One parameter per request

```python
{
    "names": ["model.layers.0.self_attn.q_proj.weight"],
    "dtypes": ["bfloat16"],
    "shapes": [[4096, 4096]]
    # No "extras", "sizes", or "packed" fields
}
```

**Code References:**
- FSDP: `skyrl_train/workers/fsdp/fsdp_worker.py:172-179`
- Megatron: `skyrl_train/workers/megatron/megatron_worker.py:477-484`
- DeepSpeed: `skyrl_train/workers/deepspeed/deepspeed_worker.py:135-142`

**Notes:**
- Sent per-parameter (one request per weight)
- No IPC handles needed (weights transferred via broadcast)
- Inference engine receives via broadcast, not from request

### Case 2: CUDA IPC - Unpacked (FSDP/DeepSpeed)

**Structure:** Batched by module/threshold (multiple parameters, one IPC handle each)

```python
{
    "names": [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight"
    ],
    "dtypes": ["bfloat16", "bfloat16", "bfloat16"],
    "shapes": [[4096, 4096], [4096, 4096], [4096, 4096]],
    "extras": [
        {"ipc_handles": {<gpu_uuid>: <handle_func, args>}},
        {"ipc_handles": {<gpu_uuid>: <handle_func, args>}},
        {"ipc_handles": {<gpu_uuid>: <handle_func, args>}}
    ],
    "packed": False
    # No "sizes" field (not packed)
}
```

**IPC Handle Structure:**
- `ipc_handles` is a dict mapping `physical_gpu_id` (UUID string) → IPC handle tuple
- IPC handle tuple: `(func, args)` where `func` is the reconstruction function and `args` contains tensor metadata
- Code: `skyrl_train/workers/fsdp/fsdp_worker.py:228-241`

**Code References:**
- FSDP: `skyrl_train/workers/fsdp/fsdp_worker.py:198-248`
- DeepSpeed: `skyrl_train/workers/deepspeed/deepspeed_worker.py:161-214`

**Notes:**
- Batched until `weight_transfer_threshold_cuda_ipc_GB` is reached
- Each parameter has its own IPC handle in `extras[i]["ipc_handles"]`
- `packed=False` indicates each tensor has separate IPC handle

### Case 3: CUDA IPC - Packed (Megatron)

**Structure:** Multiple parameters packed into one contiguous buffer (one IPC handle for all)

```python
{
    "names": [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight"
    ],
    "dtypes": ["bfloat16", "bfloat16", "bfloat16", "bfloat16"],
    "shapes": [[4096, 4096], [4096, 4096], [4096, 4096], [4096, 4096]],
    "sizes": [16777216, 16777216, 16777216, 16777216],  # numel() for each tensor
    "extras": [
        {"ipc_handles": {<gpu_uuid>: <handle_func, args>}}  # Single handle for packed_tensor
    ],
    "packed": True
}
```

**IPC Handle Structure:**
- **Single IPC handle** in `extras[0]["ipc_handles"]` for the entire `packed_tensor`
- `sizes` array tracks offset boundaries: `packed_tensor[offset:offset+size]` for each weight
- Code: `skyrl_train/workers/megatron/megatron_worker.py:513-542`

**Code References:**
- Megatron: `skyrl_train/workers/megatron/megatron_worker.py:468-545`

**Notes:**
- `packed=True` indicates all tensors are in one contiguous buffer
- `sizes` required for unpacking: `packed_tensor[offset:offset+sizes[i]].view(*shapes[i])`
- More efficient: one IPC handle instead of N handles

### Case 4: LoRA Disk Loading (Special Case)

**Structure:** Special request indicating disk-based LoRA loading

```python
{
    "names": ["lora_disk_load"],  # Special identifier
    "extras": [{"lora_disk_path": "/tmp/skyrl_lora_sync"}]
    # No "dtypes", "shapes", "sizes", or "packed" fields
}
```

**Code References:**
- FSDP LoRA: `skyrl_train/workers/fsdp/fsdp_worker.py:129-133`
- vLLM Detection: `skyrl_train/inference_engines/vllm/vllm_engine.py:285-292`

**Notes:**
- Special `names[0] == "lora_disk_load"` identifier
- `lora_disk_path` points to directory containing:
  - `adapter_model.safetensors` - LoRA weights
  - `adapter_config.json` - LoRA configuration
- Inference engine loads from disk using vLLM's `add_lora()` API

### Request Flow Summary

| Case | Transfer Method | Request Frequency | IPC Handles | Packed |
|------|----------------|------------------|-------------|--------|
| **Broadcast** | NCCL broadcast | Per parameter | None | N/A |
| **CUDA IPC (FSDP/DS)** | IPC handles | Batched by threshold | One per parameter | `False` |
| **CUDA IPC (Megatron)** | IPC handles | Per bucket | One per bucket | `True` |
| **LoRA** | Disk path | Once per sync | None | N/A |

### Key Differences

1. **Broadcast**: No `extras` field - weights transferred via process group broadcast
2. **CUDA IPC Unpacked**: `extras[i]["ipc_handles"]` - one handle per parameter, `packed=False`
3. **CUDA IPC Packed**: `extras[0]["ipc_handles"]` - one handle for all, `packed=True`, includes `sizes`
4. **LoRA**: Special `names` identifier, `lora_disk_path` in extras, no weight data in request

---

## Key Insights for Abstraction

### Common Patterns
1. **Process group setup**: All backends use same initialization flow
2. **Weight extraction**: Different methods but same goal (get weights from sharded model)
3. **Transfer mechanism selection**: Based on `use_cuda_ipc` flag
4. **Synchronization**: All use barriers and CUDA synchronization
5. **Cleanup**: Similar cleanup operations across backends

### Differences to Abstract
1. **Weight extraction**: Backend-specific (FSDP vs Megatron vs DeepSpeed)
2. **Sharding handling**: `.full_tensor()`, `GatheredParameters`, or conversion tasks
3. **IPC packing**: Megatron packs, FSDP/DeepSpeed don't (but could!)
4. **Inference engine APIs**: Direct calls (vLLM) vs API abstraction (SGLang) vs HTTP (Remote)
5. **Weight loading**: Direct call (vLLM) vs custom loader (SGLang)

### Abstraction Opportunities
1. **Weight extraction interface**: `get_weights_for_sync()` method per backend
2. **Transfer mechanism**: Strategy pattern (BroadcastStrategy vs IPCStrategy)
3. **Packing logic**: Generic PyTorch code - can be shared across backends
4. **Inference engine interface**: Common `load_weights()` abstraction
5. **Cleanup operations**: Standardized cleanup sequence

---

## Configuration

**Key Config Parameters:**
- `generator.weight_sync_backend`: "nccl" or "gloo"
- `generator.weight_transfer_threshold_cuda_ipc_GB`: Batch size threshold for CUDA IPC
- `trainer.placement.colocate_all`: Enables CUDA IPC optimizations
- `generator.override_existing_update_group`: Whether to override existing process groups

---

## Performance Considerations

1. **CUDA IPC vs Broadcast**: IPC is faster for colocated setups (zero-copy GPU memory)
2. **Packing**: Reduces IPC handle overhead (Megatron advantage)
3. **Batching**: Groups weights by module/threshold to reduce communication overhead
4. **Memory efficiency**: `torch.cuda.ipc_collect()` and `empty_cache()` prevent memory leaks

---

## LoRA Weight Sync (Special Case)

### Overview

LoRA (Low-Rank Adaptation) weight synchronization follows a **different path** than full model weight sync. Currently, **only FSDP (training) + vLLM (inference) support LoRA**.

### Key Differences from Full Model Sync

| Aspect | Full Model Weights | LoRA Adapters |
|--------|-------------------|---------------|
| **Transfer Method** | Broadcast or CUDA IPC | Disk-based (shared filesystem) |
| **Training Backends** | FSDP, Megatron, DeepSpeed | FSDP only |
| **Inference Engines** | vLLM, SGLang, Remote | vLLM only |
| **Format** | Raw tensors | PEFT format (safetensors + config JSON) |
| **Size** | Full model (~GBs) | Small adapters (<1% of model) |

### Why Only FSDP + vLLM?

**Training Side (FSDP):**
- Uses `HFModelWrapper` with LoRA parameters (`lora_rank`, `lora_alpha`, `lora_dropout`)
- Has `collect_lora_params()` utility that works with FSDP-wrapped PEFT models
- Code: `skyrl_train/workers/fsdp/fsdp_worker.py:55`, `72-74`
- Uses `FSDP.summon_full_params()` to extract LoRA weights
- Code: `skyrl_train/distributed/fsdp_utils.py:532-549`

**Why Not Megatron/DeepSpeed?**
- **Megatron**: Uses `AutoBridge.from_hf_pretrained()` (different model format), would need format conversion
- **DeepSpeed**: Uses `HFModelWrapper` but lacks LoRA collection logic (would need DeepSpeed-specific implementation)

**Inference Side (vLLM):**
- Has native LoRA support via `add_lora()` API
- Code: `skyrl_train/inference_engines/vllm/vllm_engine.py:374-379`
- Can load LoRA adapters from disk using `LoRARequest`

**Why Not SGLang?**
- SGLang does not have LoRA support in its API

### Why Disk Transfer Instead of Broadcast/IPC?

**1. vLLM API Constraint**
- vLLM's `add_lora()` API requires a **file path**, not raw tensors
- Function: `LoRARequest(lora_name=..., lora_int_id=..., lora_path=lora_path)`
- Code: `skyrl_train/inference_engines/vllm/vllm_engine.py:377`

**2. Standard PEFT Format**
- LoRA adapters must be in standard PEFT format:
  - `adapter_model.safetensors` - weights in safetensors format
  - `adapter_config.json` - LoRA configuration (rank, alpha, target_modules, etc.)
- Code: `skyrl_train/workers/fsdp/fsdp_worker.py:123-125`

**3. Size & Performance**
- LoRA adapters are **much smaller** (<1% of model size)
- Disk I/O overhead is acceptable for small files
- Full model weights benefit more from broadcast/IPC optimization

**4. Simplicity**
- Reuses vLLM's existing, tested LoRA loading infrastructure
- Ensures format compatibility (config + weights)
- Avoids custom tensor transfer logic

### LoRA Sync Flow

```
FSDP Worker: broadcast_to_inference_engines()
  └─> Check if LoRA: self._is_lora (rank > 0)
      └─> _save_lora_adapters_and_sync()
          ├─> collect_lora_params(module=self.model.model)
          │   └─> Uses FSDP.summon_full_params() + get_peft_model_state_dict()
          ├─> Save to disk:
          │   ├─> save_file(lora_params, "adapter_model.safetensors")
          │   └─> json.dump(peft_config, "adapter_config.json")
          └─> Send disk path to inference engine:
              └─> inference_engine_client.update_named_weights({
                  "names": ["lora_disk_load"],
                  "extras": [{"lora_disk_path": lora_sync_path}]
              })

vLLM Engine: update_named_weights()
  └─> Check if LoRA disk loading request: _is_lora_disk_loading_request()
      └─> _load_lora_from_disk(lora_path)
          └─> llm.llm_engine.add_lora(LoRARequest(lora_path=lora_path))
              └─> [vLLM loads from disk]
```

### Key Functions

**Training Side:**
- `fsdp_worker.py: _save_lora_adapters_and_sync()` - Collects and saves LoRA adapters
  - Code: `skyrl_train/workers/fsdp/fsdp_worker.py:104-135`
- `fsdp_utils.py: collect_lora_params()` - Extracts LoRA params from FSDP-wrapped model
  - Code: `skyrl_train/distributed/fsdp_utils.py:532-549`

**Inference Side:**
- `vllm_engine.py: _load_lora_from_disk()` - Loads LoRA from disk using vLLM API
  - Code: `skyrl_train/inference_engines/vllm/vllm_engine.py:374-379`
- `vllm_engine.py: _is_lora_disk_loading_request()` - Detects LoRA disk loading requests
  - Code: `skyrl_train/inference_engines/vllm/vllm_engine.py:285-292`

### Configuration

**Required Config:**
- `trainer.policy.model.lora.rank` - LoRA rank (set to 0 to disable)
- `trainer.policy.model.lora.alpha` - LoRA alpha scaling factor
- `trainer.policy.model.lora.dropout` - LoRA dropout rate
- `trainer.policy.model.lora.lora_sync_path` - **Shared filesystem path** for adapter sync
  - Must be accessible to all training and inference workers
  - Example: `"/tmp/skyrl_lora_sync"` or shared NFS path

### Code References

- FSDP LoRA Detection: `skyrl_train/workers/fsdp/fsdp_worker.py:55`, `156-162`
- LoRA Collection & Save: `skyrl_train/workers/fsdp/fsdp_worker.py:104-135`
- LoRA Params Collection Utility: `skyrl_train/distributed/fsdp_utils.py:532-549`
- vLLM LoRA Loading: `skyrl_train/inference_engines/vllm/vllm_engine.py:374-391`
- LoRA Request Detection: `skyrl_train/inference_engines/vllm/vllm_engine.py:285-292`

### Notes

- **Early Return**: When LoRA is detected, `broadcast_to_inference_engines()` returns early after syncing LoRA adapters (assumes base model already synced)
- **Shared Filesystem Required**: `lora_sync_path` must be accessible to all workers (training + inference)
- **Format Compatibility**: Uses standard PEFT format for compatibility with vLLM's loader
- **Future Work**: Could potentially add LoRA support for Megatron/DeepSpeed/SGLang, but would require:
  - Format conversion (Megatron)
  - Collection logic implementation (DeepSpeed)
  - Inference engine API support (SGLang)

---

## References

### Training Entry Points
- Main sync call: `skyrl_train/trainer.py:262-271` (during training loop)
- Initial sync: `skyrl_train/trainer.py:130-147` (before training starts)
- Sync method: `skyrl_train/trainer.py:869-872`
- Init method: `skyrl_train/trainer.py:492-501`

### Worker Implementations
- Base Worker: `skyrl_train/workers/worker.py:256-311` (init_weight_sync_state)
- FSDP Worker:
  - Broadcast: `skyrl_train/workers/fsdp/fsdp_worker.py:137-195`
  - IPC: `skyrl_train/workers/fsdp/fsdp_worker.py:197-273`
- Megatron Worker:
  - Bucket initialization: `skyrl_train/workers/megatron/megatron_worker.py:282-329`
  - Broadcast: `skyrl_train/workers/megatron/megatron_worker.py:472-499`
  - IPC: `skyrl_train/workers/megatron/megatron_worker.py:500-557`
- DeepSpeed Worker:
  - Broadcast: `skyrl_train/workers/deepspeed/deepspeed_worker.py:120-156`
  - IPC: `skyrl_train/workers/deepspeed/deepspeed_worker.py:158-238`

### Inference Engine Implementations
- vLLM Engine:
  - Init: `skyrl_train/inference_engines/vllm/vllm_engine.py:88-113`
  - Broadcast receive: `skyrl_train/inference_engines/vllm/vllm_engine.py:115-127`
  - IPC receive: `skyrl_train/inference_engines/vllm/vllm_engine.py:129-186`
  - API routing: `skyrl_train/inference_engines/vllm/vllm_engine.py:381-414`
- SGLang Engine:
  - Init: `skyrl_train/inference_engines/sglang/sglang_engine.py:256-272`
  - Custom loader: `skyrl_train/inference_engines/sglang/sglang_engine.py:108-162`
  - Broadcast: `skyrl_train/inference_engines/sglang/sglang_engine.py:313-326`
  - IPC: `skyrl_train/inference_engines/sglang/sglang_engine.py:274-312`
- Remote Engine:
  - Init: `skyrl_train/inference_engines/remote_inference_engine.py:161-182`
  - Update weights: `skyrl_train/inference_engines/remote_inference_engine.py:184-216`

### Supporting Code
- Inference Engine Client: `skyrl_train/inference_engines/inference_engine_client.py:355-384`
- Distributed Utils: `skyrl_train/distributed/utils.py:47-96` (init_custom_process_group)
- Weights Manager: `skyrl_train/weights_manager.py:31-121` (legacy wrapper)
