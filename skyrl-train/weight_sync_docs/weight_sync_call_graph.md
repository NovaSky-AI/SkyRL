# Weight Sync End-to-End Call Graph (Interface Methods Only)

## Complete Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ CONTROLLER PROCESS (trainer.py)                                             │
│                                                                             │
│  WeightSyncManager                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              │ 1. sync_weights()
                              │    (WeightSyncManager.sync_weights)
                              │
                              │    └─> _trigger_training_sync()
                              │        [Ray RPC to training Ray actor]
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TRAINING RAY ACTOR (Worker, rank 0)                                         │
│                                                                             │
│  WeightExtractor          WeightTransferSender                              │
│                                                                             │
│  2. extract_weights(model, generator_dtype)                                │
│     (WeightExtractor.extract_weights)                                      │
│     │                                                                       │
│     └─> Yields WeightChunk objects (multiple params per chunk)             │
│         Iterator[WeightChunk]                                               │
│         │                                                                   │
│         WeightChunk:                                                        │
│         {                                                                   │
│           names: List[str],                                                 │
│           tensors: List[torch.Tensor],                                      │
│           dtypes: List[torch.dtype],                                        │
│           shapes: List[Tuple[int, ...]],                                    │
│           module_name: Optional[str],                                       │
│           total_size_bytes: Optional[int],                                  │
│           packed: bool                                                      │
│         }                                                                   │
│                                                                             │
│  3. send_weights(weight_chunks, inference_client)                          │
│     (WeightTransferSender.send_weights)                                     │
│     │                                                                       │
│     ├─ For each WeightChunk:                                               │
│     │   │                                                                   │
│     │   ├─ Broadcast Path:                                                 │
│     │   │   ├─ Create WeightUpdateRequest from WeightChunk                 │
│     │   │   ├─ inference_client.update_named_weights(request)              │
│     │   │   │   [Ray RPC to inference Ray actor]                           │
│     │   │   └─ torch.distributed.broadcast(...)                            │
│     │   │       [Process Group]                                             │
│     │   │                                                                   │
│     │   └─ CUDA IPC Path:                                                  │
│     │       ├─ Create IPC handles (reduce_tensor)                          │
│     │       ├─ Gather handles (all_gather_object)                          │
│     │       ├─ Create WeightUpdateRequest with IPC handles                 │
│     │       └─ inference_client.update_named_weights(request)              │
│     │           [Ray RPC to inference Ray actor]                           │
│     │                                                                       │
│     └─ Return                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              │ Ray RPC (update_named_weights.remote)
                              │ Carries WeightUpdateRequest
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ INFERENCE RAY ACTOR (vLLM/SGLang/Remote)                                    │
│                                                                             │
│  WeightTransferReceiver    WeightLoader                                     │
│                                                                             │
│  4. receive_and_load_weights(request, loader)                              │
│     (WeightTransferReceiver.receive_and_load_weights)                       │
│     │                                                                       │
│     ├─ Broadcast Path:                                                      │
│     │   ├─ Receive tensors via torch.distributed.broadcast(...)            │
│     │   │   [Process Group]                                                 │
│     │   └─ loader.load_weights(request)                                    │
│     │                                                                       │
│     └─ CUDA IPC Path:                                                       │
│         ├─ Extract IPC handles from request.extras                         │
│         ├─ Open IPC handles (torch.from_handle)                            │
│         ├─ Reconstruct tensors                                              │
│         └─ loader.load_weights(request)                                    │
│                                                                             │
│  5. load_weights(request)                                                   │
│     (WeightLoader.load_weights)                                             │
│     │                                                                       │
│     └─ Load weights into inference model                                    │
│         (vLLM: model.load_weights(), SGLang: custom loader, etc.)          │
│                                                                             │
│  6. Return                                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              │ [Ray RPC completes]
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ CONTROLLER PROCESS (trainer.py)                                            │
│                                                                             │
│  7. sync_weights() completes                                                │
│     Training loop continues                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Initialization Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ CONTROLLER PROCESS (trainer.py)                                             │
│                                                                             │
│  WeightSyncManager                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              │ 1. initialize()
                              │    (WeightSyncManager.initialize)
                              │
                              ├─> _initialize_training_senders()
                              │    [Ray RPC to training Ray actor]
                              │
                              └─> _initialize_inference_receivers()
                                   [Ray RPC to inference Ray actor]
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TRAINING RAY ACTOR (Worker, rank 0)                                         │
│                                                                             │
│  WeightTransferSender                                                       │
│                                                                             │
│  2. initialize(inference_client, process_group)                              │
│     (WeightTransferSender.initialize)                                      │
│     │                                                                       │
│     └─ Set up process groups, IPC handles, etc.                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ INFERENCE RAY ACTOR (vLLM/SGLang/Remote)                                    │
│                                                                             │
│  WeightTransferReceiver                                                     │
│                                                                             │
│  3. initialize(master_addr, master_port, rank_offset,                      │
│                world_size, group_name, backend)                             │
│     (WeightTransferReceiver.initialize)                                    │
│     │                                                                       │
│     └─ Join process group, set up IPC receiving                            │
│         Returns: dist.ProcessGroup                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Sequence Diagram

```
Controller          Training Actor          Inference Actor
    │                      │                      │
    │──initialize()───────▶│                      │
    │                      │                      │
    │                      │──initialize()───────┼──▶
    │                      │                      │
    │                      │◀──process_group─────│
    │◀─────────────────────│                      │
    │                      │                      │
    │──sync_weights()──────▶│                      │
    │                      │                      │
    │                      │──extract_weights()──▶│
    │                      │◀──WeightChunk───────│
    │                      │                      │
    │                      │──send_weights()──────┼──▶
    │                      │                      │
    │                      │                      │──receive_and_load_weights()──▶
    │                      │                      │
    │                      │                      │──load_weights()──────────────▶
    │                      │                      │◀──────────────────────────────│
    │                      │                      │◀──────────────────────────────│
    │                      │◀─────────────────────│
    │◀─────────────────────│                      │
    │                      │                      │
```

## Data Structures

### WeightChunk (Training Side)
```python
WeightChunk(
    names: List[str],                    # ["layers.0.attention.q_proj.weight", ...]
    tensors: List[torch.Tensor],         # [tensor_q, tensor_k, tensor_v, ...]
    dtypes: List[torch.dtype],           # [float16, float16, float16, ...]
    shapes: List[Tuple[int, ...]],       # [(4096, 4096), (4096, 4096), ...]
    module_name: Optional[str],          # "layers.0.attention"
    total_size_bytes: Optional[int],     # Sum of all tensor sizes
    packed: bool                         # False (or True for Megatron)
)
```

### WeightUpdateRequest (Transfer)
```python
WeightUpdateRequest(
    names: List[str],                    # ["layers.0.attention.q_proj.weight", ...]
    dtypes: List[str],                   # ["float16", "float16", ...]
    shapes: List[List[int]],             # [[4096, 4096], [4096, 4096], ...]
    sizes: Optional[List[int]],          # [size_q, size_k, size_v, ...]
    extras: Optional[List[Dict]],        # [{"ipc_handles": {...}}, ...]
    packed: bool                         # False (or True for Megatron)
)
```

## Key Interface Methods

### WeightSyncManager (Controller)
- `initialize()` - Initialize sync state
- `sync_weights()` - Trigger weight sync

### WeightExtractor (Training Ray Actor)
- `extract_weights(model, generator_dtype)` → `Iterator[WeightChunk]`

### WeightTransferSender (Training Ray Actor)
- `initialize(inference_client, process_group)` - Set up sender side
- `send_weights(weight_chunks, inference_client)` - Send weights

### WeightTransferReceiver (Inference Ray Actor)
- `initialize(master_addr, master_port, ...)` → `ProcessGroup` - Set up receiver side
- `receive_and_load_weights(request, loader)` - Receive and load weights

### WeightLoader (Inference Ray Actor)
- `load_weights(request)` - Load weights into model

## Process Boundaries

1. **Controller → Training Ray Actor**: Ray RPC
   - `WeightSyncManager.sync_weights()` → `WeightExtractor.extract_weights()` + `WeightTransferSender.send_weights()`

2. **Training Ray Actor → Inference Ray Actor**: Ray RPC
   - `WeightTransferSender.send_weights()` → `WeightTransferReceiver.receive_and_load_weights()`
   - Via `InferenceEngineClient.update_named_weights()`

3. **Training Rank 0 ↔ Inference Ranks**: Process Group (Broadcast)
   - `torch.distributed.broadcast()` within `send_weights()` / `receive_and_load_weights()`

4. **Training Ranks ↔ Inference Ranks**: CUDA IPC (CUDA IPC Path)
   - IPC handles created in `send_weights()`, opened in `receive_and_load_weights()`
