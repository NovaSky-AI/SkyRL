# Weight Sync Interface Design

## 1. Overview
### Goals
- Support every existing transfer path (broadcast, CUDA IPC, packed CUDA IPC) without duplicating backend logic.
- Make it easy to extend to more transfer strategies, e.g., Ray Direct Transport.
- Cleanly separate controller responsibilities from training/inference Ray actors.
- Provide interfaces that are easy to review, implement, and test incrementally.

### High-Level Picture
1. **Controller (trainer.py)** holds an inexpensive `WeightTransferStrategy` factory. It never touches heavy resources; it just tells workers how to build senders/receivers.
2. **Training actors** own extracting weights from the training backend + the “send” half of a transfer strategy.
3. **Inference actors** own the "receive half of the transfer strategy + loading weights in the inference engine.
4C. **Shared data structures** (`WeightChunk`, `WeightUpdateRequest`, `WeightSyncInitInfo`) keep metadata flowing between the pieces.

## 2. Component Responsibilities
| Component | Lives Where | Responsibilities |
|-----------|-------------|------------------|
| **WeightExtractor** | Training actors (backend-specific) | Gather backend weights, remove sharding, convert dtype/device, perform optional module grouping & batching, yield `WeightChunk`s. |
| **WeightTransferSender** | Training actor rank 0 (strategy-specific) | Consume `WeightChunk`s, create `WeightUpdateRequest`s, execute broadcast/IPC sends, notify inference via Ray RPC. |
| **WeightTransferReceiver** | Inference actors (strategy-specific) | Provide pull-based iteration (e.g., `receive_weights()`) that loaders use to fetch tensors from broadcast/IPC sources. |
| **WeightLoader** | Inference actors (engine-specific) | Drive the receive+load workflow by invoking the receiver (e.g., `load_weights(receiver)`), then applying tensors to the inference model. |
| **WeightChunk** | Shared | Container describing a group of parameters (names, dtypes, shapes, tensors, metadata) plus total size for batching. |
| **WeightUpdateRequest** | Shared (strategy-specific) | Transport schema for senders/receivers (broadcast metadata vs. IPC handles vs. packed handles). |
| **WeightSyncInitInfo** | Shared | Info emitted by training actor rank 0 that inference actors must know during initialization (e.g., process-group endpoints). |
| **WeightTransferStrategy** | Controller | Factory that ties everything together: creates senders/receivers, produces `WeightSyncInitInfo`, exposes request type. |

## 3. How Components Work Together

```
Initialization Flow
───────────────────
Controller
    │ Ray RPC: initialize_weight_sync(...)
    ▼
Training actors
    │ strategy.create_init_info()  (one actor returns the init info)
    │ strategy.create_sender()
    │→ WeightSyncInitInfo
          │
          ▼
Controller (fan-out)
    │ send init info via Ray RPC
    ▼
Inference actors
    └─ strategy.create_receiver()
```

1. **Initialization**
   - Controller calls `training_actor.initialize_weight_sync(strategy, cfg, inference_client)` on every training actor.
   - Training actors call `strategy.create_init_info()` / `strategy.create_sender()`, producing a `WeightSyncInitInfo` that the controller forwards to inference actors, which in turn call `strategy.create_receiver()`.

```
Sync Flow
─────────
Training actors (all ranks)
    └─ WeightExtractor.extract_weights()  <-- invoked on every training actor
        (helper ranks join collectives)

Training actor rank 0
    └─ WeightTransferSender.send_chunks()
          ├─ Broadcast: torch.distributed.broadcast()
          └─ CUDA IPC: reduce_tensor() + all_gather_object()

Inference actors
    └─ WeightLoader.load_weights(receiver, request)
          └─ pulls tensors via WeightTransferReceiver.receive_weights()
```

2. **Synchronization Loop**
   - All training actors invoke `WeightExtractor.extract_weights()` so backend collectives (e.g., `all_gather_object`) engage every rank. The yielded `WeightChunk`s feed directly into `WeightTransferSender.send_chunks()`.
   - Sender (`send_chunks()` / `create_request()`) iterates the chunks, builds `WeightUpdateRequest`s, and either broadcasts tensors via `_model_update_group` or creates CUDA IPC handles (using `reduce_tensor()` + `all_gather_object`) and ships metadata through Ray RPC.
   - Inference actors receive request dicts via `InferenceEngineClient.update_named_weights()`. The loader (`load_weights(receiver, request)`) now drives the receive+load path by pulling tensors from `WeightTransferReceiver.receive_weights()`.
3. **Clean Separation**
   - Backend quirks (state_dict vs. named_parameters, ZeRO gather, Megatron packing) stay inside the extractor.
   - Transfer mechanics (process groups vs. IPC) stay inside sender/receiver pairs.
   - Engine quirks (vLLM, SGLang, Remote) stay in loaders.

## 4. Detailed Interfaces
### 4.1 WeightExtractor
```python
class WeightExtractor(ABC):
    """Extracts weights from training backend models.

    Subclasses implement backend-specific logic to extract model weights,
    handle sharding, and prepare them for transfer to inference engines.
    """

    @abstractmethod
    def extract_weights(self, dtype: torch.dtype) -> Iterator[WeightChunk]:
        """Extract weights from the model as WeightChunk objects.

        Implementations should:
        - Gather sharded weights into full tensors
        - Convert tensors to the specified dtype for inference
        - Ensure tensors are contiguous in memory
        - Optionally group related parameters (e.g., QKV for efficiency)

        Args:
            dtype: Target dtype for inference (e.g., torch.bfloat16, torch.float16)

        Yields:
            WeightChunk objects containing model parameters ready for transfer
        """
        ...
```
**Responsibilities**
- Gather backend weights, undo sharding, convert to inference dtype/device
- Ensure contiguous tensor layout
- Optionally perform module grouping or size-based batching
- Yield WeightChunk instances ready for transfer

**Implementations**
- `FSDPWeightExtractor` – FSDP-sharded models with optional module grouping
- `MegatronWeightExtractor` – Megatron model-parallel models with optional bucketing
- `DeepSpeedWeightExtractor` – DeepSpeed ZeRO-sharded models with ZeRO-3 gathering support

### 4.2 WeightTransferSender
```python
class WeightTransferSender(ABC):
    """Strategy-specific component that sends WeightChunk data to inference actors."""
    async def send_chunks(
        self,
        chunks: Iterator[WeightChunk],
        inference_client: InferenceEngineClient,
    ) -> None:
        """Iterate WeightChunk objects, execute the transfer primitive, notify inference actors."""
        ...

    def create_request(self, chunk: WeightChunk) -> WeightUpdateRequest:
        """Convert a WeightChunk into this strategy's WeightUpdateRequest."""
        ...
```
**Responsibilities**
- Consume the `WeightChunk` iterator exactly once per sync.
- Convert each chunk into the strategy’s `WeightUpdateRequest` type.
- Execute the transfer primitive (broadcast or CUDA IPC) and coordinate process groups / IPC handles.
- Call `InferenceEngineClient.update_named_weights()` with the serialized request data.

**Typical subclasses**
- `BroadcastTransferSender` – gathers tensors (if needed) and issues `torch.distributed.broadcast()` on the `_model_update_group`.
- `CudaIpcTransferSender` – creates per-tensor IPC handles via `reduce_tensor()`, gathers metadata via `all_gather_object`, sends handle info over Ray RPC.
- `PackedCudaIpcTransferSender` – packs multiple tensors into one contiguous buffer before creating IPC handles (Megatron-style).

### 4.3 WeightTransferReceiver
```python
class WeightTransferReceiver(ABC):
    """Strategy-specific component that streams WeightChunk data to loaders."""
    async def receive_weights(
        self, request: WeightUpdateRequest
    ) -> AsyncIterator[Tuple[str, torch.Tensor]]:
        """Yield `(name, tensor)` tuples by pulling data from broadcast/IPC channels."""
        ...
```
**Responsibilities**
- Provide an iterator-like `receive_weights()` entry point that loaders can pull from (broadcast receive or CUDA IPC handle open/reconstruct).
- Hide transport details (process groups, IPC handles) from the loader.

**Typical subclasses**
- `BroadcastTransferReceiver` – allocates tensors locally and receives via `torch.distributed.broadcast()` before yielding `(name, tensor)` pairs.
- `CudaIpcTransferReceiver` – opens IPC handles via `from_handle`, maps them to device tensors, yields them lazily.
- `PackedCudaIpcTransferReceiver` – opens a single packed IPC buffer and slices it into individual tensors before yielding.

### 4.4 WeightLoader
```python
class WeightLoader(ABC):
    """Engine-specific loader that drives the receive + load workflow."""
    async def load_weights(
        self, receiver: WeightTransferReceiver, request: WeightUpdateRequest
    ) -> None:
        """Drive the receive+load workflow by pulling tensors from the receiver and applying them."""
        ...
```
**Responsibilities**
- Drive the loading workflow (e.g., `load_weights(receiver, request)`), letting engine sub-processes pull tensors from the receiver.
- Implement the engine-specific mechanics for swapping weights (e.g., `model.load_weights`, tokenizer manager APIs, HTTP uploads).

**Typical subclasses**
- `VLLMWeightLoader` – cooperates with vLLM `WorkerWrap` subprocesses to apply tensors.
- `SGLangWeightLoader` – streams tensors into SGLang tokenizer/engine APIs.

### 4.5 WeightChunk
```python
@dataclass
class WeightChunk:
    names: List[str]
    dtypes: List[str]  # String representation (e.g., "torch.bfloat16")
    shapes: List[List[int]]
    tensors: List[torch.Tensor]
    module_name: Optional[str] = None

    @cached_property
    def total_numel(self) -> int:
        """Calculate total number of elements across all tensors."""
        return sum(t.numel() for t in self.tensors)

    @cached_property
    def total_size_bytes(self) -> int:
        """Calculate total memory footprint in bytes."""
        return sum(t.numel() * t.element_size() for t in self.tensors)
```
**Responsibilities**
- Compact representation of grouped parameters and associated metadata.
- Track total element count and byte size for batching heuristics (cached properties, auto-calculated from tensors).

### 4.6 WeightUpdateRequest
```python
class WeightUpdateRequest(ABC): ...

@dataclass
class BroadcastWeightUpdateRequest(WeightUpdateRequest): ...
@dataclass
class CudaIpcWeightUpdateRequest(WeightUpdateRequest): ...
@dataclass
class PackedCudaIpcWeightUpdateRequest(WeightUpdateRequest): ...
```
**Responsibilities**
- Provide typed schemas for each transfer strategy.
- Carry all metadata needed to reconstruct tensors (names/dtypes/shapes plus IPC handles and sizes).
- Remain trivially serializable (strings/lists/dicts) for Ray RPC.

**Typical subclasses**
- `BroadcastWeightUpdateRequest` – names/dtypes/shapes only (broadcast path).
- `CudaIpcWeightUpdateRequest` – metadata plus per-tensor IPC handles.
- `PackedCudaIpcWeightUpdateRequest` – metadata plus sizes and a single handle map for packed tensors.

### 4.7 WeightSyncInitInfo
```python
class WeightSyncInitInfo(ABC):
    """Initialization-only data shared from the training actor rank 0 to inference actors."""
@dataclass
class TorchDistributedWeightSyncInitInfo(WeightSyncInitInfo): ...
@dataclass
class EmptyWeightSyncInitInfo(WeightSyncInitInfo): ...
```
**Responsibilities**
- Encapsulate initialization-only data produced by training actor rank 0 that inference actors must know (e.g., process-group endpoints).
- Keep strategy setup data explicit and typed.

**Typical subclasses**
- `TorchDistributedWeightSyncInitInfo` – master addr/port/world size/group name/backend for broadcast strategies.
- `EmptyWeightSyncInitInfo` – placeholder when no extra init data is required (e.g., CUDA IPC).

### 4.8 WeightTransferStrategy
```python
class WeightTransferStrategy(ABC):
    """Controller-side factory for building transfer senders/receivers."""
    async def create_sender(..., sync_info: WeightSyncInitInfo) -> WeightTransferSender:
        """Build and initialize a training-side sender."""
        ...

    async def create_receiver(..., sync_info: WeightSyncInitInfo, rank_offset: int) -> WeightTransferReceiver:
        """Build and initialize an inference-side receiver."""
        ...

    async def create_init_info(... ) -> WeightSyncInitInfo:
        """Produce initialization info (e.g., process-group endpoints) on training actors."""
        ...

    def get_request_type(self) -> Type[WeightUpdateRequest]:
        """Return the WeightUpdateRequest implementation this strategy uses."""
        ...
```
**Responsibilities**
- Choose the correct sender/receiver/request/sync-info implementations based on config.
- Produce `WeightSyncInitInfo` on training actors and deliver it to inference actors.
- Encapsulate any strategy-specific initialization (process group wiring, CUDA contexts) within `create_sender/receiver`.

**Typical subclasses**
- `BroadcastTransferStrategy` – builds broadcast sender/receiver, emits `TorchDistributedWeightSyncInitInfo`.
- `CudaIpcTransferStrategy` – builds CUDA IPC sender/receiver, emits `EmptyWeightSyncInitInfo`.
- `PackedCudaIpcTransferStrategy` – CUDA IPC variant that packs tensors before transfer.

## 5. File Structure

After the migration is complete, the file structure will be:

```
skyrl_train/
├── weight_sync/                         # Minimal abstraction module
│   ├── __init__.py                     # Public API exports
│   ├── base.py                         # Data structures (WeightChunk, WeightUpdateRequest, WeightSyncInitInfo)
│   ├── weight_extractor.py             # WeightExtractor interface
│   ├── weight_loader.py                # WeightLoader interface
│   ├── weight_transfer_strategy.py     # WeightTransferStrategy, WeightTransferSender, WeightTransferReceiver interfaces
│   ├── broadcast_strategy.py           # BroadcastTransferStrategy + sender + receiver
│   ├── cuda_ipc_strategy.py           # CudaIpcTransferStrategy + sender + receiver
│   └── packed_cuda_ipc_strategy.py    # PackedCudaIpcTransferStrategy + sender + receiver
│
├── workers/
│   ├── worker.py                       # Base Worker (modified)
│   │   └── initialize_weight_sync()   # New method
│   ├── fsdp/
│   │   └── fsdp_worker.py             # FSDPWeightExtractor lives here
│   ├── megatron/
│   │   └── megatron_worker.py         # MegatronWeightExtractor lives here
│   └── deepspeed/
│       └── deepspeed_worker.py        # DeepSpeedWeightExtractor lives here
│
├── inference_engines/
│   ├── inference_engine_client.py      # Modified for new coordination
│   ├── vllm/
│   │   └── vllm_engine.py             # VLLMWeightLoader lives here
│   ├── sglang/
│   │   └── sglang_engine.py           # SGLangWeightLoader lives here
│   └── remote_inference_engine.py      # RemoteWeightLoader lives here
│
└── trainer.py                          # Modified controller
    ├── _create_strategy()              # Strategy factory
    └── init_weight_sync_state()        # Updated coordination
```

## 6. Key Decisions
- **Strategy factory on controller**
  - *Alternative:* Controller initializes send/receive resources directly or a global manager runs on training actors.
  - *Reason:* Process groups and CUDA contexts must be created on the actors that use them. The controller should only describe *how* to create instances; each actor then calls `create_sender/receiver` locally, keeping heavy initialization where it belongs.

- **`WeightChunk` aggregates multiple parameters**
  - *Alternative:* One tensor per request (like naive broadcast today).
  - *Reason:* FlashRL and packed CUDA IPC require related tensors (e.g., q/k/v) to travel together. Grouping also mirrors the existing `NamedWeightsUpdateRequest` lists and allows batching thresholds to work on meaningful units.

- **Grouping & batching handled inside extractor**
  - *Alternative:* Let senders re-batch chunks before transfer.
  - *Reason:* Extractors know the model structure (module boundaries, dtype conversions) and can group intelligently (FlashRL). Senders should remain transfer-focused (broadcast vs. IPC) and operate on ready-made `WeightChunk`s.

- **Loader drives receiver**
  - *Alternative:* Receiver pushes data into loader (loader just accepts named tensors).
  - *Reason:* Some engines (e.g., vLLM) perform the actual weight loading inside worker subprocesses (`WorkerWrap`). Those subprocesses need to pull tensors on demand. Exposing a pull-based `receive_weights()` on the receiver and letting `WeightLoader.load_weights(receiver, request)` drive the process keeps that flow aligned with existing engine threading models.

## 7. Migration Plan

The migration refactors each component layer (extraction → loading → sending → integration) horizontally across all backends. Interfaces are introduced on-demand, and optimizations (grouping/batching/packing) are preserved throughout.

### Overview

**Phase 1: Refactor Weight Extraction**
- Introduce `WeightExtractor` interface and `WeightChunk` dataclass
- Add extractor implementations in each worker backend (FSDP, Megatron, DeepSpeed)
- Preserve module grouping (FlashRL) and batching (Megatron packing)
- External APIs unchanged; still produce `NamedWeightsUpdateRequest`

**Phase 2: Refactor Weight Loading and Receiving**
- Introduce `WeightLoader` interface
- Add loader implementations in each inference engine (vLLM, SGLang, Remote)
- Standardize `receive_weights()` async iterator pattern locally
- Preserve multi-param chunk handling for QKV fusion
- External APIs unchanged; still accept legacy parameters

**Phase 3: Refactor Weight Sending**
- Introduce `WeightTransferSender` interface
- Create strategy files with sender implementations (broadcast, CUDA IPC, packed CUDA IPC)
- Preserve Megatron packing optimization
- Workers use senders internally but still produce legacy requests

**Phase 4: Integrate Strategy Interface**
- Introduce `WeightTransferStrategy`, `WeightTransferReceiver`, typed `WeightUpdateRequest`, `WeightSyncInitInfo`
- Complete strategy implementations (add receivers, factories)
- Update controller orchestration (`trainer.py`)
- Replace legacy `NamedWeightsUpdateRequest` with typed requests
- Remove legacy helper methods and coordination code

See [migration_plan_detailed.md](./migration_plan_detailed.md) for complete implementation guide with code examples and testing requirements.
