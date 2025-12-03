# Weight Sync Interface Design (Review-Friendly)

## 1. Overview
### Goals
- Support every existing transfer path (broadcast, CUDA IPC, packed CUDA IPC) without duplicating backend logic.
- Make it easy to extend to more transfer backends, e.g., Ray Direct Transport.
- Cleanly separate controller responsibilities from training/inference Ray actors.
- Provide interfaces that are easy to review, implement, and test incrementally.

### High-Level Picture
1. **Controller (trainer.py)** holds an inexpensive `WeightTransferStrategy` factory. It never touches heavy resources; it just tells workers how to build senders/receivers.
2. **Training actors** own extraction and the “send” half of a transfer strategy. All actors enter `WeightExtractor.extract_weights()` so collectives run across ranks, but the **training actor rank 0** drives orchestration and streams `WeightChunk`s to the sender.
3. **Inference actors** own the “receive + load” half. They never deal with training-backend quirks—only transfer strategy details and engine-specific loading.
4. **Shared data structures** (`WeightChunk`, `WeightUpdateRequest`, `WeightSyncInitInfo`) keep metadata flowing between the pieces.

## 2. Component Responsibilities (Cheat Sheet)
| Component | Lives Where | Responsibilities |
|-----------|-------------|------------------|
| **WeightExtractor** | Training actors (backend-specific) | Gather backend weights, remove sharding, convert dtype/device, perform optional module grouping & batching, yield `WeightChunk`s. |
| **WeightTransferSender** | Training actor rank 0 (strategy-specific) | Consume `WeightChunk`s, create `WeightUpdateRequest`s, execute broadcast/IPC sends, notify inference via Ray RPC. |
| **WeightTransferReceiver** | Inference actors (strategy-specific) | Parse request dicts, execute receive half of transfer (broadcast or IPC), reconstruct tensors, hand off to loader. |
| **WeightLoader** | Inference actors (engine-specific) | Apply incoming tensors to the inference model (e.g., `model.load_weights`, tokenizer manager APIs, HTTP endpoints). |
| **WeightChunk** | Shared | Container describing a group of parameters (names, dtypes, shapes, tensors, metadata) plus total size for batching. |
| **WeightUpdateRequest** | Shared (strategy-specific) | Transport schema for senders/receivers (broadcast metadata vs. IPC handles vs. packed handles). |
| **WeightSyncInitInfo** | Shared | Info emitted by training actor rank 0 that inference actors must know during initialization (e.g., process-group endpoints). |
| **WeightTransferStrategy** | Controller | Factory that ties everything together: creates senders/receivers, produces `WeightSyncInitInfo`, exposes request type. |

## 3. How Components Work Together
1. **Initialization**
   - Controller calls `training_actor.initialize_weight_sync(strategy, cfg, inference_client)`.
   - Training actor rank 0 asks the strategy for `WeightSyncInitInfo` (e.g., master_addr/port) and a configured sender. Controller forwards the info to inference actors, which call `create_receiver`.
2. **Synchronization Loop**
   - Training actor rank 0 calls `WeightExtractor.extract_weights()` (instantiated once per worker). Extraction gathers tensors, handles dtype/device conversion, optionally groups/batches them, and yields `WeightChunk`s.
   - Sender iterates the chunks, builds `WeightUpdateRequest`s, and either (a) broadcasts tensors via `_model_update_group` or (b) creates CUDA IPC handles, gathers them over the default training PG, and sends metadata via Ray RPC.
   - Inference actors receive request dicts from `InferenceEngineClient.update_named_weights()`, parse them into typed requests, materialize tensors (broadcast receive or `from_handle`), and pass them to the engine-specific loader.
3. **Clean Separation**
   - Backend quirks (state_dict vs. named_parameters, ZeRO gather, Megatron packing) stay inside the extractor.
   - Transfer mechanics (process groups vs. IPC) stay inside sender/receiver pairs.
   - Engine quirks (vLLM, SGLang, Remote) stay in loaders.

## 4. Detailed Interfaces
### 4.1 WeightExtractor
```python
class WeightExtractor(ABC):
    """Backend-specific extractor that produces WeightChunk objects."""
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._group_by_module = False
        self._batch_by_threshold = False

    def _should_group_by_module(self) -> bool: ...
    def _should_batch_by_threshold(self) -> bool: ...

    def _group_by_module(self, params: Dict[str, torch.Tensor]) -> Dict[str, List[str]]: ...
    def _batch_by_threshold(
        self,
        chunks: Iterator[WeightChunk],
        threshold_gb: float,
    ) -> Iterator[List[WeightChunk]]: ...

    @abstractmethod
    async def _extract_raw_weights(
        self, model: Any, generator_dtype: torch.dtype
    ) -> Dict[str, torch.Tensor]: ...

    async def extract_weights(
        self, model: Any, generator_dtype: torch.dtype
    ) -> Iterator[WeightChunk]: ...
```
**Responsibilities**
1. Gather backend weights, undo sharding, convert to inference dtype/device, ensure contiguous layout.
2. Use internal flags to perform module grouping or size-based batching (defaults to `False`; subclasses set them in `__init__`).
3. Yield `WeightChunk` instances ready for transfer (one per iterator step). Senders must not re-batch.
4. Hide backend specifics from the rest of the pipeline.

### 4.2 WeightTransferSender
```python
class WeightTransferSender(ABC):
    """Strategy-specific component that sends WeightChunk data to inference actors."""
    async def send_chunks(
        self,
        chunks: Iterator[WeightChunk],
        inference_client: InferenceEngineClient,
    ) -> None: ...

    def create_request(self, chunk: WeightChunk) -> WeightUpdateRequest: ...
```
**Responsibilities**
- Consume the `WeightChunk` iterator exactly once per sync.
- Convert each chunk into the strategy’s `WeightUpdateRequest` type.
- Execute the transfer primitive (broadcast or CUDA IPC) and coordinate process groups / IPC handles.
- Call `InferenceEngineClient.update_named_weights()` with the serialized request data.

### 4.3 WeightTransferReceiver
```python
class WeightTransferReceiver(ABC):
    """Strategy-specific component that receives WeightChunk data and hands it to loaders."""
    def parse_request(self, request_dict: Dict[str, Any]) -> WeightUpdateRequest: ...
    async def receive_and_load_weights(
        self, request: WeightUpdateRequest, loader: WeightLoader
    ) -> None: ...
```
**Responsibilities**
- Validate and deserialize request dicts into typed requests.
- Perform the receive half of the transfer (broadcast receive or CUDA IPC handle open/reconstruct).
- Produce `named_tensors` for the loader and call it.

### 4.4 WeightLoader
```python
class WeightLoader(ABC):
    """Engine-specific loader that applies named tensors to the inference model."""
    async def load_weights(
        self, named_tensors: Dict[str, torch.Tensor]
    ) -> None: ...
```
**Responsibilities**
- Implement the engine-specific mechanics for swapping weights (e.g., `model.load_weights`, tokenizer manager APIs, HTTP uploads).
- Shield receivers from engine quirks.

### 4.5 WeightChunk
```python
@dataclass
class WeightChunk:
    names: List[str]
    tensors: List[torch.Tensor]
    dtypes: List[torch.dtype]
    shapes: List[Tuple[int, ...]]
    module_name: Optional[str] = None
    total_size_bytes: Optional[int] = None
    packed: bool = False
```
**Responsibilities**
- Compact representation of grouped parameters and associated metadata.
- Track total byte size for batching heuristics.
- Flag whether tensors are already packed contiguously.

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

### 4.8 WeightTransferStrategy
```python
class WeightTransferStrategy(ABC):
    """Controller-side factory for building transfer senders/receivers."""
    async def create_sender(..., sync_info: WeightSyncInitInfo) -> WeightTransferSender: ...
    async def create_receiver(..., sync_info: WeightSyncInitInfo, rank_offset: int) -> WeightTransferReceiver: ...
    async def create_init_info(... ) -> WeightSyncInitInfo: ...
    def get_request_type(self) -> Type[WeightUpdateRequest]: ...
```
- **Responsibilities**
- Choose the correct sender/receiver/request/sync-info implementations based on config.
- Produce `WeightSyncInitInfo` on training actor rank 0 and deliver it to inference actors.
- Encapsulate any strategy-specific initialization (process group wiring, CUDA contexts) within `create_sender/receiver`.

## 5. Interaction Flow
1. **Initialization**
   1. Controller → Training actor rank 0: call `initialize_weight_sync(...)`.
   2. Training actor rank 0 → Strategy: `create_init_info()` and `create_sender()`; helper training actors run whatever backend collectives the extractor issues.
   3. Controller → Inference actors: deliver `sync_info`; inference actors call `create_receiver()`.
2. **Sync Loop**
   1. All training actors invoke `WeightExtractor.extract_weights()`. The implementation executes backend collectives, so non-zero ranks must enter the method to participate; rank 0 drives orchestration and also streams the yielded `WeightChunk`s to the sender.
   2. Sender (running on training actor rank 0) iterates the resulting `WeightChunk`s, builds `WeightUpdateRequest`s, and executes broadcast or IPC logic.
   3. Inference actors receive request dicts via `InferenceEngineClient.update_named_weights()`, parse them, reconstruct tensors, and load them via `WeightLoader`.
3. **Process Group Notes**
   - Broadcast uses `_model_update_group` spanning training actor rank 0 plus all inference ranks so that tensors can be fanned out in one collective call.
   - CUDA IPC handle gathering (e.g., `all_gather_object`) runs **only** on the training process group; inference actors are idle during that phase and simply open the handles later when the metadata arrives via Ray RPC.

## 6. Key Decisions (Why)
- **`WeightChunk` aggregates multiple parameters**
  - *Alternative:* One tensor per request (like naive broadcast today).
  - *Reason:* FlashRL and packed CUDA IPC require related tensors (e.g., q/k/v) to travel together. Grouping also mirrors the existing `NamedWeightsUpdateRequest` lists and allows batching thresholds to work on meaningful units.

- **Strategy-specific `WeightUpdateRequest` dataclasses**
  - *Alternative:* Single dict with optional fields/extras.
  - *Reason:* Typed dataclasses capture exactly what each strategy needs (broadcast metadata vs. IPC handles vs. packed sizes). That improves static checking, prevents missing fields, and keeps serialization straightforward (lists/dicts only).

- **Strategy factory on controller**
  - *Alternative:* Controller initializes send/receive resources directly or a global manager runs on training actors.
  - *Reason:* Process groups and CUDA contexts must be created on the actors that use them. The controller should only describe *how* to create instances; each actor then calls `create_sender/receiver` locally, keeping heavy initialization where it belongs.

- **Grouping & batching handled inside extractor**
  - *Alternative:* Let senders re-batch chunks before transfer.
  - *Reason:* Extractors know the model structure (module boundaries, dtype conversions) and can group intelligently (FlashRL). Senders should remain transfer-focused (broadcast vs. IPC) and operate on ready-made `WeightChunk`s.

- **Internal flags for grouping/batching**
  - *Alternative:* Public API flags on `extract_weights(...)`.
  - *Reason:* Keeps the method signature stable while still letting subclasses react to config. Flags live in the base class (`_group_by_module`, `_batch_by_threshold`), and subclasses override/enable them without exposing extra arguments to callers.

- **Keep dtypes/shapes as strings/lists**
  - *Alternative:* Use `torch.dtype` objects or tuple-of-int shapes.
  - *Reason:* Strings and lists serialize trivially over Ray RPC without custom hooks. Receivers can convert back to `torch.dtype` as needed, but the transport format stays simple and language-agnostic.

## 7. Migration Plan
1. **Introduce interfaces** – land abstract classes and data structures; keep adapters for old code paths.
2. **Implement extractors** – backend-specific `_extract_raw_weights()` plus flag settings for grouping/batching.
3. **Implement senders** – broadcast and CUDA IPC senders that consume `WeightChunk`s.
4. **Implement receivers** – inference engines parse new `WeightUpdateRequest` types.
5. **Implement loaders** – engine-specific load logic extracted into `WeightLoader`s.
6. **Wire strategy** – controller instantiates `WeightTransferStrategy`; actors call factory methods.
7. **Cleanup** – remove legacy paths and `NamedWeightsUpdateRequest` once all backends use the new pipeline.
