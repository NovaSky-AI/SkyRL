# Weight Sync Interface Design

## High-Level Architecture (Distributed)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CONTROLLER PROCESS                               │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │         WeightTransferStrategy (trainer.py)                  │  │
│  │  (Factory for creating senders/receivers)                     │  │
│  │  - create_sender()                                           │  │
│  │  - create_receiver()                                         │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                            │                                        │
│                            │ Config / Ray RPC                      │
│                            ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │         TRAINING RAY ACTORS (Worker)                         │  │
│  │  ┌──────────────┐         ┌──────────────────┐             │  │
│  │  │  Extractor   │────────▶│ TransferSender   │             │  │
│  │  │  (Backend)   │         │  (Created via     │             │  │
│  │  └──────────────┘         │   strategy)      │             │  │
│  │                            └──────────────────┘             │  │
│  │                                    │                         │  │
│  │                                    │ Process Group / IPC     │  │
│  └────────────────────────────────────┼─────────────────────────┘  │
└────────────────────────────────────────┼─────────────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    │  Cross-Process Communication            │
                    │  • Process Groups (NCCL/Gloo)           │
                    │  • CUDA IPC Handles                     │
                    │  • Ray RPC (via InferenceEngineClient)  │
                    └────────────────────┬────────────────────┘
                                         │
┌────────────────────────────────────────┼─────────────────────────────┐
│  ┌────────────────────────────────────┼──────────────────────────┐  │
│  │         INFERENCE RAY ACTORS (vLLM/SGLang)                    │  │
│  │                                    │                          │  │
│  │  ┌──────────────────┐         ┌──────────────┐              │  │
│  │  │TransferReceiver  │────────▶│   Loader     │              │  │
│  │  │ (Created via     │         │  (Engine)    │              │  │
│  │  │  strategy)       │         │              │              │  │
│  │  └──────────────────┘         └──────────────┘              │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                    INFERENCE PROCESSES (Ray Actors)                 │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Insight:** Components run on **different processes**:
- **Controller**: `WeightTransferStrategy` factory runs on controller (trainer.py) - lightweight
- **Training side**: `WeightExtractor` + `WeightTransferSender` (created via strategy) run on training Ray actors
- **Inference side**: `WeightTransferReceiver` (created via strategy) + `WeightLoader` run on inference Ray actors
- **Initialization**: Happens on workers, not controller (process groups must be initialized where they're used)
- **Communication**: Process groups, IPC handles, Ray RPC

## Core Components

### 1. WeightExtractor (Training Backend Interface)

**Purpose:** Extract weights from sharded models in backend-specific ways

**Location:** Runs on training Ray actor

**Interface:**
```python
class WeightExtractor(ABC):
    """Extracts weights from sharded training models for sync."""

    def __init__(self, cfg: DictConfig):
        """Initialize extractor with config."""
        self.cfg = cfg
        # Internal flags - subclasses can override in __init__ or via methods
        self._group_by_module = False
        self._batch_by_threshold = False

    def _group_by_module(self, params: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
        """
        Group parameter names by module (for FlashRL compatibility).
        Helper method for subclasses to use.

        Args:
            params: Dictionary mapping parameter names to tensors

        Returns:
            Dictionary mapping module names to lists of parameter names
        """
        pass

    def _batch_by_threshold(
        self,
        chunks: Iterator[WeightChunk],
        threshold_gb: float
    ) -> Iterator[List[WeightChunk]]:
        """
        Batch chunks until total size exceeds threshold.
        Helper method for subclasses to use.

        Args:
            chunks: Iterator of WeightChunk objects
            threshold_gb: Threshold in GB

        Yields:
            Lists of WeightChunk objects batched by threshold
        """
        pass

    @abstractmethod
    async def _extract_raw_weights(
        self,
        model: Any,
        generator_dtype: torch.dtype
    ) -> Dict[str, torch.Tensor]:
        """
        Extract raw weights from model (backend-specific implementation).
        Must handle: sharding, device placement, dtype conversion, format conversion.

        Args:
            model: The training model (backend-specific wrapper)
            generator_dtype: Target dtype for inference engine

        Returns:
            Dictionary mapping parameter names to tensors (already on device, converted dtype)
        """
        pass

    async def extract_weights(
        self,
        model: Any,
        generator_dtype: torch.dtype
    ) -> Iterator[WeightChunk]:
        """
        Extract weights from the model, yielding chunks ready for transfer.
        Base implementation uses internal flags for grouping/batching.
        Subclasses can override to customize behavior.

        Args:
            model: The training model (backend-specific wrapper)
            generator_dtype: Target dtype for inference engine.
                           Weights are converted to this dtype during extraction.

        Yields:
            WeightChunk objects containing weight data and metadata.
            Weights are already converted to generator_dtype.
            Chunks are grouped/batched based on internal flags.
        """
        pass
```

**Implementations:**
- `FSDPWeightExtractor` - Uses `state_dict()` + `full_tensor()`, optionally groups by module, optionally batches by threshold
- `MegatronWeightExtractor` - Uses `bridge.export_hf_weights()`, uses threshold buckets (not module grouping), packs tensors
- `DeepSpeedWeightExtractor` - Uses `named_parameters()` + `GatheredParameters`, optionally groups by module, optionally batches by threshold

**Key Points:**
- Base class has internal flags `_group_by_module` and `_batch_by_threshold` (default False)
- Subclasses can set these flags in `__init__` or override `_should_group_by_module()` / `_should_batch_by_threshold()`
- Subclasses can override `extract_weights()` for completely custom behavior (e.g., Megatron uses threshold buckets)
- Base class provides helper methods `_group_by_module()` and `_batch_by_threshold()` for subclasses to use

**Example Subclass:**
```python
class FSDPWeightExtractor(WeightExtractor):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        # Set flags based on config
        self._group_by_module = (
            cfg.generator.weight_sync_backend == "nccl" and
            cfg.trainer.placement.colocate_all
        )
        self._batch_by_threshold = (
            cfg.generator.weight_sync_backend == "nccl" and
            cfg.trainer.placement.colocate_all and
            hasattr(cfg.generator, "weight_transfer_threshold_cuda_ipc_GB")
        )
```


### 2. WeightTransferSender (Training Side Transfer)

**Purpose:** Handle sending weights from training to inference processes

**Location:** Runs on training Ray actor (rank 0)

**Interface:**
```python
class WeightTransferSender(ABC):
    """Handles sending weights from training to inference processes."""

    @abstractmethod
    async def send_chunks(
        self,
        chunks: Iterator[WeightChunk],
        inference_client: InferenceEngineClient
    ) -> None:
        """
        Send weight chunks to inference engines.
        Runs on training Ray actor (rank 0).
        Chunks are already batched/grouped by the extractor.

        Args:
            chunks: Iterator of weight chunks to send (each chunk contains multiple params)
            inference_client: Client for inference engine communication
        """
        pass

    @abstractmethod
    def create_request(self, chunk: WeightChunk) -> WeightUpdateRequest:
        """
        Create a WeightUpdateRequest from a WeightChunk.

        Args:
            chunk: Weight chunk to convert

        Returns:
            WeightUpdateRequest appropriate for this transfer strategy
        """
        pass
```

**Note:** Initialization happens in `TransferStrategy.create_sender()`. The sender is ready to use immediately after creation.

**Implementations:**
- `BroadcastTransferSender` - Broadcasts via process group, creates `BroadcastWeightUpdateRequest`. Sends chunks immediately.
- `CudaIpcTransferSender` - Creates IPC handles, creates `CudaIpcWeightUpdateRequest`. Sends chunks as they come (already batched by extractor).
- `PackedCudaIpcTransferSender` - Packs tensors, creates `PackedCudaIpcWeightUpdateRequest`. Sends chunks as they come (already batched by extractor).

### 3. WeightTransferReceiver (Inference Side Transfer)

**Purpose:** Handle receiving weights on inference processes

**Location:** Runs on inference Ray actor

**Interface:**
```python
class WeightTransferReceiver(ABC):
    """Handles receiving weights on inference processes."""

    @abstractmethod
    async def receive_and_load_weights(
        self,
        request: WeightUpdateRequest,
        loader: WeightLoader
    ) -> None:
        """
        Receive weights and load into inference model.
        Runs on inference Ray actor.

        Args:
            request: Weight update request (strategy-specific type)
            loader: Weight loader to use for loading weights
        """
        pass

    @abstractmethod
    def parse_request(self, request_dict: Dict[str, Any]) -> WeightUpdateRequest:
        """
        Parse a request dict (from Ray RPC) into a WeightUpdateRequest.

        Args:
            request_dict: Dictionary from Ray RPC

        Returns:
            WeightUpdateRequest appropriate for this transfer strategy
        """
        pass
```

**Note:** Initialization happens in `TransferStrategy.create_receiver()`. The receiver is ready to use immediately after creation.

**Implementations:**
- `BroadcastTransferReceiver` - Receives via broadcast, parses `BroadcastWeightUpdateRequest`
- `CudaIpcTransferReceiver` - Opens IPC handles, parses `CudaIpcWeightUpdateRequest`
- `PackedCudaIpcTransferReceiver` - Unpacks tensors, parses `PackedCudaIpcWeightUpdateRequest`

### 4. WeightLoader (Inference Engine Interface)

**Purpose:** Load weights into inference engines

**Location:** Runs on inference Ray actor

**Interface:**
```python
class WeightLoader(ABC):
    """Loads weights into inference engines."""

    @abstractmethod
    async def load_weights(
        self,
        named_tensors: Dict[str, torch.Tensor]
    ) -> None:
        """
        Load weights from named tensors into the inference model.

        Args:
            named_tensors: Dictionary mapping parameter names to tensors
        """
        pass
```

**Note:** The receiver extracts tensors from the request and passes them to the loader as a simple dict. This keeps the loader engine-specific but transfer-agnostic.

**Implementations:**
- `VLLMWeightLoader` - Direct `model.load_weights(named_tensors)` API
- `SGLangWeightLoader` - Custom loader function + SGLang APIs
- `RemoteWeightLoader` - HTTP endpoints

### 5. WeightChunk (Data Structure)

**Purpose:** Standardized representation of a batch/group of weight parameters

**Key Change:** Represents **multiple parameters**, not just one

```python
@dataclass
class WeightChunk:
    """Represents a batch/group of weights ready for transfer."""
    # Lists of parameter metadata (one entry per parameter in the chunk)
    names: List[str]
    tensors: List[torch.Tensor]  # One tensor per parameter
    dtypes: List[torch.dtype]
    shapes: List[Tuple[int, ...]]

    # Optional metadata for the entire chunk
    module_name: Optional[str] = None  # For grouping (e.g., "layers.0.attention")
    total_size_bytes: Optional[int] = None  # Total size of all tensors in chunk
    packed: bool = False  # Whether tensors are packed into contiguous memory
```

**Rationale:**
- Groups related parameters together (e.g., all attention weights for a module)
- Extractor handles grouping logic (knows model structure)
- Matches current `NamedWeightsUpdateRequest` structure (lists of names/shapes/dtypes)
- Supports packing (Megatron) where multiple tensors share one IPC handle

### 6. WeightUpdateRequest (Backend-Specific)

**Purpose:** Request structure for weight updates, backend-specific

**Rationale:** Different transfer strategies have different requirements:
- **Broadcast**: Simple metadata (names, shapes, dtypes)
- **CUDA IPC**: Needs IPC handles
- **Packed CUDA IPC**: Needs sizes for unpacking

**Base Interface:**
```python
class WeightUpdateRequest(ABC):
    """Base interface for weight update requests."""

    @property
    @abstractmethod
    def names(self) -> List[str]:
        """Parameter names."""
        pass

    @property
    @abstractmethod
    def dtypes(self) -> List[str]:
        """Parameter dtypes (string representation)."""
        pass

    @property
    @abstractmethod
    def shapes(self) -> List[List[int]]:
        """Parameter shapes."""
        pass
```

**Implementations:**

```python
@dataclass
class BroadcastWeightUpdateRequest(WeightUpdateRequest):
    """Request for broadcast-based weight transfer."""
    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]

@dataclass
class CudaIpcWeightUpdateRequest(WeightUpdateRequest):
    """Request for CUDA IPC-based weight transfer."""
    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]
    ipc_handles: List[Dict[int, Any]]
    # ipc_handles[i] maps GPU ID to IPC handle for parameter names[i]
    # Keys are physical GPU IDs, values are IPC handle objects

@dataclass
class PackedCudaIpcWeightUpdateRequest(WeightUpdateRequest):
    """Request for packed CUDA IPC weight transfer."""
    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]
    sizes: List[int]  # Size in bytes per parameter (for unpacking)
    ipc_handles: Dict[int, Any]
    # ipc_handles maps GPU ID to IPC handle for the entire packed bucket
    # The bucket contains all parameters in names concatenated together
    # sizes[i] indicates the size (in bytes) of parameter names[i] within the packed bucket
```

**Rationale:**
- **Type Safety**: Each transfer strategy has explicit request structure
- **Clear Contracts**: Sender/receiver know what to expect
- **Extensibility**: Easy to add new request types for new transfer strategies

### 7. WeightSyncInfo (Data Structure)

**Purpose:** Encapsulates information needed for weight sync initialization

**Location:** Created on training side, passed to inference side via controller

**Note:** The structure is strategy-specific. Each transfer strategy defines what information it needs.

```python
# Base interface (abstract)
class WeightSyncInfo(ABC):
    """Base interface for weight sync information."""
    pass

@dataclass
class TorchDistributedWeightSyncInfo(WeightSyncInfo):
    """Information needed for weight sync using torch.distributed process groups."""
    master_addr: str  # Master address for process group (from training rank 0)
    master_port: int  # Master port for process group (from training rank 0)
    world_size: int  # Total world size (training rank 0 + all inference ranks)
    group_name: str  # Name for the process group
    backend: str  # Backend for process group (nccl/gloo)

@dataclass
class EmptyWeightSyncInfo(WeightSyncInfo):
    """No sync info needed - strategy uses existing process groups or other mechanisms."""
    pass
```

**Rationale:**
- Training rank 0 determines master_addr/master_port
- This info needs to be shared with all inference engines
- Clean data structure for passing sync configuration
- Serializable for Ray RPC
- Strategy-specific: Different strategies may need different sync info

### 8. WeightTransferStrategy (Factory/Strategy)

**Purpose:** Factory for creating transfer senders/receivers

**Location:** Runs on controller (trainer.py) - lightweight strategy object

**Interface:**
```python
class WeightTransferStrategy(ABC):
    """Factory for creating weight transfer senders and receivers."""

    @abstractmethod
    async def create_sender(
        self,
        cfg: DictConfig,
        inference_client: InferenceEngineClient,
        sync_info: WeightSyncInfo
    ) -> WeightTransferSender:
        """
        Create and initialize a sender instance for training side.
        Called on training Ray actor (rank 0).
        Performs strategy-specific initialization internally.

        Args:
            cfg: Configuration for the sender
            inference_client: Client for communicating with inference engines
            sync_info: Weight sync information (strategy-specific type)

        Returns:
            WeightTransferSender instance, fully initialized and ready to use
        """
        pass

    @abstractmethod
    async def create_receiver(
        self,
        cfg: DictConfig,
        sync_info: WeightSyncInfo,
        rank_offset: int
    ) -> WeightTransferReceiver:
        """
        Create and initialize a receiver instance for inference side.
        Called on inference Ray actor.
        Performs strategy-specific initialization internally.

        Args:
            cfg: Configuration for the receiver
            sync_info: Weight sync information (strategy-specific type, from training rank 0)
            rank_offset: Rank offset for this inference engine

        Returns:
            WeightTransferReceiver instance, fully initialized and ready to use
        """
        pass

    @abstractmethod
    def get_request_type(self) -> Type[WeightUpdateRequest]:
        """Return the WeightUpdateRequest type for this strategy."""
        pass

    @abstractmethod
    async def create_sync_info(
        self,
        cfg: DictConfig,
        inference_client: InferenceEngineClient
    ) -> WeightSyncInfo:
        """
        Create WeightSyncInfo for this strategy.
        Called on training Ray actor (rank 0).

        Args:
            cfg: Configuration
            inference_client: Client for communicating with inference engines

        Returns:
            WeightSyncInfo instance appropriate for this strategy
        """
        pass
```

**Implementations:**
- `BroadcastTransferStrategy`
  - Creates `BroadcastTransferSender` and `BroadcastTransferReceiver`
  - Uses `TorchDistributedWeightSyncInfo` with `backend="nccl"` or `backend="gloo"`
  - Requires `_model_update_group` (training rank 0 + all inference ranks) for broadcast
- `CudaIpcTransferStrategy`
  - Creates `CudaIpcTransferSender` and `CudaIpcTransferReceiver`
  - Uses `EmptyWeightSyncInfo` (no process group needed)
  - Uses default training process group (already exists) for `all_gather_object` (to gather IPC handles from training ranks)
  - Does NOT use `_model_update_group` - weights transferred via IPC handles sent via Ray RPC
- `PackedCudaIpcTransferStrategy`
  - Creates `PackedCudaIpcTransferSender` and `PackedCudaIpcTransferReceiver`
  - Uses `EmptyWeightSyncInfo` (no process group needed)
  - Similar to `CudaIpcTransferStrategy` - uses default training process group, not `_model_update_group`

**Key Points:**
- **Lightweight**: Just a factory, no heavy initialization
- **Controller**: Lives on controller, passed to workers via Ray RPC or config
- **Factory Pattern**: Each side creates their own sender/receiver instances
- **Type Safety**: Knows which `WeightUpdateRequest` and `WeightSyncInfo` types to use
- **Encapsulation**: Strategy-specific logic (process groups, IPC handles, etc.) is hidden in implementations

## End-to-End Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ CONTROLLER (trainer.py)                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              │ 1. sync_weights() [Ray RPC]
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TRAINING RAY ACTOR (rank 0)                                                 │
│                                                                             │
│  2. sync_weights()                                                          │
│     │                                                                       │
│     ├─ 3. extractor.extract_weights()                                      │
│     │   │   (Grouping/batching determined internally from config)           │
│     │   │                                                                   │
│     │   ├─ 3a. _extract_raw_weights() (backend-specific)                   │
│     │   │   └─> Dict[str, torch.Tensor] (gathered, on device, converted)  │
│     │   │                                                                   │
│     │   └─ 3b. Group/batch based on config                                  │
│     │       └─> Yields WeightChunk objects (grouped/batched as needed)        │
│     │                                                                       │
│     ├─ 4. transfer_sender.send_chunks(chunks_iterator)                     │
│     │   │   (Sender iterates over chunks, sends immediately)               │
│     │   │                                                                   │
│     │   ├─ Broadcast Path:                                                 │
│     │   │   ├─ For each chunk: Create WeightUpdateRequest                 │
│     │   │   ├─ inference_client.update_named_weights() [Ray RPC]          │
│     │   │   └─ torch.distributed.broadcast() [Process Group]               │
│     │   │                                                                   │
│     │   └─ CUDA IPC Path:                                                  │
│     │       ├─ For each chunk:                                             │
│     │       │   ├─ Create IPC handles (reduce_tensor)                     │
│     │       │   ├─ Gather handles via all_gather_object                   │
│     │       │   ├─ Create WeightUpdateRequest with IPC handles            │
│     │       │   └─ inference_client.update_named_weights() [Ray RPC]       │
│                                                                             │
│  7. Return                                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              │ Ray RPC (update_named_weights.remote)
                              │ Carries WeightUpdateRequest (as dict)
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ INFERENCE RAY ACTOR (vLLM/SGLang)                                          │
│                                                                             │
│  7. update_named_weights(request_dict: Dict[str, Any])                    │
│     │                                                                       │
│     ├─ 8. transfer_receiver.parse_request(request_dict)                    │
│     │   └─> Returns WeightUpdateRequest (strategy-specific type)           │
│     │                                                                       │
│     ├─ 9. transfer_receiver.receive_and_load_weights(request, loader)      │
│     │   │                                                                   │
│     │   ├─ Broadcast Path:                                                    │
│     │   │   ├─ 10. Receive via torch.distributed.broadcast() → named_tensors│
│     │   │   └─ 11. loader.load_weights(named_tensors)                        │
│     │   │                                                                   │
│     │   └─ CUDA IPC Path:                                                   │
│     │       ├─ 10. Extract IPC handles from request.ipc_handles              │
│     │       ├─ 11. Open IPC handles (from_handle)                           │
│     │       ├─ 12. Reconstruct tensors → named_tensors                       │
│     │       └─ 13. loader.load_weights(named_tensors)                        │
│     │                                                                       │
│     └─ 14. Return                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Design Principles

### 1. Separation of Concerns
- **Extraction** (backend-specific) - How to get weights from sharded model
- **Transfer Sending** (mechanism-specific) - How to send weights (broadcast/IPC)
- **Transfer Receiving** (mechanism-specific) - How to receive weights
- **Loading** (engine-specific) - How to load weights into inference model

### 2. Strategy Pattern
- Different transfer strategies are pluggable
- Backends can choose optimal strategy based on configuration
- Easy to add new strategies (e.g., RDMA, tensor parallelism)

### 3. Batching and Grouping
- Base `WeightExtractor` has internal flags `_group_by_module` and `_batch_by_threshold` (default False)
- Subclasses can set flags in `__init__` or override `_should_group_by_module()` / `_should_batch_by_threshold()`
- Provides helper methods `_group_by_module()` and `_batch_by_threshold()` for subclasses
- Subclasses can override `extract_weights()` for completely custom behavior (e.g., Megatron threshold buckets)
- Yielding enables memory-efficient progressive processing
- Sender sends chunks immediately (already batched by extractor)

### 4. Backend Agnostic
- Training backends implement `WeightExtractor`
- Inference engines implement `WeightLoader`
- Transfer strategies are independent of backends

### 5. Separate Sender/Receiver Classes
- **Clean separation**: Sender and receiver are different classes
- **Different processes**: Sender on training, receiver on inference
- **Different responsibilities**: Sender creates handles/broadcasts, receiver receives/loads
- **Easier to reason about**: Each class has a single, clear responsibility

## Example Usage

### Controller Side (trainer.py)

```python
# Setup on controller - lightweight strategy factory
if cfg.generator.use_cuda_ipc:
    transfer_strategy = CudaIpcTransferStrategy()
else:
    transfer_strategy = BroadcastTransferStrategy()

# Strategy is passed to workers (via config or Ray RPC)
# Workers create their own sender/receiver instances
```

### Training Side (Ray Actor)

```python
# On training Ray actor (rank 0)
# Called via Ray RPC from controller: initialize_weight_sync(strategy, cfg, inference_client)

async def initialize_weight_sync(
    strategy: WeightTransferStrategy,
    cfg: DictConfig,
    inference_client: InferenceEngineClient
) -> WeightSyncInfo:
    """Initialize weight sync on training side, return sync info."""
    # Strategy creates sync info (strategy-specific - may be empty for CUDA IPC)
    sync_info = await strategy.create_sync_info(cfg, inference_client)

    # Create sender (initialization happens here - strategy-specific setup)
    transfer_sender = await strategy.create_sender(cfg, inference_client, sync_info)

    # Store sender for later use
    self._transfer_sender = transfer_sender

    return sync_info

# Later, when syncing weights:
async def sync_weights():
    extractor = FSDPWeightExtractor(cfg)

    # Extract weights (grouping/batching determined automatically from config)
    weight_chunks_iterator = extractor.extract_weights(model, generator_dtype)

    # Send chunks (sender iterates and sends immediately - chunks already batched)
    await self._transfer_sender.send_chunks(weight_chunks_iterator, inference_client)
```

### Inference Side (Ray Actor)

```python
# On inference Ray actor
# Called via Ray RPC from controller: initialize_weight_sync(strategy, cfg, sync_info, rank_offset)

async def initialize_weight_sync(
    strategy: WeightTransferStrategy,
    cfg: DictConfig,
    sync_info: WeightSyncInfo,
    rank_offset: int
) -> None:
    """Initialize weight sync on inference side."""
    loader = VLLMWeightLoader(engine)

    # Create receiver (initialization happens here - strategy-specific setup)
    transfer_receiver = await strategy.create_receiver(cfg, sync_info, rank_offset)

    # Store receiver for later use
    self._transfer_receiver = transfer_receiver
    self._loader = loader

# Later, when receiving weight updates:
async def handle_weight_update(request_dict: Dict[str, Any]):
    # Parse request dict into strategy-specific WeightUpdateRequest
    request = self._transfer_receiver.parse_request(request_dict)

    # Receive weights and load
    await self._transfer_receiver.receive_and_load_weights(request, self._loader)
```

## Key Design Decisions

### 1. Why Separate Extractor/Transfer/Loader?

**Benefits:**
- **Modularity**: Each component can be tested independently
- **Flexibility**: Mix and match backends with transfer strategies
- **Extensibility**: Add new backends/engines without touching transfer logic
- **Testability**: Mock components for unit testing

**Example:** FSDP can use either Broadcast or CUDA IPC without code duplication

### 2. Why WeightChunk Represents Multiple Parameters?

**Benefits:**
- **Grouping**: Extractor groups related parameters together (e.g., all attention weights for a module)
- **Model Awareness**: Extractor knows model structure and can group intelligently
- **Packing**: Supports Megatron's packed tensors (multiple params → one IPC handle)
- **Matches Reality**: Current `NamedWeightsUpdateRequest` uses lists for multiple params

**Example:** One `WeightChunk` for `layers.0.attention` contains `[q_proj.weight, k_proj.weight, v_proj.weight]`. The extractor creates chunks grouped by module.

### 3. Why Separate Sender/Receiver Classes?

**Critical for Distributed Architecture:**
- **Different Processes**: Sender runs on training Ray actors, receiver on inference Ray actors
- **Different Responsibilities**:
  - Sender: Creates IPC handles, initiates broadcast
  - Receiver: Receives via process group/IPC, loads weights
- **Clean Separation**: Each class has a single responsibility
- **Easier Testing**: Can test sender and receiver independently

**Alternative Considered:** Single transfer strategy with sender/receiver methods
- **Rejected**: Would require complex process detection and state management
- **Current**: Clean separation, each class knows its role

### 4. Why Separate WeightUpdateRequest Types?

**Benefits:**
- **Type Safety**: Each backend has explicit structure (broadcast vs IPC vs packed)
- **Clear Contracts**: Sender/receiver know exactly what to expect
- **Extensibility**: Easy to add new request types for new backends
- **Validation**: Each type can validate its own structure
- **IDE Support**: Better autocomplete and type checking

**Alternative Considered:** Single `WeightUpdateRequest` with optional fields
- **Rejected**: Less type-safe, harder to validate, unclear contracts
- **Current**: Backend-specific types with clear interfaces

### 5. Why WeightTransferStrategy Factory Pattern?

**Benefits:**
- **Lightweight Controller**: Strategy is just a factory, no heavy initialization
- **Worker Ownership**: Each side creates and owns their sender/receiver
- **Initialization on Workers**: Process groups initialized where they're used
- **Type Safety**: Strategy knows which request types to create
- **Flexibility**: Easy to swap strategies without changing worker code

**Alternative Considered:** Manager on controller that initializes everything
- **Rejected**: Process groups must be initialized on workers, not controller
- **Current**: Factory pattern, workers create and initialize their own instances

## Distributed Process Considerations

### Process Distribution

**Controller:**
- `WeightTransferStrategy` (factory) runs on **controller** (trainer.py)
- Lightweight, just creates sender/receiver instances
- Workers create strategy based on config (no serialization needed)
- Orchestrates initialization: calls training actors to initialize, gets `WeightSyncInfo`, distributes to inference actors

**Training Side:**
- `WeightExtractor` runs on **training Ray actor** (one per GPU/rank)
- `initialize_weight_sync()` runs on **training Ray actor** (rank 0):
  - Determines master_addr/master_port
  - Creates `WeightSyncInfo`
  - Calls `strategy.create_sender()` which sets up process groups internally
  - Returns `WeightSyncInfo` to controller
- `sender.send_chunks()` runs on **training Ray actor** (rank 0)

**Inference Side:**
- `initialize_weight_sync()` runs on **inference Ray actor**:
  - Receives `WeightSyncInfo` from controller
  - Calls `strategy.create_receiver()` which joins process group internally
- `receiver.receive_and_load_weights()` runs on **inference Ray actor**
- `WeightLoader` runs on **inference Ray actor**
- Called via `InferenceEngineClient.update_named_weights()` → Ray RPC → inference actor

### Communication Mechanisms

1. **Ray RPC**: `InferenceEngineClient` → `engine.update_named_weights.remote()`
   - Used to trigger weight updates on inference engines
   - Handles routing to multiple engines
   - Carries `WeightUpdateRequest` (serializable)

2. **Process Groups**: Training rank 0 ↔ All inference ranks (for broadcast strategy)
   - Created during `create_sender()` / `create_receiver()` (strategy-specific)
   - Used for NCCL/Gloo broadcast
   - CUDA IPC strategy uses default training process group (already exists) for coordination

3. **CUDA IPC Handles**: Shared GPU memory
   - Created on training side (`reduce_tensor()`)
   - Sent via Ray RPC in `WeightUpdateRequest.ipc_handles` field
   - Opened on inference side (`from_handle()`) to access shared memory

### Key Implications

1. **State Management**: Each side maintains its own state
   - Training: Process group, IPC handles
   - Inference: Process group, model state

2. **Serialization**: Requests must be serializable for Ray RPC
   - IPC handles use `reduce_tensor()` for serialization
   - `WeightUpdateRequest` dataclasses are serializable (can be converted to dict for Ray RPC)

3. **Synchronization**: Barriers and synchronization happen within each process group
   - Training ranks synchronize via training process group
   - Inference ranks synchronize via inference process group
   - Cross-group sync via process group barriers

## Migration Path

### Phase 1: Extract Interfaces
- Define abstract base classes (`WeightExtractor`, `WeightTransferSender`, `WeightTransferReceiver`, `WeightLoader`)
- Define `WeightChunk` and `WeightUpdateRequest` data structures
- Keep existing implementations as-is
- Add adapter layer to bridge old → new

### Phase 2: Refactor Backends
- Implement `WeightExtractor` base class with grouping/batching determination logic
- Implement `_extract_raw_weights()` for each backend (FSDP, Megatron, DeepSpeed)
- Subclasses determine grouping/batching from config automatically
- Gradually move logic from `broadcast_to_inference_engines()` to extractors

### Phase 3: Refactor Transfer Sending
- Implement `WeightTransferSender` classes (Broadcast, CUDA IPC)
- Move broadcast/IPC logic from workers to senders
- Update to work with `WeightChunk` objects

### Phase 4: Refactor Transfer Receiving
- Implement `WeightTransferReceiver` classes (Broadcast, CUDA IPC)
- Move receiving logic from inference engines to receivers
- Update to work with `WeightUpdateRequest`

### Phase 5: Refactor Loading
- Implement `WeightLoader` for each engine (vLLM, SGLang, Remote)
- Move loading logic from engines to loaders

### Phase 6: Integrate WeightTransferStrategy
- Implement `WeightTransferStrategy` factory on controller
- Update `trainer.py` to create and pass strategy to workers
- Workers create sender/receiver from strategy

### Phase 7: Cleanup
- Remove old code paths
- Update all call sites to use new interfaces
- Remove `NamedWeightsUpdateRequest` (replace with backend-specific `WeightUpdateRequest` types)

## Design Decisions

1. **Packing Logic**: Packing is handled by the transfer sender (strategy-specific).
   - **Decision**: Sender packs (generic PyTorch code, runs on training side)
   - **Rationale**: Packing is a transfer optimization, not extraction logic

2. **Process Group Management**: Process groups are owned by the transfer sender/receiver.
   - **Decision**: Transfer sender/receiver own process groups (strategy-specific)
   - **Rationale**: Process groups are transfer-specific, encapsulated in sender/receiver implementations

3. **Batching/Thresholding**: Batching and grouping logic lives in the extractor base class.
   - **Decision**: Base `WeightExtractor` has internal flags `_group_by_module` and `_batch_by_threshold` (default False). Subclasses can set these flags in `__init__` or override `_should_group_by_module()` / `_should_batch_by_threshold()`. Provides helper methods `_group_by_module()` and `_batch_by_threshold()` for subclasses to use.
   - **Rationale**: 
     - Grouping/batching logic is reusable across backends
     - Flags are internal (not in public API) but configurable by subclasses
     - Subclasses can set flags based on config or override methods for custom logic
     - Subclasses can override `extract_weights()` for completely custom behavior
     - Extractor yields chunks progressively for memory efficiency
     - Sender sends chunks immediately (no accumulation needed - already batched)

4. **Receiver Initialization**: Handled via `WeightTransferStrategy.create_receiver()`.
   - **Decision**: Encapsulated in strategy factory method
   - **Rationale**: Already discussed above - initialization happens in `create_receiver()`

5. **Error Handling**: Fail fast (current behavior).
   - **Decision**: Fail fast
   - **Rationale**: Keep current behavior, can add retry mechanism later if needed

6. **WeightUpdateRequest Migration**: Eventually remove `NamedWeightsUpdateRequest`.
   - **Decision**: Remove `NamedWeightsUpdateRequest` after migration
   - **Rationale**: Replace with strategy-specific `WeightUpdateRequest` types

7. **Strategy Passing**: Workers create strategy based on config.
   - **Decision**: Workers read config and create strategy
   - **Rationale**: Simplest approach, no serialization needed, config already available on workers

## Next Steps

1. **Review this design** - Get feedback on architecture
2. **Define detailed interfaces** - Method signatures, return types, exceptions
3. **Create implementation plan** - Phased migration approach
4. **Prototype** - Implement one backend + engine combination first
