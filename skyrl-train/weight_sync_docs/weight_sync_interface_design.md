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
    
    @abstractmethod
    async def extract_weights(
        self, 
        model: Any,
        generator_dtype: torch.dtype
    ) -> Iterator[WeightChunk]:
        """
        Extract weights from the model, yielding chunks ready for transfer.
        Each chunk represents a batch/group of related parameters.
        
        Args:
            model: The training model (FSDP/Megatron/DeepSpeed wrapped)
            generator_dtype: Target dtype for inference engine (e.g., float16, bfloat16).
                           Weights are converted to this dtype during extraction.
                           Determined by:
                           1. User config: `cfg.generator.model_dtype` (string, e.g., "bfloat16")
                           2. Converted via `str_to_torch_dtype()` to `torch.dtype`
                           3. Default: "bfloat16" (from ppo_base_config.yaml)
                           4. Must match the dtype the inference engine expects
        
        Returns:
            Iterator of WeightChunk objects containing weight data and metadata.
            Weights in chunks are already converted to generator_dtype.
        """
        pass
```

**Implementations:**
- `FSDPWeightExtractor` - Uses `state_dict()` + `full_tensor()`, groups by module
- `MegatronWeightExtractor` - Uses `bridge.export_hf_weights()`, groups into buckets
- `DeepSpeedWeightExtractor` - Uses `named_parameters()` + `GatheredParameters`, groups by module

### 2. WeightTransferSender (Training Side Transfer)

**Purpose:** Handle sending weights from training to inference processes

**Location:** Runs on training Ray actor (rank 0)

**Interface:**
```python
class WeightTransferSender(ABC):
    """Handles sending weights from training to inference processes."""
    
    @abstractmethod
    async def send_chunk(
        self,
        chunk: WeightChunk,
        inference_client: InferenceEngineClient
    ) -> None:
        """
        Send a single weight chunk.
        Runs on training Ray actor (rank 0).
        Handles batching internally (e.g., threshold-based batching for CUDA IPC).
        May accumulate chunks and send when threshold reached.
        
        Args:
            chunk: Weight chunk to send (contains multiple params)
            inference_client: Client for inference engine communication
        """
        pass
    
    @abstractmethod
    async def flush(
        self,
        inference_client: InferenceEngineClient
    ) -> None:
        """
        Flush any accumulated chunks that haven't been sent yet.
        Should be called after all chunks have been sent via send_chunk().
        
        Args:
            inference_client: Client for inference engine communication
        """
        pass
    
    @abstractmethod
    def create_request(self, chunk: WeightChunk) -> WeightUpdateRequest:
        """
        Create a WeightUpdateRequest from a WeightChunk.
        Backend-specific implementation.
        
        Args:
            chunk: Weight chunk to convert
            
        Returns:
            WeightUpdateRequest appropriate for this sender
        """
        pass
```

**Note:** Initialization happens in `TransferStrategy.create_sender()`, which sets up process groups, IPC handles, etc. The sender is ready to use immediately after creation.

**Implementations:**
- `BroadcastTransferSender` - Broadcasts via process group, creates `BroadcastWeightUpdateRequest`
- `CudaIpcTransferSender` - Creates IPC handles, creates `CudaIpcWeightUpdateRequest`
- `PackedCudaIpcTransferSender` - Packs tensors, creates `PackedCudaIpcWeightUpdateRequest`

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
            request: Weight update request (backend-specific type)
            loader: Weight loader to use for loading weights
        """
        pass
    
    @abstractmethod
    def parse_request(self, request_dict: Dict[str, Any]) -> WeightUpdateRequest:
        """
        Parse a request dict (from Ray RPC) into a WeightUpdateRequest.
        Backend-specific implementation.
        
        Args:
            request_dict: Dictionary from Ray RPC
            
        Returns:
            WeightUpdateRequest appropriate for this receiver
        """
        pass
```

**Note:** Initialization happens in `TransferStrategy.create_receiver()`, which joins process groups, sets up IPC receiving, etc. The receiver is ready to use immediately after creation.

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

**Implementations:**
- `VLLMWeightLoader` - Direct `model.load_weights(named_tensors)` API
- `SGLangWeightLoader` - Custom loader function + SGLang APIs
- `RemoteWeightLoader` - HTTP endpoints

**Note:** The receiver extracts tensors from the request and passes them to the loader as a simple dict. This keeps the loader engine-specific but transfer-agnostic.

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
- Groups related parameters together (e.g., all attention weights)
- Enables batching optimizations (threshold-based sending)
- Matches current `NamedWeightsUpdateRequest` structure (lists of names/shapes/dtypes)
- Supports packing (Megatron) where multiple tensors share one IPC handle

### 6. WeightUpdateRequest (Backend-Specific)

**Purpose:** Request structure for weight updates, backend-specific

**Rationale:** Different transfer backends have different requirements:
- **Broadcast**: Simple metadata (names, shapes, dtypes)
- **CUDA IPC**: Needs IPC handles in extras
- **Megatron (packed)**: Needs sizes for unpacking

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
    # Example: ipc_handles[0] = {0: handle_gpu0, 1: handle_gpu1, ...}
    # where keys are physical GPU IDs and values are IPC handle objects

@dataclass
class PackedCudaIpcWeightUpdateRequest(WeightUpdateRequest):
    """Request for packed CUDA IPC (Megatron)."""
    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]
    sizes: List[int]  # Size in bytes per parameter (for unpacking)
    ipc_handles: Dict[int, Any]
    # ipc_handles maps GPU ID to IPC handle for the entire packed bucket
    # The bucket contains all parameters in names concatenated together
    # sizes[i] indicates the size (in bytes) of parameter names[i] within the packed bucket
    # Example: ipc_handles = {0: handle_gpu0, 1: handle_gpu1, ...}
```

**Rationale:**
- **Type Safety**: Each backend has explicit structure
- **Clear Contracts**: Sender/receiver know what to expect
- **Extensibility**: Easy to add new request types for new backends

### 7. WeightSyncInfo (Data Structure)

**Purpose:** Encapsulates information needed for weight sync initialization

**Location:** Created on training side, passed to inference side via controller

**Note:** The structure is strategy-specific. Each transfer strategy defines what information it needs.

```python
# Base interface (abstract)
class WeightSyncInfo(ABC):
    """Base interface for weight sync information."""
    pass

# Broadcast strategy - uses process group for actual weight transfer
@dataclass
class TorchDistributedWeightSyncInfo(WeightSyncInfo):
    """Information needed for weight sync using torch.distributed process groups."""
    master_addr: str  # Master address for process group (from training rank 0)
    master_port: int  # Master port for process group (from training rank 0)
    world_size: int  # Total world size (training rank 0 + all inference ranks)
    group_name: str  # Name for the process group
    backend: str  # Backend for process group (nccl/gloo) - strategy-specific

# CUDA IPC strategy - doesn't need process group for transfer (uses IPC handles via Ray RPC)
@dataclass
class EmptyWeightSyncInfo(WeightSyncInfo):
    """No sync info needed - CUDA IPC uses default training process group and Ray RPC."""
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
        sync_info: WeightSyncInfo  # Strategy-specific sync info type
    ) -> WeightTransferSender:
        """
        Create and initialize a sender instance for training side.
        Called on training Ray actor (rank 0).
        Sets up process groups, IPC handles, etc. internally.
        
        Args:
            cfg: Configuration for the sender
            inference_client: Client for communicating with inference engines
            sync_info: Weight sync information (master_addr, master_port, etc.)
            
        Returns:
            WeightTransferSender instance, fully initialized and ready to use
        """
        pass
    
    @abstractmethod
    async def create_receiver(
        self, 
        cfg: DictConfig,
        sync_info: WeightSyncInfo,  # Strategy-specific sync info type
        rank_offset: int
    ) -> WeightTransferReceiver:
        """
        Create and initialize a receiver instance for inference side.
        Called on inference Ray actor.
        Joins process group, sets up IPC receiving, etc. internally.
        
        Args:
            cfg: Configuration for the receiver
            sync_info: Weight sync information (from training rank 0)
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
    def get_sync_info_type(self) -> Type[WeightSyncInfo]:
        """Return the WeightSyncInfo type for this strategy."""
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
        Determines master_addr/master_port and calculates world_size.
        
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
- **Type Safety**: Knows which `WeightUpdateRequest` type to use

## End-to-End Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ CONTROLLER (trainer.py)                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              │ 1. sync_weights()
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TRAINING RAY ACTOR (rank 0)                                                 │
│                                                                             │
│  2. broadcast_to_inference_engines()                                       │
│     │                                                                       │
│     ├─ 3. extractor.extract_weights()                                      │
│     │   └─> Yields WeightChunk (multiple params per chunk)                 │
│     │                                                                       │
│     ├─ 4. transfer_sender.send_weights()                                   │
│     │   │                                                                   │
│     │   ├─ Broadcast Path:                                                  │
│     │   │   ├─ 5. For each WeightChunk:                                    │
│     │   │   │   ├─ Create WeightUpdateRequest                              │
│     │   │   │   ├─ inference_client.update_named_weights() [Ray RPC]       │
│     │   │   │   └─ torch.distributed.broadcast() [Process Group]          │
│     │   │                                                                   │
│     │   └─ CUDA IPC Path:                                                  │
│     │       ├─ 5. For each WeightChunk:                                    │
│     │       │   ├─ Create IPC handles (reduce_tensor)                     │
│     │       │   ├─ Gather handles via all_gather_object                    │
│     │       │   ├─ Create WeightUpdateRequest with IPC handles             │
│     │       │   └─ inference_client.update_named_weights() [Ray RPC]       │
│     │                                                                       │
│     └─ 6. Return ObjectRef                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              │ Ray RPC (update_named_weights)
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ INFERENCE RAY ACTOR (vLLM/SGLang)                                          │
│                                                                             │
│  7. update_named_weights(request: WeightUpdateRequest)                     │
│     │                                                                       │
│     ├─ 8. transfer_receiver.receive_and_load_weights()                     │
│     │   │                                                                   │
│     │   ├─ Broadcast Path:                                                  │
│     │   │   ├─ 9. Receive via torch.distributed.broadcast()                │
│     │   │   └─ 10. loader.load_weights(request)                             │
│     │   │                                                                   │
│     │   └─ CUDA IPC Path:                                                   │
│     │       ├─ 9. Extract IPC handles from request.extras                   │
│     │       ├─ 10. Open IPC handles (from_handle)                           │
│     │       ├─ 11. Reconstruct tensors                                      │
│     │       └─ 12. loader.load_weights(request)                              │
│     │                                                                       │
│     └─ 13. Return                                                           │
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

### 3. Iterator Pattern
- Weight extraction yields chunks progressively
- Enables streaming for large models
- Allows batching/thresholding during transfer

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
    # Strategy creates sync info (encapsulates master_addr/master_port determination, world_size calculation)
    sync_info = await strategy.create_sync_info(cfg, inference_client)
    
    # Create sender (initialization happens here - sets up process groups, etc.)
    transfer_sender = await strategy.create_sender(cfg, inference_client, sync_info)
    
    # Store sender for later use
    self._transfer_sender = transfer_sender
    
    return sync_info

# Later, when syncing weights:
async def sync_weights():
    extractor = FSDPWeightExtractor(model, cfg)
    
    # Extract weights and send (sender handles batching internally)
    async for chunk in extractor.extract_weights(model, generator_dtype):
        await self._transfer_sender.send_chunk(chunk, inference_client)
    
    # Flush any remaining chunks
    await self._transfer_sender.flush(inference_client)
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
    
    # Create receiver (initialization happens here - joins process group, etc.)
    transfer_receiver = await strategy.create_receiver(cfg, sync_info, rank_offset)
    
    # Store receiver for later use
    self._transfer_receiver = transfer_receiver
    self._loader = loader

# Later, when receiving weight updates:
async def handle_weight_update(request_dict: Dict[str, Any]):
    # Parse request dict into backend-specific WeightUpdateRequest
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
- **Batching**: Groups related parameters together (e.g., all attention weights)
- **Efficiency**: Enables threshold-based sending (current implementation)
- **Packing**: Supports Megatron's packed tensors (multiple params → one IPC handle)
- **Matches Reality**: Current `NamedWeightsUpdateRequest` uses lists for multiple params

**Example:** One `WeightChunk` for `layers.0.attention` contains `[q_proj.weight, k_proj.weight, v_proj.weight]`

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

**Alternative Considered:** WeightSyncManager on controller that initializes everything
- **Rejected**: Process groups must be initialized on workers, not controller
- **Current**: Factory pattern, workers create and initialize their own instances

## Distributed Process Considerations

### Process Distribution

**Controller:**
- `WeightSyncManager` runs on **controller** (trainer.py)
- Coordinates via Ray RPC with training and inference actors

**Training Side:**
- `WeightExtractor` runs on **training Ray actor** (one per GPU/rank)
- `WeightTransferSender.send_weights()` runs on **training Ray actor** (rank 0)

**Inference Side:**
- `WeightTransferReceiver.receive_and_load_weights()` runs on **inference Ray actor**
- `WeightLoader` runs on **inference Ray actor**
- Called via `InferenceEngineClient.update_named_weights()` → Ray RPC → inference actor

### Communication Mechanisms

1. **Ray RPC**: `InferenceEngineClient` → `engine.update_named_weights.remote()`
   - Used to trigger weight updates on inference engines
   - Handles routing to multiple engines
   - Carries `WeightUpdateRequest` (serializable)

2. **Process Groups**: Training rank 0 ↔ All inference ranks
   - Created during `initialize()` on sender/receiver
   - Used for NCCL/Gloo broadcast

3. **CUDA IPC Handles**: Shared GPU memory
   - Created on training side (`reduce_tensor()`)
   - Sent via Ray RPC in `WeightUpdateRequest.extras`
   - Opened on inference side (`from_handle()`) to access shared memory

### Key Implications

1. **State Management**: Each side maintains its own state
   - Training: Process group, IPC handles
   - Inference: Process group, model state

2. **Serialization**: Requests must be serializable for Ray RPC
   - IPC handles use `reduce_tensor()` for serialization
   - `WeightUpdateRequest` must be JSON-serializable (dataclass → dict)

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
- Implement `WeightExtractor` for each backend (FSDP, Megatron, DeepSpeed)
- Gradually move logic from `broadcast_to_inference_engines()` to extractors
- Update to yield `WeightChunk` objects (multiple params per chunk)

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

## Open Questions

1. **Packing Logic**: Should packing be part of `WeightExtractor` or `CudaIpcTransferSender`?
   - **Option A**: Extractor packs (Megatron-specific)
   - **Option B**: Sender packs (generic, reusable)
   - **Recommendation**: Sender packs (generic PyTorch code, runs on training side)

2. **Process Group Management**: Who owns `_model_update_group`?
   - **Option A**: Transfer sender owns it (training side)
   - **Option B**: Separate `ProcessGroupManager`
   - **Recommendation**: Transfer sender (it's transfer-specific, lives on training actor)

3. **Batching/Thresholding**: Where does batching logic live?
   - **Option A**: In `WeightExtractor` (knows module structure)
   - **Option B**: In `TransferSender.send_weights()` (knows transfer constraints)
   - **Recommendation**: Transfer sender (transfer-specific optimization, runs on training)

4. **Receiver Initialization**: How to coordinate receiver initialization?
   - **Option A**: Via `InferenceEngineClient.init_weight_update_communicator()` (current)
   - **Option B**: Part of `WeightSyncManager.initialize()`
   - **Recommendation**: Keep existing pattern, wrap in receiver's `initialize()`

5. **Error Handling**: How to handle partial failures?
   - **Option A**: Fail fast (current behavior)
   - **Option B**: Retry mechanism
   - **Option C**: Partial sync with error reporting
   - **Recommendation**: Start with fail-fast, add retry later

6. **WeightUpdateRequest Migration**: Should we keep `NamedWeightsUpdateRequest` as alias?
   - **Option A**: Keep as alias, migrate gradually
   - **Option B**: Replace immediately
   - **Recommendation**: Keep as alias initially, migrate in Phase 7
   
7. **Strategy Passing**: How should strategy be passed from controller to workers?
   - **Option A**: Via config (serialized)
   - **Option B**: Via Ray RPC (pass strategy object)
   - **Option C**: Workers create strategy based on config
   - **Recommendation**: Option C (workers read config, create strategy) - simplest, no serialization needed

## Next Steps

1. **Review this design** - Get feedback on architecture
2. **Clarify open questions** - Make decisions on design choices
3. **Define detailed interfaces** - Method signatures, return types, exceptions
4. **Create implementation plan** - Phased migration approach
5. **Prototype** - Implement one backend + engine combination first
