# Weight Sync Migration Plan - Detailed Implementation Guide

**Strategy:** Refactor each component layer (extraction → loading/receiving → sending) horizontally across all backends before integrating the strategy interface. Interfaces are introduced on-demand when needed. Optimizations (grouping/batching) are preserved in each phase.

---

## Phase 1: Refactor Weight Extraction (PRs #1a, #1b, #1c)
**Goal:** Replace inline extraction logic with `WeightExtractor` implementations, preserving all existing optimizations.

### Interfaces Introduced

Create `skyrl_train/weight_sync/` directory with:

**`base.py`** - `WeightExtractor` abstract base class and `WeightChunk` dataclass:
```python
from dataclasses import dataclass
from typing import Iterator, List, Optional
import torch

@dataclass
class WeightChunk:
    """Represents one or more model parameters to be transferred."""
    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]
    tensors: Optional[List[torch.Tensor]] = None
    total_size_bytes: Optional[int] = None

class WeightExtractor(ABC):
    """Extracts weights from training backend models."""
    
    def __init__(self, cfg, model):
        self.cfg = cfg
        self.model = model
        self._group_by_module = False
        self._batch_by_threshold = False
    
    def extract_weights(self, model) -> Iterator[WeightChunk]:
        """Public API: extract weights with grouping/batching applied."""
        chunks = self._extract_raw_weights(model)
        if self._group_by_module:
            chunks = self._group_by_module_impl(chunks)
        if self._batch_by_threshold:
            chunks = self._batch_by_threshold_impl(chunks)
        return chunks
    
    @abstractmethod
    def _extract_raw_weights(self, model) -> Iterator[WeightChunk]:
        """Subclass hook: yield raw per-parameter chunks."""
        ...
    
    def _group_by_module_impl(self, chunks: Iterator[WeightChunk]) -> Iterator[WeightChunk]:
        """Helper: group related params (e.g., QKV for FlashRL)."""
        # Default implementation (can be overridden)
        ...
    
    def _batch_by_threshold_impl(self, chunks: Iterator[WeightChunk]) -> Iterator[WeightChunk]:
        """Helper: batch small tensors for efficiency."""
        # Default implementation (can be overridden)
        ...
```

### PR #1a: FSDP

**Add `FSDPWeightExtractor` class in `skyrl_train/workers/fsdp/fsdp_worker.py`:**

```python
from skyrl_train.weight_sync import WeightExtractor, WeightChunk

class FSDPWeightExtractor(WeightExtractor):
    """Extracts weights from FSDP-sharded models."""
    
    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        # Preserve module grouping for FlashRL
        self._group_by_module = cfg.generator.get("use_flashrl", False)
    
    def _extract_raw_weights(self, model) -> Iterator[WeightChunk]:
        """Wrap existing full_tensor() + DTensor conversion logic."""
        for name, param in model.named_parameters():
            # Existing FSDP gathering logic
            full_tensor = full_tensor(param, ...)  # Simplified
            dtype_str = str(full_tensor.dtype)
            shape = list(full_tensor.shape)
            
            yield WeightChunk(
                names=[name],
                dtypes=[dtype_str],
                shapes=[shape],
                tensors=[full_tensor],
                total_size_bytes=full_tensor.numel() * full_tensor.element_size()
            )
    
    def _group_by_module_impl(self, chunks: Iterator[WeightChunk]) -> Iterator[WeightChunk]:
        """Override for FSDP-specific QKV grouping."""
        # Group q_proj, k_proj, v_proj into single chunk
        buffer = []
        for chunk in chunks:
            buffer.append(chunk)
            if self._should_flush_group(buffer):
                yield self._merge_chunks(buffer)
                buffer = []
        if buffer:
            yield self._merge_chunks(buffer)
```

**Refactor `FSDPPolicyWorkerBase.broadcast_to_inference_engines()`:**

```python
async def broadcast_to_inference_engines(self, inference_engine_client):
    extractor = FSDPWeightExtractor(self.cfg, model=self.model)
    chunks = list(extractor.extract_weights(self.model))

    # Continue with old sending logic (convert chunks to NamedWeightsUpdateRequest)
    request = self._chunks_to_legacy_request(chunks)
    await inference_engine_client.update_weights(**request)

def _chunks_to_legacy_request(self, chunks: List[WeightChunk]) -> Dict:
    """Convert chunks to legacy NamedWeightsUpdateRequest format."""
    names, dtypes, shapes = [], [], []
    for chunk in chunks:
        names.extend(chunk.names)
        dtypes.extend(chunk.dtypes)
        shapes.extend(chunk.shapes)
    return {"names": names, "dtypes": dtypes, "shapes": shapes}
```

**Testing:**
- Unit test: `FSDPWeightExtractor` yields correct chunks with proper metadata
- Unit test: Module grouping works when FlashRL enabled
- Integration test: `broadcast_to_inference_engines()` produces identical `NamedWeightsUpdateRequest`
- Integration test: Weight equality check (old vs. new extraction)

### PR #1b: Megatron

**Add `MegatronWeightExtractor` class in `skyrl_train/workers/megatron/megatron_worker.py`:**

```python
from skyrl_train.weight_sync import WeightExtractor, WeightChunk

class MegatronWeightExtractor(WeightExtractor):
    """Extracts weights from Megatron model-parallel models."""
    
    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        # Preserve batching for packing optimization
        self._batch_by_threshold = True
    
    def _extract_raw_weights(self, model) -> Iterator[WeightChunk]:
        """Wrap existing export_hf_weights() logic."""
        hf_weights = export_hf_weights(model, ...)  # Existing function
        for name, tensor in hf_weights.items():
            yield WeightChunk(
                names=[name],
                dtypes=[str(tensor.dtype)],
                shapes=[list(tensor.shape)],
                tensors=[tensor],
                total_size_bytes=tensor.numel() * tensor.element_size()
            )
```

**Refactor `MegatronPolicyWorkerBase.broadcast_to_inference_engines()` similarly**

### PR #1c: DeepSpeed

**Add `DeepSpeedWeightExtractor` class in `skyrl_train/workers/deepspeed/deepspeed_worker.py`:**

```python
from skyrl_train.weight_sync import WeightExtractor, WeightChunk

class DeepSpeedWeightExtractor(WeightExtractor):
    """Extracts weights from DeepSpeed ZeRO-sharded models."""
    
    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        # Preserve module grouping for FlashRL
        self._group_by_module = cfg.generator.get("use_flashrl", False)
    
    def _extract_raw_weights(self, model) -> Iterator[WeightChunk]:
        """Wrap existing GatheredParameters logic."""
        with GatheredParameters(model.parameters(), modifier_rank=0):
            for name, param in model.named_parameters():
                yield WeightChunk(
                    names=[name],
                    dtypes=[str(param.dtype)],
                    shapes=[list(param.shape)],
                    tensors=[param],
                    total_size_bytes=param.numel() * param.element_size()
                )
```

**Refactor `DeepSpeedPolicyWorkerBase.broadcast_to_inference_engines()` similarly**

### Outcome

Extraction abstracted with all optimizations preserved; sending/loading unchanged.

---

## Phase 2: Refactor Weight Loading and Receiving (PRs #2a, #2b, #2c)
**Goal:** Replace inline loading logic with `WeightLoader` + standardize local `receive_weights()` async iterator.

### Interfaces Introduced

**Add to `base.py`:**

```python
class WeightLoader(ABC):
    """Loads received weights into inference engine."""
    
    @abstractmethod
    async def load_weights(self, receiver: AsyncIterator[Tuple[str, torch.Tensor]], request: Dict) -> None:
        """Pull weights from receiver and load into engine."""
        ...
```

### PR #2a: vLLM

**Add `VLLMWeightLoader` class in `skyrl_train/inference_engines/vllm/vllm_engine.py`:**

```python
from skyrl_train.weight_sync import WeightLoader

class VLLMWeightLoader(WeightLoader):
    """Loads weights into vLLM engine."""
    
    def __init__(self, engine):
        self.engine = engine
    
    async def load_weights(self, receiver: AsyncIterator[Tuple[str, torch.Tensor]], request: Dict) -> None:
        """Pull weights from receiver and apply to vLLM model."""
        async for name, tensor in receiver:
            # Preserve multi-param chunk handling for FlashRL
            if isinstance(name, list):
                # Grouped params (e.g., QKV fusion)
                for n, t in zip(name, tensor):
                    self.engine.worker.model.load_weights([(n, t)])
            else:
                # Single param
                self.engine.worker.model.load_weights([(name, tensor)])
```

**Add local receiver helper methods:**

```python
async def _receive_weights_broadcast(self, request: Dict):
    """Async generator yielding (name, tensor) tuples via broadcast."""
    for name, dtype_str, shape in zip(request["names"], request["dtypes"], request["shapes"]):
        tensor = torch.empty(shape, dtype=parse_dtype(dtype_str), device="cuda")
        torch.distributed.broadcast(tensor, src=0, group=self._model_update_group)
        yield name, tensor

async def _receive_weights_cuda_ipc(self, request: Dict):
    """Async generator yielding (name, tensor) tuples via CUDA IPC."""
    for name in request["names"]:
        handle = request["ipc_handles"][name]
        tensor = torch.cuda._from_handle(handle, ...)
        yield name, tensor
```

**Refactor existing methods:**

```python
async def update_weights(self, names, dtypes, shapes):
    request = {"names": names, "dtypes": dtypes, "shapes": shapes}
    loader = VLLMWeightLoader(self.engine)
    receiver = self._receive_weights_broadcast(request)
    await loader.load_weights(receiver, request)

async def update_weights_cuda_ipc(self, ipc_handles, ...):
    request = {"names": list(ipc_handles.keys()), "ipc_handles": ipc_handles}
    loader = VLLMWeightLoader(self.engine)
    receiver = self._receive_weights_cuda_ipc(request)
    await loader.load_weights(receiver, request)
```

**Testing:**
- Unit test: loader correctly applies weights from async iterator
- Unit test: FlashRL QKV fusion works with grouped chunks
- Integration test: `update_weights()` behavior unchanged
- Integration test: Weight equality check

### PR #2b: SGLang

Similar to vLLM - add `SGLangWeightLoader` in `sglang_engine.py` with multi-param handling.

### PR #2c: Remote

Add `RemoteWeightLoader` in `remote_inference_engine.py` with no-op implementation.

### Outcome

Loading/receiving abstracted with optimizations preserved; sending unchanged.

---

## Phase 3: Refactor Weight Sending (PRs #3a, #3b)
**Goal:** Implement `WeightTransferSender` locally in workers, preserve all optimizations, still use legacy coordination.

### Interfaces Introduced

**Add to `base.py`:**

```python
class WeightTransferSender(ABC):
    """Sends weight chunks from training actors to inference actors."""
    
    @abstractmethod
    async def send_chunks(self, chunks: Iterator[WeightChunk]) -> None:
        """Send weight chunks (one at a time)."""
        ...
    
    @abstractmethod
    def get_request(self) -> Dict:
        """Build the request object after sending."""
        ...
```

### PR #3a: Broadcast Sender

**Create `skyrl_train/weight_sync/broadcast_strategy.py`:**

```python
from skyrl_train.weight_sync import WeightTransferSender, WeightChunk

class BroadcastWeightTransferSender(WeightTransferSender):
    """Sends weights via torch.distributed.broadcast."""
    
    def __init__(self, process_group):
        self.process_group = process_group
        self.sent_chunks = []
    
    async def send_chunks(self, chunks: Iterator[WeightChunk]) -> None:
        """Broadcast each chunk's tensors."""
        for chunk in chunks:
            self.sent_chunks.append(chunk)
            for tensor in chunk.tensors:
                torch.distributed.broadcast(tensor, src=0, group=self.process_group)
    
    def get_request(self) -> Dict:
        """Build legacy NamedWeightsUpdateRequest."""
        names, dtypes, shapes = [], [], []
        for chunk in self.sent_chunks:
            names.extend(chunk.names)
            dtypes.extend(chunk.dtypes)
            shapes.extend(chunk.shapes)
        return {"names": names, "dtypes": dtypes, "shapes": shapes}
```

**Refactor workers:**

```python
async def broadcast_to_inference_engines(self, inference_engine_client):
    # Phase 1: extraction with grouping/batching
    extractor = self._create_extractor()
    chunks = extractor.extract_weights(self.model)

    # Phase 3: sending
    if self.cfg.generator.weight_sync_backend == "nccl":
        sender = BroadcastWeightTransferSender(process_group=self._model_update_group)
    else:
        sender = self._create_cuda_ipc_sender()  # To be implemented in PR #3b

    await sender.send_chunks(chunks)
    request = sender.get_request()  # Legacy format

    # Old coordination
    await inference_engine_client.update_weights(**request)
```

### PR #3b: CUDA IPC Senders

**Create `skyrl_train/weight_sync/cuda_ipc_strategy.py`:**

```python
class CudaIpcWeightTransferSender(WeightTransferSender):
    """Sends weights via CUDA IPC handles."""
    
    def __init__(self):
        self.ipc_handles = {}
        self.sent_chunks = []
    
    async def send_chunks(self, chunks: Iterator[WeightChunk]) -> None:
        """Get IPC handles for each tensor."""
        for chunk in chunks:
            self.sent_chunks.append(chunk)
            for name, tensor in zip(chunk.names, chunk.tensors):
                handle = torch.multiprocessing.reductions.reduce_tensor(tensor)
                self.ipc_handles[name] = handle
        
        # Coordinate via all_gather_object
        torch.distributed.all_gather_object([self.ipc_handles], ...)
    
    def get_request(self) -> Dict:
        """Build legacy request with ipc_handles."""
        names, dtypes, shapes = [], [], []
        for chunk in self.sent_chunks:
            names.extend(chunk.names)
            dtypes.extend(chunk.dtypes)
            shapes.extend(chunk.shapes)
        return {
            "names": names,
            "dtypes": dtypes,
            "shapes": shapes,
            "ipc_handles": self.ipc_handles
        }
```

**Create `skyrl_train/weight_sync/packed_cuda_ipc_strategy.py`:**

```python
class PackedCudaIpcWeightTransferSender(WeightTransferSender):
    """Sends weights via packed CUDA IPC (Megatron optimization)."""
    
    async def send_chunks(self, chunks: Iterator[WeightChunk]) -> None:
        """Pack multiple tensors into contiguous memory."""
        all_tensors = []
        for chunk in chunks:
            all_tensors.extend(chunk.tensors)
        
        # Pack tensors
        packed_tensor = torch.cat([t.flatten() for t in all_tensors])
        self.ipc_handle = torch.multiprocessing.reductions.reduce_tensor(packed_tensor)
        self.sizes = [t.numel() for t in all_tensors]
        
        # Coordinate
        torch.distributed.all_gather_object([{"handle": self.ipc_handle, "sizes": self.sizes}], ...)
    
    def get_request(self) -> Dict:
        """Build legacy request with packed format."""
        return {
            "names": [name for chunk in self.sent_chunks for name in chunk.names],
            "dtypes": [dt for chunk in self.sent_chunks for dt in chunk.dtypes],
            "shapes": [s for chunk in self.sent_chunks for s in chunk.shapes],
            "ipc_handle": self.ipc_handle,
            "sizes": self.sizes
        }
```

**Update workers to select sender:**

```python
def _create_cuda_ipc_sender(self):
    if self.cfg.generator.pack_weights and isinstance(self.extractor, MegatronWeightExtractor):
        return PackedCudaIpcWeightTransferSender()
    else:
        return CudaIpcWeightTransferSender()
```

**Testing:**
- Integration test per backend × sender combination
- Verify Megatron packing still works
- Verify FlashRL grouped chunks are sent correctly
- Performance benchmarks (packing efficiency)

### Outcome

All components (extraction, loading, sending) now use new interfaces internally, still coordinated via old controller and legacy request format.

---

## Phase 4: Integrate Strategy Interface and Update Coordination (PR #4)
**Goal:** Wire up `WeightTransferStrategy`, introduce typed `WeightUpdateRequest` subclasses, update controller orchestration.

### Interfaces Introduced

**Add to `base.py`:**

```python
# Abstract classes
class WeightTransferReceiver(ABC):
    """Receives weight chunks on inference actors."""
    
    @abstractmethod
    async def receive_weights(self, request: WeightUpdateRequest) -> AsyncIterator[Tuple[str, torch.Tensor]]:
        """Async generator yielding (name, tensor) tuples."""
        ...

class WeightTransferStrategy(ABC):
    """Factory for creating senders, receivers, and init info."""
    
    @abstractmethod
    async def create_sender(self, ...) -> WeightTransferSender:
        """Build sender on training actors."""
        ...
    
    @abstractmethod
    async def create_receiver(self, ..., sync_info: WeightSyncInitInfo, rank_offset: int) -> WeightTransferReceiver:
        """Build receiver on inference actors."""
        ...
    
    @abstractmethod
    async def create_init_info(self, ...) -> WeightSyncInitInfo:
        """Produce initialization info on training actors."""
        ...
    
    @abstractmethod
    def get_request_type(self) -> Type[WeightUpdateRequest]:
        """Return the WeightUpdateRequest implementation."""
        ...

# Data structures
@dataclass
class WeightUpdateRequest(ABC):
    """Base class for weight update requests."""
    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]

@dataclass
class BroadcastWeightUpdateRequest(WeightUpdateRequest):
    """Request for broadcast-based weight transfer."""
    pass

@dataclass
class CudaIpcWeightUpdateRequest(WeightUpdateRequest):
    """Request for CUDA IPC-based weight transfer."""
    ipc_handles: Dict[str, Any]

@dataclass
class PackedCudaIpcWeightUpdateRequest(WeightUpdateRequest):
    """Request for packed CUDA IPC transfer."""
    ipc_handle: Any
    sizes: List[int]

@dataclass
class WeightSyncInitInfo(ABC):
    """Base class for weight sync initialization info."""
    pass

@dataclass
class TorchDistributedWeightSyncInitInfo(WeightSyncInitInfo):
    """Initialization info for torch.distributed-based strategies."""
    master_addr: str
    master_port: int
    world_size: int
    group_name: str
    backend: str

@dataclass
class EmptyWeightSyncInitInfo(WeightSyncInitInfo):
    """Empty initialization info when no shared config needed."""
    pass
```

### Complete Strategy Implementations

**Update `broadcast_strategy.py`:**

```python
class BroadcastWeightTransferReceiver(WeightTransferReceiver):
    """Receives weights via torch.distributed.broadcast."""
    
    def __init__(self, process_group):
        self.process_group = process_group
    
    async def receive_weights(self, request: BroadcastWeightUpdateRequest) -> AsyncIterator[Tuple[str, torch.Tensor]]:
        """Async generator yielding (name, tensor) tuples."""
        for name, dtype_str, shape in zip(request.names, request.dtypes, request.shapes):
            tensor = torch.empty(shape, dtype=parse_dtype(dtype_str), device="cuda")
            torch.distributed.broadcast(tensor, src=0, group=self.process_group)
            yield name, tensor

class BroadcastTransferStrategy(WeightTransferStrategy):
    """Strategy for broadcast-based weight transfer."""
    
    async def create_sender(self, process_group, ...) -> BroadcastWeightTransferSender:
        return BroadcastWeightTransferSender(process_group)
    
    async def create_receiver(self, sync_info: TorchDistributedWeightSyncInitInfo, rank_offset: int) -> BroadcastWeightTransferReceiver:
        # Initialize process group on receiver side
        process_group = init_custom_process_group(
            master_addr=sync_info.master_addr,
            master_port=sync_info.master_port,
            world_size=sync_info.world_size,
            rank=rank_offset,
            group_name=sync_info.group_name,
            backend=sync_info.backend
        )
        return BroadcastWeightTransferReceiver(process_group)
    
    async def create_init_info(self, master_addr, master_port, world_size, group_name, backend) -> TorchDistributedWeightSyncInitInfo:
        return TorchDistributedWeightSyncInitInfo(
            master_addr=master_addr,
            master_port=master_port,
            world_size=world_size,
            group_name=group_name,
            backend=backend
        )
    
    def get_request_type(self) -> Type[WeightUpdateRequest]:
        return BroadcastWeightUpdateRequest
```

**Update senders to return typed requests:**

```python
# In BroadcastWeightTransferSender
def get_request(self) -> BroadcastWeightUpdateRequest:
    names, dtypes, shapes = [], [], []
    for chunk in self.sent_chunks:
        names.extend(chunk.names)
        dtypes.extend(chunk.dtypes)
        shapes.extend(chunk.shapes)
    return BroadcastWeightUpdateRequest(names=names, dtypes=dtypes, shapes=shapes)
```

### Controller Orchestration

**Update `trainer.py`:**

```python
def _create_strategy(self) -> WeightTransferStrategy:
    """Factory based on config."""
    if self.cfg.generator.weight_sync_backend == "nccl":
        return BroadcastTransferStrategy(...)
    elif self.cfg.generator.weight_sync_backend == "cuda_ipc":
        if self.cfg.generator.pack_weights and self.cfg.trainer.training_backend == "megatron":
            return PackedCudaIpcTransferStrategy(...)
        return CudaIpcTransferStrategy(...)

def init_weight_sync_state(self):
    strategy = self._create_strategy()

    # Training actors: create sender and init info
    sync_info = ray.get(self.policy_model.async_run_ray_method(
        "pass_through", "initialize_weight_sync", strategy))

    # Inference actors: create receiver with init info
    ray.get(self.inference_engine_client.initialize_weight_sync(strategy, sync_info))
```

**Update workers:**

```python
async def initialize_weight_sync(self, strategy: WeightTransferStrategy):
    """Called at startup; all ranks participate."""
    # Rank 0 creates init info
    if torch.distributed.get_rank() == 0:
        sync_info = await strategy.create_init_info(...)
    else:
        sync_info = None

    # All ranks create sender (for process group participation)
    self.sender = await strategy.create_sender(...)
    self.extractor = self._create_extractor()  # From Phase 1

    return sync_info

async def broadcast_to_inference_engines(self, inference_engine_client):
    """Simplified - components already created."""
    chunks = self.extractor.extract_weights(self.model)  # Grouping/batching already applied
    await self.sender.send_chunks(chunks)
    request = self.sender.get_request()  # Now typed

    # New coordination
    await inference_engine_client.update_weights(request)
```

**Update inference engines:**

```python
async def initialize_weight_sync(self, strategy: WeightTransferStrategy, sync_info: WeightSyncInitInfo):
    """Called at startup."""
    self.receiver = await strategy.create_receiver(..., sync_info)
    self.loader = self._create_loader()  # From Phase 2

async def update_weights(self, request: WeightUpdateRequest):
    """Simplified - loader drives receiver."""
    await self.loader.load_weights(self.receiver, request)
```

### Remove Legacy Code

- Delete old `update_weights(names, dtypes, shapes)` signatures
- Delete `update_weights_cuda_ipc()` separate methods
- Delete `NamedWeightsUpdateRequest` TypedDict
- Remove local `_receive_weights_*()` helper methods
- Remove `_chunks_to_legacy_request()` helpers

### Testing

- Full integration test suite for all combinations
- Verify identical behavior to Phase 3
- Performance benchmarks:
  - FlashRL QKV fusion
  - Megatron packing efficiency
  - Latency/memory vs. baseline

### Outcome

Complete migration to new interface; all code paths using strategy pattern; all optimizations preserved.

---

## Summary

**Key Principles:**
1. **Interfaces on-demand** - Only introduce what's needed in each phase
2. **Optimizations preserved** - Grouping/batching/packing maintained from the start
3. **No controller changes until Phase 4** - Workers/engines refactor internally first
4. **Small, reviewable PRs** - Phases 1-3 can be 9+ separate PRs
5. **Independent testing** - Each phase validates against previous phase
6. **Low risk** - External APIs unchanged until Phase 4
7. **Co-location** - Extractors with workers, loaders with engines

**By Phase:**
- **Phase 1:** 3 PRs - Extractors for FSDP, Megatron, DeepSpeed
- **Phase 2:** 3 PRs - Loaders for vLLM, SGLang, Remote
- **Phase 3:** 2 PRs - Broadcast sender, CUDA IPC senders
- **Phase 4:** 1 PR - Strategy integration and controller update

**Total:** 9 PRs, each independently testable and reviewable.
