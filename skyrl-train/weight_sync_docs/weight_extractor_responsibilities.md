# WeightExtractor Responsibilities

## Overview

The `WeightExtractor` is responsible for extracting weights from backend-specific training models and preparing them for transfer to inference engines. Beyond grouping/batching, it handles several critical transformations.

## Current Responsibilities (Beyond Grouping/Batching)

### 1. **Backend-Specific Weight Access**

Each backend stores weights differently, requiring different access methods:

**FSDP**:
- Uses `model.state_dict()` with `StateDictType.SHARDED_STATE_DICT`
- Returns sharded parameters (DTensor or regular tensors)
- Code: `params = self.model.model.state_dict()` (line 165)

**Megatron**:
- Uses `bridge.export_hf_weights()` to convert Megatron format → HuggingFace format
- Handles tensor parallel (TP) and expert parallel (EP) sharding
- Code: `per_tensor_param = self.bridge.export_hf_weights(self.actor_module, show_progress=False)` (line 474)

**DeepSpeed**:
- Uses `model.named_parameters()` to iterate parameters
- Handles ZeRO sharding (especially ZeRO-3)
- Code: `for name, param in model.named_parameters()` (line 131)

---

### 2. **Gathering Sharded Weights**

Distributed training shards weights across GPUs. Extractors must gather them:

**FSDP**:
- DTensor parameters need `.full_tensor()` to gather shards
- Code: `param = param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param` (line 223)

**Megatron**:
- `export_hf_weights()` handles gathering across TP/EP ranks internally
- Returns full tensors in HuggingFace format

**DeepSpeed ZeRO-3**:
- Uses `GatheredParameters` context manager to allgather sharded params
- Code: `with deepspeed.zero.GatheredParameters([param], enabled=self.zero_stage == 3):` (line 183)
- For ZeRO-1/2, params are already local

---

### 3. **Device Placement**

Weights must be moved to the correct device (typically CUDA) for transfer:

**All Backends**:
- Move to `torch.cuda.current_device()`
- Code examples:
  - FSDP: `device = torch.cuda.current_device()` (line 222)
  - Megatron: `device = torch.cuda.current_device()` (line 469)
  - DeepSpeed: Implicit in `GatheredParameters` context

---

### 4. **Dtype Conversion**

Convert weights from training dtype to inference engine dtype (`generator_dtype`):

**All Backends**:
- Convert to `generator_dtype` (typically `bfloat16` or `float16` for inference)
- Code examples:
  - FSDP: `param = param.to(generator_dtype)` (line 224)
  - Megatron: `tensor.to(device=device, dtype=generator_dtype)` (line 510)
  - DeepSpeed: `weight = weight.to(generator_dtype)` (line 185)

**Why**: Training may use `float32` for precision, but inference engines use lower precision for speed/memory.

---

### 5. **Memory Layout Optimization**

Prepare tensors for efficient transfer:

**Contiguous Memory**:
- FSDP: `weight = param.detach().contiguous()` (line 225)
- Ensures tensor is in contiguous memory layout for IPC handles

**Packing (Megatron CUDA IPC)**:
- Packs multiple tensors into single contiguous buffer
- Code: `packed_tensor[offset : offset + size].copy_(tensor.detach().view(-1))` (line 525)
- Reduces IPC handle overhead (one handle instead of many)

---

### 6. **Format Conversion (Megatron)**

**Megatron → HuggingFace Format**:
- Megatron uses custom parameter layout (TP/EP sharded)
- Must convert to HuggingFace format for inference engines
- Code: `self.bridge.export_hf_weights(self.actor_module, ...)` (line 474, 503)
- Handles tensor parallel and expert parallel unsharding

---

### 7. **Shape Handling (DeepSpeed ZeRO-3)**

**ZeRO-3 Shape Metadata**:
- ZeRO-3 shards parameters, so `param.shape` may not reflect full shape
- Must use `param.ds_shape` for ZeRO-3
- Code: `shape = param.shape if self.zero_stage != 3 else param.ds_shape` (line 197)

---

### 8. **Module Grouping (FlashRL)**

**Group Parameters by Module**:
- Groups related parameters (e.g., all attention weights together)
- Needed for FlashRL's fused QKV allocation
- Code: `module_name = ".".join(param_name.split(".")[:-2])` (line 204, 168)
- Creates `module_to_params` dictionary

**Why**: FlashRL allocates new storage per parameter, but vLLM fuses q/k/v weights. Grouping ensures proper fused allocation.

---

### 9. **LoRA Handling (FSDP)**

**LoRA-Specific Logic**:
- Detects LoRA models: `if self._is_lora:`
- Uses different sync path: `_save_lora_adapters_and_sync()`
- Skips regular weight extraction for LoRA adapters
- Code: Lines 156-162

---

### 10. **State Dict Configuration (FSDP)**

**FSDP State Dict Type**:
- Sets `StateDictType.SHARDED_STATE_DICT` for FSDP v1
- Ensures proper sharded state dict access
- Code: `FSDP.set_state_dict_type(self.model.model, state_dict_type=StateDictType.SHARDED_STATE_DICT, ...)` (lines 147-151)

---

## Summary Table

| Responsibility | FSDP | Megatron | DeepSpeed | Notes |
|---------------|------|----------|-----------|-------|
| **Weight Access** | `state_dict()` | `export_hf_weights()` | `named_parameters()` | Backend-specific |
| **Gathering** | `.full_tensor()` for DTensor | Handled by `export_hf_weights()` | `GatheredParameters` (ZeRO-3) | Unshard distributed weights |
| **Device Placement** | ✅ CUDA | ✅ CUDA | ✅ CUDA | Move to GPU |
| **Dtype Conversion** | ✅ `to(generator_dtype)` | ✅ `to(generator_dtype)` | ✅ `to(generator_dtype)` | Training → Inference dtype |
| **Contiguous Memory** | ✅ `.contiguous()` | ✅ Packed buffer | ✅ `.clone()` | Memory layout |
| **Format Conversion** | N/A | ✅ Megatron → HF | N/A | Megatron-specific |
| **Shape Handling** | N/A | N/A | ✅ `ds_shape` (ZeRO-3) | ZeRO-3-specific |
| **Module Grouping** | ✅ (CUDA IPC) | ❌ (uses buckets) | ✅ (CUDA IPC) | FlashRL requirement |
| **Packing** | ❌ | ✅ (CUDA IPC) | ❌ | Megatron optimization |
| **LoRA Handling** | ✅ Special path | ❌ | ❌ | FSDP-specific |
| **State Dict Config** | ✅ `SHARDED_STATE_DICT` | N/A | N/A | FSDP v1-specific |

---

## Key Insights

1. **Backend Diversity**: Each backend requires different extraction logic due to different sharding strategies and formats.

2. **Sharding Complexity**: Gathering sharded weights is a major responsibility - FSDP (DTensor), Megatron (TP/EP), DeepSpeed (ZeRO-1/2/3) all handle it differently.

3. **Format Conversion**: Megatron uniquely requires format conversion (Megatron → HuggingFace), adding complexity.

4. **Memory Optimization**: Contiguous memory and packing (Megatron) optimize for IPC transfer efficiency.

5. **Dtype Handling**: Always converts to inference dtype, which may differ from training dtype.

6. **Special Cases**: LoRA (FSDP), ZeRO-3 shapes (DeepSpeed), FlashRL grouping add backend-specific complexity.

---

## Design Implications

When designing the `WeightExtractor` interface:

1. **Must handle backend-specific access patterns** (state_dict vs named_parameters vs export_hf_weights)
2. **Must handle sharding** (gathering logic varies by backend)
3. **Must handle dtype conversion** (training → inference dtype)
4. **Must handle device placement** (move to CUDA)
5. **Must handle memory layout** (contiguous, packing)
6. **Must handle format conversion** (Megatron → HF)
7. **Must handle grouping/batching** (module grouping, threshold batching)
8. **Must handle special cases** (LoRA, ZeRO-3 shapes, FlashRL)

The extractor is the **translation layer** between backend-specific training formats and transfer-ready inference formats.
