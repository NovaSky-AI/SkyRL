# Step 1: initialization

* Worker.init_weight_sync_state
  * Inference_engine_client.init_weight_update_communicator -> broadcast metadata
  * Init_custom_process_group
* InferenceEngine init_weight_update_communicator
  * receives metadata ->
  * vllm: init_custom_process_group, sglang: self.engine.tokenizer_manager.init_weights_update_group(obj, None)


```
  What happens

  Sets up a distributed process group connecting:
  • Training rank 0 (policy model)
  • All inference engine ranks (vLLM/SGLang workers)


  Code flow

  1. Training side (rank 0 only):
    • Gets master address (node IP)
    • Binds to a random port
    • Calculates world size: 1 (training) + num_inference_engines * tp_size * pp_size * dp_size
    • Calls inference_engine_client.init_weight_update_communicator() (async)
    • Calls init_custom_process_group() (async) to join as rank 0
    • Stores _model_update_group for later broadcasts
  2. Inference engine side:
    • Each engine receives master_addr, master_port, rank_offset, world_size, group_name, backend
    • Calculates its rank: torch.distributed.get_rank() + rank_offset
    • Joins the process group via init_custom_process_group()
    • Stores _model_update_group


  Differences across backends

  Policy backends (FSDP/Megatron/DeepSpeed):
  • All inherit init_weight_sync_state from base Worker class
  • Same implementation
  • Some set self.use_cuda_ipc = True during __init__ if:
    • weight_sync_backend == "nccl" AND
    • colocate_all == True
  • Megatron also initializes weight conversion tasks/buckets for CUDA IPC

  Inference engines:
  • vLLM: Direct init_custom_process_group() call
  • SGLang: Uses SGLang's tokenizer_manager.init_weights_update_group()
  • Remote: HTTP POST to /init_weight_update_communicator or /init_weights_update_group


  Common vs different

  Common:
  • Process group creation
  • Master address/port discovery
  • World size calculation
  • Rank assignment logic

  Different:
  • How inference engines join (direct vs HTTP vs SGLang API)
  • CUDA IPC setup (only for colocated NCCL)
  • Megatron-specific bucket preparation

  Questions

  1. Does this match your understanding?
  2. For abstraction, should we standardize how inference engines join (direct vs HTTP vs SGLang API)?
  3. Should CUDA IPC detection be part of the abstraction or remain backend-specific?
```


Step 2: Weight gathering & Preparation

PolicyWorker.broadcast_to_inference_engines

1. common preparation

```
  Common setup (all backends)

  1. Prefix cache reset (if enabled): async task to reset prefix cache
    • Code: skyrl_train/workers/fsdp/fsdp_worker.py:141-143, skyrl_train/workers/megatron/megatron_worker.py:463-465,
      skyrl_train/workers/deepspeed/deepspeed_worker.py:124-126
  2. CUDA cache clearing: torch.cuda.empty_cache()
    • Code: skyrl_train/workers/fsdp/fsdp_worker.py:145, skyrl_train/workers/megatron/megatron_worker.py:467,
      skyrl_train/workers/deepspeed/deepspeed_worker.py:128
  3. Dtype conversion: convert to generator.model_dtype
    • Code: skyrl_train/workers/fsdp/fsdp_worker.py:139, skyrl_train/workers/megatron/megatron_worker.py:461,
      skyrl_train/workers/deepspeed/deepspeed_worker.py:122
  4. LoRA check: if LoRA, handle separately (FSDP only)
    • Code: skyrl_train/workers/fsdp/fsdp_worker.py:156-162
```

2. get params state_dict

3.1. broadcast - non-cuda ipc path
  1 broadcast schema: inference_engine_client.update_named_weights
  2 torch.distributed.broadcast

3.2 broadcast - cuda ipc path
```
  • Group parameters by module name
    • Code: skyrl_train/workers/fsdp/fsdp_worker.py:201-208, skyrl_train/workers/deepspeed/deepspeed_worker.py:164-172
  • Create IPC handles per parameter (or bucket)
    • Code: skyrl_train/workers/fsdp/fsdp_worker.py:220-226, skyrl_train/workers/megatron/megatron_worker.py:532,
      skyrl_train/workers/deepspeed/deepspeed_worker.py:186
  • Batch IPC handles until threshold reached
    • Code: skyrl_train/workers/fsdp/fsdp_worker.py:244-247, skyrl_train/workers/megatron/megatron_worker.py:544-545,
      skyrl_train/workers/deepspeed/deepspeed_worker.py:210-213
  • Use all_gather_object to share handles across ranks
    • Code: skyrl_train/workers/fsdp/fsdp_worker.py:229-230, skyrl_train/workers/megatron/megatron_worker.py:534-535,
      skyrl_train/workers/deepspeed/deepspeed_worker.py:189-190
```
batches are pushed via inference_engine_client.update_named_weights(weights_update_request)


```
  CUDA IPC path differences

  All backends follow a similar pattern but differ in gathering:
  1. FSDP: param.full_tensor() if DTensor
    • Code: skyrl_train/workers/fsdp/fsdp_worker.py:223
  2. Megatron: Uses buckets from get_conversion_tasks(); can pack multiple tensors into one IPC handle
    • Code: skyrl_train/workers/megatron/megatron_worker.py:285 (bucket initialization),
      skyrl_train/workers/megatron/megatron_worker.py:502-542 (packed tensor creation)
  3. DeepSpeed: GatheredParameters context for ZeRO-3
    • Code: skyrl_train/workers/deepspeed/deepspeed_worker.py:183

  All CUDA IPC paths:
  • Group parameters by module name
    • Code: skyrl_train/workers/fsdp/fsdp_worker.py:201-208, skyrl_train/workers/deepspeed/deepspeed_worker.py:164-172
  • Create IPC handles per parameter (or bucket)
    • Code: skyrl_train/workers/fsdp/fsdp_worker.py:220-226, skyrl_train/workers/megatron/megatron_worker.py:532,
      skyrl_train/workers/deepspeed/deepspeed_worker.py:186
  • Batch IPC handles until threshold reached
    • Code: skyrl_train/workers/fsdp/fsdp_worker.py:244-247, skyrl_train/workers/megatron/megatron_worker.py:544-545,
      skyrl_train/workers/deepspeed/deepspeed_worker.py:210-213
  • Use all_gather_object to share handles across ranks
    • Code: skyrl_train/workers/fsdp/fsdp_worker.py:229-230, skyrl_train/workers/megatron/megatron_worker.py:534-535,
      skyrl_train/workers/deepspeed/deepspeed_worker.py:189-190


  Common vs different

  Common:
  • Prefix cache reset
    • Code: skyrl_train/workers/fsdp/fsdp_worker.py:141-143, skyrl_train/w
      skyrl_train/workers/deepspeed/deepspeed_worker.py:124-126
  • Dtype conversion
    • Code: skyrl_train/workers/fsdp/fsdp_worker.py:188, skyrl_train/worke
      skyrl_train/workers/deepspeed/deepspeed_worker.py:150
  • Iterating through parameters
    • Code: skyrl_train/workers/fsdp/fsdp_worker.py:168, skyrl_train/worke
      skyrl_train/workers/deepspeed/deepspeed_worker.py:131
  • Creating update requests with names/dtypes/shapes
    • Code: skyrl_train/workers/fsdp/fsdp_worker.py:172-179, skyrl_train/w
      skyrl_train/workers/deepspeed/deepspeed_worker.py:135-142

  Different:
  • Weight extraction method:
    • FSDP: state_dict() → handles DTensor
      • Code: skyrl_train/workers/fsdp/fsdp_worker.py:165
    • Megatron: bridge.export_hf_weights() → format conversion
      • Code: skyrl_train/workers/megatron/megatron_worker.py:474
    • DeepSpeed: named_parameters() → ZeRO-3 gathering
      • Code: skyrl_train/workers/deepspeed/deepspeed_worker.py:131

  Questions for abstraction

  1. Should weight extraction be abstracted into a common interface (e.g., get_weights_for_sync()) that each backend
     implements?
  2. Should sharding handling be part of the abstraction, or remain backend-specific?
  3. For CUDA IPC, should batching/packing be abstracted or backend-specific?
```


```

  Step 3: Weight Transfer (Broadcast/IPC)


  What happens

  Weights are transferred from training rank 0 to all inference engine ranks using either NCCL broadcast or CUDA IPC.

  Two transfer mechanisms


  Mechanism A: NCCL Broadcast (standard path)

  Used when:
  • use_cuda_ipc == False (non-colocated or Gloo backend)
  • Remote inference engines (CUDA IPC not supported)

  Flow:
  1. Training side: Broadcast via process group
    • Code: skyrl_train/workers/fsdp/fsdp_worker.py:189-190 (FSDP), skyrl_train/workers/megatron/megatron_worker.py:494
      (Megatron), skyrl_train/workers/deepspeed/deepspeed_worker.py:151 (DeepSpeed)
    • Training rank 0 broadcasts: torch.distributed.broadcast(param.data, 0, group=self._model_update_group)
    • All ranks participate (barrier after each param)
    • Code: skyrl_train/workers/fsdp/fsdp_worker.py:195, skyrl_train/workers/megatron/megatron_worker.py:499,
      skyrl_train/workers/deepspeed/deepspeed_worker.py:156
  2. Inference engine side: Receive and load
    • vLLM: Creates empty tensor, receives via broadcast, loads weights
      • Code: skyrl_train/inference_engines/vllm/vllm_engine.py:115-127
      • Code: skyrl_train/inference_engines/vllm/vllm_engine.py:121-122 (broadcast receive),
        skyrl_train/inference_engines/vllm/vllm_engine.py:125 (load weights)
    • SGLang: Uses SGLang's distributed update API
      • Code: skyrl_train/inference_engines/sglang/sglang_engine.py:318-323
    • Remote: HTTP POST to /update_weights or /update_weights_from_distributed
      • Code: skyrl_train/inference_engines/remote_inference_engine.py:208-216


  Mechanism B: CUDA IPC (optimized path)

  Used when:
  • use_cuda_ipc == True (colocated + NCCL backend)
  • Local inference engines only

  Flow:
  1. Training side: Create IPC handles
    • FSDP/DeepSpeed: Per-parameter IPC handles
      • Code: skyrl_train/workers/fsdp/fsdp_worker.py:220-226
      • Code: skyrl_train/workers/fsdp/fsdp_worker.py:226 (reduce_tensor creates handle),
        skyrl_train/workers/fsdp/fsdp_worker.py:229-230 (all_gather_object shares handles)
    • Megatron: Can pack multiple tensors into one IPC handle
      • Code: skyrl_train/workers/megatron/megatron_worker.py:513-542
      • Code: skyrl_train/workers/megatron/megatron_worker.py:532 (single IPC handle for packed tensor)
  2. Training side: Send IPC handles via async request
    • Code: skyrl_train/workers/fsdp/fsdp_worker.py:248 (FSDP), skyrl_train/workers/megatron/megatron_worker.py:545
      (Megatron), skyrl_train/workers/deepspeed/deepspeed_worker.py:214 (DeepSpeed)
    • Batched by size threshold: weight_transfer_threshold_cuda_ipc_GB
    • Code: skyrl_train/workers/fsdp/fsdp_worker.py:244-247, skyrl_train/workers/megatron/megatron_worker.py:544-545,
      skyrl_train/workers/deepspeed/deepspeed_worker.py:210-213
  3. Inference engine client: Route to engines
    • Code: skyrl_train/inference_engines/inference_engine_client.py:383-384 (routes to all engines)
    • Code: skyrl_train/inference_engines/vllm/vllm_engine.py:395-407 (vLLM checks for IPC handles)
  4. Inference engine side: Open IPC handles and load
    • vLLM: Opens handles, reconstructs tensors, loads weights
      • Code: skyrl_train/inference_engines/vllm/vllm_engine.py:129-186
      • Code: skyrl_train/inference_engines/vllm/vllm_engine.py:153-158 (packed),
        skyrl_train/inference_engines/vllm/vllm_engine.py:174-180 (unpacked),
        skyrl_train/inference_engines/vllm/vllm_engine.py:183 (load weights)
    • SGLang: Serializes request, uses custom weight loader
      • Code: skyrl_train/inference_engines/sglang/sglang_engine.py:280-312
      • Code: skyrl_train/inference_engines/sglang/sglang_engine.py:288-311 (serialization + custom loader)


  Differences across inference engines


  vLLM

  • Broadcast: Direct torch.distributed.broadcast call
    • Code: skyrl_train/inference_engines/vllm/vllm_engine.py:115-127
  • CUDA IPC: Supports both packed and unpacked tensors
    • Code: skyrl_train/inference_engines/vllm/vllm_engine.py:129-186
  • Uses collective_rpc for distributed calls
    • Code: skyrl_train/inference_engines/vllm/vllm_engine.py:396-414


  SGLang

  • Broadcast: Uses SGLang's update_weights_from_distributed API
    • Code: skyrl_train/inference_engines/sglang/sglang_engine.py:318-323
  • CUDA IPC: Serializes request into tensor, uses custom weight loader
    • Code: skyrl_train/inference_engines/sglang/sglang_engine.py:280-312
  • Uses tokenizer_manager API
    • Code: skyrl_train/inference_engines/sglang/sglang_engine.py:311


  Remote

  • Broadcast only: HTTP POST to /update_weights or /update_weights_from_distributed
    • Code: skyrl_train/inference_engines/remote_inference_engine.py:184-216
  • No CUDA IPC support
    • Code: skyrl_train/inference_engines/remote_inference_engine.py:192-195
  • Backend-specific endpoints
    • Code: skyrl_train/inference_engines/remote_inference_engine.py:196-201


  Common vs different

  Common:
  • Process group usage for broadcast
    • Code: skyrl_train/workers/fsdp/fsdp_worker.py:190, skyrl_train/inference_engines/vllm/vllm_engine.py:122
  • Barrier synchronization after each parameter/batch
    • Code: skyrl_train/workers/fsdp/fsdp_worker.py:195, skyrl_train/workers/fsdp/fsdp_worker.py:260
  • Final weight loading into model
    • Code: skyrl_train/inference_engines/vllm/vllm_engine.py:125, skyrl_train/inference_engines/sglang/sglang_engine.py:323

  Different:
  • Transfer mechanism selection:
    • Training: Based on use_cuda_ipc flag
      • Code: skyrl_train/workers/fsdp/fsdp_worker.py:167 (check), skyrl_train/workers/fsdp/fsdp_worker.py:197 (else branch)
    • Inference: Based on presence of ipc_handles in request
      • Code: skyrl_train/inference_engines/vllm/vllm_engine.py:395
  • IPC handle creation:
    • FSDP/DeepSpeed: Per-parameter handles
      • Code: skyrl_train/workers/fsdp/fsdp_worker.py:226
    • Megatron: Can pack multiple tensors
      • Code: skyrl_train/workers/megatron/megatron_worker.py:513-542
  • Inference engine API:
    • vLLM: Direct PyTorch calls + collective_rpc
      • Code: skyrl_train/inference_engines/vllm/vllm_engine.py:115-186
    • SGLang: SGLang-specific APIs
      • Code: skyrl_train/inference_engines/sglang/sglang_engine.py:274-323
    • Remote: HTTP endpoints
      • Code: skyrl_train/inference_engines/remote_inference_engine.py:184-216


  Questions for abstraction

  1. Should transfer mechanism selection (broadcast vs IPC) be abstracted into a strategy pattern?
  2. Should IPC handle creation be abstracted, or remain backend-specific?
  3. Should inference engine weight loading be abstracted, or remain engine-specific?
  4. How should remote engines be handled in the abstraction (HTTP vs direct calls)?
```
