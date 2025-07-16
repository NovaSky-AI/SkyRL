import os
import random
from collections import defaultdict
from datetime import timedelta
from typing import List, Union, Optional
from jaxtyping import Float

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload, MixedPrecision
import torch.distributed.checkpoint as dcp

from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    get_state_dict,
    StateDictOptions,
)
from skyrl_train.distributed.strategy import DistributedStrategy
from skyrl_train.models import Actor
from skyrl_train.distributed.utils import ModelOrModelOptimPair
from skyrl_train.distributed.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    init_fn,
    get_fsdp_wrap_policy,
    PrecisionType,
    create_device_mesh,
    fsdp2_clip_grad_norm_,
    fsdp2_get_full_state_dict,
    apply_fsdp2,
    get_sharding_strategy,
    get_constant_schedule_with_warmup,
    offload_fsdp_model_to_cpu,
    load_fsdp_model_to_gpu,
    offload_fsdp_optimizer,
    load_fsdp_optimizer,
    get_fsdp_state_ctx,
    fsdp_version,
    fsdp2_load_full_state_dict,
)

from packaging import version

if version.parse(torch.__version__) >= version.parse("2.6"):
    from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy
elif version.parse(torch.__version__) >= version.parse("2.4"):
    from torch.distributed._composable.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy
else:
    CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy = None, None, None


class FSDPStrategy(DistributedStrategy):
    """
    The strategy for training with FSDP.
    """

    def __init__(
        self,
        fsdp_config,
        optimizer_config=None,
        fsdp_strategy: str = "fsdp",
        seed: int = 42,
        micro_train_batch_size_per_gpu=1,
        train_batch_size=1,
    ) -> None:
        super().__init__()
        assert fsdp_strategy in ("fsdp", "fsdp2"), f"Unsupported FSDP strategy: {fsdp_strategy}"
        self.fsdp_config = fsdp_config
        self.optimizer_config = optimizer_config
        self.fsdp_strategy = fsdp_strategy
        self.max_norm = optimizer_config.max_grad_norm if optimizer_config is not None else 1.0
        self.train_batch_size = train_batch_size
        self.micro_train_batch_size_per_gpu = micro_train_batch_size_per_gpu
        self.seed = seed
        self.device_mesh = None

        # if we are using fsdp 1 or cpu offload is off for fsdp2, then we need to manually offload weights/optimizer to cpu
        self.manual_offload = self.fsdp_strategy == "fsdp" or not self.fsdp_config.get("cpu_offload")
        if self.optimizer_config is not None:
            self.manual_offload_optimizer = (
                self.optimizer_config.get("offload_after_step", True) and self.manual_offload
            )
        else:
            self.manual_offload_optimizer = False

        self.time_steps = defaultdict(int)

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_distributed(self, timeout=timedelta(minutes=30)) -> None:
        self.set_seed(self.seed)

        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank != -1:
            torch.cuda.set_device(local_rank)

        # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        self.world_size = dist.get_world_size()
        self.accumulated_gradient = self.train_batch_size // self.micro_train_batch_size_per_gpu // self.world_size

        self.device_mesh = create_device_mesh(world_size=self.world_size, fsdp_size=self.fsdp_config.fsdp_size)

    def offload_to_cpu(self, model, optimizer, pin_memory=True, non_blocking=True):
        """
        Offload model weights and optimizer to CPU memory.

        For all cases except fsdp2 with cpu_offload=True, we need to manually offload weights/optimizer to cpu.
        """
        if isinstance(model, Actor):
            model = model.model
        else:
            model = model

        if self.manual_offload:
            print(f"TGRIGGS_MEM: offload_to_cpu with manual_offload=True")
            offload_fsdp_model_to_cpu(model, empty_cache=True)

            if optimizer is not None and self.manual_offload_optimizer:
                print(f"TGRIGGS_MEM: offload_fsdp_optimizer with manual_offload_optimizer=True")
                offload_fsdp_optimizer(optimizer)
        else:
            print(f"TGRIGGS_MEM: offload_to_cpu with manual_offload=False")

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def backload_to_gpu(self, model, optimizer, non_blocking=True):
        """Reload model weights back to GPU."""
        if isinstance(model, Actor):
            model = model.model
        else:
            model = model

        # if we are using fsdp 1 or cpu offload is off for fsdp2, then we need to manually backload weights/optimizer to gpu
        if self.manual_offload:
            load_fsdp_model_to_gpu(model)
            if optimizer is not None and self.manual_offload_optimizer:
                load_fsdp_optimizer(optimizer, torch.cuda.current_device())

        torch.cuda.synchronize()

    def backward(self, loss: torch.Tensor, model, optimizer: optim.Optimizer, **kwargs) -> None:
        """Perform backward pass"""
        loss.backward()

    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model,
        scheduler,
        name="model",
        **kwargs,
    ) -> Optional[Float[torch.Tensor, "1"]]:
        """Perform optimizer step"""
        grad_norm = None
        if isinstance(model, Actor):
            model = model.model

        if self.max_norm > 0:
            # NOTE (sumanthrh): All `grad_norm`s returned here are the original grad norms before clipping.
            if isinstance(model, FSDP):
                grad_norm = model.clip_grad_norm_(max_norm=self.max_norm)
            elif isinstance(model, FSDPModule):
                grad_norm = fsdp2_clip_grad_norm_(model.parameters(), max_norm=self.max_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.max_norm)

        # Skip update if gradient norm is not finite
        if grad_norm is not None and not torch.isfinite(grad_norm):
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                print(f"WARN: rank {rank} grad_norm is not finite: {grad_norm}")
            else:
                print(f"WARN: grad_norm is not finite: {grad_norm}")
            optimizer.zero_grad()
            return grad_norm

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
        return grad_norm

    def prepare(
        self, *models_or_model_optim_pairs: ModelOrModelOptimPair
    ) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        """Prepare models and optimizers with FSDP"""
        ret = []
        for arg in models_or_model_optim_pairs:
            if isinstance(arg, tuple):
                assert len(arg) == 3, f'Expect (model, optimizer, scheduler) pair, got a tuple with size "{len(arg)}"'
                ret.append(self._fsdp_init_train_model(*arg))
            else:
                ret.append(self._fsdp_init_eval_model(arg))

        return ret[0] if len(ret) == 1 else ret

    def _fsdp_init_model(self, model, is_train=True, is_actor=False):
        # Initialize FSDP wrapping policy
        wrap_policy = get_fsdp_wrap_policy(module=model, config=self.fsdp_config.get("wrap_policy", None))

        # Setup mixed precision
        mixed_precision_config = self.fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get("param_dtype", "bf16"))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get("reduce_dtype", "fp32"))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get("buffer_dtype", "fp32"))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        cpu_offload = None

        # sharding strategy
        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # Wrap model with FSDP
        if self.fsdp_strategy == "fsdp":
            # cpu offloading will always be none for models that train with FSDP due to correctness issues with gradient accumulation -
            # see https://docs.pytorch.org/docs/stable/fsdp.html
            if not is_train and self.fsdp_config.get("cpu_offload", False):
                cpu_offload = CPUOffload(offload_params=True)
            fsdp_module = FSDP(
                model.model if is_actor else model,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=wrap_policy,
                device_id=torch.cuda.current_device(),
                sharding_strategy=sharding_strategy,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif self.fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=True
            )
            if self.fsdp_config.get("cpu_offload", False):
                cpu_offload = CPUOffloadPolicy(pin_memory=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": self.fsdp_config.get("reshard_after_forward", True),
            }
            actor_module = model.model if is_actor else model
            full_state = actor_module.state_dict()
            actor_module = apply_fsdp2(actor_module, fsdp_kwargs, self.fsdp_config)
            actor_module = fsdp2_load_full_state_dict(actor_module, full_state, cpu_offload)
            fsdp_module = actor_module
        else:
            raise NotImplementedError(f"{self.fsdp_strategy} not implemented")

        return fsdp_module

    def _fsdp_init_train_model(self, model, optimizer, scheduler):
        """Initialize a model for training with FSDP"""
        is_actor = isinstance(model, Actor)
        fsdp_module = self._fsdp_init_model(model, is_train=True, is_actor=is_actor)

        optim_config = self.optimizer_config
        if optim_config is not None:
            actor_optimizer = optim.AdamW(
                fsdp_module.parameters(),
                lr=optim_config.lr,
                betas=optim_config.adam_betas,
                weight_decay=optim_config.weight_decay,
            )

            #  TODO(csy): add other schedulers, add more to config
            actor_lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=actor_optimizer, num_warmup_steps=optim_config.num_warmup_steps
            )
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        if is_actor:
            model.model = fsdp_module
        else:
            model = fsdp_module
            
        print("wrapped type:", type(model.model))
        print("FSDP modules:", FSDP.fsdp_modules(model.model))

        return model, actor_optimizer, actor_lr_scheduler

    def _fsdp_init_eval_model(self, model):
        """Initialize a model for evaluation with FSDP"""
        is_actor = isinstance(model, Actor)
        fsdp_module = self._fsdp_init_model(model, is_train=False, is_actor=is_actor)

        if is_actor:
            model.model = fsdp_module
        else:
            model = fsdp_module

        return model

    def all_reduce(self, data, op="mean"):
        """Perform all_reduce across all processes"""
        assert op in ("mean", "max", "sum")
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_reduce(v, op)
            return ret
        else:
            is_tensor = True
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
                is_tensor = False
            is_cpu_tensor = data.device.type == "cpu"

            if is_cpu_tensor:
                data = data.to(torch.cuda.current_device())
            if op == "mean":
                data /= self.world_size
            dist.all_reduce(data, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM)
            if is_cpu_tensor:
                data = data.cpu()
            return data.item() if not is_tensor else data

    def all_gather(self, data):
        """Perform all_gather across all processes"""
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_gather(v)
            return ret
        else:
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
            is_cpu_tensor = data.device.type == "cpu"

            ret = [torch.zeros_like(data).to(torch.cuda.current_device()) for _ in range(self.world_size)]
            dist.all_gather(ret, data.to(torch.cuda.current_device()))
            return torch.cat(ret).cpu() if is_cpu_tensor else torch.cat(ret)

    def _unwrap_model(self, model) -> nn.Module:
        """Unwrap model from Actor or FSDP"""
        # Handle Actor wrapper
        if isinstance(model, Actor):
            return self._unwrap_model(model.model)

        # For FSDP2 models, check if the FSDP model itself has the necessary attributes
        model_type = type(model).__name__
        if "FSDP" in model_type:
            has_config = hasattr(model, "config")
            has_lm_head = hasattr(model, "lm_head")
            has_generate = hasattr(model, "generate")
            if has_config and (has_lm_head or has_generate):
                return model

        # Check for FSDP v1 unwrapping
        if hasattr(model, "_fsdp_wrapped_module"):
            return model._fsdp_wrapped_module

        # If no unwrapping needed, return the original model
        return model

    def save_ckpt(
        self,
        model,
        ckpt_dir,
        global_step,
        node_local_rank,
        optimizer=None,
        scheduler=None,
        client_state={},
        tag=None,
    ):
        """Save model checkpoint for FSDP"""
        import warnings
        from torch.distributed.fsdp import ShardedStateDictConfig, ShardedOptimStateDictConfig, StateDictType

        if node_local_rank == 0:
            os.makedirs(ckpt_dir, exist_ok=True)

        # Wait for checkpoint directory to be created.
        dist.barrier()

        # Extract the actual model for saving
        save_model = model
        if isinstance(model, Actor):
            print(f"TGRIGGS_MEM: saving an Actor model")
            save_model = model.model
        else:
            print(f"TGRIGGS_MEM: saving a non-Actor model")

        if self.fsdp_strategy not in ("fsdp", "fsdp2"):
            raise ValueError(f"Unsupported FSDP strategy: {self.fsdp_strategy}")

        # Set up state dict configurations for sharded saving
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)
        print("TGRIGGS: save_model type:", type(save_model))
        
        print("TGRIGGS: FSDP modules:", FSDP.fsdp_modules(save_model))
        print("TGRIGGS: Sub-modules:", [k for k, _ in save_model.named_modules()])
        
        # DEBUG: Check parameter ID alignment
        if optimizer is not None:
            print(f"TGRIGGS_DEBUG: Checking parameter ID alignment...")
            
            # Get model parameter IDs
            model_param_ids = set(id(p) for p in save_model.parameters())
            print(f"TGRIGGS_DEBUG: Model has {len(model_param_ids)} parameters")
            
            # Get optimizer parameter IDs
            optim_param_ids = set()
            for group in optimizer.param_groups:
                for param in group['params']:
                    optim_param_ids.add(id(param))
            print(f"TGRIGGS_DEBUG: Optimizer has {len(optim_param_ids)} parameters")
            
            # Check for mismatches
            missing_in_model = optim_param_ids - model_param_ids
            missing_in_optim = model_param_ids - optim_param_ids
            
            if missing_in_model:
                print(f"TGRIGGS_DEBUG: ERROR - {len(missing_in_model)} optimizer params not found in model!")
                print(f"TGRIGGS_DEBUG: Missing param IDs: {list(missing_in_model)[:10]}...")  # Show first 10
                
            if missing_in_optim:
                print(f"TGRIGGS_DEBUG: WARNING - {len(missing_in_optim)} model params not found in optimizer")
                
            if not missing_in_model and not missing_in_optim:
                print(f"TGRIGGS_DEBUG: ✅ All parameter IDs match perfectly")
            
            # Additional debug: check if this is the FSDP2 case
            print(f"TGRIGGS_DEBUG: FSDP strategy: {self.fsdp_strategy}")
            print(f"TGRIGGS_DEBUG: Model type: {type(save_model).__name__}")
            print(f"TGRIGGS_DEBUG: Optimizer type: {type(optimizer).__name__}")
        
        model_shard, optim_shard = get_state_dict(
            save_model,
            [optimizer] if optimizer is not None else [],
            options=StateDictOptions(full_state_dict=False, cpu_offload=True)
        )

        # --- 3) Save via DCP; each rank writes its own file under `ckpt_dir` ---
        rank = self.get_rank()
        world_size = self.world_size
        optim_path = os.path.join(ckpt_dir, f"optim_world_size_{world_size}_rank_{rank}.pt")
        dcp.save(
            {"model": model_shard, "optim": optim_shard},
            checkpoint_id=optim_path,
        )




        # Define paths for saving individual rank files
        rank = self.get_rank()
        world_size = self.world_size
        model_path = os.path.join(ckpt_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        optim_path = os.path.join(ckpt_dir, f"optim_world_size_{world_size}_rank_{rank}.pt")
        extra_path = os.path.join(ckpt_dir, f"extra_state_world_size_{world_size}_rank_{rank}.pt")

        # Save using appropriate FSDP context
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with get_fsdp_state_ctx(save_model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
            # with FSDP.state_dict_type(save_model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                # Get and save model state dict
                # model_state_dict = save_model.state_dict()
                # self.print(f"[rank-{rank}]: Saving model to {os.path.abspath(model_path)}")
                # torch.save(model_state_dict, model_path)

                # del model_state_dict
                
                # Get and save optimizer state dict if optimizer is provided
                # if optimizer is not None:
                #     model_shard, optim_shard = get_state_dict(
                #         save_model,
                #         optimizer,
                #         options=StateDictOptions(full_state_dict=False, cpu_offload=True)
                #     )

                #     # --- 3) Save via DCP; each rank writes its own file under `ckpt_dir` ---
                #     dcp.save(
                #         {"model": model_shard, "optim": optim_shard},
                #         checkpoint_id=optim_path,
                #     )
                    
                    # with FSDP.state_dict_type(
                    #     model,
                    #     StateDictType.SHARDED_STATE_DICT,
                    #     optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=False)
                    # ):
                    # optimizer_state_dict = FSDP.optim_state_dict(model, optimizer)
                    # self.print(f"[rank-{rank}]: Saving optim to {os.path.abspath(optim_path)}")
                    # torch.save(optimizer_state_dict, optim_path)
                    # del optimizer_state_dict
                        
                        # def deep_debug_optimizer_state(optimizer, stage_name):
                        #     """Comprehensive debugging of optimizer state"""
                        #     print(f"\n=== OPTIMIZER STATE DEBUG - {stage_name} ===")
                            
                        #     # 1. Basic optimizer info
                        #     print(f"Optimizer type: {type(optimizer).__name__}")
                        #     print(f"Optimizer state keys: {list(optimizer.state.keys())}")
                        #     print(f"Number of param groups: {len(optimizer.param_groups)}")
                            
                        #     # 2. Check all attributes of optimizer
                        #     optimizer_attrs = [attr for attr in dir(optimizer) if not attr.startswith('_')]
                        #     print(f"Optimizer attributes: {optimizer_attrs}")
                            
                        #     # 3. Check for FSDP-specific attributes
                        #     fsdp_attrs = [attr for attr in dir(optimizer) if 'fsdp' in attr.lower()]
                        #     if fsdp_attrs:
                        #         print(f"FSDP-related attributes: {fsdp_attrs}")
                            
                        #     # 4. Deep dive into state structure
                        #     total_tensors = 0
                        #     total_gpu_memory = 0
                        #     state_summary = {}
                            
                        #     for group_idx, param_group in enumerate(optimizer.param_groups):
                        #         print(f"\n--- Param Group {group_idx} ---")
                        #         print(f"Group keys: {list(param_group.keys())}")
                        #         print(f"Number of params: {len(param_group['params'])}")
                                
                        #         for param_idx, param in enumerate(param_group['params']):
                        #             param_state = optimizer.state.get(param, {})
                        #             if not param_state:
                        #                 continue
                                        
                        #             param_info = {
                        #                 'param_id': id(param),
                        #                 'param_shape': param.shape,
                        #                 'param_device': param.device,
                        #                 'param_dtype': param.dtype,
                        #                 'state_keys': list(param_state.keys()),
                        #                 'tensor_states': {}
                        #             }
                                    
                        #             # Check each state tensor
                        #             for state_key, state_value in param_state.items():
                        #                 if isinstance(state_value, torch.Tensor):
                        #                     total_tensors += 1
                        #                     tensor_memory = state_value.element_size() * state_value.numel()
                        #                     if state_value.device.type == 'cuda':
                        #                         total_gpu_memory += tensor_memory
                                            
                        #                     param_info['tensor_states'][state_key] = {
                        #                         'tensor_id': id(state_value),
                        #                         'shape': state_value.shape,
                        #                         'device': state_value.device,
                        #                         'dtype': state_value.dtype,
                        #                         'memory_bytes': tensor_memory,
                        #                         'data_ptr': state_value.data_ptr() if state_value.device.type == 'cuda' else None,
                        #                         'storage_id': id(state_value.storage()),
                        #                         'is_contiguous': state_value.is_contiguous(),
                        #                     }
                                            
                        #                     # Check for any views or aliases
                        #                     try:
                        #                         param_info['tensor_states'][state_key]['storage_offset'] = state_value.storage_offset()
                        #                         param_info['tensor_states'][state_key]['stride'] = state_value.stride()
                        #                     except:
                        #                         pass
                                    
                        #             state_summary[f'group_{group_idx}_param_{param_idx}'] = param_info
                                    
                        #             # Print summary for first few params
                        #             if param_idx < 3:  # Only print first 3 params to avoid spam
                        #                 print(f"  Param {param_idx}: shape={param.shape}, device={param.device}")
                        #                 for state_key, tensor_info in param_info['tensor_states'].items():
                        #                     print(f"    {state_key}: shape={tensor_info['shape']}, device={tensor_info['device']}, "
                        #                           f"id={tensor_info['tensor_id']}, data_ptr={tensor_info['data_ptr']}")
                            
                        #     print(f"\n--- SUMMARY ---")
                        #     print(f"Total state tensors: {total_tensors}")
                        #     print(f"Total GPU memory in optimizer state: {total_gpu_memory / (1024**3):.3f} GB")
                            
                        #     # 5. Check current GPU memory
                        #     if torch.cuda.is_available():
                        #         allocated = torch.cuda.memory_allocated() / (1024**3)
                        #         reserved = torch.cuda.memory_reserved() / (1024**3)
                        #         print(f"Current GPU memory - Allocated: {allocated:.3f} GB, Reserved: {reserved:.3f} GB")
                            
                        #     return state_summary
                        
                        # # Debug BEFORE state_dict()
                        # state_before = deep_debug_optimizer_state(optimizer, "BEFORE state_dict()")
                        
                        # # Call state_dict()
                        # print(f"\n🔍 CALLING optimizer.state_dict()...")
                        # optimizer_state_dict = optimizer.state_dict()
                        # print(f"✅ optimizer.state_dict() completed")
                        
                        # # Debug AFTER state_dict()
                        # state_after = deep_debug_optimizer_state(optimizer, "AFTER state_dict()")
                        
                        # # Compare states in detail
                        # print(f"\n=== DETAILED COMPARISON ===")
                        # changes_detected = False
                        
                        # for param_key in state_before.keys():
                        #     if param_key not in state_after:
                        #         print(f"❌ MISSING: {param_key} not found in after state")
                        #         changes_detected = True
                        #         continue
                                
                        #     before_param = state_before[param_key]
                        #     after_param = state_after[param_key]
                            
                        #     # Compare basic param info
                        #     if before_param['param_id'] != after_param['param_id']:
                        #         print(f"❌ PARAM ID CHANGED: {param_key} - {before_param['param_id']} -> {after_param['param_id']}")
                        #         changes_detected = True
                            
                        #     if before_param['state_keys'] != after_param['state_keys']:
                        #         print(f"❌ STATE KEYS CHANGED: {param_key} - {before_param['state_keys']} -> {after_param['state_keys']}")
                        #         changes_detected = True
                            
                        #     # Compare tensor states
                        #     for state_key in before_param['tensor_states'].keys():
                        #         if state_key not in after_param['tensor_states']:
                        #             print(f"❌ TENSOR MISSING: {param_key}.{state_key}")
                        #             changes_detected = True
                        #             continue
                                    
                        #         before_tensor = before_param['tensor_states'][state_key]
                        #         after_tensor = after_param['tensor_states'][state_key]
                                
                        #         # Check critical properties
                        #         critical_props = ['tensor_id', 'device', 'data_ptr', 'storage_id']
                        #         for prop in critical_props:
                        #             if prop in before_tensor and prop in after_tensor:
                        #                 if before_tensor[prop] != after_tensor[prop]:
                        #                     print(f"❌ {prop.upper()} CHANGED: {param_key}.{state_key} - {before_tensor[prop]} -> {after_tensor[prop]}")
                        #                     changes_detected = True
                        
                        # if not changes_detected:
                        #     print("✅ No detectable changes in optimizer state structure")
                        
                        # # Additional check: verify that offloading would actually work
                        # print(f"\n=== OFFLOAD SIMULATION TEST ===")
                        # gpu_tensors_found = 0
                        # for param_group in optimizer.param_groups:
                        #     for param in param_group["params"]:
                        #         if param in optimizer.state:
                        #             for key, value in optimizer.state[param].items():
                        #                 if isinstance(value, torch.Tensor) and value.device.type == 'cuda':
                        #                     gpu_tensors_found += 1
                        #                     print(f"GPU tensor found: param_id={id(param)}, state_key={key}, "
                        #                           f"device={value.device}, shape={value.shape}, data_ptr={value.data_ptr()}")
                        
                        # print(f"Total GPU tensors that would be offloaded: {gpu_tensors_found}")
                        
                        # self.print(f"[rank-{rank}]: Saving optim to {os.path.abspath(optim_path)}")
                        # torch.save(optimizer_state_dict, optim_path)
                        # del optimizer_state_dict
                
                
                
                # Get scheduler state dict if scheduler is provided
                lr_scheduler_state_dict = {}
                if scheduler is not None:
                    lr_scheduler_state_dict = scheduler.state_dict()
    
                # Create extra state dict with client state and any additional info
                extra_state_dict = {
                    "lr_scheduler": lr_scheduler_state_dict,
                    "client_state": client_state,
                    "tag": tag,
                    "fsdp_strategy": self.fsdp_strategy,
                    "world_size": world_size,
                    "rank": rank,
                    "global_step": global_step,
                    "rng": self.get_rng_state(),  # Add RNG state for reproducibility
                }
                
                # del optimizer_state_dict
                del lr_scheduler_state_dict

                # Save extra state
                self.print(f"[rank-{rank}]: Saving extra_state to {os.path.abspath(extra_path)}")
                torch.save(extra_state_dict, extra_path)

        # Final barrier to ensure all operations complete
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self.print(f"[rank-{rank}]: Checkpoint saved to {ckpt_dir}")

    def load_ckpt(
        self,
        model,
        ckpt_dir,
        optimizer=None,
        scheduler=None,
        tag=None,
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
        load_module_only=False,
    ):
        """Load model checkpoint for FSDP"""
        import warnings
        from torch.distributed.fsdp import ShardedStateDictConfig, ShardedOptimStateDictConfig, StateDictType

        if ckpt_dir is None:
            raise ValueError("ckpt_dir cannot be None")
        elif not os.path.exists(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

        # Extract the actual model for loading
        load_model = model
        if isinstance(model, Actor):
            load_model = model.model

        # Define paths for loading individual rank files
        rank = self.get_rank()
        world_size = self.world_size
        model_path = os.path.join(ckpt_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        optim_path = os.path.join(ckpt_dir, f"optim_world_size_{world_size}_rank_{rank}.pt")
        extra_path = os.path.join(ckpt_dir, f"extra_state_world_size_{world_size}_rank_{rank}.pt")

        # Check if checkpoint files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        if not os.path.exists(extra_path):
            raise FileNotFoundError(f"Extra state checkpoint not found: {extra_path}")

        # Optimizer path is optional since we may not save optimizer states initially
        optim_exists = os.path.exists(optim_path)

        self.print(f"[rank-{rank}]: Loading model from {os.path.abspath(model_path)}")
        self.print(f"[rank-{rank}]: Loading extra_state from {os.path.abspath(extra_path)}")
        if optim_exists:
            self.print(f"[rank-{rank}]: Loading optim from {os.path.abspath(optim_path)}")

        # Load state dictionaries from disk
        model_state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        extra_state_dict = torch.load(extra_path, map_location="cpu", weights_only=False)

        optimizer_state_dict = {}
        if optim_exists and load_optimizer_states and not load_module_only:
            optimizer_state_dict = torch.load(optim_path, map_location="cpu", weights_only=False)

        # Extract scheduler state from extra state
        lr_scheduler_state_dict = extra_state_dict.get("lr_scheduler", {})

        # Set up state dict configurations for sharded loading
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True)

        # Load using appropriate FSDP context
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with get_fsdp_state_ctx(load_model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                # Load model state dict
                load_model.load_state_dict(model_state_dict, strict=load_module_strict)
                self.print(f"[rank-{rank}]: Successfully loaded model state dict")

                # Load optimizer state dict if optimizer object is provided and loading is requested
                if optimizer is not None and load_optimizer_states and not load_module_only and optimizer_state_dict:
                    optimizer.load_state_dict(optimizer_state_dict)
                    self.print(f"[rank-{rank}]: Successfully loaded optimizer state")

                # Load scheduler state dict if scheduler object is provided and loading is requested
                if scheduler is not None and load_lr_scheduler_states and not load_module_only:
                    scheduler.load_state_dict(lr_scheduler_state_dict)
                    self.print(f"[rank-{rank}]: Successfully loaded scheduler state")

        # Load RNG state for reproducibility
        if "rng" in extra_state_dict:
            self.load_rng_state(extra_state_dict["rng"])

        # Wait for all ranks to finish loading
        dist.barrier()

        # Create states dict with extra information
        client_state = extra_state_dict.get("client_state", {})
        states = {
            "client_state": client_state,
            "tag": extra_state_dict.get("tag", tag),
            "fsdp_strategy": extra_state_dict.get("fsdp_strategy", self.fsdp_strategy),
            "world_size": extra_state_dict.get("world_size", world_size),
            "rank": extra_state_dict.get("rank", rank),
            "global_step": extra_state_dict.get("global_step", 0),  # Include global_step in return
        }

        self.print(f"[rank-{rank}]: Checkpoint loaded successfully from {ckpt_dir}")

        return ckpt_dir, states

    # TODO (erictang000): Test in multi-node setting
    def save_hf_model(self, model: Union[Actor, nn.Module], output_dir: str, tokenizer=None, **kwargs) -> None:
        """Save model in HuggingFace safetensors format using FSDP's full state dict gathering"""

        # Step 1: Create output directory (rank 0 only)
        if self.is_rank_0():
            os.makedirs(output_dir, exist_ok=True)
            self.print(f"[rank-0]: Created output directory: {output_dir}")

        # Step 2: Extract models - get both the model for saving metadata and the FSDP model for state dict
        model_to_save = self._unwrap_model(model)  # For saving config/metadata
        fsdp_model = model.model if isinstance(model, Actor) else model  # For state dict collection

        # Validate that we have a proper HuggingFace model
        if not hasattr(model_to_save, "config") or not hasattr(model_to_save, "save_pretrained"):
            raise ValueError("Model must be a HuggingFace model with config and save_pretrained method")

        # Step 3: Determine FSDP version and collect full state dict
        fsdp_ver = fsdp_version(fsdp_model)
        self.print(f"[rank-{self.get_rank()}]: Detected FSDP version: {fsdp_ver}")

        if fsdp_ver == 2:
            # Use FSDP2 API - collects on rank 0 only
            output_state_dict = fsdp2_get_full_state_dict(fsdp_model, cpu_offload=True, rank0_only=True)
        elif fsdp_ver == 1:
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

            options = StateDictOptions(full_state_dict=True, cpu_offload=True, broadcast_from_rank0=False)
            output_state_dict = get_model_state_dict(fsdp_model, options=options)
            if not self.is_rank_0():
                output_state_dict.clear()
        else:
            raise ValueError(f"Unsupported FSDP version: {fsdp_ver}")

        # Step 4: Save on rank 0 only
        if self.is_rank_0():
            # Save the model in HuggingFace format using safetensors
            model_to_save.save_pretrained(
                output_dir, state_dict=output_state_dict, safe_serialization=True, **kwargs  # Always use safetensors
            )

            # Save config
            model_to_save.config.save_pretrained(output_dir)

            # Save tokenizer if provided
            if tokenizer is not None:
                tokenizer.save_pretrained(output_dir)

            self.print(f"[rank-0]: Successfully saved model to {output_dir}")

        dist.barrier()
