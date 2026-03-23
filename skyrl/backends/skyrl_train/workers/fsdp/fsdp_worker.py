import io
from typing import TYPE_CHECKING

from loguru import logger
import ray
import torch
import torch.nn as nn
import torch.distributed
from torch.distributed.fsdp.api import ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
)
from transformers import AutoConfig

from skyrl.train.utils.trainer_utils import (
    get_rope_scaling_config,
    get_rope_theta_config,
)

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from skyrl.backends.skyrl_train.distributed.fsdp_strategy import FSDPStrategy
from skyrl.backends.skyrl_train.distributed.fsdp_utils import (
    fsdp_version,
    get_init_weight_context_manager,
)
from skyrl.backends.skyrl_train.training_batch import (
    TrainingInputBatch,
    TrainingOutputBatch,
)
from skyrl.backends.skyrl_train.weight_sync import (
    LoraLoadRequest,
    WeightChunk,
    WeightExtractor,
)
from skyrl.backends.skyrl_train.weight_sync.weight_extractor_utils import (
    yield_module_grouped_chunks,
)
from skyrl.backends.skyrl_train.workers.model_wrapper import (
    HFModelWrapper,
    get_llm_for_sequence_regression,
)
from skyrl.backends.skyrl_train.workers.worker import (
    CriticWorkerBase,
    PolicyWorkerBase,
    RefWorkerBase,
)
from skyrl.train.utils.utils import str_to_torch_dtype

if TYPE_CHECKING:
    from skyrl.train.config.config import InferenceEngineConfig


def _patch_moe_experts_for_fsdp2(model: nn.Module):
    """Patch MoE expert modules that skip unused experts during forward.

    Some MoE implementations only iterate over experts that received tokens
    (via expert_hit/nonzero). With FSDP2 + non-reentrant gradient checkpointing,
    this causes checkpoint recompute assertion failures because the variable
    computation graph saves different tensors between forward and recompute.

    The fix replaces the selective expert loop with one that iterates ALL experts
    unconditionally, ensuring a deterministic computation graph.

    Supports:
    - Qwen3MoeSparseMoeBlock (transformers <= 4.57.x): nn.ModuleList of Qwen3MoeMLP
    - Qwen3MoeExperts (transformers >= 4.58.x): fused 3D parameter tensors
    """
    import torch.nn.functional as F

    patched = 0
    module_classes = set()
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__
        module_classes.add(cls_name)

        # transformers <= 4.57.x: individual MLP expert modules in nn.ModuleList
        if cls_name == "Qwen3MoeSparseMoeBlock":

            def _make_patched_sparse_forward(mod):
                def patched_forward(hidden_states):
                    batch_size, sequence_length, hidden_dim = hidden_states.shape
                    hidden_states_flat = hidden_states.view(-1, hidden_dim)
                    router_logits = mod.gate(hidden_states_flat)

                    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
                    routing_weights, selected_experts = torch.topk(routing_weights, mod.top_k, dim=-1)
                    if mod.norm_topk_prob:
                        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
                    routing_weights = routing_weights.to(hidden_states_flat.dtype)

                    final_hidden_states = torch.zeros_like(hidden_states_flat)
                    expert_mask = torch.nn.functional.one_hot(
                        selected_experts, num_classes=mod.num_experts
                    ).permute(2, 1, 0)

                    # Iterate ALL experts with identical code path (no branching)
                    # to ensure deterministic tensor count for gradient checkpointing.
                    for expert_idx in range(mod.num_experts):
                        expert_layer = mod.experts[expert_idx]
                        idx, top_x = torch.where(expert_mask[expert_idx])
                        # Always index into hidden_states — for empty experts top_x is
                        # an empty tensor, so current_state has 0 rows but same hidden_dim.
                        current_state = hidden_states_flat[top_x]
                        current_hidden_states = expert_layer(current_state)
                        current_hidden_states = current_hidden_states * routing_weights[top_x, idx, None]
                        final_hidden_states.index_add_(
                            0, top_x, current_hidden_states.to(hidden_states_flat.dtype)
                        )

                    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
                    return final_hidden_states, router_logits
                return patched_forward

            module.forward = _make_patched_sparse_forward(module)
            patched += 1
            logger.info(f"Patched {cls_name} at '{name}' to iterate all experts for FSDP2")

        # transformers >= 4.58.x: fused 3D parameter tensors
        elif cls_name == "Qwen3MoeExperts":

            def _make_patched_fused_forward(mod):
                def patched_forward(hidden_states, top_k_index, top_k_weights):
                    final_hidden_states = torch.zeros_like(hidden_states)
                    with torch.no_grad():
                        expert_mask = torch.nn.functional.one_hot(
                            top_k_index, num_classes=mod.num_experts
                        )
                        expert_mask = expert_mask.permute(2, 1, 0)

                    for expert_idx in range(mod.num_experts):
                        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                        if token_idx.numel() > 0:
                            current_state = hidden_states[token_idx]
                        else:
                            current_state = hidden_states[:0]

                        gate, up = nn.functional.linear(
                            current_state, mod.gate_up_proj[expert_idx]
                        ).chunk(2, dim=-1)
                        current_hidden_states = mod.act_fn(gate) * up
                        current_hidden_states = nn.functional.linear(
                            current_hidden_states, mod.down_proj[expert_idx]
                        )

                        if token_idx.numel() > 0:
                            current_hidden_states = (
                                current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
                            )
                            final_hidden_states.index_add_(
                                0, token_idx,
                                current_hidden_states.to(final_hidden_states.dtype),
                            )

                    return final_hidden_states
                return patched_forward

            module.forward = _make_patched_fused_forward(module)
            patched += 1
            logger.info(f"Patched {cls_name} at '{name}' to iterate all experts for FSDP2")

    logger.info(f"MoE expert scan: found classes={module_classes}, patched={patched}")
    return patched


def _patch_checkpoint_for_moe():
    """Patch non-reentrant gradient checkpointing for MoE compatibility.

    Non-reentrant checkpointing tracks every tensor saved during forward via
    pack_hook/unpack_hook. During backward, it recomputes the forward and expects
    the same tensors to be saved. MoE models with data-dependent routing produce
    different computation graphs (different experts active), so:
    - Different number of tensors saved (2016 vs 176 for Qwen3-30B-A3B)
    - Holder handles missing for the current graph task ID

    This patches _checkpoint_hook to make unpack_hook tolerant of missing handles
    by triggering a fresh recompute when a holder's gid is missing.
    """
    import torch.utils.checkpoint as _cp

    # Suppress tensor count validation
    _cp._CheckpointFrame.check_recomputed_tensors_match = lambda self, gid: None

    def _patched_checkpoint_hook_init(self, frame):
        import uuid as _uuid
        import weakref as _weakref

        def pack_hook(x):
            holder = _cp._Holder()
            frame.weak_holders.append(_weakref.ref(holder))
            if frame.metadata_fn is not None:
                with torch.no_grad():
                    frame.x_metadatas.append(frame.metadata_fn(x))
            return holder

        def unpack_hook(holder):
            gid = torch._C._current_graph_task_id()
            if gid == -1:
                gid = int(_uuid.uuid4())

            if not frame.is_recomputed.get(gid, False):
                ctx = frame.input_saver.grad_fn
                args = ctx.get_args(ctx.saved_tensors)
                try:
                    with _cp._recomputation_hook(
                        _weakref.ref(frame), gid
                    ), torch.autograd.enable_grad():
                        frame.recompute_fn(*args)
                except _cp._StopRecomputationError:
                    pass
                frame.is_recomputed[gid] = True

            # Tolerant handle lookup: if gid not in holder.handles,
            # the tensor wasn't recreated during recompute (MoE routing difference).
            # Return a zero tensor as a safe fallback — gradients through unused
            # expert paths are zero anyway.
            if gid not in holder.handles or holder.handles[gid] is None:
                # Find any available recomputed tensor to infer shape/dtype/device
                if gid in frame.recomputed and frame.recomputed[gid]:
                    sample = next(iter(frame.recomputed[gid].values()))
                    return torch.zeros_like(sample)
                return torch.tensor(0.0)

            handle = holder.handles[gid]
            if handle in frame.recomputed.get(gid, {}):
                ret = frame.recomputed[gid][handle]
            else:
                ret = torch.tensor(0.0)
            holder.handles[gid] = None
            return ret

        if frame.unpack_error_cb is not None:
            def unpack_hook_with_error_cb(holder):
                try:
                    return unpack_hook(holder)
                except _cp.CheckpointError as e:
                    frame.unpack_error_cb(e)
            torch.autograd.graph.saved_tensors_hooks.__init__(self, pack_hook, unpack_hook_with_error_cb)
        else:
            torch.autograd.graph.saved_tensors_hooks.__init__(self, pack_hook, unpack_hook)

    _cp._checkpoint_hook.__init__ = _patched_checkpoint_hook_init


class FSDPWeightExtractor(WeightExtractor):
    """Extracts weights from FSDP-sharded models.

    Args:
        model: FSDP model to extract weights from
        group_by_module: If True, group parameters by module (e.g., for FlashRL QKV fusion)
        batch_size_threshold_gb: If > 0, batch complete modules together until threshold is reached
    """

    def __init__(self, model: torch.nn.Module, group_by_module: bool = False, batch_size_threshold_gb: float = 0.0):
        self.model = model
        self.group_by_module = group_by_module
        self.batch_size_threshold_gb = batch_size_threshold_gb

    def extract_weights(self, dtype: torch.dtype):
        """Extract weights from FSDP model.

        Args:
            dtype: Target dtype for inference

        Yields:
            WeightChunk objects (one per parameter, or grouped by module)
        """
        # Configure state_dict type for FSDP v1
        if fsdp_version(self.model) == 1:
            FSDP.set_state_dict_type(
                self.model,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        # Get state dict (handles FSDP sharding)
        params = self.model.state_dict()

        if not self.group_by_module:
            # Simple path: yield one chunk per parameter
            for name, param in params.items():
                tensor = self._gather_tensor(param).to(dtype).detach().contiguous()
                yield WeightChunk(
                    names=[name],
                    dtypes=[str(dtype)],
                    shapes=[list(tensor.shape)],
                    tensors=[tensor],
                )
        else:
            for chunk in yield_module_grouped_chunks(
                params=params,
                dtype=dtype,
                gather_tensor_fn=self._gather_tensor,
                get_shape_fn=lambda name, param, tensor: list(tensor.shape),
                batch_size_threshold_gb=self.batch_size_threshold_gb,
            ):
                yield chunk

    def get_weight_metadata(self, dtype: torch.dtype) -> dict:
        """Return weight metadata without materializing full tensors.

        Uses state_dict() to get clean parameter names (FSDP strips the
        _fsdp_wrapped_module prefix), matching extract_weights behavior.
        The sharded tensors returned by state_dict() are not gathered;
        we only read their shape.
        """
        if fsdp_version(self.model) == 1:
            FSDP.set_state_dict_type(
                self.model,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        names = []
        dtype_names = []
        shapes = []
        dtype_name = str(dtype).split(".")[-1]
        for name, param in self.model.state_dict().items():
            names.append(name)
            dtype_names.append(dtype_name)
            shapes.append(list(param.shape))
        return {"names": names, "dtype_names": dtype_names, "shapes": shapes}

    def _gather_tensor(self, param: torch.Tensor) -> torch.Tensor:
        """Gather sharded tensor into full tensor."""
        device = torch.cuda.current_device()
        return param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param


class FSDPPolicyWorkerBase(PolicyWorkerBase):
    def offload_to_cpu(self, pin_memory=True, non_blocking=True, offload_optimizer=True, offload_model=True):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(
            self.model, self.optimizer, pin_memory, non_blocking, offload_optimizer, offload_model
        )

    def backload_to_gpu(self, non_blocking=True, backload_optimizer=True, backload_model=True):
        self.strategy.backload_to_gpu(self.model, self.optimizer, non_blocking, backload_optimizer, backload_model)

    def init_model(self, model_path, num_training_steps: int = None):
        assert self.cfg.strategy in ("fsdp", "fsdp2")
        strategy = FSDPStrategy(
            fsdp_config=self.cfg.policy.fsdp_config,
            optimizer_config=self.cfg.policy.optimizer_config,
            model_config=self.cfg.policy.model,
            fsdp_strategy=self.cfg.strategy,
            seed=self.cfg.seed,
            micro_train_batch_size_per_gpu=self.cfg.micro_train_batch_size_per_gpu,
            num_training_steps=num_training_steps,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        self._is_lora = self.cfg.policy.model.lora.rank > 0

        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # FSDP2 handles tied embeddings correctly via broadcast + tie_weights(),
        # so meta tensor init is always safe. FSDP1 needs CPU init for tied embeddings.
        use_meta = True if self.cfg.strategy == "fsdp2" else (not getattr(model_config, "tie_word_embeddings", False))
        init_context = get_init_weight_context_manager(
            use_meta_tensor=use_meta, mesh=self.strategy.device_mesh
        )
        with init_context():

            wrapped_model = HFModelWrapper(
                model_path,
                use_flash_attention_2=self.cfg.flash_attn,
                # NOTE (sumanthrh): Model initialization should always be in fp32
                # during training
                bf16=False,
                lora_rank=self.cfg.policy.model.lora.rank,
                lora_alpha=self.cfg.policy.model.lora.alpha,
                lora_dropout=self.cfg.policy.model.lora.dropout,
                lora_init_method=self.cfg.policy.model.lora.init_method,
                target_modules=self.cfg.policy.model.lora.target_modules,
                exclude_modules=self.cfg.policy.model.lora.exclude_modules,
                sequence_parallel_size=self.cfg.policy.sequence_parallel_size,
                use_sample_packing=self.cfg.use_sample_packing,
                use_torch_compile=self.cfg.policy.use_torch_compile,
                rope_scaling=get_rope_scaling_config(self.cfg),
                rope_theta=get_rope_theta_config(self.cfg),
                model_config_kwargs=self.cfg.policy.model_config_kwargs,
            )
            # in-place patch
            self._seq_parallel_monkey_patch(model=wrapped_model.model)

            if self.cfg.gradient_checkpointing:
                use_reentrant = self.cfg.gradient_checkpointing_use_reentrant
                wrapped_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
                )
                logger.info(
                    f"Gradient checkpointing enabled inside init_context: "
                    f"use_reentrant={use_reentrant}"
                )

        is_moe = getattr(model_config, "num_experts", None) is not None or \
                 getattr(model_config, "num_local_experts", None) is not None
        needs_expert_patch = (
            is_moe
            and self.cfg.strategy == "fsdp2"
            and self.cfg.gradient_checkpointing
            and not self.cfg.gradient_checkpointing_use_reentrant
        )
        if needs_expert_patch:
            _patch_moe_experts_for_fsdp2(wrapped_model.model)

        self.model, self.optimizer, self.scheduler = strategy.prepare(
            (wrapped_model, None, None),
        )
        assert (
            self.optimizer is not None and self.scheduler is not None
        ), "FSDP preparation should create optimizer and scheduler"

    async def init_weight_sync_state(self, inference_engine_client, inference_engine_cfg: "InferenceEngineConfig"):
        # Call super first to set _transfer_strategy_cls and create sender/receivers
        await super().init_weight_sync_state(inference_engine_client, inference_engine_cfg)

        # Initialize weight extractor
        # TODO(haochen): Now module grouping (in order to support FlashRL) is only enabled for the CUDA IPC
        # transfer strategy, we can enable it for other strategies as well.
        from skyrl.backends.skyrl_train.weight_sync import CudaIpcTransferStrategy

        group_by_module = self._transfer_strategy_cls is CudaIpcTransferStrategy
        self.weight_extractor = FSDPWeightExtractor(
            self.model.model,
            group_by_module=group_by_module,
            batch_size_threshold_gb=(
                inference_engine_cfg.weight_transfer_threshold_cuda_ipc_GB if group_by_module else 0.0
            ),
        )

    async def _save_lora_adapters_and_sync(self, peft_model, lora_sync_path, inference_engine_client):
        """Collect LoRA parameters, save and call inference engine to load."""
        import json
        import os
        from dataclasses import asdict

        from safetensors.torch import save_file

        from skyrl.backends.skyrl_train.distributed.fsdp_utils import (
            collect_lora_params,
        )

        lora_params = collect_lora_params(module=self.model.model)

        if torch.distributed.get_rank() == 0:
            os.makedirs(lora_sync_path, exist_ok=True)

            peft_config = asdict(peft_model.peft_config.get("default", {}))
            peft_config["task_type"] = peft_config["task_type"].value
            peft_config["peft_type"] = peft_config["peft_type"].value
            peft_config["target_modules"] = list(peft_config["target_modules"])

            # Save LoRA parameters and config
            save_file(lora_params, os.path.join(lora_sync_path, "adapter_model.safetensors"))
            with io.open(os.path.join(lora_sync_path, "adapter_config.json"), "w", encoding="utf-8") as f:
                json.dump(peft_config, f, ensure_ascii=False, indent=4)

            # Send LoRA disk loading request to inference engine.
            from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
                RemoteInferenceClient,
            )

            if isinstance(inference_engine_client, RemoteInferenceClient):
                await inference_engine_client.update_lora_from_disk(lora_sync_path)
            else:
                lora_request = LoraLoadRequest(lora_path=lora_sync_path)
                await inference_engine_client.update_named_weights(lora_request)

        torch.distributed.barrier()

    async def broadcast_to_inference_engines(self, inference_engine_client, inference_engine_cfg):
        use_prefix_cache = inference_engine_cfg.enable_prefix_caching
        generator_dtype = str_to_torch_dtype(inference_engine_cfg.model_dtype)
        cache_reset_task = None
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            cache_reset_task = inference_engine_client.reset_prefix_cache()

        torch.cuda.empty_cache()

        # Check if this is a LoRA model
        peft_model = getattr(self.model.model, "_fsdp_wrapped_module", self.model.model)

        if self._is_lora:
            assert hasattr(peft_model, "peft_config"), "LoRA model should have peft_config"

            # assume base model is already synced, sync LoRA adapters
            lora_sync_path = self.cfg.policy.model.lora.lora_sync_path
            await self._save_lora_adapters_and_sync(peft_model, lora_sync_path, inference_engine_client)
        else:
            # Extract and send weights using the sender created at init time
            weight_iterator = self.weight_extractor.extract_weights(generator_dtype)
            weight_metadata = self.weight_extractor.get_weight_metadata(generator_dtype)
            await self._weight_transfer_sender.send_chunks(
                weight_iterator,
                weight_metadata=weight_metadata,
            )

        if cache_reset_task is not None:
            await cache_reset_task
        torch.cuda.empty_cache()
        torch.distributed.barrier()

    def get_weight_statistics(self):
        """Compute lightweight statistics for model weights"""
        raise NotImplementedError()

    def _set_pad_token_id(self, pad_token_id):
        # NOTE (sumanthrh): self.model -> HFModelWrapper; self.model.model -> AutoModelForCausalLM
        self.model.model.config.pad_token_id = pad_token_id

    def forward(
        self,
        data: TrainingInputBatch,
    ) -> TrainingOutputBatch:
        """Run forward pass on data in inference mode.

        Reshard the model after forward pass to redistribute memory and allow for offloading to cpu.
        """
        output = super().forward(data)
        # unshard the root FSDP module (https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes)
        if self._world_size > 1 and fsdp_version(self.model.model) == 1:
            self.model.model._handle.reshard(True)
        return output


class FSDPCriticWorkerBase(CriticWorkerBase):
    def offload_to_cpu(self, pin_memory=True, non_blocking=True, offload_optimizer=True, offload_model=True):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(
            self.model, self.optimizer, pin_memory, non_blocking, offload_optimizer, offload_model
        )

    def backload_to_gpu(self, non_blocking=True, backload_optimizer=True, backload_model=True):
        self.strategy.backload_to_gpu(self.model, self.optimizer, non_blocking, backload_optimizer, backload_model)

    def init_model(self, model_path, num_training_steps: int = None):
        assert self.cfg.strategy in ("fsdp", "fsdp2")
        strategy = FSDPStrategy(
            fsdp_config=self.cfg.critic.fsdp_config,
            optimizer_config=self.cfg.critic.optimizer_config,
            fsdp_strategy=self.cfg.strategy,
            seed=self.cfg.seed,
            micro_train_batch_size_per_gpu=self.cfg.micro_train_batch_size_per_gpu,
            num_training_steps=num_training_steps,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        use_meta = True if self.cfg.strategy == "fsdp2" else (not getattr(model_config, "tie_word_embeddings", False))
        init_context = get_init_weight_context_manager(
            use_meta_tensor=use_meta, mesh=self.strategy.device_mesh
        )
        with init_context():
            critic = get_llm_for_sequence_regression(
                model_path,
                "critic",
                use_flash_attention_2=self.cfg.flash_attn,
                # NOTE (sumanthrh): Model initialization should always be in fp32
                # during training
                bf16=False,
                lora_rank=self.cfg.critic.model.lora.rank,
                lora_alpha=self.cfg.critic.model.lora.alpha,
                lora_dropout=self.cfg.critic.model.lora.dropout,
                target_modules=self.cfg.critic.model.lora.target_modules,
                exclude_modules=self.cfg.critic.model.lora.exclude_modules,
                value_head_prefix=self.cfg.algorithm.value_head_prefix,
                init_value_head=self.cfg.policy.model.path == self.cfg.critic.model.path,
                sequence_parallel_size=self.cfg.critic.sequence_parallel_size,
                use_sample_packing=self.cfg.use_sample_packing,
                model_config_kwargs=self.cfg.critic.model_config_kwargs,
            )
            self._seq_parallel_monkey_patch(model=critic, use_parent_class=True)

            if self.cfg.gradient_checkpointing:
                use_reentrant = self.cfg.gradient_checkpointing_use_reentrant
                critic.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
                )
                logger.info(
                    f"Critic gradient checkpointing enabled inside init_context: "
                    f"use_reentrant={use_reentrant}"
                )

        is_moe = getattr(model_config, "num_experts", None) is not None or \
                 getattr(model_config, "num_local_experts", None) is not None
        needs_expert_patch = (
            is_moe
            and self.cfg.strategy == "fsdp2"
            and self.cfg.gradient_checkpointing
            and not self.cfg.gradient_checkpointing_use_reentrant
        )
        if needs_expert_patch:
            _patch_moe_experts_for_fsdp2(critic)

        # prepare models/optimizers...
        self.model, self.optimizer, self.scheduler = strategy.prepare(
            (critic, None, None),
        )
        assert self.optimizer is not None

    def forward(
        self,
        data: TrainingInputBatch,
    ) -> TrainingOutputBatch:
        """Run forward pass on data in inference mode.

        Reshard the model after forward pass to redistribute memory and allow for offloading to cpu.
        """
        output = super().forward(data)
        # unshard the root FSDP module (https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes)
        if self._world_size > 1 and fsdp_version(self.model.model) == 1:
            self.model.model._handle.reshard(True)
        return output


class FSDPRefWorkerBase(RefWorkerBase):
    def offload_to_cpu(self, pin_memory=True, non_blocking=True, **kwargs):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(self.model, None, pin_memory, non_blocking)

    def backload_to_gpu(self, non_blocking=True, **kwargs):
        self.strategy.backload_to_gpu(self.model, None, non_blocking)

    def init_model(self, model_path):
        assert self.cfg.strategy in ("fsdp", "fsdp2")
        strategy = FSDPStrategy(
            fsdp_config=self.cfg.ref.fsdp_config,
            fsdp_strategy=self.cfg.strategy,
            seed=self.cfg.seed,
            micro_train_batch_size_per_gpu=self.cfg.micro_train_batch_size_per_gpu,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        use_meta = True if self.cfg.strategy == "fsdp2" else (not getattr(model_config, "tie_word_embeddings", False))
        init_context = get_init_weight_context_manager(
            use_meta_tensor=use_meta, mesh=self.strategy.device_mesh
        )

        with init_context():
            wrapped_model = HFModelWrapper(
                model_path,
                use_flash_attention_2=self.cfg.flash_attn,
                bf16=self.cfg.bf16,
                sequence_parallel_size=self.cfg.ref.sequence_parallel_size,
                use_sample_packing=self.cfg.use_sample_packing,
                rope_scaling=get_rope_scaling_config(self.cfg),
                rope_theta=get_rope_theta_config(self.cfg),
                model_config_kwargs=self.cfg.ref.model_config_kwargs,
            )
            self._seq_parallel_monkey_patch(model=wrapped_model.model)

        self.model = strategy.prepare(wrapped_model)
        self.model.eval()

    def forward(
        self,
        data: TrainingInputBatch,
    ) -> TrainingOutputBatch:
        """Run forward pass on data in inference mode.

        Reshard the model after forward pass to redistribute memory and allow for offloading to cpu.
        """
        output = super().forward(data)
        # unshard the root FSDP module (https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes)
        if self._world_size > 1 and fsdp_version(self.model.model) == 1:
            self.model.model._handle.reshard(True)
        return output


# Ray remote actors
PolicyWorker = ray.remote(num_gpus=1)(FSDPPolicyWorkerBase)
CriticWorker = ray.remote(num_gpus=1)(FSDPCriticWorkerBase)
RefWorker = ray.remote(num_gpus=1)(FSDPRefWorkerBase)
