import os
import shutil
from collections import defaultdict
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import megatron.core.parallel_state as mpu
import ray
import torch
import torch.distributed
import torch.nn as nn
from huggingface_hub import snapshot_download
from loguru import logger
from megatron.bridge import AutoBridge
from megatron.bridge.peft.canonical_lora import CanonicalLoRA
from megatron.bridge.peft.lora import LoRA
from megatron.core.optimizer import ChainedOptimizer, DistributedOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from omegaconf import OmegaConf
from transformers import AutoConfig

from skyrl.backends.skyrl_train.distributed.dispatch import MeshRank, WorkerOutput
from skyrl.backends.skyrl_train.distributed.megatron.megatron_strategy import (
    MegatronStrategy,
)
from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
    broadcast_object_across_pp_ranks,
    freeze_moe_router,
    get_model_config,
    get_moe_metrics,
    print_model_size,
)
from skyrl.backends.skyrl_train.distributed.megatron.optimizer import (
    get_megatron_optimizer,
    get_megatron_optimizer_param_scheduler,
    init_megatron_optim_config,
)
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    SKYRL_LORA_ADAPTER_NAME,
)
from skyrl.backends.skyrl_train.training_batch import (
    TrainingInputBatch,
    TrainingOutputBatch,
)
from skyrl.backends.skyrl_train.utils.profiler import Profiler
from skyrl.backends.skyrl_train.weight_sync import (
    LoraLoadRequest,
    WeightChunk,
    WeightExtractor,
)
from skyrl.backends.skyrl_train.workers.megatron.adapter_store import (
    AdapterStore,
    LoraSignature,
    iter_opts,
)
from skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper import (
    MegatronModelWrapper,
)
from skyrl.backends.skyrl_train.workers.worker import (
    CriticWorkerBase,
    PolicyWorkerBase,
    RefWorkerBase,
)
from skyrl.backends.skyrl_train.workers.worker_utils import (
    BaseBatchIterator,
    BatchIterator,
    TokenBasedBatchIterator,
    all_reduce_metrics,
    get_microbatch_iterator,
    reduce_metrics,
)
from skyrl.env_vars import SKYRL_WORKER_NCCL_TIMEOUT_IN_S
from skyrl.train.config.config import MegatronDDPConfig, get_config_as_dict
from skyrl.train.utils.utils import str_to_torch_dtype, update_model_config
from skyrl.utils.tok import get_tokenizer

if TYPE_CHECKING:
    from skyrl.backends.skyrl_train.inference_engines.base import (
        InferenceEngineInterface,
    )
    from skyrl.train.config.config import InferenceEngineConfig

import skyrl.backends.skyrl_train.workers.megatron.model_bridges as _  # noqa: F401  # register extra bridges


class MegatronWeightExtractor(WeightExtractor):
    """Extracts weights from Megatron model-parallel models.

    Uses Megatron's bridge to export weights in HuggingFace format.

    Args:
        bridge: Megatron AutoBridge instance for weight conversion
        actor_module: The actor module to extract weights from
        enable_bucketing: If True, group parameters into size-based buckets for packing
        bucket_size_threshold_GB: Size threshold in GB for bucketing (only used if enable_bucketing=True)
        training_dtype: Training dtype for size calculation (only used if enable_bucketing=True)
    """

    def __init__(
        self,
        bridge,
        actor_module,
        enable_bucketing: bool = False,
        bucket_size_threshold_GB: float = 1.0,
        training_dtype: torch.dtype = torch.bfloat16,
    ):
        self.bridge = bridge
        self.actor_module = actor_module
        self.enable_bucketing = enable_bucketing
        self.bucket_size_threshold_GB = bucket_size_threshold_GB
        self.training_dtype = training_dtype

        # Defer bucket init to first extract_weights call.
        # At __init__ time the model may be CPU-offloaded (colocate_all),
        # so param.numel()==0 and bucketing collapses to a single bucket.
        # By the time extract_weights runs, the dispatch has already
        # called prepare_for_weight_sync → _ensure_on_gpu.
        self.bucket_index_groups = None
        self._buckets_initialized = False

    def _init_param_buckets(self):
        """Compute bucket boundaries (index groups) from parameter sizes.

        Only the bucket *structure* (which task indices go in which bucket) is
        persisted.  The actual ``WeightConversionTask`` objects are rebuilt on
        every ``extract_weights`` call so that mapping objects start with clean
        PP-collective caches, avoiding stale cached state across offload/reload
        and training cycles.

        Tasks that participate in grouped export (e.g., fused MoE expert
        weights) are collected first and placed into dedicated buckets so that
        all tasks sharing the same ``group_key`` end up in a single
        ``export_hf_weights`` call.  The bridge's
        ``_accumulate_grouped_export`` requires every task for a group to be
        present in one call; splitting them across buckets causes expert
        weights to never be yielded.
        """
        weight_conversion_tasks = self.bridge.get_conversion_tasks(self.actor_module)

        def calculate_size_in_bytes(param, tp_size, ep_size):
            if param is None:
                size_in_bytes = None
            else:
                prec_to_bytes = {
                    torch.bfloat16: 2,
                    torch.float32: 4,
                }
                scale = prec_to_bytes[self.training_dtype] / prec_to_bytes[param.dtype]
                size_in_bytes = param.element_size() * param.numel() * tp_size * ep_size * scale
            return broadcast_object_across_pp_ranks(size_in_bytes)

        sizes = [
            calculate_size_in_bytes(
                task.param_weight,
                task.mapping.tp_size,
                task.mapping.ep_size if task.mapping.is_expert else 1,
            )
            for task in weight_conversion_tasks
        ]

        # ---- Separate grouped-export tasks from regular tasks ----
        # Grouped-export tasks (is_grouped_export=True, e.g. FusedGatedExpertMapping /
        # FusedExpertMapping for MoE expert weights) must ALL be present in a single
        # export_hf_weights call for the bridge's _accumulate_grouped_export to produce
        # the fused tensor.  Collect them by group_key and give each group its own bucket.
        grouped_task_indices: dict[str, list[int]] = {}  # group_key -> list of task indices
        regular_task_indices: list[int] = []

        for idx, task in enumerate(weight_conversion_tasks):
            if getattr(task.mapping, "is_grouped_export", False):
                gk = getattr(task.mapping, "group_key", None)
                grouped_task_indices.setdefault(gk, []).append(idx)
            else:
                regular_task_indices.append(idx)

        self.bucket_index_groups: list[list[int]] = []

        # Pack grouped-export tasks into buckets by size, keeping each
        # group_key's tasks together (they must not be split across calls).
        curr_size = 0
        threshold = self.bucket_size_threshold_GB * 1024**3
        for gk, indices in grouped_task_indices.items():
            group_size = sum(sizes[idx] for idx in indices if sizes[idx] is not None)
            if not self.bucket_index_groups or curr_size + group_size > threshold:
                self.bucket_index_groups.append([])
                curr_size = 0
            self.bucket_index_groups[-1].extend(indices)
            curr_size += group_size

        # Bucket regular (non-grouped) tasks by size as before.
        if regular_task_indices:
            self.bucket_index_groups.append([])
            curr_size = 0
            for idx in regular_task_indices:
                size = sizes[idx]
                if curr_size + size > threshold:
                    self.bucket_index_groups.append([])
                    curr_size = 0
                self.bucket_index_groups[-1].append(idx)
                curr_size += size

    def get_weight_metadata(self, dtype: torch.dtype) -> dict:
        """Return weight metadata without keeping tensors in memory.

        On first call, runs export_hf_weights to discover HF names and shapes
        (tensors are discarded immediately). Result is cached for subsequent calls.
        TODO (aaron): find a better way to get all metadata without materializing tensors.
        """
        if hasattr(self, "_weight_metadata_cache"):
            return self._weight_metadata_cache

        self._ensure_buckets_initialized()
        names = []
        dtype_names = []
        shapes = []
        dtype_name = str(dtype).split(".")[-1]
        # Collect parameter metadata in the same order
        # as provided by `.extract_weights`.
        if not self.enable_bucketing:
            for name, tensor in self.bridge.export_hf_weights(
                self.actor_module,
                show_progress=False,
                conversion_tasks=None,
            ):
                names.append(name)
                dtype_names.append(dtype_name)
                shapes.append(list(tensor.shape))
                del tensor
        else:
            # Build fresh tasks each sync so mapping objects have clean
            # PP-collective caches; reuse the pre-computed bucket structure.
            fresh_tasks = self.bridge.get_conversion_tasks(self.actor_module)
            for index_group in self.bucket_index_groups:
                bucket_tasks = [fresh_tasks[i] for i in index_group]
                for name, tensor in self.bridge.export_hf_weights(
                    self.actor_module,
                    show_progress=False,
                    conversion_tasks=bucket_tasks,
                ):
                    names.append(name)
                    shapes.append(list(tensor.shape))
                    dtype_names.append(dtype_name)
                    del tensor

        self._weight_metadata_cache = {"names": names, "dtype_names": dtype_names, "shapes": shapes}
        return self._weight_metadata_cache

    def _ensure_buckets_initialized(self):
        """Lazily initialize param buckets on first use (model must be on GPU)."""
        if self._buckets_initialized:
            return
        if self.enable_bucketing:
            self._init_param_buckets()
        self._buckets_initialized = True

    def extract_weights(self, dtype: torch.dtype):
        """Extract weights from Megatron model.

        Args:
            dtype: Target dtype for inference

        Yields:
            WeightChunk objects (one per parameter, or one per bucket if bucketing enabled)
        """
        self._ensure_buckets_initialized()
        device = torch.cuda.current_device()

        if not self.enable_bucketing:
            # No bucketing: yield one chunk per parameter
            hf_params_generator = self.bridge.export_hf_weights(
                self.actor_module,
                show_progress=False,
                conversion_tasks=None,
            )

            for name, tensor in hf_params_generator:
                tensor = tensor.to(device=device, dtype=dtype, non_blocking=True)

                yield WeightChunk(
                    names=[name],
                    dtypes=[str(dtype)],
                    shapes=[list(tensor.shape)],
                    tensors=[tensor],
                )
        else:
            # Build fresh tasks each sync so mapping objects have clean
            # PP-collective caches; reuse the pre-computed bucket structure.
            fresh_tasks = self.bridge.get_conversion_tasks(self.actor_module)

            for index_group in self.bucket_index_groups:
                bucket_tasks = [fresh_tasks[i] for i in index_group]
                hf_params_generator = self.bridge.export_hf_weights(
                    self.actor_module,
                    show_progress=False,
                    conversion_tasks=bucket_tasks,
                )

                # Collect all parameters in this bucket into one chunk
                names = []
                dtypes_list = []
                shapes = []
                tensors = []

                for name, tensor in hf_params_generator:
                    # Move to device and convert dtype
                    tensor = tensor.to(device=device, dtype=dtype, non_blocking=True)

                    names.append(name)
                    dtypes_list.append(str(dtype))
                    shapes.append(list(tensor.shape))
                    tensors.append(tensor)

                # Yield one chunk containing all parameters in this bucket
                if tensors:
                    yield WeightChunk(
                        names=names,
                        dtypes=dtypes_list,
                        shapes=shapes,
                        tensors=tensors,
                    )


class MegatronWorker:
    def init_configs(
        self,
        model_path,
        megatron_config,
        model_config_kwargs,
        transformer_config_kwargs,
        bf16=True,
        flash_attn=False,
        lora_config=None,
    ):
        """
        Initialize the Megatron-Bridge bridge and provider objects + hf_config and tokenizer
        """
        tokenizer = get_tokenizer(model_path, trust_remote_code=True)
        hf_config_original = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        # VLM detection mirrors the FSDP path: a non-null ``vision_config`` on the
        # HF config means a vision tower is present. Megatron's TransformerConfig
        # has no such field, so this must be read off the HF config.
        self.is_vlm = hasattr(hf_config_original, "vision_config") and hf_config_original.vision_config is not None

        override_config_kwargs = {
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        override_config_kwargs.update(model_config_kwargs.get("model_config", {}))
        hf_config = update_model_config(hf_config_original, override_config_kwargs=override_config_kwargs)

        transformer_config_kwargs = (
            transformer_config_kwargs
            if isinstance(transformer_config_kwargs, dict)
            else OmegaConf.to_container(transformer_config_kwargs, resolve=True)
        )

        if not self.cfg.gradient_checkpointing:
            for key in ("recompute_granularity", "recompute_method", "recompute_num_layers"):
                transformer_config_kwargs[key] = None

        bridge = AutoBridge.from_hf_pretrained(model_path, trust_remote_code=True)
        provider = bridge.to_megatron_provider()

        # Workaround for megatron-bridge CONFIG_MAPPING dropping None values:
        # MLA models like Moonlight-16B have q_lora_rank=None (no Q compression),
        # but CONFIG_MAPPING skips None so the MCoreMLATransformerConfig default
        # (512) is used instead, causing the wrong model architecture to be built.
        # see: https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c8eb587c5fd43163dbcd9c40980225b3fe1981f8/src/megatron/bridge/recipes/moonlight/moonlight_16b.py#L60
        if hasattr(provider, "q_lora_rank") and hasattr(hf_config, "q_lora_rank"):
            provider.q_lora_rank = hf_config.q_lora_rank

        # Workaround for transformers v5 moving rope_theta into rope_parameters
        # (previously it was a top-level config attribute). megatron-bridge's
        # CONFIG_MAPPING reads config.rope_theta which no longer exists in v5,
        # causing it to fall back to the default rotary_base of 10000.
        rope_params = getattr(hf_config, "rope_parameters", None) or getattr(hf_config, "rope_scaling", None)
        if isinstance(rope_params, dict) and "rope_theta" in rope_params:
            provider.rotary_base = rope_params["rope_theta"]

        provider.tensor_model_parallel_size = megatron_config.tensor_model_parallel_size
        provider.pipeline_model_parallel_size = megatron_config.pipeline_model_parallel_size
        provider.pipeline_dtype = torch.bfloat16 if bf16 else torch.float32
        provider.context_parallel_size = megatron_config.context_parallel_size
        provider.expert_model_parallel_size = megatron_config.expert_model_parallel_size
        provider.expert_tensor_parallel_size = megatron_config.expert_tensor_parallel_size
        provider.sequence_parallel = megatron_config.tensor_model_parallel_size > 1
        provider.attention_backend = "flash" if flash_attn else "fused"
        provider.variable_seq_lengths = True
        provider.masked_softmax_fusion = True
        # Apply explicit MoE config fields to the provider.
        # These replace the previously hardcoded values and can be further
        # overridden by transformer_config_kwargs if needed.
        provider.moe_token_dispatcher_type = megatron_config.moe_token_dispatcher_type
        provider.moe_router_load_balancing_type = megatron_config.moe_router_load_balancing_type
        provider.moe_aux_loss_coeff = megatron_config.moe_aux_loss_coeff
        provider.moe_router_dtype = megatron_config.moe_router_dtype
        provider.moe_grouped_gemm = megatron_config.moe_grouped_gemm
        if megatron_config.moe_router_score_function is not None:
            provider.moe_router_score_function = megatron_config.moe_router_score_function
        if megatron_config.moe_router_enable_expert_bias is not None:
            provider.moe_router_enable_expert_bias = megatron_config.moe_router_enable_expert_bias
        provider.moe_enable_routing_replay = megatron_config.moe_enable_routing_replay

        # Apply any additional transformer config kwargs (can override the above).
        for k, v in transformer_config_kwargs.items():
            setattr(provider, k, v)
        provider.finalize()

        self.provider = provider
        self.bridge = bridge

        # strategy.hf_config is the on-disk source-of-truth used by
        # save_hf_configs and must NOT carry runtime overrides like
        # mtp_num_layers=0; assign the un-mutated AutoConfig here.
        self.strategy.hf_config = hf_config_original
        self.tokenizer = tokenizer
        self.enable_router_replay = megatron_config.moe_enable_routing_replay

    def configure_lora(self, lora_config, lora_type: Optional[str] = "lora"):
        if lora_type == "lora":
            self.lora_cls = LoRA(
                target_modules=(
                    ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
                    if lora_config.target_modules == "all-linear"
                    else lora_config.target_modules
                ),
                dim=lora_config.rank,
                alpha=lora_config.alpha,
                dropout=lora_config.dropout,
                lora_A_init_method=lora_config.init_method,
                lora_B_init_method="zero",
                exclude_modules=[] if lora_config.exclude_modules is None else lora_config.exclude_modules,
                lora_dtype=torch.bfloat16 if self.cfg.bf16 else torch.float32,
            )
        elif lora_type == "canonical_lora":
            self.lora_cls = CanonicalLoRA(
                target_modules=(
                    [
                        "linear_q",
                        "linear_k",
                        "linear_v",
                        "linear_proj",
                        "linear_fc1_up",
                        "linear_fc1_gate",
                        "linear_fc2",
                    ]
                    if lora_config.target_modules == "all-linear"
                    else lora_config.target_modules
                ),
                dim=lora_config.rank,
                alpha=lora_config.alpha,
                dropout=lora_config.dropout,
                lora_A_init_method=lora_config.init_method,
                lora_B_init_method="zero",
                exclude_modules=[] if lora_config.exclude_modules is None else lora_config.exclude_modules,
            )

    def make_megatron_module(
        self,
        wrap_with_ddp: bool = True,
        ddp_config: Optional[Union[MegatronDDPConfig, Dict[str, Any]]] = None,
        lora_config: Optional[Dict[str, Any]] = None,
        lora_type: Optional[str] = "lora",
        bf16: bool = True,
    ) -> List[nn.Module]:
        """
        Creates a megatron GPTModel (optionally DDP wrapped) using the bridge.
        """
        from megatron.core.distributed.distributed_data_parallel_config import (
            DistributedDataParallelConfig,
        )

        if lora_config is not None:
            self.configure_lora(lora_config, lora_type)

            def lora_pre_wrap_hook(model):
                lora_model = self.lora_cls(model, training=True)
                self.lora_cls.set_params_to_save(lora_model)

                return lora_model

            self.provider.register_pre_wrap_hook(lora_pre_wrap_hook)

        default_ddp_config = DistributedDataParallelConfig()
        if wrap_with_ddp:
            default_ddp_config.use_distributed_optimizer = True
        if ddp_config is not None:
            for k, v in get_config_as_dict(ddp_config).items():
                setattr(default_ddp_config, k, v)
        model = self.provider.provide_distributed_model(
            ddp_config=default_ddp_config, wrap_with_ddp=wrap_with_ddp, bf16=bf16
        )
        return model

    def _forward_logprobs(self, data: TrainingInputBatch) -> torch.Tensor:
        """Run a Megatron inference forward over ``data`` and return per-sample logprobs.

        Passes the full mini batch to ``MegatronModelWrapper.forward``. Supports token-based
        micro-batching via ``max_tokens_per_microbatch`` (padding micro-batches to a uniform
        size as Megatron's pipeline schedule requires, then reordering back to input order).

        Returns:
            CPU tensor of shape ``[batch_size, response_length]`` in original sample order.
        """
        from skyrl.backends.skyrl_train.utils.replay_utils import clear_router_replay

        self._drop_pixel_values_on_non_first_pp_stage(data)

        use_token_batching = self.cfg.max_tokens_per_microbatch > 0

        if use_token_batching:
            microbatch_iterator = get_microbatch_iterator(
                data,
                micro_batch_size=self.cfg.micro_forward_batch_size_per_gpu,
                max_tokens_per_microbatch=self.cfg.max_tokens_per_microbatch,
            )
        else:
            microbatch_iterator = None

        # Build micro-batch dicts expected by policy.forward_mini_batch
        micro_dicts = []
        device = torch.cuda.current_device()

        if microbatch_iterator is not None:
            micro_batches = microbatch_iterator
        else:
            micro_batches = data.chunk(self.cfg.micro_forward_batch_size_per_gpu)

        for micro in micro_batches:
            micro.to(device)
            attention_mask = micro["attention_mask"]
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            rollout_expert_indices = micro.get("rollout_expert_indices")
            if rollout_expert_indices is not None:
                rollout_expert_indices = rollout_expert_indices.to(torch.int32)

            vlm_inputs = {}
            if micro.get("pixel_values") is not None:
                vlm_inputs["pixel_values"] = micro.get("pixel_values")
            if micro.get("image_grid_thw") is not None:
                vlm_inputs["image_grid_thw"] = micro.get("image_grid_thw")

            micro_dicts.append(
                {
                    "sequences": micro["sequences"],
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "num_actions": micro.metadata["response_length"],
                    "rollout_expert_indices": (rollout_expert_indices if self.enable_router_replay else None),
                    "sub_seq_lengths": micro.get("sub_seq_lengths"),
                    **vlm_inputs,
                }
            )

        if use_token_batching:
            # Pad microbatches to uniform batch size for Megatron compatibility
            max_micro_bsz = max(m["sequences"].shape[0] for m in micro_dicts) if micro_dicts else 1
            for i, m in enumerate(micro_dicts):
                micro_dicts[i] = self._pad_microbatch_to_size(m, max_micro_bsz)
            mbs = max_micro_bsz
        else:
            mbs = micro_dicts[0]["sequences"].shape[0] if micro_dicts else 1

        self.model.eval()
        seq_len = micro_dicts[0]["sequences"].shape[1]
        with torch.no_grad():
            log_probs = self.model.forward(
                micro_batches=micro_dicts,
                seq_len=seq_len,
                micro_batch_size=mbs,
                temperature=self.cfg.algorithm.temperature,
            )

        log_probs = log_probs.to("cpu")

        if use_token_batching and microbatch_iterator is not None:
            # Need to strip padded samples and reorder back to original order
            output = TrainingOutputBatch({"output": log_probs})
            output.metadata = data.metadata
            # The output from Megatron is concatenated across microbatches.
            # We need to extract only the real (non-padded) samples and reorder.
            output = self._reorder_megatron_forward_output(output, microbatch_iterator, micro_dicts, mbs)
        else:
            output = TrainingOutputBatch({"output": log_probs})
            output.metadata = data.metadata

        clear_router_replay()
        return output["output"]

    def _reorder_megatron_forward_output(
        self, output: TrainingOutputBatch, microbatch_iterator, micro_dicts, padded_mbs
    ) -> TrainingOutputBatch:
        """Reorder forward output from token-based microbatching back to original sample order."""
        if not isinstance(microbatch_iterator, TokenBasedBatchIterator):
            return output

        # With PP > 1 only the last pipeline stage produces real per-sample logprobs;
        # other stages return a dummy placeholder (e.g. [1, 1]). There is nothing to
        # reorder there, and indexing it by microbatch would raise — so return as-is,
        # matching how the non-token-batched path leaves the placeholder untouched.
        if not mpu.is_pipeline_last_stage(ignore_virtual=True):
            return output

        log_probs = output["output"]  # shape: [total_padded_samples, num_actions]

        # Split by padded_mbs, take only real samples, reorder
        all_log_probs = log_probs.split(padded_mbs, dim=0)

        # Build original-order tensor
        batch_size = microbatch_iterator.data.batch_size
        num_actions = log_probs.shape[1]
        reordered = torch.zeros((batch_size, num_actions), dtype=log_probs.dtype, device=log_probs.device)

        for mb_idx, original_indices in enumerate(microbatch_iterator._microbatches):
            mb_log_probs = all_log_probs[mb_idx]
            for sample_idx, original_idx in enumerate(original_indices):
                reordered[original_idx] = mb_log_probs[sample_idx]

        result = TrainingOutputBatch({"output": reordered})
        result.metadata = output.metadata
        return result

    def _pad_microbatch_to_size(self, micro_dict: dict, target_batch_size: int) -> dict:
        """Pad a forward or forward_backward micro-batch dict to target_batch_size with dummy samples.

        Padded samples have loss_mask/action_mask=0 so they don't contribute to the loss
        (forward micro-batches carry neither key, so this is inert there). This is needed
        because Megatron's forward_backward_func requires uniform micro_batch_size across all
        microbatches (especially with PP > 1). Scalar keys (``num_actions``,
        ``num_microbatches``, ``num_real_microbatches``) are passed through unchanged.

        Defined on the base worker so the shared ``_forward_logprobs`` path works for
        policy, ref, and critic workers alike.
        """
        current_bsz = micro_dict["sequences"].shape[0]
        if current_bsz >= target_batch_size:
            return micro_dict

        pad_count = target_batch_size - current_bsz
        device = micro_dict["sequences"].device

        padded = {}
        for key, value in micro_dict.items():
            if key in ("num_actions", "num_microbatches", "num_real_microbatches"):
                padded[key] = value
                continue
            if value is None:
                padded[key] = None
                continue
            if isinstance(value, torch.Tensor):
                if key == "loss_mask":
                    # Pad with zeros so padded samples don't contribute to loss
                    pad_tensor = torch.zeros((pad_count, *value.shape[1:]), dtype=value.dtype, device=device)
                elif key == "attention_mask":
                    # Give each dummy row a single valid token, so the row is non-degenerate:
                    # it avoids a fully-masked row (NaN in dense attention's softmax) and a
                    # zero-length cu_seqlens segment (rejected by the packed/THD kernel).
                    # The row is still excluded from the loss via loss_mask/action_mask=0.
                    pad_tensor = torch.zeros((pad_count, *value.shape[1:]), dtype=value.dtype, device=device)
                    pad_tensor[:, 0] = 1
                elif key == "position_ids":
                    # position_ids for padded samples
                    seq_len = value.shape[1]
                    pad_tensor = torch.arange(seq_len, device=device).unsqueeze(0).expand(pad_count, -1)
                elif key == "action_mask":
                    # action_mask should be zeros for padded samples
                    pad_tensor = torch.zeros((pad_count, *value.shape[1:]), dtype=value.dtype, device=device)
                else:
                    pad_tensor = torch.zeros((pad_count, *value.shape[1:]), dtype=value.dtype, device=device)
                padded[key] = torch.cat([value, pad_tensor], dim=0)
            else:
                padded[key] = value

        return padded

    def save_hf_model(self, export_dir: str, tokenizer):
        # Save model in HuggingFace safetensors format
        self.strategy.save_hf_model(
            self.bridge,
            self.model,
            export_dir,
            tokenizer=tokenizer,
        )

    def _get_module_for_offload(self):
        # The underlying offloadable module is `self.actor_module` instead of `self.model`.
        return self.actor_module

    def _drop_pixel_values_on_non_first_pp_stage(self, data: TrainingInputBatch) -> None:
        """
        Drop ``pixel_values`` from the batch on every pipeline stage except the first.
        Do this prior to moving to GPU to save memory
        """
        if mpu.get_pipeline_model_parallel_rank() != 0 and data.get("pixel_values") is not None:
            data["pixel_values"] = None


class MegatronPolicyWorkerBase(MegatronWorker, PolicyWorkerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: MegatronModelWrapper = None
        self.actor_module: List[nn.Module] = None
        self.scheduler: OptimizerParamScheduler = None
        self.optimizer: DistributedOptimizer = None
        self.profiler: Profiler = None
        self._is_lora = self.cfg.policy.model.lora.rank > 0
        # Per-worker store of LoRA adapter snapshots. Allocated only for the
        # LoRA path; FFT runs single-tenant exactly as before.
        self.adapter_store: Optional[AdapterStore] = AdapterStore() if self._is_lora else None

    def init_worker_process_group(self):
        """
        Override DistributedTorchRayActor.init_worker_process_group to use megatron distributed setup to create the mesh.
        """
        if not torch.distributed.is_initialized():
            # Ensure CUDA device is set before process group init — required when
            # using split "cpu:gloo,cuda:nccl" backend to avoid 'invalid device ordinal'
            # errors during NCCL communicator creation in subgroups.
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            # Default torch dist pg init timeout is 10 minutes (600 seconds)
            torch.distributed.init_process_group(
                backend="cpu:gloo,cuda:nccl", timeout=timedelta(seconds=SKYRL_WORKER_NCCL_TIMEOUT_IN_S)
            )

        # Explicitly wrap torch.distributed.broadcast in torch.no_grad() to avoid a warning in Megatron training where the
        # autograd engine tries to track gradients through the default Torch kernel. This fixes a deprecated behaviour in
        # PyTorch, preventing potential silent errors in future versions.

        if not getattr(torch.distributed, "_skyrl_broadcast_no_grad_patched", False):
            _orig_broadcast = torch.distributed.broadcast

            def _broadcast_no_grad(*args, **kwargs):
                with torch.no_grad():
                    return _orig_broadcast(*args, **kwargs)

            torch.distributed.broadcast = _broadcast_no_grad
            torch.distributed._skyrl_broadcast_no_grad_patched = True

        self.strategy = MegatronStrategy(
            megatron_config=self.cfg.policy.megatron_config,
            optimizer_config=self.cfg.policy.optimizer_config,
            seed=self.cfg.seed,
            is_lora=self._is_lora,
            node_local_rank=self._local_rank,
        )
        self.strategy.setup_distributed()

        self.mesh_rank = MeshRank(
            dp=mpu.get_data_parallel_rank(),
            sp=mpu.get_context_parallel_rank(),
            tp=mpu.get_tensor_model_parallel_rank(),
            pp=mpu.get_pipeline_model_parallel_rank(),
            world_size=self._world_size,
            dp_size=mpu.get_data_parallel_world_size(),
            pp_size=mpu.get_pipeline_model_parallel_world_size(),
        )

    def init_model(self, model_path, num_training_steps: int = 1e9):
        """
        Initialize the model, optimizer, and scheduler for the policy worker.
        """
        # initialize the bridge and provider objects
        self.init_configs(
            model_path,
            self.cfg.policy.megatron_config,
            self.cfg.policy.megatron_config.model_config_kwargs,
            self.cfg.policy.megatron_config.transformer_config_kwargs,
            bf16=self.cfg.bf16,
            flash_attn=self.cfg.flash_attn,
        )

        if self.enable_router_replay:
            from skyrl.backends.skyrl_train.utils.replay_utils import (
                patch_topk_router_layer_number,
            )

            patch_topk_router_layer_number()

        # Freeze MoE router params before optimizer build.
        # Megatron's DistributedOptimizer reads requires_grad at construction.
        if self.cfg.policy.megatron_config.freeze_moe_router:
            if self._rank == 0:
                logger.info("freeze_moe_router=True: freezing MoE router params")
            self.provider.register_pre_wrap_hook(freeze_moe_router)

        # wrap with DDP for training
        self.actor_module = self.make_megatron_module(
            wrap_with_ddp=True,
            ddp_config=self.cfg.policy.megatron_config.ddp_config,
            lora_config=self.cfg.policy.model.lora if self._is_lora else None,
            lora_type=self.cfg.policy.megatron_config.lora_config.lora_type,
            bf16=self.cfg.bf16,
        )

        if self._local_rank == 0 and not os.path.exists(
            model_path
        ):  # if not local path, try downloading model weights from huggingface
            snapshot_download(model_path)  # will be no-op if already downloaded
        torch.distributed.barrier()

        if self._rank == 0:
            print_model_size(self.actor_module[0])

        # create profiler
        if self.cfg.policy.megatron_config.torch_profiler_config.enable:
            self.profiler = Profiler(self.cfg.policy.megatron_config.torch_profiler_config)

        # create optimizer (skipped for inference-only flows; Megatron's
        # DistributedOptimizer eagerly materializes fp32 master + AdamW state
        # on GPU, which OOMs large MoE models on memory-constrained nodes)
        if self.cfg.policy.inference_only_init:
            self.optimizer = None
            self.scheduler = None
        else:
            optim_config = init_megatron_optim_config(
                self.cfg.policy.optimizer_config, self.cfg.policy.megatron_config.optimizer_config_kwargs
            )
            self.optimizer = get_megatron_optimizer(self.actor_module, optim_config)

            # create scheduler
            self.scheduler = get_megatron_optimizer_param_scheduler(
                optimizer=self.optimizer,
                config=self.cfg.policy.optimizer_config,
                num_training_steps=num_training_steps,
            )

        # create worker model
        self.model = MegatronModelWrapper(
            config=self.cfg,
            actor_module=self.actor_module,
            actor_optimizer=self.optimizer,
            policy_loss_fn=self.policy_loss_fn,
            is_vlm=self.is_vlm,
        )

        self.empty_cuda_cache = self.cfg.policy.megatron_config.empty_cuda_cache

        # Enable expandable_segments after init so model weights stay in IPC-compatible
        # standard CUDA memory; only subsequent activations use expandable segments.
        self._set_expandable_segments(True)

    def forward(
        self,
        data: TrainingInputBatch,
        loss_fn: Optional[str] = None,
        loss_fn_config: Optional[Dict[str, Any]] = None,
    ) -> WorkerOutput:
        """Forward pass.

        - Without ``loss_fn``: runs Megatron's pipeline inference and returns a
          :class:`WorkerOutput` with per-sample ``loss_fn_outputs`` (``logprobs``
          key) and empty ``metrics``.
        - With ``loss_fn`` (e.g., ``"cross_entropy"``): runs the SFT loss through Megatron's
          pipeline schedule with ``forward_only=True`` (no backward) and returns a
          :class:`WorkerOutput` with per-sample ``loss_fn_outputs`` plus scalar
          ``metrics`` (including ``"loss"``).
        """
        from skyrl.backends.skyrl_train.utils.replay_utils import clear_router_replay

        if loss_fn is None:
            # Megatron inference forward path: emit per-sample logprobs. Token-based
            # micro-batching (when `max_tokens_per_microbatch > 0`) is handled inside
            # `_forward_logprobs`, which also reorders back to the original sample order.
            log_probs = self._forward_logprobs(data)
            loss_fn_outputs = [{"logprobs": log_probs[i].tolist()} for i in range(log_probs.shape[0])]
            return WorkerOutput(loss_fn_outputs=loss_fn_outputs, metrics={})

        self.model.eval()

        micro_batch_size = self.cfg.micro_forward_batch_size_per_gpu
        all_metrics = defaultdict(list)
        all_loss_fn_outputs: List[Dict[str, Any]] = []

        self._drop_pixel_values_on_non_first_pp_stage(data)
        # Move data to GPU
        data.to(torch.cuda.current_device())

        # Build micro-batch dicts expected by forward_backward_mini_batch
        micro_buffer = []
        for experience in BatchIterator(data, micro_batch_size, drop_last=False):
            sequences = experience.sequences
            attention_mask = experience.attention_mask
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            rollout_expert_indices = experience.rollout_expert_indices
            if rollout_expert_indices is not None:
                rollout_expert_indices = rollout_expert_indices.to(torch.int32)

            vlm_inputs = {}
            if experience.pixel_values is not None:
                vlm_inputs["pixel_values"] = experience.pixel_values
            if experience.image_grid_thw is not None:
                vlm_inputs["image_grid_thw"] = experience.image_grid_thw

            micro_buffer.append(
                {
                    "sequences": sequences,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "num_actions": experience.num_actions,
                    "old_action_log_probs": experience.action_log_probs,
                    "base_action_log_probs": experience.base_action_log_probs,
                    "advantages": experience.advantages,
                    "loss_mask": experience.loss_mask,
                    "rollout_action_logprobs": experience.rollout_logprobs,
                    "action_mask": experience.action_mask,
                    "rollout_expert_indices": rollout_expert_indices if self.enable_router_replay else None,
                    "sub_seq_lengths": experience.sub_seq_lengths,
                    **vlm_inputs,
                }
            )

        for m_batch in micro_buffer:
            m_batch["num_microbatches"] = len(micro_buffer)

        if not micro_buffer:
            return WorkerOutput()

        seq_len = micro_buffer[0]["sequences"].shape[1]
        micro_bsz = micro_buffer[0]["sequences"].shape[0]

        with torch.no_grad():
            metrics_list = self.model.forward_backward_mini_batch(
                micro_batches=micro_buffer,
                seq_len=seq_len,
                micro_batch_size=micro_bsz,
                temperature=self.cfg.algorithm.temperature,
                loss_fn=loss_fn,
                loss_fn_config=loss_fn_config,
                forward_only=True,
            )

        if self.empty_cuda_cache:
            torch.cuda.empty_cache()

        # Aggregate metrics across micro-batches
        for metrics in metrics_list:
            if metrics is None:
                continue
            if "loss_fn_outputs" in metrics:
                all_loss_fn_outputs.extend(metrics.pop("loss_fn_outputs"))
            for k, v in metrics.items():
                all_metrics[k].append(v)

        resolved_loss_name = loss_fn or self.cfg.algorithm.policy_loss_type
        sum_loss_metrics = resolved_loss_name != "cross_entropy"

        status = reduce_metrics(all_metrics, sum_loss_metrics=sum_loss_metrics)
        group = mpu.get_data_parallel_group(with_context_parallel=False)
        status = all_reduce_metrics(status, self.strategy, group=group, sum_loss_metrics=sum_loss_metrics)

        clear_router_replay()
        return WorkerOutput(loss_fn_outputs=all_loss_fn_outputs, metrics=status)

    def forward_backward(
        self,
        data: TrainingInputBatch,
        loss_fn: Optional[str] = None,
        loss_fn_config: Optional[Dict[str, Any]] = None,
    ) -> WorkerOutput:
        """
        Perform forward and backward passes for a batch, handling micro-batching internally.

        The batch is split into micro batches based on micro_train_batch_size_per_gpu,
        or by token count if max_tokens_per_microbatch is configured.
        Megatron Core's forward_backward_func handles gradient accumulation internally.

        Args:
            data: TrainingInputBatch (already DP-sharded by WorkerDispatch/MeshDispatch)
            loss_fn: Optional loss function name (e.g., "cross_entropy", "ppo").
                     If provided, overrides the config's policy_loss_type.
            loss_fn_config: Optional config overrides for the loss function.

        Returns:
            :class:`WorkerOutput` with per-sample ``loss_fn_outputs`` and scalar
            ``metrics`` (all-reduced across DP).
        """
        from skyrl.backends.skyrl_train.utils.replay_utils import clear_router_replay

        self.model.train()
        for chunk in self.actor_module:
            # if use distributed optimizer, zero grad buffer will be handled by optimizer
            chunk.zero_grad_buffer()

        all_metrics = defaultdict(list)

        self._drop_pixel_values_on_non_first_pp_stage(data)
        # Move data to GPU
        data.to(torch.cuda.current_device())

        use_token_batching = self.cfg.max_tokens_per_microbatch > 0

        if use_token_batching:
            microbatch_iterator = get_microbatch_iterator(
                data,
                micro_batch_size=self.cfg.micro_train_batch_size_per_gpu,
                max_tokens_per_microbatch=self.cfg.max_tokens_per_microbatch,
            )
        else:
            microbatch_iterator = None

        # Build micro-batch dicts expected by forward_backward_mini_batch.
        # Token-based batching yields TrainingInputBatch microbatches (converted to
        # Experience here); sample-based BatchIterator yields Experience directly.
        micro_buffer = []

        if microbatch_iterator is not None:
            experiences = (BaseBatchIterator.batch_to_experience(mb) for mb in microbatch_iterator)
        else:
            experiences = BatchIterator(data, self.cfg.micro_train_batch_size_per_gpu, drop_last=False)

        for experience in experiences:
            attention_mask = experience.attention_mask
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            rollout_expert_indices = experience.rollout_expert_indices
            if rollout_expert_indices is not None:
                rollout_expert_indices = rollout_expert_indices.to(torch.int32)

            vlm_inputs = {}
            if experience.pixel_values is not None:
                vlm_inputs["pixel_values"] = experience.pixel_values
            if experience.image_grid_thw is not None:
                vlm_inputs["image_grid_thw"] = experience.image_grid_thw

            micro_buffer.append(
                {
                    "sequences": experience.sequences,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "num_actions": experience.num_actions,
                    "old_action_log_probs": experience.action_log_probs,
                    "base_action_log_probs": experience.base_action_log_probs,
                    "advantages": experience.advantages,
                    "loss_mask": experience.loss_mask,
                    "rollout_action_logprobs": experience.rollout_logprobs,
                    "action_mask": experience.action_mask,
                    "rollout_expert_indices": rollout_expert_indices if self.enable_router_replay else None,
                    # used with global sequence packing (None when token-based batching is active)
                    "sub_seq_lengths": experience.sub_seq_lengths,
                    **vlm_inputs,
                }
            )

        # Count microbatches that carry real (non-padding) samples. Token-based batching
        # appends fully-padding microbatches (loss_mask all zero) so every DP rank runs the
        # same number of forward passes; those contribute 0 to KL/entropy and to mean metrics
        # but would otherwise inflate the denominators. `num_real_microbatches` lets the loss
        # normalize KL/entropy over real microbatches only.
        num_real_microbatches = sum(1 for m in micro_buffer if m["loss_mask"].sum().item() > 0)
        for m_batch in micro_buffer:
            m_batch["num_microbatches"] = len(micro_buffer)
            m_batch["num_real_microbatches"] = num_real_microbatches

        if not micro_buffer:
            return WorkerOutput()

        seq_len = micro_buffer[0]["sequences"].shape[1]

        if use_token_batching:
            # With token-based batching, microbatches may have different batch sizes.
            # Megatron's forward_backward_func requires uniform micro_batch_size,
            # so pad all microbatches to the max batch size across microbatches.
            max_micro_bsz = max(m["sequences"].shape[0] for m in micro_buffer)
            micro_buffer = [self._pad_microbatch_to_size(m, max_micro_bsz) for m in micro_buffer]
            micro_bsz = max_micro_bsz
        else:
            micro_bsz = micro_buffer[0]["sequences"].shape[0]

        # Gate on first PP/TP/CP rank so we emit exactly one line per DP rank
        # (matches how status all-reduce treats metrics as identical within a DP group).
        if (
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == 0
            and mpu.get_context_parallel_rank() == 0
        ):
            real_tokens = int(sum(int(mb["attention_mask"].sum().item()) for mb in micro_buffer))
            num_microbatches = len(micro_buffer)
            dp_rank = mpu.get_data_parallel_rank()
            logger.info(
                f"sequence packing | dp_rank={dp_rank} microbatches_this_step={num_microbatches} "
                f"seq_len={seq_len} tokens={real_tokens}"
            )

        metrics_list = self.model.forward_backward_mini_batch(
            micro_batches=micro_buffer,
            seq_len=seq_len,
            micro_batch_size=micro_bsz,
            temperature=self.cfg.algorithm.temperature,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
        )

        if self.empty_cuda_cache:
            torch.cuda.empty_cache()

        # Aggregate metrics across micro-batches
        all_loss_fn_outputs = []  # Handle separately from scalar metrics
        for m_batch, metrics in zip(micro_buffer, metrics_list):
            # Extract loss_fn_outputs before reduce_metrics (it's not a scalar metric)
            if "loss_fn_outputs" in metrics:
                all_loss_fn_outputs.extend(metrics.pop("loss_fn_outputs"))
            # Skip fully-padding microbatches: their metrics (clip_ratio=0, policy_entropy=0,
            # ...) are meaningless and would drag down the mean-reduced metrics. Summed
            # metrics (e.g. policy_loss) are unaffected since padding contributes 0, but
            # excluding them here keeps both reductions correct.
            if m_batch["loss_mask"].sum().item() == 0:
                continue
            for k, v in metrics.items():
                all_metrics[k].append(v)

        # TODO: SFT path still averages metrics across microbatches and workers.
        # This needs to be unified with the RL path which sums.
        resolved_loss_name = loss_fn or self.cfg.algorithm.policy_loss_type
        sum_loss_metrics = resolved_loss_name != "cross_entropy"

        # Reduce across microbatches and all-reduce metrics across DP ranks
        # (metrics should be identical within DP groups, i.e., across TP/PP/SP ranks)
        # NOTE: Sum loss metrics because scaling is already applied at the advantage level
        status = reduce_metrics(all_metrics, sum_loss_metrics=sum_loss_metrics)
        if self.optimizer is not None:
            status["policy_lr"] = self.optimizer.param_groups[0]["lr"]

        # Token-based batching diagnostics: total microbatches this rank ran and how many
        # were purely-padding (added to equalize the microbatch count across DP ranks).
        # Added before all-reduce so they are averaged across DP (num_microbatches is
        # identical on every rank; num_padding_microbatches reports the per-rank average).
        if use_token_batching:
            status["num_microbatches"] = float(len(micro_buffer))
            status["num_padding_microbatches"] = float(len(micro_buffer) - num_real_microbatches)

        group = mpu.get_data_parallel_group(with_context_parallel=False)
        status = all_reduce_metrics(status, self.strategy, group=group, sum_loss_metrics=sum_loss_metrics)

        # Collect MoE aux metrics averaged across microbatches (all-reduced across ranks
        # inside get_moe_metrics) aggregating after per-microbatch scalar metrics.
        total_num_microbatches = len(micro_buffer)
        model_config = get_model_config(self.actor_module[0])
        num_moe_experts = getattr(model_config, "num_moe_experts", None)
        moe_metrics: Dict[str, Any] = {}
        if num_moe_experts is not None and num_moe_experts > 1:
            moe_loss_scale = 1.0 / max(1, total_num_microbatches)
            moe_metrics = get_moe_metrics(
                loss_scale=moe_loss_scale,
                per_layer_logging=self.cfg.policy.megatron_config.moe_per_layer_logging,
            )
            # moe_metrics will only be non-empty if "moe_router_load_balancing_type" is set to "aux_loss", "seq_aux_loss", or "global_aux_loss"
            if moe_metrics:
                for k, v in moe_metrics.items():
                    status[k] = v

        clear_router_replay()

        return WorkerOutput(loss_fn_outputs=all_loss_fn_outputs, metrics=status)

    def optim_step(self) -> Optional[float]:
        """
        Perform optimizer step.

        Note: Unlike FSDP workers, Megatron doesn't need manual gradient scaling here
        because Megatron Core's forward_backward_func handles loss scaling internally.

        Returns:
            The gradient norm (before scaling, after clipping), or None if unavailable.
        """
        if self.optimizer is None:
            raise RuntimeError("optim_step called but policy.inference_only_init=True (no optimizer constructed)")
        grad_norm = self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler, name="actor")

        # Reset counter for next accumulation cycle
        self._micro_batches_accumulated = 0

        if grad_norm is not None:
            grad_norm = grad_norm.detach().cpu().item() if hasattr(grad_norm, "item") else grad_norm
        return grad_norm

    def get_lr(self) -> Optional[float]:
        """
        Get current learning rate from optimizer.

        Handles both regular optimizers and ChainedOptimizer. Returns None when
        the worker was initialized with ``policy.inference_only_init=True``.
        """
        if self.optimizer is None:
            return None
        if isinstance(self.optimizer, ChainedOptimizer):
            return self.optimizer.chained_optimizers[0].param_groups[0]["lr"]
        return self.optimizer.param_groups[0]["lr"]

    def set_lr(self, learning_rate: float) -> None:
        """
        Set learning rate for the optimizer.

        Handles both regular optimizers and ChainedOptimizer (used with
        distributed optimizer). Updates all param_groups across all
        underlying optimizers.

        Note: This bypasses the scheduler. The next scheduler.step() call
        will override this value unless the scheduler is configured for
        constant LR. No-op when ``policy.inference_only_init=True``.
        """
        if self.optimizer is None:
            return
        if isinstance(self.optimizer, ChainedOptimizer):
            # ChainedOptimizer wraps multiple optimizers (e.g., for different param groups)
            for opt in self.optimizer.chained_optimizers:
                for param_group in opt.param_groups:
                    param_group["lr"] = learning_rate
        else:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = learning_rate

    async def init_weight_sync_state(self, inference_engine_client, inference_engine_cfg: "InferenceEngineConfig"):
        # Call super first to set _transfer_strategy_cls and create sender/receivers
        await super().init_weight_sync_state(inference_engine_client, inference_engine_cfg)

        # Initialize weight extractor with bucketing enabled for all strategies
        self.weight_extractor = MegatronWeightExtractor(
            bridge=self.bridge,
            actor_module=self.actor_module,
            enable_bucketing=True,
            bucket_size_threshold_GB=inference_engine_cfg.weight_transfer_threshold_cuda_ipc_GB,
            training_dtype=torch.bfloat16 if self.cfg.bf16 else torch.float32,
        )

    async def _save_lora_adapters_and_sync(
        self, lora_sync_path, inference_engine_client, lora_name: str = SKYRL_LORA_ADAPTER_NAME
    ):
        """Export LoRA adapter weights via Megatron-Bridge and tell the inference engine to load them.

        All ranks participate in the collective export (TP/PP/EP gathering is
        handled internally by the bridge).  Only rank 0 writes to disk and
        sends the ``LoraLoadRequest``.
        """
        import json

        from megatron.bridge.models.conversion.peft_bridge import (
            build_adapter_config_dict,
            infer_target_modules_from_adapter_weights,
        )
        from safetensors.torch import save_file

        adapter_state = {}
        for name, tensor in self.bridge.export_adapter_weights(self.actor_module, cpu=True, show_progress=False):
            adapter_state[f"base_model.model.{name}"] = tensor.clone().float()

        if torch.distributed.get_rank() == 0:
            os.makedirs(lora_sync_path, exist_ok=True)

            target_modules = infer_target_modules_from_adapter_weights(adapter_state.keys())
            base_model_name_or_path = str(
                getattr(self.bridge.hf_pretrained, "model_name_or_path", "")
                or getattr(self.bridge.hf_pretrained, "name_or_path", "")
            )
            adapter_config = build_adapter_config_dict(
                self.lora_cls,
                target_modules=target_modules,
                base_model_name_or_path=base_model_name_or_path,
            )

            save_file(adapter_state, os.path.join(lora_sync_path, "adapter_model.safetensors"))
            with open(os.path.join(lora_sync_path, "adapter_config.json"), "w", encoding="utf-8") as f:
                json.dump(adapter_config, f, ensure_ascii=False, indent=4)

            # Send LoRA disk loading request to inference engine.
            from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
                RemoteInferenceClient,
            )

            if isinstance(inference_engine_client, RemoteInferenceClient):
                await inference_engine_client.load_lora_adapter(lora_name, lora_sync_path)
            else:
                lora_request = LoraLoadRequest(lora_path=lora_sync_path, lora_name=lora_name)
                await inference_engine_client.update_named_weights(lora_request)

        torch.distributed.barrier()

    async def broadcast_to_inference_engines(
        self,
        inference_engine_client: "InferenceEngineInterface",
        inference_engine_cfg: "InferenceEngineConfig",
        model_id: Optional[str] = None,
    ):
        use_prefix_cache = inference_engine_cfg.enable_prefix_caching
        generator_dtype = str_to_torch_dtype(inference_engine_cfg.model_dtype)
        cache_reset_task = None
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            cache_reset_task = inference_engine_client.reset_prefix_cache()

        torch.cuda.empty_cache()

        if self._is_lora and not self.cfg.policy.megatron_config.lora_config.merge_lora:
            # AdapterStore.swap_to has already made `model_id` the live adapter
            # before we get here; sync that adapter to vLLM under its own name
            # so sample(model=<model_id>) routes correctly. Single-tenant
            # (model_id=None) keeps the legacy shared path + name.
            lora_name, lora_sync_path = self._resolve_lora_sync_target(model_id)
            await self._save_lora_adapters_and_sync(lora_sync_path, inference_engine_client, lora_name=lora_name)
        else:
            # Extract and send weights using the sender created at init time.
            # Disable expandable_segments around the send: under colocate_all the
            # CUDA-IPC path calls cudaIpcGetMemHandle, which is incompatible with the
            # VMM addresses expandable segments uses.
            with self._expandable_segments_disabled_for_sync():
                weight_metadata = self.weight_extractor.get_weight_metadata(generator_dtype)
                await self._weight_transfer_sender.send_chunks(
                    self.weight_extractor.extract_weights(generator_dtype),
                    weight_metadata=weight_metadata,
                )

        if cache_reset_task is not None:
            await cache_reset_task
        torch.cuda.empty_cache()
        torch.distributed.barrier()

    def _set_pad_token_id(self, pad_token_id):
        # this already gets set in the init_model method
        pass

    # ------------------------------------------------------------------
    # Multi-LoRA / AdapterStore Ray-callable methods
    # ------------------------------------------------------------------

    def prime_optimizer_state(self) -> None:
        """Materialise DistributedOptimizer state (exp_avg / exp_avg_sq).

        Adam's state tensors are allocated lazily on the first non-trivial
        step; without priming, the pristine snapshot would miss them.
        Megatron exposes ``_init_optimizer_states_with_dummy_values()`` which
        zero-fills grads + steps once + zero_grads, leaving the model weights
        unchanged.
        """
        if not self._is_lora:
            raise RuntimeError("prime_optimizer_state is only used on the LoRA path")
        for _opt in iter_opts(self.optimizer):
            init_fn = getattr(_opt, "_init_optimizer_states_with_dummy_values", None)
            if init_fn is not None:
                init_fn()

    def register_pristine_adapter(self) -> None:
        """Capture the current (freshly-initialised) LoRA state as the
        pristine slot. Must be called once per worker, after
        prime_optimizer_state.
        """
        if self.adapter_store is None:
            raise RuntimeError("AdapterStore not initialised (FFT path)")
        signature = LoraSignature.from_lora_config(
            self.cfg.policy.model.lora,
            lora_type=self.cfg.policy.megatron_config.lora_config.lora_type,
        )
        self.adapter_store.register_pristine(self.actor_module, self.optimizer, signature)

    def register_adapter(self, model_id: str) -> None:
        """Register a new LoRA adapter slot. The first call uses the live
        state as the slot; subsequent calls seed from pristine.
        """
        if self.adapter_store is None:
            raise RuntimeError("AdapterStore not initialised (FFT path)")
        signature = self.adapter_store.signature
        if signature is None:
            raise RuntimeError("register_adapter called before register_pristine_adapter")
        self.adapter_store.create(model_id, self.actor_module, self.optimizer, signature)

    def delete_adapter(self, model_id: str) -> None:
        if self.adapter_store is None:
            raise RuntimeError("AdapterStore not initialised (FFT path)")
        self.adapter_store.delete(model_id)
        # Drop the per-tenant safetensors subdir written by
        # _save_lora_adapters_and_sync. Rank 0 wrote it; rank 0 cleans it.
        # Other ranks no-op. Best-effort — log on failure but don't propagate.
        if self._rank == 0:
            _, lora_sync_path = self._resolve_lora_sync_target(model_id)
            base_sync_path = self.cfg.policy.model.lora.lora_sync_path
            if lora_sync_path != base_sync_path:
                try:
                    shutil.rmtree(lora_sync_path)
                except FileNotFoundError:
                    pass  # already gone, fine
                except OSError as e:
                    logger.warning(f"Failed to remove lora_sync subdir {lora_sync_path}: {e}")

    def swap_to_adapter(self, model_id: str) -> None:
        """Make ``model_id`` the live adapter on this worker. No-op if it
        already is. Issues local tensor.copy_()s + dp_group barriers.
        """
        if self.adapter_store is None:
            return  # FFT path: no-op
        self.adapter_store.swap_to(model_id, self.actor_module, self.optimizer)

    def adapter_store_state(self) -> dict:
        """Diagnostic: return current_id + registered model_ids. Cheap; useful
        for tests."""
        if self.adapter_store is None:
            return {"enabled": False}
        return {
            "enabled": True,
            "current_id": self.adapter_store.current_id,
            "registered": self.adapter_store.registered_ids(),
            "num_adapters": self.adapter_store.num_adapters(),
        }


class MegatronRefWorkerBase(MegatronWorker, RefWorkerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: MegatronModelWrapper = None
        self.actor_module: List[nn.Module] = None

    def forward(self, data: TrainingInputBatch) -> WorkerOutput:
        """Run inference forward pass.

        Returns a :class:`WorkerOutput` whose ``loss_fn_outputs`` carries one
        per-sample dict with key ``"logprobs"``. Token-based micro-batching (when
        ``max_tokens_per_microbatch > 0``) is handled inside ``_forward_logprobs``.
        """
        log_probs = self._forward_logprobs(data)
        loss_fn_outputs = [{"logprobs": log_probs[i].tolist()} for i in range(log_probs.shape[0])]
        return WorkerOutput(loss_fn_outputs=loss_fn_outputs, metrics={})

    def init_worker_process_group(self):
        """
        Override DistributedTorchRayActor.init_worker_process_group to use megatron distributed setup to create the mesh.
        """
        if not torch.distributed.is_initialized():
            # Ensure CUDA device is set before process group init — required when
            # using split "cpu:gloo,cuda:nccl" backend to avoid 'invalid device ordinal'
            # errors during NCCL communicator creation in subgroups.
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            # Default torch dist pg init timeout is 10 minutes (600 seconds)
            torch.distributed.init_process_group(
                backend="cpu:gloo,cuda:nccl", timeout=timedelta(seconds=SKYRL_WORKER_NCCL_TIMEOUT_IN_S)
            )

        self.strategy = MegatronStrategy(
            megatron_config=self.cfg.ref.megatron_config,
            optimizer_config=None,
            seed=self.cfg.seed,
            node_local_rank=self._local_rank,
        )
        self.strategy.setup_distributed()

        self.mesh_rank = MeshRank(
            dp=mpu.get_data_parallel_rank(),
            sp=mpu.get_context_parallel_rank(),
            tp=mpu.get_tensor_model_parallel_rank(),
            pp=mpu.get_pipeline_model_parallel_rank(),
            world_size=self._world_size,
            dp_size=mpu.get_data_parallel_world_size(),
            pp_size=mpu.get_pipeline_model_parallel_world_size(),
        )

    def init_model(self, model_path, num_training_steps: int = 1e9):
        """
        Initialize the model for the ref worker.
        """
        # initialize the bridge and provider objects
        self.init_configs(
            model_path,
            self.cfg.ref.megatron_config,
            self.cfg.ref.megatron_config.model_config_kwargs,
            self.cfg.ref.megatron_config.transformer_config_kwargs,
            bf16=self.cfg.bf16,
            flash_attn=self.cfg.flash_attn,
        )

        self.actor_module = self.make_megatron_module(
            wrap_with_ddp=False,
            ddp_config=None,
            bf16=self.cfg.bf16,
        )

        # download model weights from huggingface (need to be done for ref worker as well, else errors when colocate_all=False)
        if self._local_rank == 0 and not os.path.exists(
            model_path
        ):  # if not local path, try downloading model weights from huggingface
            snapshot_download(model_path)  # will be no-op if already downloaded
        torch.distributed.barrier()

        # load weights
        if self._rank == 0:
            print_model_size(self.actor_module[0])

        # create worker model
        # Ref worker does not handle VLM inputs during SFT, so is_vlm stays False (the wrapper default).
        self.model = MegatronModelWrapper(config=self.cfg, actor_module=self.actor_module)

        self._set_expandable_segments(True)

    def _set_pad_token_id(self, pad_token_id):
        # this already gets set in the init_model method
        pass


class MegatronCriticWorkerBase(MegatronWorker, CriticWorkerBase):
    def __init__(self, **kwargs):
        raise NotImplementedError()


PolicyWorker = ray.remote(num_gpus=1)(MegatronPolicyWorkerBase)
RefWorker = ray.remote(num_gpus=1)(MegatronRefWorkerBase)
CriticWorker = ray.remote(num_gpus=1)(MegatronCriticWorkerBase)
