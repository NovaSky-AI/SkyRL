"""
WorkerDispatch: Manages all actor groups with automatic offload/onload.

Automatically handles GPU placement:
- Tracks which model is currently on GPU
- If colocation is enabled, offloads other models when one is requested

The trainer interacts with the worker dispatch if all models are always on GPU.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import ray
from ray import ObjectRef

from omegaconf import DictConfig
from skyrl_train.config import SkyRLConfig
from skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch
from skyrl_train.workers.worker import PPORayActorGroup
from skyrl_train.distributed.dispatch import MeshDispatch


@dataclass
class GPUState:
    """Tracks what's on GPU for a model."""

    model_on_gpu: bool = False
    optimizer_on_gpu: bool = False


class WorkerDispatch:
    """
    Unified dispatch layer that manages all actor groups (policy, critic, ref).

    Handles automatic offload/onload when colocate_all=True.
    """

    def __init__(
        self,
        cfg: Union[SkyRLConfig, DictConfig],
        policy_actor_group: PPORayActorGroup,
        critic_actor_group: Optional[PPORayActorGroup] = None,
        ref_actor_group: Optional[PPORayActorGroup] = None,
        inference_engine_client: Optional[InferenceEngineClient] = None,
    ):
        self.cfg = cfg
        self.colocate_all = cfg.trainer.placement.colocate_all
        self.colocate_policy_ref = cfg.trainer.placement.colocate_policy_ref

        # Inference engine client for weight sync (optional)
        self._inference_engine_client = inference_engine_client

        # Actor groups by name.
        # TODO: Remove these role-specific identifiers. We will move to using model IDs and add support for generic models beyond these.
        self._actor_groups: Dict[str, PPORayActorGroup] = {"policy": policy_actor_group}
        if critic_actor_group is not None:
            self._actor_groups["critic"] = critic_actor_group
        if ref_actor_group is not None:
            self._actor_groups["ref"] = ref_actor_group

        # GPU state tracking (only matters when colocated)
        self._gpu_state: Dict[str, GPUState] = {name: GPUState() for name in self._actor_groups.keys()}

    def get_lcm_dp_size(self) -> int:
        """Get LCM of all models' dp_size."""
        import math

        dp_size = self._actor_groups["policy"].actor_infos[0].rank.dp_size
        if "critic" in self._actor_groups:
            dp_size = math.lcm(dp_size, self._actor_groups["critic"].actor_infos[0].rank.dp_size)
        if "ref" in self._actor_groups:
            dp_size = math.lcm(dp_size, self._actor_groups["ref"].actor_infos[0].rank.dp_size)
        return dp_size

    def _should_manage_offload(self, model: str) -> bool:
        """Check if we need to manage offload for this model."""
        if self.colocate_all:
            return True
        if self.colocate_policy_ref and model in ("policy", "ref"):
            return True
        return False

    def _get_colocation_group(self, model: str) -> List[str]:
        """Get which models share GPU with the given model."""
        if self.colocate_all:
            return list(self._actor_groups.keys())
        elif self.colocate_policy_ref and model in ("policy", "ref"):
            return [m for m in ["policy", "ref"] if m in self._actor_groups]
        return [model]

    def _ensure_on_gpu(self, model: str, need_optimizer: bool = True, need_model: bool = True) -> None:
        """Ensure model is on GPU, offloading others in same colocation group if needed."""
        if not self._should_manage_offload(model):
            return

        if model not in self._actor_groups:
            return

        group = self._get_colocation_group(model)

        # Offload others in the same colocation group
        for other in group:
            if other != model and other in self._actor_groups:
                state = self._gpu_state[other]
                if state.model_on_gpu or state.optimizer_on_gpu:
                    self._actor_groups[other].offload_to_cpu()
                    self._gpu_state[other] = GPUState()

        # Backload requested model
        state = self._gpu_state[model]
        needs_backload = (need_model and not state.model_on_gpu) or (need_optimizer and not state.optimizer_on_gpu)

        if needs_backload:
            self._actor_groups[model].backload_to_gpu(
                backload_optimizer=need_optimizer,
                backload_model=need_model,
            )
            if need_model:
                self._gpu_state[model].model_on_gpu = True
            if need_optimizer:
                self._gpu_state[model].optimizer_on_gpu = True

    def _offload(self, model: str, offload_optimizer: bool = True, offload_model: bool = True) -> None:
        """Offload model to CPU."""
        if not self._should_manage_offload(model):
            return

        if model not in self._actor_groups:
            return

        self._actor_groups[model].offload_to_cpu(
            offload_optimizer=offload_optimizer,
            offload_model=offload_model,
        )

        if offload_model:
            self._gpu_state[model].model_on_gpu = False
        if offload_optimizer:
            self._gpu_state[model].optimizer_on_gpu = False

    def mark_all_offloaded(self) -> None:
        """Mark all models as offloaded (call after build_models when colocate_all)."""
        for model in self._actor_groups:
            self._gpu_state[model] = GPUState()

    def forward(self, model: str, data: TrainingInputBatch) -> TrainingOutputBatch:
        """Run inference forward pass. Only loads model (not optimizer)."""
        self._ensure_on_gpu(model, need_optimizer=False, need_model=True)

        refs = self._actor_groups[model].async_run_ray_method("mesh", "forward", data=data)
        results = ray.get(refs)

        output = concatenate_outputs_after_mesh_dispatch(self._actor_groups[model].actor_infos, results)
        return output

    def stage_data(self, data: TrainingInputBatch) -> ObjectRef:
        """
        Put training data in Ray object store for efficient access.

        Call this once to avoid repeated serialization when dispatching
        data to workers

        Args:
            data: Full training batch to stage in the ray object store

        Returns:
            ObjectRef to the staged data
        """
        return ray.put(data)

    def forward_backward(
        self,
        model: str,
        data: TrainingInputBatch,
        loss_fn: Optional[str] = None,
        loss_fn_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Run forward/backward pass. Needs model + optimizer.

        Args:
            model: Model identifier ("policy", "critic", or "ref")
            data: Training batch data
            loss_fn: Optional loss function name (e.g., "cross_entropy", "ppo").
                     If provided, overrides the config's policy_loss_type.
            loss_fn_config: Optional config overrides for the loss function
                           (e.g., {"clip_low_threshold": 0.9} for PPO)

        Returns:
            Dictionary of training metrics
        """
        self._ensure_on_gpu(model, need_optimizer=True, need_model=True)

        # Only pass kwargs that are not None (critic worker doesn't accept loss_fn)
        kwargs = {}
        if loss_fn is not None:
            kwargs["loss_fn"] = loss_fn
        if loss_fn_config is not None:
            kwargs["loss_fn_config"] = loss_fn_config

        refs = self._actor_groups[model].async_run_ray_method("mesh", "forward_backward", data, **kwargs)
        statuses = ray.get(refs)

        self._save_memory_snapshot(model, "forward_backward")

        # With DP>1, each rank returns loss_fn_outputs for its data chunk.
        # Concatenate them in rank order to get the full batch's outputs.
        # Scalar metrics (loss, lr) are already all-reduced, so use statuses[0] for those.
        if len(statuses) > 1 and statuses[0] and "loss_fn_outputs" in statuses[0]:
            all_loss_fn_outputs = []
            for status in statuses:
                all_loss_fn_outputs.extend(status.pop("loss_fn_outputs", []))
            result = statuses[0]
            result["loss_fn_outputs"] = all_loss_fn_outputs
            return result

        return statuses[0]

    def forward_backward_from_staged(
        self,
        model: str,
        data_ref: ObjectRef,
        start_idx: int,
        end_idx: int,
        loss_fn: Optional[str] = None,
        loss_fn_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Run forward/backward pass using pre-staged data from object store.

        Workers fetch the full batch from object store and slice locally,
        avoiding repeated serialization for each mini-batch.

        Args:
            model: Model name ("policy" or "critic")
            data_ref: ObjectRef to staged TrainingInputBatch (from stage_data)
            start_idx: Start index for mini-batch slice
            end_idx: End index for mini-batch slice

        Returns:
            Aggregated metrics dict from training
        """
        self._ensure_on_gpu(model, need_optimizer=True, need_model=True)

        # Only pass kwargs that are not None (critic worker doesn't accept loss_fn)
        kwargs = {}
        if loss_fn is not None:
            kwargs["loss_fn"] = loss_fn
        if loss_fn_config is not None:
            kwargs["loss_fn_config"] = loss_fn_config

        # Use specialized dispatch that passes ObjectRef + indices instead of data
        refs = MeshDispatch.dispatch_from_staged(
            self._actor_groups[model].actor_infos,
            "forward_backward_from_staged",
            data_ref=data_ref,
            start_idx=start_idx,
            end_idx=end_idx,
            **kwargs,
        )
        statuses = ray.get(refs)

        self._save_memory_snapshot(model, "forward_backward")
        return statuses[0]

    def optim_step(self, model: str) -> Optional[float]:
        """Run optimizer step. Model should already be on GPU from forward_backward."""
        refs = self._actor_groups[model].async_run_ray_method("pass_through", "optim_step")
        grad_norms = ray.get(refs)

        self._save_memory_snapshot(model, "optim_step")
        return grad_norms[0]

    def set_lr(self, model: str, learning_rate: float) -> None:
        """Set learning rate for model's optimizer.

        This directly updates the optimizer's param_groups on all workers,
        bypassing the scheduler. Useful for external learning rate schedules.
        """
        self._ensure_on_gpu(model, need_optimizer=True, need_model=False)
        ray.get(self._actor_groups[model].async_run_ray_method("pass_through", "set_lr", learning_rate=learning_rate))

    def _save_memory_snapshot(self, model: str, tag: str) -> None:
        """Save memory snapshot on workers."""
        ray.get(
            self._actor_groups[model].async_run_ray_method("pass_through", "save_memory_snapshot", tag=f"{model}_{tag}")
        )

    def save_checkpoint(self, model: str, ckpt_dir: str, tokenizer=None) -> None:
        """Save checkpoint for model."""
        self._ensure_on_gpu(model, need_optimizer=True, need_model=True)

        ray.get(
            self._actor_groups[model].async_run_ray_method(
                "pass_through", "save_checkpoint", ckpt_dir=ckpt_dir, tokenizer=tokenizer
            )
        )

    def load_checkpoint(
        self,
        model: str,
        ckpt_dir: str,
        load_optimizer_states: bool = True,
        load_lr_scheduler_states: bool = True,
    ) -> None:
        """Load checkpoint for model."""
        self._ensure_on_gpu(model, need_optimizer=load_optimizer_states, need_model=True)

        ray.get(
            self._actor_groups[model].async_run_ray_method(
                "pass_through",
                "load_checkpoint",
                ckpt_dir=ckpt_dir,
                load_optimizer_states=load_optimizer_states,
                load_lr_scheduler_states=load_lr_scheduler_states,
            )
        )

    def save_hf_model(self, model: str, export_dir: str, tokenizer) -> None:
        """Save model in HuggingFace format."""
        self._ensure_on_gpu(model, need_optimizer=False, need_model=True)

        ray.get(self._actor_groups[model].async_run_ray_method("pass_through", "save_hf_model", export_dir, tokenizer))

    def init_model(self, model: str, model_path: str, num_training_steps: Optional[int] = None) -> None:
        """Initialize model from path. Offloads others in colocation group first."""
        # Offload others in colocation group before init
        if self._should_manage_offload(model):
            group = self._get_colocation_group(model)
            for other in group:
                if other != model and other in self._actor_groups:
                    state = self._gpu_state[other]
                    if state.model_on_gpu or state.optimizer_on_gpu:
                        self._actor_groups[other].offload_to_cpu()
                        self._gpu_state[other] = GPUState()

        kwargs = {"model_path": model_path}
        if num_training_steps is not None:
            kwargs["num_training_steps"] = num_training_steps

        ray.get(self._actor_groups[model].async_init_model(**kwargs))

        # After init, model is on GPU
        self._gpu_state[model].model_on_gpu = True
        self._gpu_state[model].optimizer_on_gpu = model != "ref"  # ref has no optimizer

    def init_weight_sync_state(self, inference_engine_client) -> None:
        """Initialize weight sync state for policy model."""
        ray.get(
            self._actor_groups["policy"].async_run_ray_method(
                "pass_through", "init_weight_sync_state", inference_engine_client
            )
        )

    def broadcast_to_inference_engines(self, inference_engine_client) -> None:
        """Broadcast policy weights to inference engines."""
        ray.get(
            self._actor_groups["policy"].async_run_ray_method(
                "pass_through", "broadcast_to_inference_engines", inference_engine_client
            )
        )

    def prepare_for_weight_sync(self) -> None:
        """Prepare for weight sync: ensure policy model is on GPU, offload optimizer."""
        if not self.colocate_all:
            return
        # Ensure policy model is on GPU (will offload others in colocation group)
        self._ensure_on_gpu("policy", need_optimizer=False, need_model=True)
        # Offload optimizer if it's on GPU
        if self._gpu_state["policy"].optimizer_on_gpu:
            self._offload("policy", offload_optimizer=True, offload_model=False)

    def finish_weight_sync(self) -> None:
        """Finish weight sync: offload model."""
        if not self.colocate_all:
            return
        self._offload("policy", offload_optimizer=False, offload_model=True)

    async def save_weights_for_sampler(self) -> None:
        """
        Tinker API method to prepare updated parameters for sampling.

        Syncs weights to inference engine for sampling.
        """
        if self._inference_engine_client is None:
            raise RuntimeError(
                "Cannot save_weights_for_sampler: no inference_engine_client configured. "
                "Pass inference_engine_client to WorkerDispatch constructor or call set_inference_engine_client()."
            )

        # Sync weights to inference engine
        self.prepare_for_weight_sync()
        if self.colocate_all:
            await self._inference_engine_client.wake_up(tags=["weights"])
        self.broadcast_to_inference_engines(self._inference_engine_client)
        self.finish_weight_sync()
        if self.colocate_all:
            await self._inference_engine_client.wake_up(tags=["kv_cache"])

    def set_inference_engine_client(self, inference_engine_client: InferenceEngineClient) -> None:
        """Set the inference engine client for weight sync.

        This can be called after construction if the client isn't available at init time.
        """
        self._inference_engine_client = inference_engine_client

    def empty_cache(self, model: Optional[str] = None) -> None:
        """Empty GPU cache for model(s)."""
        if model is not None:
            ray.get(self._actor_groups[model].async_run_ray_method("pass_through", "empty_cache"))
        else:
            refs = []
            for group in self._actor_groups.values():
                refs.extend(group.async_run_ray_method("pass_through", "empty_cache"))
            ray.get(refs)

    def get_node_ids(self) -> List[str]:
        """Get unique node IDs from all actor groups."""
        all_node_ids = []
        for group in self._actor_groups.values():
            node_ids = ray.get(group.async_run_ray_method("pass_through", "get_ray_node_id"))
            all_node_ids.extend(node_ids)
        return list(set(all_node_ids))
