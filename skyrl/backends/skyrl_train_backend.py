"""SkyRL-Train backend for TinkerEngine."""

import asyncio
import io
import os
import tarfile
import tempfile

import ray
import torch
from pydantic import BaseModel
from ray.util.placement_group import placement_group
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from skyrl.backends.backend import AbstractBackend
from skyrl.backends.renderer import VLLMRenderer, render_model_input
from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import (
    InferenceEngineClient,
)
from skyrl.backends.skyrl_train.inference_engines.ray_wrapped_inference_engine import (
    create_ray_wrapped_inference_engines,
)
from skyrl.backends.skyrl_train.training_batch import (
    TensorList,
    TrainingInputBatch,
    pad_training_input_batch,
)
from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
from skyrl.backends.skyrl_train.workers.worker_dispatch import WorkerDispatch
from skyrl.env_vars import _SKYRL_USE_NEW_INFERENCE, SKYRL_RAY_PG_TIMEOUT_IN_S
from skyrl.tinker import types
from skyrl.train.config import SkyRLTrainConfig, get_config_as_yaml_str
from skyrl.train.utils.utils import (
    ResolvedPlacementGroup,
    get_ray_pg_ready_with_timeout,
    initialize_ray,
)
from skyrl.utils.log import logger
from skyrl.utils.tok import get_tokenizer


class SkyRLTrainBackendOverrides(BaseModel, extra="allow"):
    """Configuration overrides for the SkyRL-Train backend.

    All keys are applied as overrides to the default SkyRL-Train config.
    """

    pass


class FSDPBackendOverrides(SkyRLTrainBackendOverrides):
    strategy: str = "fsdp2"


class MegatronBackendOverrides(SkyRLTrainBackendOverrides):
    strategy: str = "megatron"


def _build_skyrl_train_config(
    base_model: str,
    overrides: SkyRLTrainBackendOverrides,
    lora_config: types.LoraConfig | None = None,
) -> SkyRLTrainConfig:
    """Build config for SkyRL-Train workers using default config with overrides.

    Args:
        base_model: HuggingFace model path
        config_container: Backend configuration
        lora_config: LoRA configuration if using LoRA
    """

    # Apply user overrides from backend_config
    user_overrides = dict(overrides.model_extra)
    # override base model path
    # NOTE: It is better to add this as a part of the CLI overrides since we have post_init logic
    # that resolves other config derived from the policy model path.
    user_overrides["trainer.policy.model.path"] = base_model
    user_overrides["trainer.critic.model.path"] = base_model
    cfg = SkyRLTrainConfig.from_cli_overrides(user_overrides)

    # Disable scheduler - Tinker manages learning rate externally via set_lr()
    cfg.trainer.policy.optimizer_config.scheduler = "constant_with_warmup"
    cfg.trainer.policy.optimizer_config.num_warmup_steps = 0
    cfg.trainer.critic.optimizer_config.scheduler = "constant_with_warmup"
    cfg.trainer.critic.optimizer_config.num_warmup_steps = 0

    # TODO(tyler): Support KL Loss
    cfg.trainer.algorithm.use_kl_loss = False

    assert overrides.strategy in (
        "fsdp2",
        "megatron",
    ), "Only fsdp and megatron are supported for SkyRL-Train backend"
    cfg.trainer.strategy = overrides.strategy

    # Apply LoRA configuration
    if lora_config is not None and lora_config.rank > 0:
        cfg.trainer.policy.model.lora.rank = lora_config.rank
        cfg.trainer.policy.model.lora.alpha = int(lora_config.alpha)

    logger.info("SkyRL-Train config:\n%s", get_config_as_yaml_str(cfg))
    return cfg


class SkyRLTrainBackend(AbstractBackend):
    """SkyRL-Train backend for supervised training."""

    def __init__(self, base_model: str, config: SkyRLTrainBackendOverrides):
        logger.warning("=" * 80)
        logger.warning("SkyRLTrainBackend is currently EXPERIMENTAL!")
        logger.warning("=" * 80)

        if ray is None:
            raise ImportError(
                "SkyRLTrainBackend requires `ray`. Install the appropriate extras (e.g. `--extra skyrl_train`)."
            )

        self.base_model = base_model
        # NOTE: We currently have two config attributes "config" which is just config overrides and "_cfg" which is the actual config object. This is a temporary state given that the Tinker engine expects a .config attribute
        self.config = config
        self._model_ids_to_role: dict[str, str] = {}
        self._model_metadata: dict[str, types.ModelMetadata] = {}
        self._cfg = None
        self._dispatch: WorkerDispatch | None = None
        self._colocate_pg: ResolvedPlacementGroup | None = None
        self._tokenizer: AutoTokenizer = get_tokenizer(self.base_model)
        self._inference_engine_client = None
        self._inference_engines_initialized = False
        self._renderer = None

        # New inference infrastructure
        self._server_groups: list = []
        self._inference_router = None

    def has_model(self, model_id: str) -> bool:
        return model_id in self._model_ids_to_role

    def _get_role(self, model_id: str) -> str:
        try:
            return self._model_ids_to_role[model_id]
        except KeyError as exc:
            raise ValueError(f"Model {model_id} not found") from exc

    def _get_batch_role(self, model_ids: list[str]) -> str:
        if not model_ids:
            return "policy"
        unique_model_ids = set(model_ids)
        if len(unique_model_ids) != 1:
            raise ValueError(f"Mixed model_ids in one batch are not supported: {sorted(unique_model_ids)}")
        return self._get_role(next(iter(unique_model_ids)))

    def _split_model_pass_batch_by_model_id(
        self,
        prepared_batch: types.PreparedModelPassBatch,
    ) -> list[types.PreparedModelPassBatch]:
        """Split a mixed model-pass batch into per-model sub-batches.

        The engine batches pending forward/forward_backward requests across all
        models. Worker dispatch still executes one logical training model at a
        time, so mixed batches must be partitioned here while preserving
        request-level boundaries.
        """
        unique_model_ids = list(dict.fromkeys(prepared_batch.all_model_ids))
        if len(unique_model_ids) <= 1:
            return [prepared_batch]

        batch_fields = (
            "all_model_inputs",
            "all_targets",
            "all_token_weights",
            "all_sampling_logprobs",
            "all_advantages",
            "all_values",
            "all_returns",
            "all_model_ids",
            "all_loss_fns",
            "all_loss_fn_configs",
        )

        request_slices_by_model_id: dict[str, list[tuple[str, str, int, int]]] = {}
        for request_id, model_id, start_idx, end_idx in prepared_batch.request_batch_slices:
            # Validate early so an unknown model_id still surfaces clearly.
            self._get_role(model_id)
            request_slices_by_model_id.setdefault(model_id, []).append((request_id, model_id, start_idx, end_idx))

        sub_batches = []
        for request_slices in request_slices_by_model_id.values():
            sub_batch_data = {field: [] for field in batch_fields}
            sub_request_batch_slices = []

            for request_id, model_id, start_idx, end_idx in request_slices:
                sub_start_idx = len(sub_batch_data["all_model_inputs"])
                for field in batch_fields:
                    sub_batch_data[field].extend(getattr(prepared_batch, field)[start_idx:end_idx])
                sub_end_idx = len(sub_batch_data["all_model_inputs"])
                sub_request_batch_slices.append((request_id, model_id, sub_start_idx, sub_end_idx))

            sub_batches.append(
                types.PreparedModelPassBatch(
                    request_batch_slices=sub_request_batch_slices,
                    **sub_batch_data,
                )
            )

        return sub_batches

    def _build_policy(self, PolicyWorker):
        cfg = self._cfg
        colocate_all = cfg.trainer.placement.colocate_all
        pg = self._colocate_pg

        if colocate_all:
            assert pg is not None, "placement group must be created when colocate_all=True"
            num_policy_gpus = cfg.trainer.placement.policy_num_gpus_per_node * cfg.trainer.placement.policy_num_nodes
            num_rollout_gpus = (
                cfg.generator.inference_engine.num_engines
                * cfg.generator.inference_engine.tensor_parallel_size
                * cfg.generator.inference_engine.pipeline_parallel_size
                * cfg.generator.inference_engine.data_parallel_size
            )
            assert (
                num_policy_gpus == num_rollout_gpus
            ), "num_policy_gpus and num_rollout_gpus must be the same when colocating all models"

        policy_model = PPORayActorGroup(
            cfg.trainer,
            cfg.trainer.placement.policy_num_nodes,
            cfg.trainer.placement.policy_num_gpus_per_node,
            PolicyWorker,
            pg=pg,
            num_gpus_per_actor=0.2 if colocate_all else 1,
            colocate_all=colocate_all,
            sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
            record_memory=cfg.trainer.policy.record_memory,
        )

        # set to a large number for megatron scheduler init
        # lr will be managed externally via set_lr()
        policy_num_training_steps = 1e9
        ray.get(
            policy_model.async_init_model(
                cfg.trainer.policy.model.path,
                num_training_steps=policy_num_training_steps,
            )
        )
        ray.get(policy_model.async_run_ray_method("pass_through", "_set_pad_token_id", self._tokenizer.pad_token_id))

        if colocate_all:
            policy_model.offload_to_cpu()

        # Create unified dispatch that manages all actor groups
        self._dispatch = WorkerDispatch(
            cfg=cfg,
            policy_actor_group=policy_model,
            inference_engine_client=self._inference_engine_client,
        )

        # Mark the offloaded policy model in dispatch state
        if colocate_all:
            self._dispatch.mark_as_offloaded("policy")

        logger.info("init policy model done")

    def _build_critic(self, CriticWorker, lora_config: types.LoraConfig) -> None:
        cfg = self._cfg
        colocate_all = cfg.trainer.placement.colocate_all
        if colocate_all:
            num_policy_gpus = cfg.trainer.placement.policy_num_gpus_per_node * cfg.trainer.placement.policy_num_nodes
            num_critic_gpus = cfg.trainer.placement.critic_num_gpus_per_node * cfg.trainer.placement.critic_num_nodes
            assert (
                num_policy_gpus == num_critic_gpus
            ), "num_policy_gpus and num_critic_gpus must be the same when colocating policy and critic model"

        cfg.trainer.critic.model.lora.rank = lora_config.rank
        cfg.trainer.critic.model.lora.alpha = int(lora_config.alpha)
        critic_model = PPORayActorGroup(
            cfg.trainer,
            cfg.trainer.placement.critic_num_nodes,
            cfg.trainer.placement.critic_num_gpus_per_node,
            CriticWorker,
            pg=self._colocate_pg,
            num_gpus_per_actor=0.2 if colocate_all else 1,
            colocate_all=colocate_all,
            sequence_parallel_size=cfg.trainer.critic.sequence_parallel_size,
        )
        self._dispatch.register_actor_group("critic", critic_model)
        self._dispatch.init_model("critic", cfg.trainer.critic.model.path, num_training_steps=1e9)
        ray.get(critic_model.async_run_ray_method("pass_through", "_set_pad_token_id", self._tokenizer.pad_token_id))
        if colocate_all:
            critic_model.offload_to_cpu()
            self._dispatch.mark_as_offloaded("critic")
        logger.info("init critic model done")

    def init_weight_sync_state(self):
        """
        Setup the connection between policy model and inference engine for weight syncing.
        """
        self._dispatch.init_weight_sync_state(self._inference_engine_client)
        logger.info("Initialized weight sync state for policy model and inference engines.")

    def _create_legacy_inference_client(self):
        """Create legacy inference client using Ray-wrapped engines."""
        logger.info(f"Creating {self._cfg.generator.inference_engine.num_engines} Ray-wrapped inference engines")
        self._inference_engine_client = InferenceEngineClient(
            create_ray_wrapped_inference_engines_from_config(self._cfg, self._colocate_pg, self._tokenizer),
            self._tokenizer,
            self._cfg.trainer.policy.model.path,
            self._cfg.trainer.policy.model.lora,
            self._cfg.generator.inference_engine,
        )

    def _create_new_inference_client(self):
        """Create new HTTP-based inference client."""
        from skyrl.backends.skyrl_train.inference_servers.setup import (
            build_new_inference_client,
        )

        is_colocated = self._cfg.trainer.placement.colocate_all
        client, server_setup = build_new_inference_client(
            self._cfg,
            self._tokenizer,
            placement_group=self._colocate_pg if is_colocated else None,
        )
        self._inference_router = server_setup.router
        self._server_groups = server_setup.server_groups
        self._inference_engine_client = client

    def _ensure_inference_engines(self):
        """Lazily create inference engines and init weight sync on first sampling-related call."""
        if self._inference_engines_initialized:
            return

        if _SKYRL_USE_NEW_INFERENCE:
            self._create_new_inference_client()
        else:
            self._create_legacy_inference_client()

        self._dispatch.set_inference_engine_client(self._inference_engine_client)
        self.init_weight_sync_state()
        self._inference_engines_initialized = True

    def create_model(self, model_id: str, lora_config: types.LoraConfig, model_role: str = "policy") -> None:
        if model_id in self._model_ids_to_role:
            raise ValueError(f"Model '{model_id}' already exists")
        if model_role in self._model_ids_to_role.values():
            raise ValueError(f"SkyRLTrainBackend already has a '{model_role}' model")

        if model_role == "policy":
            self._cfg = _build_skyrl_train_config(self.base_model, self.config, lora_config)

            if not ray.is_initialized():
                logger.info("Initializing Ray with runtime environment")
                initialize_ray(self._cfg)

            self._colocate_pg = self._create_colocate_pg() if self._cfg.trainer.placement.colocate_all else None

            if self._cfg.trainer.strategy in ("fsdp", "fsdp2"):
                from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import (
                    PolicyWorker,
                )
            elif self._cfg.trainer.strategy == "megatron":
                from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
                    PolicyWorker,
                )
            else:
                raise ValueError(f"Unknown strategy type: {self._cfg.trainer.strategy}")

            logger.info("Building models.")
            self._build_policy(PolicyWorker)
        elif model_role == "critic":
            if "policy" not in self._model_ids_to_role.values():
                raise ValueError("Create a policy model before creating a critic model")
            if self._cfg.trainer.strategy in ("fsdp", "fsdp2"):
                from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import (
                    CriticWorker,
                )
            elif self._cfg.trainer.strategy == "megatron":
                raise NotImplementedError("Critic model support is not implemented for the Megatron backend yet")
            else:
                raise ValueError(f"Unknown strategy type: {self._cfg.trainer.strategy}")
            self._build_critic(CriticWorker, lora_config)
        else:
            raise ValueError(f"Unknown model_role: {model_role}")

        self._model_ids_to_role[model_id] = model_role
        self._model_metadata[model_id] = types.ModelMetadata(adapter_index=0, lora_config=lora_config)
        logger.info(f"Created {model_role} model {model_id} using RayPPOTrainer")

    def _create_colocate_pg(self):
        """Create a placement group for colocated training + inference."""
        ie_cfg = self._cfg.generator.inference_engine
        per_engine_gpu_count = ie_cfg.tensor_parallel_size * ie_cfg.pipeline_parallel_size * ie_cfg.data_parallel_size
        total_gpu_slots = ie_cfg.num_engines * per_engine_gpu_count

        logger.info(f"Creating placement group with {total_gpu_slots} GPU slots for colocated training+inference")
        pg = placement_group([{"GPU": 1, "CPU": 1}] * total_gpu_slots, strategy="PACK")

        logger.info("Waiting for placement group to be ready...")
        get_ray_pg_ready_with_timeout(pg, timeout=SKYRL_RAY_PG_TIMEOUT_IN_S)
        logger.info("Placement group ready!")

        return ResolvedPlacementGroup(pg)

    def delete_model(self, model_id: str) -> None:
        self._get_role(model_id)

        # Models in this backend share one Ray runtime and inference stack, so
        # deleting any model tears down the whole backend and it will be
        # re-initialized on the next create_model() call.
        logger.info(f"Deleting model {model_id}, shutting down shared SkyRL-Train runtime...")
        for group in self._server_groups:
            group.shutdown()
        self._server_groups = []
        if self._inference_router:
            self._inference_router.shutdown()
            self._inference_router = None
        ray.shutdown()
        self._model_ids_to_role = {}
        self._model_metadata = {}
        self._cfg = None
        self._dispatch = None
        self._inference_engine_client = None
        self._inference_engines_initialized = False
        self._renderer = None
        self._colocate_pg = None
        logger.info(f"Successfully deleted model {model_id}")

    def _to_training_batch(self, prepared_batch: types.PreparedModelPassBatch, role: str) -> TrainingInputBatch:
        """Convert PreparedModelPassBatch to TrainingInputBatch."""
        if not prepared_batch.all_model_inputs:
            return TrainingInputBatch({})

        if _SKYRL_USE_NEW_INFERENCE:
            if self._renderer is None:
                self._ensure_inference_engines()
                self._renderer = VLLMRenderer(self._inference_engine_client, self._cfg.trainer.policy.model.path)
            rendered_inputs = asyncio.run(self._renderer(prepared_batch.all_model_inputs))
        else:
            rendered_inputs = render_model_input(prepared_batch.all_model_inputs)

        all_input_ids = [r.prompt_ids for r in rendered_inputs]

        # SkyRL-Train shifts internally, so provide the full sequence length by
        # appending the last target token to each already-shifted input.
        full_sequences = [
            list(input_ids) + ([targets[-1]] if targets else [])
            for input_ids, targets in zip(all_input_ids, prepared_batch.all_targets)
        ]

        max_seq_len = max(len(seq) for seq in full_sequences)
        max_response_len = max(len(weights) for weights in prepared_batch.all_token_weights)

        sequences, attention_masks, loss_masks, response_masks = [], [], [], []
        action_log_probs_list, advantages_list = [], []
        values_list, returns_list = [], []

        for seq, weights, logprobs, advs, values, returns in zip(
            full_sequences,
            prepared_batch.all_token_weights,
            prepared_batch.all_sampling_logprobs,
            prepared_batch.all_advantages,
            prepared_batch.all_values,
            prepared_batch.all_returns,
        ):
            pad_len = max_seq_len - len(seq)
            sequences.append([self._tokenizer.pad_token_id] * pad_len + list(seq))
            attention_masks.append([0] * pad_len + [1] * len(seq))
            action_pad = max_response_len - len(weights)
            loss_masks.append([0.0] * action_pad + [float(w) for w in weights])
            response_masks.append([0] * action_pad + [1] * len(weights))
            action_log_probs_list.append([0.0] * action_pad + [float(lp) for lp in logprobs])
            advantages_list.append([0.0] * action_pad + [float(a) for a in advs])
            values_list.append([0.0] * action_pad + [float(v) for v in values])
            returns_list.append([0.0] * action_pad + [float(r) for r in returns])

        sequences_tensor = torch.tensor(sequences, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_masks, dtype=torch.long)
        loss_mask_tensor = torch.tensor(loss_masks, dtype=torch.float32)
        response_mask_tensor = torch.tensor(response_masks, dtype=torch.long)

        batch_dict = {
            "sequences": sequences_tensor,
            "attention_mask": attention_mask_tensor,
            "loss_mask": loss_mask_tensor,
            "response_mask": response_mask_tensor,
        }

        # Include RL fields (action_log_probs, advantages) when data is present
        has_logprobs = any(len(lp) > 0 for lp in prepared_batch.all_sampling_logprobs)
        has_advantages = any(len(a) > 0 for a in prepared_batch.all_advantages)
        has_values = any(len(v) > 0 for v in prepared_batch.all_values)
        has_returns = any(len(r) > 0 for r in prepared_batch.all_returns)
        if has_logprobs:
            batch_dict["action_log_probs"] = torch.tensor(action_log_probs_list, dtype=torch.float32)
        if has_advantages:
            batch_dict["advantages"] = torch.tensor(advantages_list, dtype=torch.float32)
        if role == "critic":
            if has_values != has_returns:
                raise ValueError("Critic batches must provide both values and returns, or neither")
            if has_values and any(
                len(values) != len(weights) or len(returns) != len(weights)
                for values, returns, weights in zip(
                    prepared_batch.all_values, prepared_batch.all_returns, prepared_batch.all_token_weights
                )
            ):
                raise ValueError("Critic batches with values/returns must align with response-token lengths")
            if has_values:
                batch_dict["values"] = torch.tensor(values_list, dtype=torch.float32)
                batch_dict["returns"] = torch.tensor(returns_list, dtype=torch.float32)

        # In mixed batches (some vision, some text-only), text-only samples
        # get an empty tensor placeholder so the TensorList length matches the batch size.
        # Empty tensors contribute zero rows when torch.cat'd downstream.
        for mm_key in ("pixel_values", "image_grid_thw"):
            values = [
                r.multi_modal_kwargs.get(mm_key) if r.multi_modal_kwargs is not None else None for r in rendered_inputs
            ]
            # Iterate through to get the first non-none value.
            # We use the reference shape to make sure subsequent stack / cat calls
            # don't run into shape errors.
            ref = next((v for v in values if v is not None), None)
            # If ref is None, then all of the values empty and we don't need to add placeholder tensors.
            if ref is not None:
                placeholder = torch.empty(0, *ref.shape[1:], dtype=ref.dtype, device=ref.device)
                batch_dict[mm_key] = TensorList([v if v is not None else placeholder for v in values])

        batch = TrainingInputBatch(batch_dict)
        batch.metadata = {"response_length": max_response_len}
        return batch

    def _pad_batch(
        self, batch: TrainingInputBatch, micro_batch_size: int | None = None
    ) -> tuple[TrainingInputBatch, int]:
        """Pad the batch so its size is divisible by the required alignment.

        The dispatch layer splits the batch evenly across DP workers, so the
        batch size must be a multiple of dp_size.  When *micro_batch_size* is
        given (needed for the Megatron backend whose ``forward_backward_func``
        doesn't support ragged micro-batches), we align to
        ``dp_size * micro_batch_size`` so each per-worker shard is also evenly
        divisible by *micro_batch_size*.

        Returns:
            (padded_batch, pad_size)
        """
        dp_size = self._dispatch.get_lcm_dp_size()
        alignment = dp_size * micro_batch_size if micro_batch_size else dp_size
        pad_size = (alignment - batch.batch_size % alignment) % alignment
        if pad_size > 0:
            logger.info(
                f"Padded batch from {batch.batch_size} to {batch.batch_size + pad_size} (alignment={alignment})"
            )
        return pad_training_input_batch(batch, pad_size), pad_size

    def _extract_metrics(self, data: dict) -> dict[str, float]:
        """Extract training metrics from dispatch return dict.

        Workers return metrics like 'loss', 'policy_loss', 'policy_entropy', etc.
        We convert to Tinker's colon-suffixed format (e.g. 'total_loss:sum').
        """
        metrics: dict[str, float] = {}

        # SFT path returns 'loss'; RL path returns 'final_loss' / 'policy_loss'
        if "loss" in data:
            metrics["total_loss:sum"] = float(data["loss"])
        elif "final_loss" in data:
            metrics["total_loss:sum"] = float(data["final_loss"])

        if "policy_loss" in data:
            metrics["pg_loss:sum"] = float(data["policy_loss"])
        if "policy_entropy" in data:
            metrics["entropy_loss:sum"] = float(data["policy_entropy"])
        if "critic_loss" in data:
            metrics["critic_loss:sum"] = float(data["critic_loss"])
        if "values_mean" in data:
            metrics["values_mean:mean"] = float(data["values_mean"])
        if "values_clipfrac" in data:
            metrics["values_clipfrac:mean"] = float(data["values_clipfrac"])
        if "response_length" in data:
            metrics["num_tokens:sum"] = float(data["response_length"])
        if "policy_lr" in data:
            metrics["policy_lr:last"] = float(data["policy_lr"])
        if "critic_lr" in data:
            metrics["critic_lr:last"] = float(data["critic_lr"])

        return metrics

    def _sleep_inference_engines(self):
        """Sleep inference engines to free GPU memory for training."""
        if self._inference_engines_initialized and self._cfg.trainer.placement.colocate_all:
            lora_cfg = self._cfg.trainer.policy.model.lora
            # TODO(team): remove once vllm fixes this
            # otherwise waking it up will output gibberish: https://github.com/vllm-project/vllm/issues/17103
            sleep_level = 1 if lora_cfg and lora_cfg.rank > 0 else 2
            if _SKYRL_USE_NEW_INFERENCE:
                asyncio.run(self._inference_engine_client.sleep(level=sleep_level))
            else:
                # Legacy client has a preset sleep level passed during create_ray_wrapped_inference_engines_from_config
                asyncio.run(self._inference_engine_client.sleep())

    def _validate_batch_role_and_loss(self, role: str, loss_fn: str):
        if role == "critic" and loss_fn not in {"ppo", "ppo_critic"}:
            raise ValueError(f"Critic batches must use loss_fn='ppo' or 'ppo_critic', got {loss_fn!r}")
        if role != "critic" and loss_fn == "ppo_critic":
            raise ValueError("loss_fn='ppo_critic' is only valid for critic models")

    def _normalize_policy_loss_request(
        self,
        role: str,
        loss_fn: str,
        loss_fn_config: dict[str, float] | None,
    ) -> tuple[str, dict[str, float] | None]:
        """Normalize public Tinker loss names/config into SkyRL-Train policy settings."""
        if role == "critic":
            return loss_fn, loss_fn_config

        if loss_fn != "ppo":
            return loss_fn, loss_fn_config

        normalized_config = dict(loss_fn_config or {})
        clip_low_threshold = normalized_config.pop("clip_low_threshold", None)
        clip_high_threshold = normalized_config.pop("clip_high_threshold", None)
        if clip_low_threshold is not None:
            normalized_config["eps_clip_low"] = 1.0 - clip_low_threshold
        if clip_high_threshold is not None:
            normalized_config["eps_clip_high"] = clip_high_threshold - 1.0
        return "regular", normalized_config or None

    def forward_backward(
        self,
        prepared_batch: types.PreparedModelPassBatch,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        if not prepared_batch.all_model_inputs:
            return {}

        self._sleep_inference_engines()
        results = {}
        for sub_batch in self._split_model_pass_batch_by_model_id(prepared_batch):
            results.update(self._forward_backward_single_model_batch(sub_batch))
        return results

    def _forward_backward_single_model_batch(
        self,
        prepared_batch: types.PreparedModelPassBatch,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        role = self._get_batch_role(prepared_batch.all_model_ids)
        loss_fn = prepared_batch.all_loss_fns[0]
        self._validate_batch_role_and_loss(role, loss_fn)
        if role == "critic" and any(
            len(values) != len(weights) or len(returns) != len(weights)
            for values, returns, weights in zip(
                prepared_batch.all_values, prepared_batch.all_returns, prepared_batch.all_token_weights
            )
        ):
            raise ValueError("Critic forward_backward requires values and returns for every response token")
        batch = self._to_training_batch(prepared_batch, role)
        micro_bs = (
            self._cfg.trainer.micro_train_batch_size_per_gpu if self._cfg.trainer.strategy == "megatron" else None
        )
        batch, pad_size = self._pad_batch(batch, micro_batch_size=micro_bs)

        loss_fn = prepared_batch.all_loss_fns[0]
        if len(set(prepared_batch.all_loss_fns)) > 1:
            logger.warning(
                "SkyRL backend received mixed loss functions %s in one batch; using '%s' for all",
                set(prepared_batch.all_loss_fns),
                loss_fn,
            )
        loss_fn_config = next((c for c in prepared_batch.all_loss_fn_configs if c is not None), None)
        loss_fn, loss_fn_config = self._normalize_policy_loss_request(role, loss_fn, loss_fn_config)
        if role == "critic":
            self._dispatch.set_algorithm_config(
                "critic",
                value_clip=(loss_fn_config or {}).get("value_clip", self._cfg.trainer.algorithm.value_clip),
            )
            data = self._dispatch.forward_backward("critic", batch)
        else:
            data = self._dispatch.forward_backward(
                role,
                batch,
                loss_fn=loss_fn,
                loss_fn_config=loss_fn_config,
            )

        # Trim padding entries from loss_fn_outputs
        if pad_size > 0 and "loss_fn_outputs" in data:
            data["loss_fn_outputs"] = data["loss_fn_outputs"][:-pad_size]

        metrics = self._extract_metrics(data)

        results = {}
        for request_id, _, start_idx, end_idx in prepared_batch.request_batch_slices:
            if "loss_fn_outputs" in data:
                loss_fn_outputs = []
                for i in range(start_idx, end_idx):
                    raw_output = data["loss_fn_outputs"][i]
                    formatted_output = {}
                    for key in ("elementwise_loss", "logprobs", "values"):
                        values = list(raw_output.get(key, []))
                        if values or key in raw_output:
                            formatted_output[key] = {
                                "data": values,
                                "dtype": "float32",
                                "shape": [len(values)],
                            }
                    loss_fn_outputs.append(formatted_output)
            else:
                loss_fn_outputs = [{} for _ in range(end_idx - start_idx)]
            results[request_id] = types.ForwardBackwardOutput(
                loss_fn_output_type="scalar",
                loss_fn_outputs=loss_fn_outputs,
                metrics=metrics,
            )
        return results

    def forward(
        self,
        prepared_batch: types.PreparedModelPassBatch,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        if not prepared_batch.all_model_inputs:
            return {}

        self._sleep_inference_engines()
        results = {}
        for sub_batch in self._split_model_pass_batch_by_model_id(prepared_batch):
            results.update(self._forward_single_model_batch(sub_batch))
        return results

    def _forward_single_model_batch(
        self,
        prepared_batch: types.PreparedModelPassBatch,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        role = self._get_batch_role(prepared_batch.all_model_ids)
        batch = self._to_training_batch(prepared_batch, role)
        micro_bs = (
            self._cfg.trainer.micro_forward_batch_size_per_gpu if self._cfg.trainer.strategy == "megatron" else None
        )
        batch, pad_size = self._pad_batch(batch, micro_batch_size=micro_bs)
        data = self._dispatch.forward(role, batch)

        # dispatch.forward() returns TrainingOutputBatch({"output": tensor[batch, max_response_len]})
        # Trim padding entries from output
        output_tensor = data["output"]
        if pad_size > 0:
            output_tensor = output_tensor[:-pad_size]

        results = {}
        for request_id, _, start_idx, end_idx in prepared_batch.request_batch_slices:
            loss_fn_outputs = []
            for i in range(start_idx, end_idx):
                # Use token weights length to determine each example's actual response length
                valid_len = len(prepared_batch.all_token_weights[i])
                start = max(output_tensor.shape[1] - valid_len, 0)
                outputs = output_tensor[i, start:].tolist()
                output_key = "values" if role == "critic" else "logprobs"
                loss_fn_outputs.append(
                    {
                        output_key: {
                            "data": outputs,
                            "dtype": "float32",
                            "shape": [len(outputs)],
                        },
                    }
                )
            results[request_id] = types.ForwardBackwardOutput(
                loss_fn_output_type="scalar",
                loss_fn_outputs=loss_fn_outputs,
                metrics={},
            )
        return results

    def optim_step(self, model_id: str, request_data: types.OptimStepInput) -> types.OptimStepOutput:
        role = self._get_role(model_id)

        # Apply learning rate from AdamParams before optimizer step
        # Note: beta1, beta2, eps are fixed at optimizer creation and cannot be changed dynamically
        adam_params = request_data.adam_params
        self._dispatch.set_lr(role, adam_params.learning_rate)

        grad_norm = self._dispatch.optim_step(role)
        logger.info(f"optim_step: lr={adam_params.learning_rate}, grad_norm={grad_norm}")

        metrics: dict[str, float] = {}
        if grad_norm is not None:
            metrics["skyrl.ai/grad_norm"] = float(grad_norm)
        metrics["skyrl.ai/learning_rate"] = adam_params.learning_rate
        return types.OptimStepOutput(metrics=metrics)

    def sample(
        self,
        prepared_batch: types.PreparedSampleBatch,
    ) -> dict[str, types.SampleOutput | types.ErrorResponse]:
        """Generate samples using inference engine.

        NOTE: Weight sync is NOT triggered automatically. The caller must call
        save_weights_for_sampler() explicitly before calling sample() if weights
        have been updated.
        """
        # 1. Ensure inference engines are initialized
        self._ensure_inference_engines()

        # 2. Validate single model
        unique_models = set(prepared_batch.all_model_ids)
        if len(unique_models) != 1:
            error = types.ErrorResponse(
                error=f"Expected exactly one model_id for sampling, got {unique_models}", status="error"
            )
            return {req_id: error for req_id, _, _, _, _ in prepared_batch.request_batch_slices}
        model_id = next(iter(unique_models))
        role = self._model_ids_to_role.get(model_id)
        if role != "policy":
            error = types.ErrorResponse(
                error=f"Sampling is only supported for policy models, got '{model_id}'", status="error"
            )
            return {req_id: error for req_id, _, _, _, _ in prepared_batch.request_batch_slices}

        # 3. Dispatch to appropriate sampling path
        if _SKYRL_USE_NEW_INFERENCE:
            return self._sample_with_remote_client(prepared_batch)
        return self._sample_with_legacy_client(prepared_batch)

    def _sample_with_legacy_client(
        self,
        prepared_batch: types.PreparedSampleBatch,
    ) -> dict[str, types.SampleOutput | types.ErrorResponse]:
        """Sample using legacy InferenceEngineClient with Ray-wrapped engines."""
        all_input_ids = [r.prompt_ids for r in render_model_input(prepared_batch.all_model_inputs)]

        async def sample_all():
            tasks = []
            for i in range(len(all_input_ids)):
                prompt_token_ids = all_input_ids[i]
                sampling_params = prepared_batch.all_sampling_params[i]

                # Pass through common fields; only stop needs name translation
                # (Tinker uses stop_strings/stop_tokens, vLLM uses stop/stop_token_ids)
                params_dict = {
                    "temperature": sampling_params.temperature,
                    "max_tokens": sampling_params.max_tokens,
                    "seed": sampling_params.seed,
                    "top_k": sampling_params.top_k,
                    "top_p": sampling_params.top_p,
                    "logprobs": 0,
                }
                if sampling_params.stop_strings:
                    params_dict["stop"] = sampling_params.stop_strings
                if sampling_params.stop_tokens:
                    params_dict["stop_token_ids"] = sampling_params.stop_tokens

                tasks.append(
                    self._inference_engine_client.sample(
                        prompt_token_ids=prompt_token_ids,
                        num_samples=1,  # Tinker batches multiple samples separately
                        sampling_params=params_dict,
                    )
                )

            return await asyncio.gather(*tasks, return_exceptions=True)

        # Backend runs in engine subprocess with no event loop
        sample_outputs = asyncio.run(sample_all())
        return self._aggregate_sample_results(prepared_batch, sample_outputs)

    def _sample_with_remote_client(
        self,
        prepared_batch: types.PreparedSampleBatch,
    ) -> dict[str, types.SampleOutput | types.ErrorResponse]:
        """Sample using RemoteInferenceClient, forwarding model input chunks directly."""

        async def sample_all():
            tasks = []
            for i in range(len(prepared_batch.all_model_inputs)):
                model_input = prepared_batch.all_model_inputs[i]
                sampling_params = prepared_batch.all_sampling_params[i]

                request_payload = {
                    "json": {
                        "prompt": model_input.model_dump(),
                        "num_samples": 1,
                        "sampling_params": sampling_params.model_dump(),
                    }
                }
                tasks.append(self._inference_engine_client.sample(request_payload))

            try:
                return await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                await self._inference_engine_client.aclose()

        sample_outputs = asyncio.run(sample_all())
        logger.info(f"Collected {len(sample_outputs)} sample outputs")
        return self._aggregate_sample_results(prepared_batch, sample_outputs)

    def _aggregate_sample_results(
        self,
        prepared_batch: types.PreparedSampleBatch,
        sample_outputs: list,
    ) -> dict[str, types.SampleOutput | types.ErrorResponse]:
        """Convert sample outputs to Tinker format. Handles both legacy and remote client outputs."""
        logger.info(f"Aggregating sample results for {len(sample_outputs)} samples")

        def _extract_sequences(output):
            """Yield (tokens, logprobs, stop_reason) from a single sample output."""
            if _SKYRL_USE_NEW_INFERENCE:
                for seq in output["sequences"]:
                    yield seq["tokens"], seq.get("logprobs"), seq.get("stop_reason")
            else:
                yield (
                    output["response_ids"][0],
                    (output.get("response_logprobs") or [[]])[0],
                    output["stop_reasons"][0],
                )

        results = {}
        for request_id, model_id, start_idx, end_idx, needs_prompt_logprobs in prepared_batch.request_batch_slices:
            sequences = []
            has_error = False
            error_msg = None

            for i in range(start_idx, end_idx):
                output = sample_outputs[i]

                # Check if sampling failed (Exception or None)
                if isinstance(output, Exception):
                    has_error = True
                    error_msg = f"Sampling failed for sample {i}: {type(output).__name__}: {str(output)}"
                    logger.error(error_msg)
                    break
                elif output is None:
                    has_error = True
                    error_msg = f"Sampling failed for sample {i}: Unknown error (output is None)"
                    logger.error(error_msg)
                    break

                for tokens, logprobs_raw, stop_reason_raw in _extract_sequences(output):
                    # Map vLLM stop reason to Tinker format
                    stop_reason = "stop" if stop_reason_raw in ("stop", "stop_token") else "length"
                    logprobs = logprobs_raw or []

                    # Ensure logprobs exist (critical for RL)
                    if not logprobs and tokens:
                        logger.warning("No logprobs returned - filling with zeros")
                        logprobs = [0.0] * len(tokens)

                    sequences.append(
                        types.GeneratedSequence(
                            tokens=tokens,
                            logprobs=logprobs,
                            stop_reason=stop_reason,
                        )
                    )

            if has_error:
                results[request_id] = types.ErrorResponse(
                    error=error_msg or "Unknown sampling error",
                    status="error",
                )
            else:
                # Note: prompt_logprobs not supported initially
                if needs_prompt_logprobs:
                    logger.warning("Prompt logprobs requested but not yet supported")

                results[request_id] = types.SampleOutput(
                    sequences=sequences,
                    prompt_logprobs=None,
                )

        return results

    def _validate_model_state(self, model_id: str) -> None:
        """Validate that model exists and is initialized."""
        self._get_role(model_id)
        if self._dispatch is None:
            raise RuntimeError("Model not initialized")

    def _create_tar_from_directory(self, source_dir: str, output_path: str) -> None:
        """Create an uncompressed tar archive from a directory."""
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Use uncompressed tar - gzip adds 5-10min CPU time on 6-7GB FSDP checkpoints
        with tarfile.open(output_path, "w") as tar:
            tar.add(source_dir, arcname=".")

    def save_checkpoint(self, output_path, model_id: str) -> None:
        """Save full training checkpoint (model + optimizer + scheduler) as tar."""
        self._validate_model_state(model_id)
        role = self._get_role(model_id)

        # Create temp directory for checkpoint
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt_dir = os.path.join(temp_dir, "checkpoint")

            # Save checkpoint directory (includes optimizer state automatically)
            self._dispatch.save_checkpoint(model=role, ckpt_dir=ckpt_dir, tokenizer=self._tokenizer)

            # Create tar archive
            self._create_tar_from_directory(ckpt_dir, output_path)

        logger.info(f"Saved checkpoint for {model_id} to {output_path}")

    def load_checkpoint(self, checkpoint_path, model_id: str) -> None:
        """Load full training checkpoint (model + optimizer + scheduler) from tar."""
        self._validate_model_state(model_id)
        role = self._get_role(model_id)

        # Extract tar to temp directory (filter='data' prevents path traversal attacks)
        with tempfile.TemporaryDirectory() as temp_dir:
            with tarfile.open(checkpoint_path, "r") as tar:
                tar.extractall(temp_dir, filter="data")

            # Load checkpoint (includes optimizer and scheduler states)
            self._dispatch.load_checkpoint(
                model=role, ckpt_dir=temp_dir, load_optimizer_states=True, load_lr_scheduler_states=True
            )

        logger.info(f"Loaded checkpoint for {model_id} from {checkpoint_path}")

    def save_sampler_checkpoint(self, output_path, model_id: str, persist: bool = True) -> None:
        """Sync weights to colocated inference engines and optionally save to disk.

        The NCCL broadcast always runs so inference engines have the latest
        policy weights.  When ``persist`` is False (the common hot-path in RL
        loops) the expensive HuggingFace model export is skipped entirely.
        """
        self._validate_model_state(model_id)
        if self._get_role(model_id) != "policy":
            raise ValueError("save_sampler_checkpoint is only supported for policy models")

        # Lazily create inference engines on first sampling-related call
        self._ensure_inference_engines()

        asyncio.run(self._dispatch.save_weights_for_sampler())
        logger.info(f"Synced weights for {model_id} to inference engines via NCCL")

        if persist:
            # TODO(tyler): For LoRA, only save the adapters instead of the full merged model
            with tempfile.TemporaryDirectory() as temp_dir:
                hf_dir = os.path.join(temp_dir, "model")
                self._dispatch.save_hf_model(model="policy", export_dir=hf_dir, tokenizer=self._tokenizer)
                self._create_tar_from_directory(hf_dir, output_path)
            logger.info(f"Saved sampler checkpoint for {model_id} to {output_path}")
        else:
            # Hot path: write a lightweight marker so the engine's checkpoint
            # bookkeeping stays consistent.  Actual weights live in GPU memory.
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            marker = f"SkyRL sampler marker for {model_id}: weights live in GPU memory (persist=False).\n".encode()
            with tarfile.open(output_path, "w") as tar:
                info = tarfile.TarInfo("MARKER")
                info.size = len(marker)
                tar.addfile(info, io.BytesIO(marker))
            logger.info(f"Synced weights for {model_id} (disk save skipped)")


def create_ray_wrapped_inference_engines_from_config(
    cfg: SkyRLTrainConfig, colocate_pg: ResolvedPlacementGroup | None, tokenizer: PreTrainedTokenizerBase
):
    engine_kwargs = {
        "num_inference_engines": cfg.generator.inference_engine.num_engines,
        "tensor_parallel_size": cfg.generator.inference_engine.tensor_parallel_size,
        "pipeline_parallel_size": cfg.generator.inference_engine.pipeline_parallel_size,
        "model_dtype": cfg.generator.inference_engine.model_dtype,
        "pretrain": cfg.trainer.policy.model.path,
        "seed": cfg.trainer.seed,
        "vllm_v1_disable_multiproc": cfg.generator.inference_engine.vllm_v1_disable_multiproc,
        "enable_prefix_caching": cfg.generator.inference_engine.enable_prefix_caching,
        "enforce_eager": cfg.generator.inference_engine.enforce_eager,
        "expert_parallel_size": cfg.generator.inference_engine.expert_parallel_size,
        "data_parallel_size": cfg.generator.inference_engine.data_parallel_size,
        "shared_pg": colocate_pg,
        "gpu_memory_utilization": cfg.generator.inference_engine.gpu_memory_utilization,
        "inference_engine_enable_sleep": cfg.trainer.placement.colocate_all,
        "async_engine": cfg.generator.inference_engine.async_engine,
        "max_num_batched_tokens": cfg.generator.inference_engine.max_num_batched_tokens,
        "max_num_seqs": cfg.generator.inference_engine.max_num_seqs,
        "tokenizer": tokenizer,
        "backend": cfg.generator.inference_engine.backend,
        "engine_init_kwargs": cfg.generator.inference_engine.engine_init_kwargs,
        "enable_ray_prometheus_stats": cfg.generator.inference_engine.enable_ray_prometheus_stats,
        "distributed_executor_backend": cfg.generator.inference_engine.distributed_executor_backend,
        "language_model_only": cfg.generator.inference_engine.language_model_only,
    }

    # Conditionally add LoRA parameters if LoRA is enabled
    if cfg.trainer.policy.model.lora.rank > 0 and cfg.trainer.strategy != "megatron":
        engine_kwargs["enable_lora"] = True
        engine_kwargs["max_lora_rank"] = cfg.trainer.policy.model.lora.rank
        engine_kwargs["sleep_level"] = 1
        engine_kwargs["max_loras"] = 1
        engine_kwargs["fully_sharded_loras"] = cfg.generator.inference_engine.fully_sharded_loras

        if cfg.generator.inference_engine.enforce_eager and cfg.generator.inference_engine.backend == "vllm":
            logger.warning(
                "LoRA is enabled but generator.inference_engine.enforce_eager=true. "
                "This combination causes significant performance degradation (2-3x slower generation). "
                "Automatically setting enforce_eager=false for better performance. "
            )
            engine_kwargs["enforce_eager"] = False

    if cfg.generator.rope_scaling is not None:
        engine_kwargs["rope_scaling"] = cfg.generator.rope_scaling
    if cfg.generator.rope_theta is not None:
        engine_kwargs["rope_theta"] = cfg.generator.rope_theta
    if cfg.generator.inference_engine.served_model_name is not None:
        engine_kwargs["served_model_name"] = cfg.generator.inference_engine.served_model_name

    return create_ray_wrapped_inference_engines(**engine_kwargs)
