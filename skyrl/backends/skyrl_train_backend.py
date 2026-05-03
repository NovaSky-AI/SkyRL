"""SkyRL-Train backend for TinkerEngine.

Uses SkyRL-Train infrastructure for supervised training with cross-entropy loss.
Currently supports a single model only.
"""

import asyncio
import os
import tarfile
import tempfile

import ray
import torch
from pydantic import BaseModel
from ray.util.placement_group import placement_group
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from skyrl.backends.backend import AbstractBackend
from skyrl.backends.renderer import render_model_input
from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import (
    InferenceEngineClient,
)
from skyrl.backends.skyrl_train.inference_engines.ray_wrapped_inference_engine import (
    create_ray_wrapped_inference_engines,
)
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)
from skyrl.backends.skyrl_train.inference_servers.server_group import ServerGroup
from skyrl.backends.skyrl_train.inference_servers.utils import build_vllm_cli_args
from skyrl.backends.skyrl_train.inference_servers.vllm_router import VLLMRouter
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
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

# Fixed LoRA adapter name used for generation requests when LoRA is active.
_SKYRL_LORA_ADAPTER_NAME = "skyrl-lora"


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
    # that will resolve other attributes such as the reference model path based on the policy model path.
    user_overrides["trainer.policy.model.path"] = base_model
    cfg = SkyRLTrainConfig.from_cli_overrides(user_overrides)

    # Disable scheduler - Tinker manages learning rate externally via set_lr()
    cfg.trainer.policy.optimizer_config.scheduler = "constant_with_warmup"
    cfg.trainer.policy.optimizer_config.num_warmup_steps = 0

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
        self._model_id: str | None = None
        self._model_metadata: types.ModelMetadata | None = None
        self._cfg = None
        self._dispatch: WorkerDispatch | None = None
        self._tokenizer: AutoTokenizer = get_tokenizer(self.base_model)
        self._inference_engine_client = None
        self._inference_engines_initialized = False

        # New inference infrastructure
        self._server_group = None
        self._inference_router = None

    def has_model(self, model_id: str) -> bool:
        return self._model_id == model_id

    def build_models(self, PolicyWorker):
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

        # Mark all models as offloaded
        if colocate_all:
            self._dispatch.mark_all_offloaded()

        logger.info("init policy model done")

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
        """Create new HTTP-based inference client.

        Possible config combinations:
        - Both external_proxy_url and external_server_urls → fully external setup
        - external_proxy_url only → proxy for both data + control plane
        - external_server_urls only → create internal router over them
        - Neither → build servers and router internally
        """
        ie_cfg = self._cfg.generator.inference_engine
        is_colocated = self._cfg.trainer.placement.colocate_all
        external_proxy_url = ie_cfg.external_proxy_url
        external_server_urls = ie_cfg.external_server_urls

        has_external_proxy = external_proxy_url is not None
        has_external_servers = external_server_urls is not None

        if has_external_proxy and has_external_servers:
            proxy_url = external_proxy_url
            server_urls = list(external_server_urls)
            logger.info(
                f"HTTP Inference: Using fully external setup - proxy_url={proxy_url}, server_urls={server_urls}"
            )

        elif has_external_proxy and not has_external_servers:
            proxy_url = external_proxy_url
            server_urls = [proxy_url]
            logger.info(f"HTTP Inference: Using external proxy for both data and control plane - proxy_url={proxy_url}")

        elif has_external_servers and not has_external_proxy:
            server_urls = list(external_server_urls)
            self._inference_router = VLLMRouter(server_urls=server_urls)
            proxy_url = self._inference_router.start()
            logger.info(
                f"HTTP Inference: Created router over external servers - "
                f"server_urls={server_urls}, proxy_url={proxy_url}"
            )

        else:
            cli_args = build_vllm_cli_args(self._cfg)

            self._server_group = ServerGroup(
                cli_args=cli_args,
                num_servers=ie_cfg.num_engines,
                placement_group=self._colocate_pg if is_colocated else None,
                enable_dp=ie_cfg.data_parallel_size > 1,
                distributed_executor_backend=ie_cfg.distributed_executor_backend,
            )
            server_infos = self._server_group.start()
            server_urls = [info.url for info in server_infos]

            self._inference_router = VLLMRouter(server_urls=server_urls)
            proxy_url = self._inference_router.start()
            logger.info(
                f"HTTP Inference: Built servers and router internally - "
                f"proxy_url={proxy_url}, server_urls={server_urls}, colocated={is_colocated}"
            )

        lora_cfg = self._cfg.trainer.policy.model.lora
        active_lora_name = _SKYRL_LORA_ADAPTER_NAME if lora_cfg and lora_cfg.rank > 0 else None
        self._inference_engine_client = RemoteInferenceClient(
            proxy_url=proxy_url,
            server_urls=server_urls,
            model_name=self._cfg.trainer.policy.model.path,
            active_lora_name=active_lora_name,
            tokenizer=self._tokenizer,
        )

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

    def create_model(self, model_id: str, lora_config: types.LoraConfig) -> None:
        if self._model_id is not None:
            raise ValueError(f"Model '{self._model_id}' already exists. Only one model supported.")

        # Build config
        self._cfg = _build_skyrl_train_config(self.base_model, self.config, lora_config)

        if not ray.is_initialized():
            logger.info("Initializing Ray with runtime environment")
            initialize_ray(self._cfg)

        # Create shared placement group only when colocating training + inference
        if self._cfg.trainer.placement.colocate_all:
            self._colocate_pg = self._create_colocate_pg()
        else:
            self._colocate_pg = None

        # Get worker types based on strategy
        if self._cfg.trainer.strategy in ("fsdp", "fsdp2"):
            from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import PolicyWorker
        elif self._cfg.trainer.strategy == "megatron":
            from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
                PolicyWorker,
            )
        else:
            raise ValueError(f"Unknown strategy type: {self._cfg.trainer.strategy}")

        logger.info("Building models.")
        self.build_models(PolicyWorker)

        self._model_id = model_id
        self._model_metadata = types.ModelMetadata(adapter_index=0, lora_config=lora_config)
        logger.info(f"Created model {model_id} using RayPPOTrainer")

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
        if self._model_id != model_id:
            raise ValueError(f"Model {model_id} not found")

        # Currently only one model at a time is supported. Shut down Ray entirely
        # and reset state; everything will be re-initialized in create_model().
        logger.info(f"Deleting model {model_id}, shutting down Ray...")
        if self._server_group:
            self._server_group.shutdown()
            self._server_group = None
        if self._inference_router:
            self._inference_router.shutdown()
            self._inference_router = None
        ray.shutdown()

        self._model_id = None
        self._model_metadata = None
        self._cfg = None
        self._dispatch = None
        self._inference_engine_client = None
        self._inference_engines_initialized = False
        self._colocate_pg = None

        logger.info(f"Successfully deleted model {model_id}")

    def _to_training_batch(self, prepared_batch: types.PreparedModelPassBatch) -> TrainingInputBatch:
        """Convert PreparedModelPassBatch to TrainingInputBatch."""
        if not prepared_batch.all_model_inputs:
            return TrainingInputBatch({})

        # Extract token IDs from ModelInput chunks
        all_input_ids = [r.prompt_ids for r in render_model_input(prepared_batch.all_model_inputs)]

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

        for seq, weights, logprobs, advs in zip(
            full_sequences,
            prepared_batch.all_token_weights,
            prepared_batch.all_sampling_logprobs,
            prepared_batch.all_advantages,
        ):
            pad_len = max_seq_len - len(seq)
            sequences.append([self._tokenizer.pad_token_id] * pad_len + list(seq))
            attention_masks.append([0] * pad_len + [1] * len(seq))
            action_pad = max_response_len - len(weights)
            loss_masks.append([0.0] * action_pad + [float(w) for w in weights])
            response_masks.append([0] * action_pad + [1] * len(weights))
            action_log_probs_list.append([0.0] * action_pad + [float(lp) for lp in logprobs])
            advantages_list.append([0.0] * action_pad + [float(a) for a in advs])

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
        if has_logprobs:
            batch_dict["action_log_probs"] = torch.tensor(action_log_probs_list, dtype=torch.float32)
        if has_advantages:
            batch_dict["advantages"] = torch.tensor(advantages_list, dtype=torch.float32)

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
        if pad_size == 0:
            return batch, 0

        new_tensors = {}
        for key, tensor in batch.items():
            if tensor is not None:
                if key == "loss_mask":
                    # Padding entries must not contribute to the loss
                    additional_dims = tensor.shape[1:]
                    padding_tensor = torch.zeros(pad_size, *additional_dims, dtype=tensor.dtype, device=tensor.device)
                else:
                    # Clone real data so shapes/dtypes are valid for the model
                    padding_tensor = tensor[torch.arange(pad_size) % tensor.shape[0]].clone()
                new_tensors[key] = torch.cat([tensor, padding_tensor], dim=0)

        padded = TrainingInputBatch(new_tensors)
        padded.metadata = batch.metadata
        logger.info(f"Padded batch from {batch.batch_size} to {batch.batch_size + pad_size} (alignment={alignment})")
        return padded, pad_size

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
        if "response_length" in data:
            metrics["num_tokens:sum"] = float(data["response_length"])

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

    def forward_backward(
        self,
        prepared_batch: types.PreparedModelPassBatch,
    ) -> dict[str, types.ForwardBackwardOutput | types.ErrorResponse]:
        if not prepared_batch.all_model_inputs:
            return {}

        self._sleep_inference_engines()
        batch = self._to_training_batch(prepared_batch)
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
        data = self._dispatch.forward_backward(
            "policy",
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
            loss_fn_outputs = []
            for i in range(start_idx, end_idx):
                raw_output = data["loss_fn_outputs"][i]
                logprobs = list(raw_output.get("logprobs", []))
                elementwise_loss = list(raw_output.get("elementwise_loss", []))
                loss_fn_outputs.append(
                    {
                        "elementwise_loss": {
                            "data": elementwise_loss,
                            "dtype": "float32",
                            "shape": [len(elementwise_loss)],
                        },
                        "logprobs": {
                            "data": logprobs,
                            "dtype": "float32",
                            "shape": [len(logprobs)],
                        },
                    }
                )
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
        batch = self._to_training_batch(prepared_batch)
        micro_bs = (
            self._cfg.trainer.micro_forward_batch_size_per_gpu if self._cfg.trainer.strategy == "megatron" else None
        )
        batch, pad_size = self._pad_batch(batch, micro_batch_size=micro_bs)
        data = self._dispatch.forward("policy", batch)

        # dispatch.forward() returns TrainingOutputBatch({"output": tensor[batch, max_response_len]})
        # Trim padding entries from output
        output_logprobs = data["output"]
        if pad_size > 0:
            output_logprobs = output_logprobs[:-pad_size]

        results = {}
        for request_id, _, start_idx, end_idx in prepared_batch.request_batch_slices:
            loss_fn_outputs = []
            for i in range(start_idx, end_idx):
                # Use token weights length to determine each example's actual response length
                valid_len = len(prepared_batch.all_token_weights[i])
                start = max(output_logprobs.shape[1] - valid_len, 0)
                logprobs = output_logprobs[i, start:].tolist()
                loss_fn_outputs.append(
                    {
                        "logprobs": {
                            "data": logprobs,
                            "dtype": "float32",
                            "shape": [len(logprobs)],
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
        if model_id != self._model_id:
            raise ValueError(f"Model {model_id} not found")

        # Apply learning rate from AdamParams before optimizer step
        # Note: beta1, beta2, eps are fixed at optimizer creation and cannot be changed dynamically
        adam_params = request_data.adam_params
        self._dispatch.set_lr("policy", adam_params.learning_rate)

        grad_norm = self._dispatch.optim_step("policy")
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
        if unique_models != {self._model_id}:
            error = types.ErrorResponse(
                error=f"Model mismatch. Expected {self._model_id}, got {unique_models}", status="error"
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
        if model_id != self._model_id:
            raise ValueError(f"Model {model_id} not found")
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

        # Create temp directory for checkpoint
        with tempfile.TemporaryDirectory() as temp_dir:
            ckpt_dir = os.path.join(temp_dir, "checkpoint")

            # Save checkpoint directory (includes optimizer state automatically)
            self._dispatch.save_checkpoint(model="policy", ckpt_dir=ckpt_dir, tokenizer=self._tokenizer)

            # Create tar archive
            self._create_tar_from_directory(ckpt_dir, output_path)

        logger.info(f"Saved checkpoint for {model_id} to {output_path}")

    def load_checkpoint(self, checkpoint_path, model_id: str) -> None:
        """Load full training checkpoint (model + optimizer + scheduler) from tar."""
        self._validate_model_state(model_id)

        # Extract tar to temp directory (filter='data' prevents path traversal attacks)
        with tempfile.TemporaryDirectory() as temp_dir:
            with tarfile.open(checkpoint_path, "r") as tar:
                tar.extractall(temp_dir, filter="data")

            # Load checkpoint (includes optimizer and scheduler states)
            self._dispatch.load_checkpoint(
                model="policy", ckpt_dir=temp_dir, load_optimizer_states=True, load_lr_scheduler_states=True
            )

        logger.info(f"Loaded checkpoint for {model_id} from {checkpoint_path}")

    def save_sampler_checkpoint(self, output_path, model_id: str, persist: bool = True) -> None:
        """Sync weights to colocated inference engines and optionally save to disk.

        The NCCL broadcast always runs so inference engines have the latest
        policy weights.  When ``persist`` is False (the common hot-path in RL
        loops) the expensive HuggingFace model export is skipped entirely.
        """
        self._validate_model_state(model_id)

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
            with tarfile.open(output_path, "w"):
                pass  # empty tar — marker only
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
