"""
Main entrypoint for training.
"""

import asyncio
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Optional

import ray
from loguru import logger
from ray.util.placement_group import placement_group
from transformers import PreTrainedTokenizerBase

from skyrl.backends.skyrl_train.inference_engines.base import InferenceEngineInterface
from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import (
    InferenceEngineClient,
)
from skyrl.backends.skyrl_train.inference_engines.remote_inference_engine import (
    create_remote_inference_engines,
)
from skyrl.env_vars import _SKYRL_USE_NEW_INFERENCE, SKYRL_RAY_PG_TIMEOUT_IN_S
from skyrl.train.config import SkyRLTrainConfig, get_config_as_yaml_str
from skyrl.train.dataset import PromptDataset
from skyrl.train.generators.base import GeneratorInterface
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.tracking import Tracking
from skyrl.train.utils.utils import (
    ResolvedPlacementGroup,
    get_ray_pg_ready_with_timeout,
    initialize_ray,
)
from skyrl.utils.tok import get_tokenizer

# NOTE (sumanthrh): We use ray heavily and thus disable `fork` start method.
# forking within ray leads to undefined behaviour and often causes hard to debug
# memory leaks.  See: https://docs.ray.io/en/latest/ray-core/patterns/fork-new-processes.html
# A common culprit is Pytorch dataloaders which use `fork` by default.
mp.set_start_method("spawn", force=True)

config_dir = str(Path(__file__).parent.parent / "config")
__all__ = ["BasePPOExp", "config_dir"]


def create_ray_wrapped_inference_engines_from_config(
    cfg: SkyRLTrainConfig,
    colocate_pg: Optional[ResolvedPlacementGroup],
    tokenizer: PreTrainedTokenizerBase,
):
    from skyrl.backends.skyrl_train.inference_engines.ray_wrapped_inference_engine import (
        create_ray_wrapped_inference_engines,
    )

    ie_cfg = cfg.generator.inference_engine
    engine_kwargs = {
        "num_inference_engines": ie_cfg.num_engines,
        "tensor_parallel_size": ie_cfg.tensor_parallel_size,
        "pipeline_parallel_size": ie_cfg.pipeline_parallel_size,
        "model_dtype": ie_cfg.model_dtype,
        "pretrain": cfg.trainer.policy.model.path,
        "seed": cfg.trainer.seed,
        "vllm_v1_disable_multiproc": ie_cfg.vllm_v1_disable_multiproc,
        "enable_prefix_caching": ie_cfg.enable_prefix_caching,
        "enforce_eager": ie_cfg.enforce_eager,
        "expert_parallel_size": ie_cfg.expert_parallel_size,
        "data_parallel_size": ie_cfg.data_parallel_size,
        "shared_pg": colocate_pg,
        "gpu_memory_utilization": ie_cfg.gpu_memory_utilization,
        "inference_engine_enable_sleep": cfg.trainer.placement.colocate_all,
        "async_engine": ie_cfg.async_engine,
        "max_num_batched_tokens": ie_cfg.max_num_batched_tokens,
        "max_num_seqs": ie_cfg.max_num_seqs,
        "tokenizer": tokenizer,
        "backend": ie_cfg.backend,
        "language_model_only": ie_cfg.language_model_only,
        "engine_init_kwargs": ie_cfg.engine_init_kwargs,
        "enable_ray_prometheus_stats": ie_cfg.enable_ray_prometheus_stats,
        "enable_return_routed_experts": ie_cfg.enable_return_routed_experts,
        "distributed_executor_backend": ie_cfg.distributed_executor_backend,
    }

    # Conditionally add LoRA parameters if LoRA is enabled
    if cfg.trainer.policy.model.lora.rank > 0 and cfg.trainer.strategy != "megatron":
        engine_kwargs["enable_lora"] = True
        engine_kwargs["max_lora_rank"] = cfg.trainer.policy.model.lora.rank
        engine_kwargs["sleep_level"] = 1
        engine_kwargs["max_loras"] = 1
        engine_kwargs["fully_sharded_loras"] = ie_cfg.fully_sharded_loras

        # TODO(devpatel): Bandaid solution, replace this once we have a better
        # solution for LoRA performance degradation on the vLLM side
        if ie_cfg.enforce_eager and ie_cfg.backend == "vllm":
            logger.warning(
                "LoRA is enabled but inference_engine.enforce_eager=true. "
                "This combination causes significant performance degradation (2-3x slower generation). "
                "Automatically setting enforce_eager=false for better performance. "
            )
            engine_kwargs["enforce_eager"] = False

    if cfg.generator.rope_scaling is not None:
        engine_kwargs["rope_scaling"] = cfg.generator.rope_scaling
    if cfg.generator.rope_theta is not None:
        engine_kwargs["rope_theta"] = cfg.generator.rope_theta
    if ie_cfg.served_model_name is not None:
        engine_kwargs["served_model_name"] = ie_cfg.served_model_name

    return create_ray_wrapped_inference_engines(**engine_kwargs)


def create_remote_inference_engines_from_config(cfg: SkyRLTrainConfig, tokenizer: PreTrainedTokenizerBase):
    # TODO(tgriggs): We may want a separate config for the model name in case
    # it's different from the name used in the OpenAI API
    ie_cfg = cfg.generator.inference_engine
    return create_remote_inference_engines(
        urls=ie_cfg.remote_urls,
        model_name=cfg.trainer.policy.model.path,
        engine_backend=ie_cfg.backend,
        tokenizer=tokenizer,
        tensor_parallel_size=ie_cfg.tensor_parallel_size,
        pipeline_parallel_size=ie_cfg.pipeline_parallel_size,
        data_parallel_size=ie_cfg.data_parallel_size,
        expert_parallel_size=ie_cfg.expert_parallel_size,
    )


class BasePPOExp:
    def __init__(self, cfg: SkyRLTrainConfig):
        """
        Initializes a PPO experiment.

        Args:
            cfg: The fully resolved SkyRLTrainConfig instance.
        """
        self.cfg = cfg
        self.tokenizer = get_tokenizer(
            self.cfg.trainer.policy.model.path,
            trust_remote_code=True,
            use_fast=not self.cfg.trainer.disable_fast_tokenizer,
            padding_side="left",
        )
        self.train_dataset = self.get_train_dataset()
        self.eval_dataset = self.get_eval_dataset()
        self.colocate_pg = self.get_colocate_pg()

        # New inference resources (created lazily when _SKYRL_USE_NEW_INFERENCE=1)
        self._server_groups = None
        self._prefill_server_groups = None
        self._decode_server_groups = None
        self._inference_router = None

    @staticmethod
    def get_cfg_as_str(cfg: SkyRLTrainConfig) -> str:
        return get_config_as_yaml_str(cfg)

    def get_train_dataset(self):
        """Initializes the training dataset.

        Returns:
            PromptDataset: The training dataset.
        """
        prompts_dataset = PromptDataset(
            datasets=self.cfg.data.train_data,
            tokenizer=self.tokenizer,
            max_prompt_length=self.cfg.trainer.max_prompt_length,
            num_workers=8,
        )
        # make sure the dataset is large enough to train on
        assert (
            len(prompts_dataset) >= self.cfg.trainer.train_batch_size
        ), f"dataset should be at least as large as `train_batch_size` {self.cfg.trainer.train_batch_size}, got size {len(prompts_dataset)}"
        return prompts_dataset

    def get_eval_dataset(self):
        """Initializes the evaluation dataset.

        Returns:
            PromptDataset: The evaluation dataset.
        """
        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.val_data:
            prompts_dataset = PromptDataset(
                datasets=self.cfg.data.val_data,
                tokenizer=self.tokenizer,
                max_prompt_length=self.cfg.trainer.max_prompt_length,
                num_workers=8,
            )
            return prompts_dataset
        return None

    def get_colocate_pg(self, timeout: int = SKYRL_RAY_PG_TIMEOUT_IN_S) -> Optional[ResolvedPlacementGroup]:
        """Initializes a placement group for colocated training.

        Creates a single placement group with per-GPU bundles for all inference
        engines. The returned wrapper computes GPU-aware bundle ordering at init time.

        Args:
            timeout (int): The timeout for the placement group to be ready.

        Returns:
            ResolvedPlacementGroup: The placement group wrapper for colocated training, or None.
        """
        if not self.cfg.trainer.placement.colocate_all:
            return None

        ie_cfg = self.cfg.generator.inference_engine
        per_engine_gpu_count = ie_cfg.tensor_parallel_size * ie_cfg.pipeline_parallel_size * ie_cfg.data_parallel_size
        total_gpu_slots = ie_cfg.num_engines * per_engine_gpu_count

        pg = placement_group(
            [{"GPU": 1, "CPU": 1}] * total_gpu_slots,
            strategy="PACK",
        )
        get_ray_pg_ready_with_timeout(pg, timeout=timeout)
        return ResolvedPlacementGroup(pg)

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """Initializes the generator.

        Returns:
            GeneratorInterface: The generator.
        """
        if cfg.generator.vision_language_generator:
            from skyrl.train.generators.skyrl_vlm_generator import SkyRLVLMGymGenerator

            generator_cls = SkyRLVLMGymGenerator
        else:
            from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator

            generator_cls = SkyRLGymGenerator

        return generator_cls(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=cfg.environment.skyrl_gym,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
        )

    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        generator: GeneratorInterface,
        colocate_pg,
    ):
        """Initializes the trainer.

        Returns:
            RayPPOTrainer: The trainer.
        """
        return RayPPOTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )

    def get_tracker(self):
        """Initializes the tracker for experiment tracking.

        Returns:
            Tracking: The tracker.
        """
        return Tracking(
            project_name=self.cfg.trainer.project_name,
            experiment_name=self.cfg.trainer.run_name,
            backends=self.cfg.trainer.logger,
            config=self.cfg,
        )

    def get_inference_client(self) -> InferenceEngineInterface:
        """Setup and return the inference engine client.

        This is a hook method that can be overridden by subclasses to customize
        inference engine creation (e.g., FlashRL, custom backends).

        Returns:
            InferenceEngineInterface: The inference engine client.
        """
        if _SKYRL_USE_NEW_INFERENCE:
            logger.info("Initializing new inference client")
            return self._get_new_inference_client()
        else:
            return self._get_legacy_inference_client()

    def _get_legacy_inference_client(self) -> InferenceEngineInterface:
        """Legacy inference client using Ray actors."""
        if self.cfg.generator.inference_engine.run_engines_locally:
            inference_engines = create_ray_wrapped_inference_engines_from_config(
                self.cfg, self.colocate_pg, self.tokenizer
            )
        else:
            inference_engines = create_remote_inference_engines_from_config(self.cfg, self.tokenizer)

        return InferenceEngineClient(
            inference_engines,
            self.tokenizer,
            self.cfg.trainer.policy.model.path,
            self.cfg.trainer.policy.model.lora,
            self.cfg.generator.inference_engine,
        )

    def _get_new_inference_client(self):
        """New inference client using HTTP endpoints.

        Returns:
            RemoteInferenceClient: The new inference client.
        """
        from skyrl.backends.skyrl_train.inference_servers.setup import (
            build_new_inference_client,
        )

        is_colocated = self.cfg.trainer.placement.colocate_all
        client, server_setup = build_new_inference_client(
            self.cfg,
            self.tokenizer,
            placement_group=self.colocate_pg if is_colocated else None,
        )
        self._inference_router = server_setup.router
        self._server_groups = server_setup.server_groups
        self._prefill_server_groups = server_setup.prefill_server_groups
        self._decode_server_groups = server_setup.decode_server_groups

        if is_colocated:
            # Callers must invoke get_inference_client() from a sync context (no running event loop).
            asyncio.run(client.sleep())
            logger.info("HTTP Inference: Colocated mode - slept inference engines after startup")

        return client

    def _setup_trainer(self):
        """Setup and return the trainer.

        Instantiates the trainer and all the associated models for training.

        Returns:
            RayPPOTrainer: The trainer.
        """
        logger.info(self.get_cfg_as_str(self.cfg))
        os.makedirs(self.cfg.trainer.export_path, exist_ok=True)
        os.makedirs(self.cfg.trainer.ckpt_path, exist_ok=True)

        if self.cfg.trainer.strategy in ("fsdp", "fsdp2"):
            from skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker import (
                CriticWorker,
                PolicyWorker,
                RefWorker,
            )
        elif self.cfg.trainer.strategy == "megatron":
            from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
                CriticWorker,
                PolicyWorker,
                RefWorker,
            )
        else:
            raise ValueError(f"Unknown strategy type: {self.cfg.trainer.strategy}")

        # NOTE (sumanthrh): Instantiate tracker before trainer init.
        # We have custom validation before this step to give better error messages.
        tracker = self.get_tracker()

        inference_engine_client = self.get_inference_client()

        generator: GeneratorInterface = self.get_generator(self.cfg, self.tokenizer, inference_engine_client)

        trainer = self.get_trainer(
            cfg=self.cfg,
            tracker=tracker,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=self.colocate_pg,
        )

        # Build the models
        trainer.build_models(PolicyWorker, CriticWorker, RefWorker)
        return trainer

    def run(self):
        trainer = self._setup_trainer()
        # Start the training loop
        asyncio.run(trainer.train())


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    # make sure that the training loop is not run on the head node.
    exp = BasePPOExp(cfg)
    exp.run()


def main() -> None:
    # Parse CLI args and build typed config
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])

    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
