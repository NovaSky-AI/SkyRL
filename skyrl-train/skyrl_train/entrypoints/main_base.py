"""
Main entrypoint for training.
"""

from typing import Union
from ray.util.placement_group import placement_group, PlacementGroup

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from skyrl_train.dataset import PromptDataset
from skyrl_train.utils import validate_cfg

from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.inference_engines.base import InferenceEngineInterface
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.remote_inference_engine import create_remote_inference_engines
from skyrl_train.utils.utils import initialize_ray, get_ray_pg_ready_with_timeout
from skyrl_train.inference_servers.utils import build_vllm_cli_args
from skyrl_train.env_vars import SKYRL_RAY_PG_TIMEOUT_IN_S, _SKYRL_USE_NEW_INFERENCE
from skyrl_train.generators.base import GeneratorInterface
from omegaconf import DictConfig
from skyrl_train.config import SkyRLConfig, get_config_as_yaml_str
from pathlib import Path
import ray

import os
import hydra
from loguru import logger
from skyrl_train.utils.tracking import Tracking
import multiprocessing as mp
import asyncio

# NOTE (sumanthrh): We use ray heavily and thus disable `fork` start method.
# forking within ray leads to undefined behaviour and often causes hard to debug
# memory leaks.  See: https://docs.ray.io/en/latest/ray-core/patterns/fork-new-processes.html
# A common culprit is Pytorch dataloaders which use `fork` by default.
mp.set_start_method("spawn", force=True)

config_dir = str(Path(__file__).parent.parent / "config")
__all__ = ["BasePPOExp", "config_dir"]


def create_ray_wrapped_inference_engines_from_config(
    cfg: Union[SkyRLConfig, DictConfig], colocate_pg, tokenizer: PreTrainedTokenizerBase
):
    from skyrl_train.inference_engines.ray_wrapped_inference_engine import create_ray_wrapped_inference_engines

    engine_kwargs = {
        "num_inference_engines": cfg.generator.num_inference_engines,
        "tensor_parallel_size": cfg.generator.inference_engine_tensor_parallel_size,
        "pipeline_parallel_size": cfg.generator.inference_engine_pipeline_parallel_size,
        "model_dtype": cfg.generator.model_dtype,
        "pretrain": cfg.trainer.policy.model.path,
        "seed": cfg.trainer.seed,
        "vllm_v1_disable_multiproc": cfg.generator.vllm_v1_disable_multiproc,
        "enable_prefix_caching": cfg.generator.enable_prefix_caching,
        "enforce_eager": cfg.generator.enforce_eager,
        "expert_parallel_size": cfg.generator.inference_engine_expert_parallel_size,
        "data_parallel_size": cfg.generator.inference_engine_data_parallel_size,
        "shared_pg": colocate_pg,
        "gpu_memory_utilization": cfg.generator.gpu_memory_utilization,
        "inference_engine_enable_sleep": cfg.trainer.placement.colocate_all,
        "async_engine": cfg.generator.async_engine,
        "max_num_batched_tokens": cfg.generator.max_num_batched_tokens,
        "max_num_seqs": cfg.generator.max_num_seqs,
        "tokenizer": tokenizer,
        "backend": cfg.generator.backend,
        "engine_init_kwargs": cfg.generator.engine_init_kwargs,
        "enable_ray_prometheus_stats": cfg.generator.enable_ray_prometheus_stats,
    }

    # Conditionally add LoRA parameters if LoRA is enabled
    if cfg.trainer.policy.model.lora.rank > 0 and cfg.trainer.strategy != "megatron":
        engine_kwargs["enable_lora"] = True
        engine_kwargs["max_lora_rank"] = cfg.trainer.policy.model.lora.rank
        engine_kwargs["sleep_level"] = 1
        engine_kwargs["max_loras"] = 1
        engine_kwargs["fully_sharded_loras"] = cfg.generator.fully_sharded_loras

        # TODO(devpatel): Bandaid solution, replace this once we have a better
        # solution for LoRA performance degradation on the vLLM side
        if cfg.generator.enforce_eager and cfg.generator.backend == "vllm":
            logger.warning(
                "LoRA is enabled but generator.enforce_eager=true. "
                "This combination causes significant performance degradation (2-3x slower generation). "
                "Automatically setting enforce_eager=false for better performance. "
            )
            engine_kwargs["enforce_eager"] = False

    if cfg.generator.rope_scaling is not None:
        engine_kwargs["rope_scaling"] = cfg.generator.rope_scaling
    if cfg.generator.rope_theta is not None:
        engine_kwargs["rope_theta"] = cfg.generator.rope_theta
    if cfg.generator.served_model_name is not None:
        engine_kwargs["served_model_name"] = cfg.generator.served_model_name

    return create_ray_wrapped_inference_engines(**engine_kwargs)


def create_remote_inference_engines_from_config(
    cfg: Union[SkyRLConfig, DictConfig], tokenizer: PreTrainedTokenizerBase
):
    # TODO(tgriggs): We may want a separate config for the model name in case
    # it's different from the name used in the OpenAI API
    return create_remote_inference_engines(
        urls=cfg.generator.remote_inference_engine_urls,
        model_name=cfg.trainer.policy.model.path,
        engine_backend=cfg.generator.backend,
        tokenizer=tokenizer,
        tensor_parallel_size=cfg.generator.inference_engine_tensor_parallel_size,
        pipeline_parallel_size=cfg.generator.inference_engine_pipeline_parallel_size,
        data_parallel_size=cfg.generator.inference_engine_data_parallel_size,
        expert_parallel_size=cfg.generator.inference_engine_expert_parallel_size,
    )


class BasePPOExp:
    def __init__(self, cfg: Union[SkyRLConfig, DictConfig]):
        """
        Initializes a PPO experiment.

        The `cfg` passed here will be the final config from Hydra, including CLI overrides.
        """
        # TODO (sumanthrh): Migrate to using SkyRLConfig
        self.cfg = cfg
        self.tokenizer = self.get_tokenizer()
        self.train_dataset = self.get_train_dataset()
        self.eval_dataset = self.get_eval_dataset()
        self.colocate_pg = self.get_colocate_pg()

        # New inference resources (created lazily when _SKYRL_USE_NEW_INFERENCE=1)
        self._server_group = None
        self._inference_router = None

    @staticmethod
    def get_cfg_as_str(cfg: Union[SkyRLConfig, DictConfig]) -> str:
        return get_config_as_yaml_str(cfg)

    def get_tokenizer(self, padding_side="left"):
        """Initializes a tokenizer for the given model."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.trainer.policy.model.path,
            trust_remote_code=True,
            use_fast=not self.cfg.trainer.disable_fast_tokenizer,
        )
        tokenizer.padding_side = padding_side
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

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

    def get_colocate_pg(self, timeout: int = SKYRL_RAY_PG_TIMEOUT_IN_S) -> PlacementGroup:
        """Initializes a placement group for colocated training.

        A single placement group that packs all the inference engines together is created.

        Args:
            timeout (int): The timeout for the placement group to be ready.

        Returns:
            PlacementGroup: The placement group for colocated training.
        """
        if self.cfg.trainer.placement.colocate_all:
            pg = placement_group(
                [{"GPU": 1, "CPU": 1}]
                * self.cfg.generator.num_inference_engines
                * self.cfg.generator.inference_engine_tensor_parallel_size
                * self.cfg.generator.inference_engine_pipeline_parallel_size
                * self.cfg.generator.inference_engine_data_parallel_size,
                strategy="PACK",
            )
            get_ray_pg_ready_with_timeout(pg, timeout=timeout)
            return pg
        else:
            return None

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """Initializes the generator.

        Returns:
            GeneratorInterface: The generator.
        """
        from skyrl_train.generators.skyrl_gym_generator import SkyRLGymGenerator

        return SkyRLGymGenerator(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=cfg.environment.skyrl_gym,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            model_name=cfg.trainer.policy.model.path,
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
        if self.cfg.generator.run_engines_locally:
            inference_engines = create_ray_wrapped_inference_engines_from_config(
                self.cfg, self.colocate_pg, self.tokenizer
            )
        else:
            inference_engines = create_remote_inference_engines_from_config(self.cfg, self.tokenizer)

        return InferenceEngineClient(inference_engines, self.tokenizer, self.cfg)

    def _get_new_inference_client(self):
        """New inference client using HTTP endpoints.

        Config combinations:
        - Colocated + external URLs → ERROR (validated earlier)
        - Neither set → Build servers internally
        - external_server_urls only → Create router over external servers
        - external_proxy_url only → Use proxy for both data + control plane
        - Both set → Fully external (proxy for data plane, servers for control plane)

        Returns:
            RemoteInferenceClient: The new inference client.
        """
        from skyrl_train.inference_servers.remote_inference_client import RemoteInferenceClient
        from skyrl_train.inference_servers.router import InferenceRouter
        from skyrl_train.inference_servers.server_group import ServerGroup

        is_colocated = self.cfg.trainer.placement.colocate_all
        external_proxy_url = self.cfg.generator.get("external_proxy_url")
        external_server_urls = self.cfg.generator.get("external_server_urls")

        has_external_proxy = external_proxy_url is not None
        has_external_servers = external_server_urls is not None

        if has_external_proxy and has_external_servers:
            # Case: Both external - fully external setup
            proxy_url = external_proxy_url
            server_urls = list(external_server_urls)
            logger.info(
                f"HTTP Inference: Using fully external setup - " f"proxy_url={proxy_url}, server_urls={server_urls}"
            )

        elif has_external_proxy and not has_external_servers:
            # Case: Proxy only - assume proxy handles control plane too
            proxy_url = external_proxy_url
            server_urls = [proxy_url]
            logger.info(
                f"HTTP Inference: Using external proxy for both data and " f"control plane - proxy_url={proxy_url}"
            )

        elif has_external_servers and not has_external_proxy:
            # Case: Servers only - create internal router over them
            server_urls = list(external_server_urls)
            self._inference_router = InferenceRouter(server_urls=server_urls)
            proxy_url = self._inference_router.start()
            logger.info(
                f"HTTP Inference: Created internal router over external "
                f"servers - server_urls={server_urls}, proxy_url={proxy_url}"
            )

        else:
            # Case: Neither - build servers and router internally
            cli_args = build_vllm_cli_args(self.cfg)

            self._server_group = ServerGroup(
                cli_args=cli_args,
                num_servers=self.cfg.generator.num_inference_engines,
                placement_group=self.colocate_pg if is_colocated else None,
                enable_dp=self.cfg.generator.inference_engine_data_parallel_size > 1,
            )
            server_infos = self._server_group.start()
            server_urls = [info.url for info in server_infos]

            self._inference_router = InferenceRouter(server_urls=server_urls)
            proxy_url = self._inference_router.start()
            logger.info(
                f"HTTP Inference: Built servers and router internally - "
                f"proxy_url={proxy_url}, server_urls={server_urls}, colocated={is_colocated}"
            )

        return RemoteInferenceClient(
            proxy_url=proxy_url,
            server_urls=server_urls,
            model_name=self.cfg.trainer.policy.model.path,
        )

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
            from skyrl_train.workers.fsdp.fsdp_worker import PolicyWorker, CriticWorker, RefWorker
        elif self.cfg.trainer.strategy == "megatron":
            from skyrl_train.workers.megatron.megatron_worker import PolicyWorker, CriticWorker, RefWorker
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
def skyrl_entrypoint(cfg: DictConfig):
    # make sure that the training loop is not run on the head node.
    exp = BasePPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
