"""
Shared ThunderAgent experiment base for the Harbor 32B recipe.

Uses ThunderAgentRouter (instead of InferenceRouter) for the new HTTP inference
layer. ThunderAgent intercepts /v1/chat/completions for program scheduling
while routing SkyRL's token-based generation endpoint through the same router.
"""

import asyncio
import os
from pathlib import Path

from loguru import logger

from skyrl.backends.skyrl_train.inference_engines.base import InferenceEngineInterface
from skyrl.env_vars import _SKYRL_USE_NEW_INFERENCE
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer


class FullyAsyncThunderAgentExp(BasePPOExp):
    """Base experiment that uses ThunderAgentRouter for inference."""

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        raise NotImplementedError(
            "FullyAsyncThunderAgentExp is a ThunderAgent inference base. "
            "Use HarborThunderAgentFullyAsyncExp for the Harbor 32B recipe."
        )

    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        generator,
        colocate_pg,
    ):
        return FullyAsyncRayPPOTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )

    def run(self):
        trainer = self._setup_trainer()
        asyncio.run(trainer.train())

    def get_inference_client(self) -> InferenceEngineInterface:
        if _SKYRL_USE_NEW_INFERENCE:
            return self._get_new_inference_client()
        raise ValueError(
            "ThunderAgent integration requires the new inference layer. "
            "Set _SKYRL_USE_NEW_INFERENCE=1 environment variable."
        )

    def _get_new_inference_client(self):
        """Override to use ThunderAgentRouter instead of InferenceRouter."""
        from skyrl.backends.skyrl_train.inference_servers.server_group import (
            ServerGroup,
        )
        from skyrl.backends.skyrl_train.inference_servers.utils import (
            build_vllm_cli_args,
        )

        from .skyrl_integration.remote_inference_client import (
            ThunderAgentRemoteInferenceClient,
        )

        ie_cfg = self.cfg.generator.inference_engine
        is_colocated = self.cfg.trainer.placement.colocate_all
        external_proxy_url = ie_cfg.external_proxy_url
        external_server_urls = ie_cfg.external_server_urls

        has_external_proxy = external_proxy_url is not None
        has_external_servers = external_server_urls is not None

        if has_external_proxy and has_external_servers:
            proxy_url = external_proxy_url
            server_urls = list(external_server_urls)
            logger.info(
                "HTTP Inference (ThunderAgent): Using fully external setup - "
                f"proxy_url={proxy_url}, server_urls={server_urls}"
            )
        elif has_external_proxy and not has_external_servers:
            raise ValueError(
                "ThunderAgent requires external_server_urls when using external_proxy_url. "
                "SkyRL fans out control-plane calls (pause, resume, weight sync) to all "
                "server_urls entries. With only a proxy URL, those calls would hit the proxy "
                "instead of the actual backends. Set external_server_urls to the list of "
                "backend URLs behind the proxy."
            )
        elif has_external_servers and not has_external_proxy:
            server_urls = list(external_server_urls)
            self._inference_router = self._create_thunder_agent_router(server_urls, ie_cfg)
            proxy_url = self._inference_router.start()
            logger.info(
                "HTTP Inference (ThunderAgent): Created router over external "
                f"servers - server_urls={server_urls}, proxy_url={proxy_url}"
            )
        else:
            cli_args = build_vllm_cli_args(self.cfg)
            self._server_group = ServerGroup(
                cli_args=cli_args,
                num_servers=ie_cfg.num_engines,
                placement_group=self.colocate_pg if is_colocated else None,
                enable_dp=ie_cfg.data_parallel_size > 1,
            )
            server_infos = self._server_group.start()
            server_urls = [info.url for info in server_infos]

            self._inference_router = self._create_thunder_agent_router(server_urls, ie_cfg)
            proxy_url = self._inference_router.start()
            logger.info(
                "HTTP Inference (ThunderAgent): Built servers and router internally - "
                f"proxy_url={proxy_url}, server_urls={server_urls}, colocated={is_colocated}"
            )

        return ThunderAgentRemoteInferenceClient(
            proxy_url=proxy_url,
            server_urls=server_urls,
            model_name=self.cfg.trainer.policy.model.path,
        )

    def _create_thunder_agent_router(self, server_urls, ie_cfg):
        from .skyrl_integration.router import ThunderAgentRouter

        thunderagent_log_file = str(Path(self.cfg.trainer.log_path) / "thunderagent.log")
        thunderagent_port = int(os.environ.get("THUNDER_AGENT_ROUTER_PORT", "8080"))
        return ThunderAgentRouter(
            server_urls=server_urls,
            port=thunderagent_port,
            log_file=thunderagent_log_file,
            router_mode=ie_cfg.thunder_agent_mode,
            backend_type=ie_cfg.backend,
            acting_token_weight=ie_cfg.thunder_agent_acting_token_weight,
            scheduler_interval=ie_cfg.thunder_agent_scheduler_interval,
            use_acting_token_decay=ie_cfg.thunder_agent_use_acting_token_decay,
            profile_enabled=ie_cfg.thunder_agent_profile_enabled,
            metrics_enabled=ie_cfg.thunder_agent_metrics_enabled,
            metrics_interval=ie_cfg.thunder_agent_metrics_interval,
        )
