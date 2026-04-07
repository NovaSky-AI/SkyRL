import copy
from argparse import Namespace
from dataclasses import dataclass
from typing import List, Optional

from loguru import logger

from skyrl.train.config import InferenceEngineConfig

from .server_group import ServerGroup
from .utils import build_router_args, get_pd_cli_args
from .vllm_router import VLLMRouter


@dataclass
class InferenceServerSetup:
    """Simple dataclass for inference server setup result."""

    router: "VLLMRouter"
    proxy_url: str
    server_urls: List[str]
    server_groups: Optional[List["ServerGroup"]] = None
    prefill_server_groups: Optional[List["ServerGroup"]] = None
    decode_server_groups: Optional[List["ServerGroup"]] = None


def create_inference_servers(
    ie_cfg: InferenceEngineConfig,
    cli_args: Namespace,
    log_path: str,
    placement_group=None,
) -> InferenceServerSetup:
    """Build server groups and router from config.

    Shared logic for ``main_base.py`` and test utilities.  Creates one
    :class:`ServerGroup` per engine (each with ``data_parallel_size``
    servers).  When ``enable_pd=True``, prefill and decode groups are
    created separately and the router is configured for PD disaggregation.

    Args:
        ie_cfg: Inference engine config.
        cli_args: vLLM CLI args from :func:`build_vllm_cli_args`.
        log_path: Log path for SkyRL logs
        placement_group: Optional resolved placement group for colocated
            training.  ``None`` when not colocated.

    Returns:
        An :class:`InferenceServerSetup` with the router, URLs, and
        server group references.
    """
    from skyrl.backends.skyrl_train.inference_servers.server_group import ServerGroup
    from skyrl.backends.skyrl_train.inference_servers.vllm_router import VLLMRouter

    gpus_per_server = ie_cfg.tensor_parallel_size * ie_cfg.pipeline_parallel_size

    if ie_cfg.enable_pd:
        pd_cli_args = get_pd_cli_args(cli_args)
        num_prefill = ie_cfg.num_prefill
        num_decode = ie_cfg.num_engines - num_prefill
        servers_per_group = ie_cfg.data_parallel_size

        prefill_server_groups = [
            ServerGroup(
                cli_args=copy.deepcopy(pd_cli_args),
                num_servers=ie_cfg.data_parallel_size,
                start_port=8000 + i * servers_per_group,
                placement_group=placement_group,
                placement_group_bundle_offset=i * gpus_per_server * servers_per_group,
                enable_dp=ie_cfg.data_parallel_size > 1,
                enable_pd=True,
                nixl_side_channel_base=5600 + i * servers_per_group,
                distributed_executor_backend=ie_cfg.distributed_executor_backend,
            )
            for i in range(num_prefill)
        ]
        prefill_infos_nested = [g.start() for g in prefill_server_groups]
        prefill_urls = [info.url for infos in prefill_infos_nested for info in infos]

        decode_bundle_offset = num_prefill * gpus_per_server * servers_per_group if placement_group else 0
        decode_server_groups = [
            ServerGroup(
                cli_args=copy.deepcopy(pd_cli_args),
                num_servers=ie_cfg.data_parallel_size,
                start_port=8000 + (num_prefill + i) * servers_per_group,
                placement_group=placement_group,
                placement_group_bundle_offset=decode_bundle_offset + i * gpus_per_server * servers_per_group,
                enable_dp=ie_cfg.data_parallel_size > 1,
                enable_pd=True,
                nixl_side_channel_base=5600 + (num_prefill + i) * servers_per_group,
                distributed_executor_backend=ie_cfg.distributed_executor_backend,
            )
            for i in range(num_decode)
        ]
        decode_infos_nested = [g.start() for g in decode_server_groups]
        decode_urls = [info.url for infos in decode_infos_nested for info in infos]

        server_urls = prefill_urls + decode_urls

        router_args = build_router_args(ie_cfg, prefill_urls=prefill_urls, decode_urls=decode_urls)
        router = VLLMRouter(router_args, log_path=log_path)
        proxy_url = router.start()
        logger.info(
            f"HTTP Inference (PD): prefill_urls={prefill_urls}, decode_urls={decode_urls}, "
            f"proxy_url={proxy_url}, colocated={placement_group is not None}"
        )
        return InferenceServerSetup(
            router=router,
            proxy_url=proxy_url,
            server_urls=server_urls,
            prefill_server_groups=prefill_server_groups,
            decode_server_groups=decode_server_groups,
        )
    else:
        server_groups = [
            ServerGroup(
                cli_args=cli_args,
                num_servers=ie_cfg.data_parallel_size,
                placement_group=placement_group,
                enable_dp=ie_cfg.data_parallel_size > 1,
                distributed_executor_backend=ie_cfg.distributed_executor_backend,
                placement_group_bundle_offset=i * gpus_per_server * ie_cfg.data_parallel_size,
            )
            for i in range(ie_cfg.num_engines)
        ]
        server_infos_nested = [g.start() for g in server_groups]
        server_urls = [info.url for infos in server_infos_nested for info in infos]

        router_args = build_router_args(ie_cfg, server_urls=server_urls)
        router = VLLMRouter(router_args, log_path=log_path)
        proxy_url = router.start()
        logger.info(
            f"HTTP Inference: proxy_url={proxy_url}, server_urls={server_urls}, "
            f"colocated={placement_group is not None}"
        )
        return InferenceServerSetup(
            router=router,
            proxy_url=proxy_url,
            server_urls=server_urls,
            server_groups=server_groups,
        )
