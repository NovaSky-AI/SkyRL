"""A pool of skycap servers, one Ray actor each.

Each actor runs one server (``skycap.CaptureService``) on a port it picks itself, so
servers never collide, and advertises its node's address. The actors sit in one
placement group whose strategy is configurable: ``SPREAD`` by default, so one
node going away takes one server rather than all of them. The generator spreads
trajectories over the pool's URLs round-robin. Given the actors, it keeps each group on
one server, which composes its rows into that node's object store.

With an exposure (``skycap.exposure``), each server also serves its harness
routes to agents inside remote sandboxes: the actor builds its own ``Exposure``
from the name and kwargs, and skycap opens and closes it with the server.
"""

from typing import Any, Dict, List, Optional, Tuple

import ray
from loguru import logger

from skyrl.backends.skyrl_train.inference_servers.common import (
    default_bind_host,
    get_node_ip,
)
from skyrl.train.generators.base import GeneratorOutput
from skyrl.train.generators.skycap_store import (
    SkycapServers,
    StoredCaptureServer,
    start_server_pool,
)

from .compose import TrialChecks, compose


@ray.remote(num_cpus=0)
class SkycapServerActor(StoredCaptureServer):
    def __init__(
        self,
        settings: Dict[str, Any],
        record_dir: Optional[str],
        ttl: float,
        exposure: Optional[Tuple[str, Dict[str, Any]]] = None,
        sample_support_logprobs: bool = False,
    ) -> None:
        from skycap import CaptureService
        from skycap.exposure import load_exposure
        from skyrl.train.generators.skycap_engine import SkyRLEngine

        node_ip = get_node_ip()
        super().__init__(
            CaptureService(
                mode="tokens",
                engine=SkyRLEngine(sample_support_logprobs=sample_support_logprobs),
                record_dir=record_dir,
                ttl=ttl,
                host=default_bind_host(node_ip),
                port=0,
                advertise_host=node_ip,
                exposure=load_exposure(exposure[0], **exposure[1]) if exposure is not None else None,
                **settings,
            )
        )

    def checks(self, samples: List[Any]) -> TrialChecks:
        return TrialChecks.of(samples)

    def compose(self, outcomes: List[Any], **compose_options: Any) -> GeneratorOutput:
        return compose(outcomes, **compose_options)


def start_servers(
    settings: Dict[str, Any],
    *,
    num_servers: int,
    num_cpus_per_server: float,
    placement_strategy: str,
    record_dir: Optional[str],
    ttl: float,
    exposure: Optional[str] = None,
    exposure_kwargs: Optional[Dict[str, Any]] = None,
    sample_support_logprobs: bool = False,
    max_concurrent_finishes: int = 64,
) -> SkycapServers:
    """``num_servers`` skycap servers in token mode, in front of SkyRL's router.

    ``settings`` are ``skycap.CaptureService``'s options (``upstream_url``, ``tokenizer``, sampling, ...).
    ``exposure`` names a ``skycap.exposure`` way in for agents in remote sandboxes, built in each actor
    with ``exposure_kwargs``; for ``external_host``, server ``i`` gets ``port + i``.
    ``sample_support_logprobs`` asks the engine for the sampler's logprob of each support member.
    """
    servers = start_server_pool(
        SkycapServerActor,
        lambda i: (settings, record_dir, ttl, exposure_for(exposure, exposure_kwargs, i), sample_support_logprobs),
        num_servers=num_servers,
        num_cpus_per_server=num_cpus_per_server,
        placement_strategy=placement_strategy,
        max_concurrent_finishes=max_concurrent_finishes,
    )
    logger.info(f"skycap serving at {servers.urls}" + (f", exposed by {exposure}" if exposure else ""))
    return servers


def exposure_for(
    exposure: Optional[str], kwargs: Optional[Dict[str, Any]], index: int
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Server ``index``'s exposure, as the name and kwargs its actor builds it from."""
    if exposure is None:
        return None
    kwargs = dict(kwargs or {})
    if exposure == "external_host":
        kwargs["port"] = kwargs.get("port", EXTERNAL_HOST_PORT) + index
    return exposure, kwargs


#: ``external_host``'s first port when none is given; server ``i`` gets ``EXTERNAL_HOST_PORT + i``.
EXTERNAL_HOST_PORT = 11500
