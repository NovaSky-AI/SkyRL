"""A pool of skycap servers, one Ray actor each.

Each actor runs one server (``skycap.CaptureService``) on a port it picks itself, so
servers never collide, and advertises its node's address. The actors sit in one
placement group whose strategy is configurable: ``SPREAD`` by default, so one
node going away takes one server rather than all of them. The generator spreads
trajectories over the pool's URLs round-robin.

With an exposure (``skycap.exposure``), each server also serves its harness
routes to agents inside remote sandboxes: the actor builds its own ``Exposure``
from the name and kwargs, and skycap opens and closes it with the server.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import ray
from loguru import logger
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from skyrl.backends.skyrl_train.inference_servers.common import (
    default_bind_host,
    get_node_ip,
)


@ray.remote(num_cpus=0)
class SkycapServerActor:
    def __init__(
        self,
        settings: Dict[str, Any],
        record_dir: Optional[str],
        ttl: float,
        exposure: Optional[Tuple[str, Dict[str, Any]]] = None,
    ) -> None:
        from skycap import CaptureService
        from skycap.exposure import load_exposure

        from .engine import SkyRLEngine

        node_ip = get_node_ip()
        self.service = CaptureService(
            mode="tokens",
            engine=SkyRLEngine(),
            record_dir=record_dir,
            ttl=ttl,
            host=default_bind_host(node_ip),
            port=0,
            advertise_host=node_ip,
            exposure=load_exposure(exposure[0], **exposure[1]) if exposure is not None else None,
            **settings,
        )

    def start(self) -> str:
        return self.service.start()

    def stop(self, timeout: float) -> bool:
        return self.service.stop(timeout)


@dataclass
class SkycapServers:
    """The running pool. ``urls`` is what the generator is given."""

    actors: List[Any]
    urls: List[str]
    pg: Any
    stop_timeout: float = 600.0
    _stopped: bool = field(default=False, repr=False)

    def stop(self) -> None:
        """Stop every server, writing the trajectories still in memory. Idempotent."""
        if self._stopped:
            return
        self._stopped = True
        flushed = ray.get([actor.stop.remote(self.stop_timeout) for actor in self.actors])
        for url, done in zip(self.urls, flushed):
            if not done:
                logger.error(f"skycap at {url} did not finish writing its trajectories within {self.stop_timeout}s")
        for actor in self.actors:
            ray.kill(actor)
        remove_placement_group(self.pg)


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
) -> SkycapServers:
    """``num_servers`` skycap servers in token mode, in front of SkyRL's router.

    ``settings`` are ``skycap.CaptureService``'s options (``upstream_url``, ``tokenizer``, sampling, ...).
    ``exposure`` names a ``skycap.exposure`` way in for agents in remote sandboxes, built in each actor
    with ``exposure_kwargs``; for ``external_host``, server ``i`` gets ``port + i``.
    """
    if num_servers < 1:
        raise ValueError("skycap.num_servers must be at least 1")
    pg = placement_group([{"CPU": num_cpus_per_server}] * num_servers, strategy=placement_strategy)
    ray.get(pg.ready())
    actors = [
        SkycapServerActor.options(
            num_cpus=num_cpus_per_server,
            scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=i),
        ).remote(settings, record_dir, ttl, exposure_for(exposure, exposure_kwargs, i))
        for i in range(num_servers)
    ]
    try:
        urls = ray.get([actor.start.remote() for actor in actors])
    except BaseException:
        # A server or its exposure failed to start. Killing the actors takes their tunnels with them
        # (skycap ties cloudflared to its server's process); nothing was handed out yet, so nothing is lost.
        for actor in actors:
            ray.kill(actor)
        remove_placement_group(pg)
        raise
    logger.info(f"skycap serving at {urls}" + (f", exposed by {exposure}" if exposure else ""))
    return SkycapServers(actors=actors, urls=urls, pg=pg)


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
