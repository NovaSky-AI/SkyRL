"""A pool of skycap servers, one Ray actor each.

Each actor runs one server (``skycap.CaptureService``) on a port it picks itself, so
servers never collide, and advertises its node's address. The actors sit in one
placement group whose strategy is configurable: ``SPREAD`` by default, so one
node going away takes one server rather than all of them. The generator spreads
trajectories over the pool's URLs round-robin.

With an exposure (``exposure.py``), each server also listens with its harness
routes alone, where the exposure says, and the actor makes that listener
reachable from agents inside sandboxes; it closes that before the server stops.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import ray
from loguru import logger
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from skyrl.backends.skyrl_train.inference_servers.common import (
    default_bind_host,
    get_node_ip,
)

from .exposure import Exposure


@ray.remote(num_cpus=0)
class SkycapServerActor:
    def __init__(
        self,
        settings: Dict[str, Any],
        record_dir: Optional[str],
        ttl: float,
        exposure: Optional[Callable[[], Exposure]] = None,
        index: int = 0,
    ) -> None:
        from skycap import CaptureService

        from .engine import SkyRLEngine

        self.exposure = exposure() if exposure is not None else None
        self.index = index
        node_ip = get_node_ip()
        harness_host, harness_port = self.exposure.bind(index) if self.exposure is not None else (None, 0)
        self.service = CaptureService(
            mode="tokens",
            engine=SkyRLEngine(),
            record_dir=record_dir,
            ttl=ttl,
            host=default_bind_host(node_ip),
            port=0,
            advertise_host=node_ip,
            harness_host=harness_host,
            harness_port=harness_port,
            **settings,
        )

    def start(self) -> Tuple[str, Optional[str]]:
        """Start serving. Returns the server's URL, and the URL its harness routes are exposed at, if any."""
        url = self.service.start()
        if self.exposure is None:
            return url, None
        assert self.service.harness_url is not None
        return url, self.exposure.open(self.service.harness_url, self.index)

    def stop(self, timeout: float) -> bool:
        # Close the way in before stopping the server behind it; the server writes what it holds either way.
        if self.exposure is not None:
            try:
                self.exposure.close()
            except Exception:
                logger.exception(f"closing skycap server {self.index}'s exposure failed")
        return self.service.stop(timeout)


@dataclass
class SkycapServers:
    """The running pool. ``urls`` is what the generator is given."""

    actors: List[Any]
    urls: List[str]
    pg: Any
    #: Server URL to the URL agents in sandboxes reach its harness routes at; empty when nothing is exposed.
    harness_urls: Dict[str, str] = field(default_factory=dict)
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
    exposure: Optional[Callable[[], Exposure]] = None,
) -> SkycapServers:
    """``num_servers`` skycap servers in token mode, in front of SkyRL's router.

    ``settings`` are ``skycap.CaptureService``'s options (``upstream_url``, ``tokenizer``, sampling, ...).
    ``exposure`` builds each server's ``Exposure`` (``exposure.exposure_factory``); None exposes nothing.
    """
    if num_servers < 1:
        raise ValueError("skycap.num_servers must be at least 1")
    pg = placement_group([{"CPU": num_cpus_per_server}] * num_servers, strategy=placement_strategy)
    ray.get(pg.ready())
    actors = [
        SkycapServerActor.options(
            num_cpus=num_cpus_per_server,
            scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=i),
        ).remote(settings, record_dir, ttl, exposure, i)
        for i in range(num_servers)
    ]
    servers = SkycapServers(actors=actors, urls=[], pg=pg)
    try:
        started = ray.get([actor.start.remote() for actor in actors])
    except BaseException:
        # A server or its exposure failed to start: close what the others opened (tunnels are processes).
        try:
            servers.stop()
        except Exception:  # noqa: BLE001 - the start failure is the error to raise
            logger.exception("stopping skycap after a failed start")
        raise
    servers.urls = [url for url, _ in started]
    servers.harness_urls = {url: exposed for url, exposed in started if exposed is not None}
    logger.info(f"skycap serving at {servers.urls}")
    if servers.harness_urls:
        logger.info(f"skycap harness routes exposed to sandboxes at {list(servers.harness_urls.values())}")
    return servers
