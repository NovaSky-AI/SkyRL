"""A pool of skycap servers, one Ray actor each.

Each actor runs one server (``SkycapService``) on a port it picks itself, so
servers never collide, and advertises its node's address. The actors sit in one
placement group whose strategy is configurable: ``SPREAD`` by default, so one
node going away takes one server rather than all of them. The generator spreads
trajectories over the pool's URLs round-robin.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import ray
from loguru import logger
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from skyrl.backends.skyrl_train.inference_servers.common import (
    default_bind_host,
    get_node_ip,
)

#: Builds a skycap backend from plain settings, inside the actor.
BackendFactory = Callable[[Dict[str, Any]], Any]


def build_tokens_backend(settings: Dict[str, Any]) -> Any:
    """skycap in token mode, in front of SkyRL's router."""
    from skycap.tokens.backend import TokensBackend
    from skycap.tokens.renderer import RenderersRenderer

    from .engine import SkyRLEngine

    return TokensBackend(
        settings["engine_url"],
        RenderersRenderer(settings["tokenizer"], size=settings["renderer_pool_size"]),
        engine=SkyRLEngine(),
        model=settings["model"],
        max_model_len=settings["max_model_len"],
        sampling_overrides=settings["sampling_overrides"],
        sampling_mask=settings["sampling_mask"],
    )


@ray.remote(num_cpus=0)
class SkycapServerActor:
    def __init__(
        self, settings: Dict[str, Any], record_dir: Optional[str], ttl: float, backend_factory: BackendFactory
    ) -> None:
        from .service import SkycapService

        node_ip = get_node_ip()
        self.service = SkycapService(
            backend_factory(settings),
            record_dir=record_dir,
            ttl=ttl,
            host=default_bind_host(node_ip),
            port=0,
            advertise_host=node_ip,
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
    backend_factory: BackendFactory = build_tokens_backend,
) -> SkycapServers:
    if num_servers < 1:
        raise ValueError("skycap.num_servers must be at least 1")
    pg = placement_group([{"CPU": num_cpus_per_server}] * num_servers, strategy=placement_strategy)
    ray.get(pg.ready())
    actors = [
        SkycapServerActor.options(
            num_cpus=num_cpus_per_server,
            scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=i),
        ).remote(settings, record_dir, ttl, backend_factory)
        for i in range(num_servers)
    ]
    urls = ray.get([actor.start.remote() for actor in actors])
    logger.info(f"skycap serving at {urls}")
    return SkycapServers(actors=actors, urls=urls, pg=pg)
