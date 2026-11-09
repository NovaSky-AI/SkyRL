"""Keep skycap rollouts in the object store of the node whose server captured them.

A skycap server actor (``StoredCaptureServer``) finishes each trajectory in its own process and
keeps the samples there (``SampleCache``); finishing returns only the integration's checks. Once a
group's trials end, the actor composes the group's rows and returns each trajectory's payload as a
Ray object owned by the caller, so the token payload never crosses HTTP or the caller's process.
``start_server_pool`` runs one such actor per placement-group bundle.

Needs the ``skycap`` extra.
"""

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import ray
from loguru import logger
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from skycap import CapturePool, CaptureService, Sample
from skycap.client import RecordLocation, Trajectory
from skyrl.train.generators.base import GeneratorOutput, StoredRolloutGroup

ChecksT = TypeVar("ChecksT")


@dataclass(frozen=True)
class CapturedFinish(Generic[ChecksT]):
    """What finishing a trajectory returns when its samples stay with the server."""

    id: str
    status: str
    # Token-mode calls whose prompt had to be rendered rather than extended.
    unbridged_calls: int
    # The integration's verdict on the samples, computed where they live.
    checks: ChecksT
    record: RecordLocation | None = None


class SampleCache:
    """Finished trajectories' samples, held by the server actor until their group is composed.

    An entry leaves when its group takes it or discards it (a retried attempt, a cancelled group),
    however long the group's slowest trial runs. Thread-safe: a server actor finishes
    trajectories concurrently.
    """

    def __init__(self) -> None:
        self._entries: Dict[str, List[Sample]] = {}
        self._lock = threading.Lock()

    def put(self, trajectory_id: str, samples: List[Sample]) -> None:
        with self._lock:
            self._entries[trajectory_id] = samples

    def take(self, trajectory_id: str) -> List[Sample]:
        with self._lock:
            samples = self._entries.pop(trajectory_id, None)
        if samples is None:
            raise KeyError(f"no samples cached for {trajectory_id}; was it finished on this server?")
        return samples

    def discard(self, trajectory_ids: Sequence[str]) -> None:
        with self._lock:
            for trajectory_id in trajectory_ids:
                self._entries.pop(trajectory_id, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class StoredCapturePool(CapturePool):
    """A pool over one server actor that finishes trajectories in the actor's process.

    ``finish`` returns a ``CapturedFinish``, not samples. ``trajectory()`` still finishes every
    trajectory it creates, so the actor caches the samples of every attempt; the caller discards
    the ones it does not compose.
    """

    def __init__(self, url: str, actor: Any, *, timeout: float = 60.0) -> None:
        super().__init__([url], timeout=timeout)
        self.actor = actor
        # The finish deadline the HTTP pool's request timeout gives, kept by the server's actor.
        self.finish_timeout = timeout

    async def finish(self, trajectory: Trajectory, annotations: Dict[str, Any]) -> CapturedFinish:
        return await self.actor.finish_trajectory.remote(
            trajectory.id, annotations, paths=trajectory.paths, timeout=self.finish_timeout
        )


#: Times a group ran again because its server actor died; summed over a step's groups.
SERVER_DIED_METRIC = "generate/errors/server_died"


class StoredPools:
    """Round-robin over stored pools, one per server actor, that skips servers whose actor died.

    A group runs on one pool (its samples are cached in that actor). ``run`` gives the group the
    next live pool, and runs it again on another if the actor dies under it, where the HTTP pool
    would fail over to another server.
    """

    def __init__(self, pools: Sequence[StoredCapturePool]) -> None:
        self.pools = list(pools)
        self._next = 0
        self._dead: set = set()

    def next(self) -> StoredCapturePool:
        for _ in range(len(self.pools)):
            index, self._next = self._next, (self._next + 1) % len(self.pools)
            if index not in self._dead:
                return self.pools[index]
        raise RuntimeError("every skycap server actor has died")

    async def run(self, group: Callable[[StoredCapturePool], Awaitable[StoredRolloutGroup]]) -> StoredRolloutGroup:
        """``group`` on a live pool; reruns after a server death count in its ``SERVER_DIED_METRIC``."""
        from ray.exceptions import RayActorError

        deaths = 0
        while True:
            pool = self.next()
            try:
                stored = await group(pool)
            except RayActorError:
                self._dead.add(self.pools.index(pool))
                deaths += 1
                logger.warning(f"skycap server {pool.urls[0]} died; running its group on another server")
                continue
            if deaths:
                if stored.group_fields.get("rollout_metrics") is None:
                    stored.group_fields["rollout_metrics"] = {}
                stored.group_fields["rollout_metrics"][SERVER_DIED_METRIC] = deaths
            return stored


def stored_group_returns(output: GeneratorOutput, *, support_id_bits: Optional[int] = None) -> Tuple[Any, ...]:
    """A server actor's ``compose_group`` result: the payload-free group, then each trajectory's payload."""
    group, payloads = StoredRolloutGroup.split_output(output, support_id_bits=support_id_bits)
    return (group, *payloads)


async def compose_stored_group(actor: Any, num_trajectories: int, *args: Any) -> StoredRolloutGroup:
    """Call ``actor.compose_group(*args)`` and attach the returned payloads, which stay on the actor's node.

    The caller owns every returned object, so a payload outlives the actor that produced it as long
    as its node does.
    """
    refs = actor.compose_group.options(num_returns=num_trajectories + 1).remote(*args)
    group: StoredRolloutGroup = await refs[0]
    group.attach_payloads(refs[1:])
    return group


class StoredCaptureServer(ABC):
    """One skycap server whose finished captures stay in its process until their group is composed.

    ``finish_trajectory`` finishes a capture here and returns the integration's ``checks`` on its
    samples; ``compose_group`` composes a group from the kept samples and returns its rows as separate
    results (``compose_stored_group``). An integration subclasses it as a Ray actor, threaded
    (``max_concurrency``) so many rollouts finish at once, each waiting on the server's own loop.
    """

    def __init__(self, service: CaptureService) -> None:
        self.service = service
        self.samples = SampleCache()

    @abstractmethod
    def checks(self, samples: List[Sample]) -> Any:
        """The integration's verdict on a finished trajectory's samples, which its generator retries on."""

    @abstractmethod
    def compose(self, outcomes: List[Any], **compose_options: Any) -> GeneratorOutput:
        """A group's rows from its outcomes, each holding its samples."""

    def start(self) -> str:
        return self.service.start()

    def stop(self, timeout: float) -> bool:
        return self.service.stop(timeout)

    def finish_trajectory(
        self, trajectory_id: str, annotations: Dict[str, Any], *, paths: str = "all", timeout: Optional[float] = None
    ) -> CapturedFinish:
        """Finish in this process, keep the samples here, and return the integration's checks on them.

        Raises ``TimeoutError`` if the server's loop hasn't finished it within ``timeout`` seconds.
        """
        result = self.service.finish(trajectory_id, annotations, timeout, paths=paths)
        self.samples.put(trajectory_id, result.samples)
        return CapturedFinish(
            result.id, result.status, result.unbridged_calls, self.checks(result.samples), record=result.record
        )

    def discard(self, trajectory_ids: List[str]) -> None:
        self.samples.discard(trajectory_ids)

    def compose_group(
        self,
        outcomes: List[Any],
        discard: List[str],
        compose_options: Dict[str, Any],
        support_id_bits: Optional[int] = None,
    ) -> Tuple[Any, ...]:
        """Compose a group from the samples cached here; see ``compose_stored_group``.

        ``outcomes`` carry everything but samples; each names the trajectory to take them from in
        ``capture_id``. ``discard`` lists the group's other finished trajectories (retried attempts).
        ``support_id_bits`` packs the payloads' sample-support member ids (``split_output``).
        """
        self.samples.discard(discard)
        outcomes = [
            replace(o, samples=self.samples.take(o.capture_id)) if o.capture_id is not None else o for o in outcomes
        ]
        return stored_group_returns(self.compose(outcomes, **compose_options), support_id_bits=support_id_bits)


@dataclass
class SkycapServers:
    """A running pool of server actors, one per placement-group bundle. ``urls`` is what a generator is given."""

    actors: List[Any]
    urls: List[str]
    placement_group: Any
    stop_timeout: float = 600.0
    _stopped: bool = field(default=False, repr=False)

    def actor_for(self, url: str) -> Any:
        return self.actors[self.urls.index(url)]

    def stop(self) -> None:
        """Stop every server, writing the trajectories still in memory, then kill the actors. Idempotent."""
        if self._stopped:
            return
        self._stopped = True
        try:
            flushed = ray.get(
                [actor.stop.remote(self.stop_timeout) for actor in self.actors], timeout=self.stop_timeout + 10
            )
            for url, done in zip(self.urls, flushed):
                if not done:
                    logger.error(f"skycap at {url} did not finish writing its trajectories within {self.stop_timeout}s")
        finally:
            for actor in self.actors:
                ray.kill(actor, no_restart=True)
            remove_placement_group(self.placement_group)


def start_server_pool(
    actor_cls: Any,
    actor_args: Callable[[int], Tuple[Any, ...]],
    *,
    num_servers: int,
    num_cpus_per_server: float,
    placement_strategy: str,
    max_concurrent_finishes: int,
    timeout: Optional[float] = None,
) -> SkycapServers:
    """``num_servers`` started actors of ``actor_cls`` (a ``StoredCaptureServer`` Ray actor class), server
    ``i`` built from ``actor_args(i)``, each on its own placement-group bundle of ``num_cpus_per_server``.

    Raises, killing whatever it started, if a server fails to start or isn't up within ``timeout`` seconds.
    """
    if num_servers < 1:
        raise ValueError("skycap.num_servers must be at least 1")
    pg = placement_group([{"CPU": num_cpus_per_server}] * num_servers, strategy=placement_strategy)
    actors: List[Any] = []
    try:
        ray.get(pg.ready(), timeout=timeout)
        for index in range(num_servers):
            actors.append(
                actor_cls.options(
                    num_cpus=num_cpus_per_server,
                    max_concurrency=max_concurrent_finishes,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg, placement_group_bundle_index=index
                    ),
                ).remote(*actor_args(index))
            )
        urls = ray.get([actor.start.remote() for actor in actors], timeout=timeout)
    except BaseException:
        # Nothing was handed out yet, so nothing is lost. Killing a server's actor also takes its
        # exposure's tunnel, which skycap ties to the server's process.
        for actor in actors:
            ray.kill(actor, no_restart=True)
        remove_placement_group(pg)
        raise
    return SkycapServers(actors=actors, urls=urls, placement_group=pg)
