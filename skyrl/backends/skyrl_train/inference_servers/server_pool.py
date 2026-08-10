"""
Generic server actor pool.
"""

import dataclasses
from typing import Any, List, Optional, Sequence, Union

import ray
from loguru import logger

from skyrl.backends.skyrl_train.inference_servers.common import ServerInfo


class ServerActorPool:
    """Generic pool that manages a list of server actors.

    This layer provides a generic pool interface which can be extended to
    support fault-tolerance, monitoring, etc. for now it's just a simple wrapper around a list of actor handles.

    Actors must implement:
      - start() -> ServerInfo
      - shutdown() -> None

    This layer is agnostic to the type of server.
    """

    def __init__(self, actors: List[Any], slots: Optional[Sequence[int]] = None):
        """
        Initialize the pool with pre-constructed actor handles.

        Args:
            actors: List of Ray actor handles
            slots: Stable deployment ordinal per actor, stamped onto each
                ``ServerInfo`` as it is resolved. Actors do not know their own
                slot -- it is a fleet-level fact, so it is applied here, at the
                layer that was told what the fleet looks like. Defaults to
                ``range(len(actors))``, which is right for a standalone pool;
                ``ServerGroup`` passes its own deployment ordinal so that DP
                peers correctly share one slot.
        """
        self._actors = actors
        self._slots = list(slots) if slots is not None else list(range(len(actors)))
        if len(self._slots) != len(actors):
            raise ValueError(f"Expected one slot per actor, got {len(self._slots)} slots for {len(actors)} actors.")
        self._server_infos: List[ServerInfo] = []
        self._start_refs: List[ray.ObjectRef] = []

    def start(self, blocking: bool = True) -> Union[List[ServerInfo], List[ray.ObjectRef]]:
        """Start all actors and collect their server infos.

        Args:
            blocking: If True (default), waits for all actors to be ready
                and returns ``List[ServerInfo]``.  If False, returns the
                ``List[ObjectRef]`` immediately without waiting.
        """
        self._start_refs = [actor.start.remote() for actor in self._actors]
        if blocking:
            self._server_infos = self._stamp(ray.get(self._start_refs))
            return self._server_infos
        return self._start_refs

    def _stamp(self, infos: Sequence[ServerInfo]) -> List[ServerInfo]:
        """Attach each actor's slot to the ``ServerInfo`` it returned."""
        return [dataclasses.replace(info, slot=slot) for info, slot in zip(infos, self._slots)]

    @property
    def slots(self) -> List[int]:
        """Stable deployment ordinal per actor, parallel to ``get_actors()``."""
        return list(self._slots)

    @property
    def server_infos(self) -> List[ServerInfo]:
        """Lazily resolved server infos.

        On first access (when ``_server_infos`` is empty), calls
        ``ray.get`` on the stored start refs to block until all actors
        are ready.
        """
        if not self._server_infos and self._start_refs:
            self._server_infos = self._stamp(ray.get(self._start_refs))
        return self._server_infos

    def replace_actor(self, idx: int, actor: Any, info: ServerInfo) -> ServerInfo:
        """Swap in a freshly restarted actor at ``idx``, keeping its slot.

        Honours the class docstring's extension promise: the pool is the one place
        that owns the actor-list-to-slot mapping, so a restart updates the handle
        and the (new) URL in place while the slot -- the identity everything else
        keys on -- is preserved by construction rather than by the caller
        remembering to pass it back.

        The caller supplies ``info`` because only it knows the new actor is ready
        (it awaited the new ``start()``); the pool supplies the slot.
        """
        if not 0 <= idx < len(self._actors):
            raise IndexError(f"replace_actor index {idx} out of range for {len(self._actors)} actors.")
        # Resolve every index BEFORE swapping: the list we are about to
        # index-assign into has to exist, and `_start_refs` still holds the dead
        # actor's ref, which must never be `ray.get`-ed again. Resolving now and
        # then dropping the refs leaves `_server_infos` as the only source.
        _ = self.server_infos
        self._start_refs = []
        self._actors[idx] = actor
        stamped = dataclasses.replace(info, slot=self._slots[idx])
        self._server_infos[idx] = stamped
        return stamped

    def get_server_urls(self) -> List[str]:
        """Get the list of server URLs."""
        return [info.url for info in self.server_infos]

    def get_actors(self) -> List[Any]:
        """Get the list of actor handles."""
        return self._actors

    def shutdown(self) -> None:
        """Shutdown all actors and kill them to release GPU memory."""
        shutdown_refs = [actor.shutdown.remote() for actor in self._actors]
        ray.get(shutdown_refs)
        for actor in self._actors:
            try:
                ray.kill(actor)
            except Exception as e:
                logger.info(f"Encountered exception while cleaning up actor {actor}: {e}")
