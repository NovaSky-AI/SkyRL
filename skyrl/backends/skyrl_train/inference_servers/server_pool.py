"""
Generic server actor pool.
"""

from typing import Any, List, Union

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

    def __init__(self, actors: List[Any]):
        """
        Initialize the pool with pre-constructed actor handles.

        Args:
            actors: List of Ray actor handles
        """
        self._actors = actors
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
            self._server_infos = ray.get(self._start_refs)
            return self._server_infos
        return self._start_refs

    @property
    def server_infos(self) -> List[ServerInfo]:
        """Lazily resolved server infos.

        On first access (when ``_server_infos`` is empty), calls
        ``ray.get`` on the stored start refs to block until all actors
        are ready.
        """
        if not self._server_infos and self._start_refs:
            self._server_infos = ray.get(self._start_refs)
        return self._server_infos

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
