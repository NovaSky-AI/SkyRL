"""
Generic server actor pool.
"""

from typing import Any, List

import ray

from skyrl_train.inference_servers.common import ServerInfo


class ServerActorPool:
    """Generic pool that manages a list of server actors.

    This layer provides a generic pool interface which can be extended to
    support fault-tolerance, monitoring, etc. for now it's just a simple wrapper around a list of actor handles.

    Actors must implement:
      - start() -> ServerInfo
      - shutdown() -> None

    This layer is agnostic to the type of server (vLLM, SGLang, etc).
    """

    def __init__(self, actors: List[Any]):
        """
        Initialize the pool with pre-constructed actor handles.

        Args:
            actors: List of Ray actor handles
        """
        self._actors = actors
        self._server_infos: List[ServerInfo] = []

    def start(self) -> List[ServerInfo]:
        """Start all actors and collect their server infos."""
        # Start all actors in parallel, wait for all to be ready
        start_refs = [actor.start.remote() for actor in self._actors]
        self._server_infos = ray.get(start_refs)
        return self._server_infos

    def get_server_infos(self) -> List[ServerInfo]:
        """Get the list of server endpoints."""
        return self._server_infos

    def get_server_urls(self) -> List[str]:
        """Get the list of server URLs."""
        return [info.url for info in self._server_infos]

    def get_actors(self) -> List[Any]:
        """Get the list of actor handles."""
        return self._actors

    def shutdown(self) -> None:
        """Shutdown all actors."""
        shutdown_refs = [actor.shutdown.remote() for actor in self._actors]
        ray.get(shutdown_refs)
