"""
Server Group - manages server actors with placement groups.
"""

import logging
from argparse import Namespace
from typing import Any, List, Optional, Type, Union

import ray
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from skyrl.backends.skyrl_train.inference_servers.common import (
    ServerInfo,
)
from skyrl.backends.skyrl_train.inference_servers.protocols import ServerActorProtocol
from skyrl.backends.skyrl_train.inference_servers.server_pool import ServerActorPool
from skyrl.train.utils.utils import ResolvedPlacementGroup

logger = logging.getLogger(__name__)

# In the colocated training case, we schedule trainig and inference actors
# in the same placement group. In SkyRL, we further schedule actors to get information
# about the GPU ID to pack actors appropriately on different nodes.
# Thus we use a fractional CPU allocation for colocated actors.
COLOCATED_ACTOR_CPU_FRACTION = 0.2


class ServerGroup:
    """
    Creates and manages a group of server actors.

    This layer handles actor creation with placement group support,
    then delegates pool management to ServerActorPool.

    Supports:
    - Basic mode: Creates its own placement group
    - Colocation mode: Uses external placement group (shared with training)
    - Data Parallel: Multiple DP-enabled servers
    - PD Disaggregation: Prefill-decode disaggregation with NIXL
    """

    def __init__(
        self,
        cli_args: Namespace,
        num_servers: int,
        start_port: int = 8000,
        placement_group: Optional[ResolvedPlacementGroup] = None,
        placement_group_bundle_offset: int = 0,
        enable_dp: bool = False,
        enable_pd: bool = False,
        nixl_side_channel_base: int = 5600,
        server_actor_cls: Optional[Type[ServerActorProtocol]] = None,
        **server_actor_kwargs: Any,
    ):
        """
        Initialize the server group.

        Args:
            cli_args: CLI arguments for the server (engine-specific).
            num_servers: Number of server instances to create.
            start_port: Base port for server ports.
            placement_group: External placement group for colocation mode.
                If None, creates an internal placement group.
            placement_group_bundle_offset: Offset for bundle indices when using
                external placement group (e.g., if training uses first N
                bundles).
            enable_dp: Enable data parallelism across servers.
            enable_pd: Enable prefill-decode disaggregation.
            nixl_side_channel_base: Base port for NIXL side channels. Each
                server will be assigned a port of nixl_side_channel_base +
                server_idx.
            server_actor_cls: Server actor class implementing
                ServerActorProtocol. Defaults to VLLMServerActor.
            **server_actor_kwargs: Additional keyword arguments to pass to the server actor class.
        """
        from skyrl.backends.skyrl_train.inference_servers.vllm_server_actor import (
            VLLMServerActor,
        )

        self._server_actor_cls = server_actor_cls or VLLMServerActor
        self._cli_args = cli_args
        self._num_servers = num_servers
        self._start_port = start_port
        self._bundle_offset = placement_group_bundle_offset
        self._enable_dp = enable_dp
        self._enable_pd = enable_pd
        self._nixl_side_channel_base = nixl_side_channel_base
        self._pool: Optional[ServerActorPool] = None
        self._internal_pg: Optional[PlacementGroup] = None
        self._server_actor_kwargs = server_actor_kwargs
        self._external_pg = placement_group

        # Extract the raw PG, reordered indices, and GPU IDs from ResolvedPlacementGroup.
        if placement_group is not None:
            self._external_pg = placement_group.pg
            self._reordered_bundle_indices = placement_group.reordered_bundle_indices
            self._bundle_gpu_ids = placement_group.bundle_gpu_ids
        else:
            self._external_pg = None
            self._reordered_bundle_indices = None
            self._bundle_gpu_ids = None

        # Query the actor class for GPU requirements
        self._num_gpus_per_server = self._server_actor_cls.compute_num_gpus_per_server(cli_args)

        logger.info(
            f"ServerGroup: actor_cls={self._server_actor_cls.__name__}, "
            f"num_servers={num_servers}, "
            f"gpus_per_server={self._num_gpus_per_server}, "
            f"enable_dp={enable_dp}, enable_pd={enable_pd}, "
            f"external_pg={'yes' if self._external_pg else 'no'}"
        )

    def _create_placement_group(self) -> PlacementGroup:
        """Create an internal placement group with per-GPU bundles."""
        total_bundles = self._num_servers * self._num_gpus_per_server
        logger.info(f"Creating placement group with {total_bundles} bundles...")
        pg = placement_group([{"CPU": 1, "GPU": 1} for _ in range(total_bundles)])
        ray.get(pg.ready())
        skyrl_pg = ResolvedPlacementGroup(pg)
        self._reordered_bundle_indices = skyrl_pg.reordered_bundle_indices
        self._bundle_gpu_ids = skyrl_pg.bundle_gpu_ids
        logger.info("Placement group ready")
        return pg

    def _get_placement_group(self) -> PlacementGroup:
        """Get the placement group (external or internal)."""
        if self._external_pg is not None:
            return self._external_pg
        if self._internal_pg is None:
            self._internal_pg = self._create_placement_group()
        return self._internal_pg

    def _create_actor_class(self, pg: PlacementGroup, start_bundle_idx: int) -> Any:
        """Create actor class with scheduling constraints for a specific bundle."""
        return ray.remote(self._server_actor_cls).options(
            num_gpus=0,  # GPU allocation managed by placement group
            num_cpus=COLOCATED_ACTOR_CPU_FRACTION,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=start_bundle_idx,
            ),
        )

    def _get_bundle_indices_for_server(self, server_idx: int) -> List[int]:
        """Get the bundle indices for a server, using reordered indices if available."""
        gpus = self._num_gpus_per_server
        logical_base = self._bundle_offset + server_idx * gpus
        if self._reordered_bundle_indices is not None:
            return [self._reordered_bundle_indices[logical_base + k] for k in range(gpus)]
        return list(range(logical_base, logical_base + gpus))

    def _get_gpu_ids_for_server(self, server_idx: int) -> Optional[List[int]]:
        """Get the physical GPU IDs for a server, using cached gpu_ids if available."""
        if self._bundle_gpu_ids is None:
            return None
        gpus = self._num_gpus_per_server
        logical_base = self._bundle_offset + server_idx * gpus
        return [self._bundle_gpu_ids[logical_base + k] for k in range(gpus)]

    def _create_actors(self) -> List[Any]:
        """Create server actors with GPU resources."""
        pg = self._get_placement_group()

        actors = []
        dp_address, dp_rpc_port = None, None

        for server_idx in range(self._num_servers):
            bundle_indices = self._get_bundle_indices_for_server(server_idx)
            start_bundle_idx = bundle_indices[0]

            ServerActorClass = self._create_actor_class(pg, start_bundle_idx)

            gpu_ids = self._get_gpu_ids_for_server(server_idx)
            server_kwargs = self._server_actor_cls.prepare_server_kwargs(
                pg,
                start_bundle_idx,
                self._num_gpus_per_server,
                _gpu_ids=gpu_ids,
                **self._server_actor_kwargs,
            )

            actor = ServerActorClass.remote(
                self._cli_args,
                self._start_port + server_idx,
                server_idx=server_idx,
                bundle_indices=bundle_indices,
                dp_size=self._num_servers if self._enable_dp else -1,
                dp_master_address=dp_address,
                dp_rpc_port=dp_rpc_port,
                enable_pd=self._enable_pd,
                nixl_side_channel_base=self._nixl_side_channel_base,
                colocated_training=self._external_pg is not None,
                **server_kwargs,
            )

            # Get DP info from server 0 which is where DP0 will be
            if self._enable_dp and server_idx == 0:
                dp_address, dp_rpc_port = ray.get(actor.get_dp_info.remote())
                logger.info(f"DP0 info: address={dp_address}, rpc_port={dp_rpc_port}")

            actors.append(actor)

        return actors

    def start(self, blocking: bool = True) -> Union[List[ServerInfo], List[ray.ObjectRef]]:
        """Create actors, start the pool, and return endpoints.

        Args:
            blocking: If True (default), waits for all servers to be ready
                and returns ``List[ServerInfo]``.  If False, creates actors
                and fires off start RPCs but returns the
                ``List[ObjectRef]`` without waiting.
        """
        logger.info(f"Starting {self._num_servers} server(s)...")
        actors = self._create_actors()
        self._pool = ServerActorPool(actors)

        if blocking:
            server_infos = self._pool.start(blocking=True)
            for i, info in enumerate(server_infos):
                logger.info(f"Server {i}: {info.url}")
            return server_infos

        return self._pool.start(blocking=False)

    @property
    def server_infos(self) -> List[ServerInfo]:
        """Lazily resolved server infos (delegates to pool)."""
        if self._pool is None:
            return []
        return self._pool.server_infos

    def get_pool(self) -> Optional[ServerActorPool]:
        """Get the underlying actor pool."""
        return self._pool

    def get_server_urls(self) -> List[str]:
        """Get the list of server URLs."""
        return [info.url for info in self.server_infos]

    def get_actors(self) -> List[Any]:
        """Get the list of actor handles."""
        return self._pool.get_actors() if self._pool else []

    def shutdown(self) -> None:
        """Shutdown all servers."""
        if self._pool:
            logger.info("Shutting down servers...")
            self._pool.shutdown()

        if self._internal_pg:
            # created pg internally, teardown
            ray.util.remove_placement_group(self._internal_pg)
