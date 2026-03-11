"""
Server Group - manages server actors with placement groups.
"""

import logging
from argparse import Namespace
from typing import Any, List, Optional, Type

import ray
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from skyrl.backends.skyrl_train.inference_servers.common import ServerInfo
from skyrl.backends.skyrl_train.inference_servers.protocols import ServerActorProtocol
from skyrl.backends.skyrl_train.inference_servers.server_pool import ServerActorPool

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
    - mp backend: CUDA_VISIBLE_DEVICES targeting within a shared PG
    """

    def __init__(
        self,
        cli_args: Namespace,
        num_servers: int,
        start_port: int = 8000,
        placement_group: Optional[PlacementGroup] = None,
        placement_group_bundle_offset: int = 0,
        enable_dp: bool = False,
        enable_pd: bool = False,
        nixl_side_channel_base: int = 5600,
        server_actor_cls: Optional[Type[ServerActorProtocol]] = None,
        distributed_executor_backend: str = "ray",
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
            distributed_executor_backend: vLLM distributed executor backend.
                ``"ray"`` (default) spawns TP/PP workers as Ray tasks.
                ``"mp"`` spawns workers as local processes and sets
                CUDA_VISIBLE_DEVICES per server.
        """
        from skyrl.backends.skyrl_train.inference_servers.vllm_server_actor import VLLMServerActor

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
        self._distributed_executor_backend = distributed_executor_backend
        self._use_mp_backend = distributed_executor_backend == "mp"
        self._external_pg = placement_group

        # Query the actor class for GPU requirements
        self._num_gpus_per_server = self._server_actor_cls.compute_num_gpus_per_server(cli_args)

        logger.info(
            f"ServerGroup: actor_cls={self._server_actor_cls.__name__}, "
            f"num_servers={num_servers}, "
            f"gpus_per_server={self._num_gpus_per_server}, "
            f"enable_dp={enable_dp}, enable_pd={enable_pd}, "
            f"distributed_executor_backend={distributed_executor_backend}, "
            f"external_pg={'yes' if self._external_pg else 'no'}"
        )

    def _create_placement_group(self) -> PlacementGroup:
        """Create an internal placement group with per-GPU bundles."""
        total_bundles = self._num_servers * self._num_gpus_per_server
        logger.info(f"Creating placement group with {total_bundles} bundles...")
        pg = placement_group([{"CPU": 1, "GPU": 1} for _ in range(total_bundles)])
        ray.get(pg.ready())
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
        capture_child_tasks = not self._use_mp_backend
        return ray.remote(self._server_actor_cls).options(
            num_gpus=0,  # GPU allocation managed by placement group
            num_cpus=COLOCATED_ACTOR_CPU_FRACTION,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=capture_child_tasks,
                placement_group_bundle_index=start_bundle_idx,
            ),
        )

    def _get_cuda_visible_devices(self, pg: PlacementGroup, bundle_start: int, bundle_count: int) -> Optional[str]:
        """Pre-compute CUDA_VISIBLE_DEVICES for a slice of bundles in a PG.

        Args:
            pg: Placement group to probe.
            bundle_start: First bundle index (0-based within the PG).
            bundle_count: Number of contiguous bundles for this server.

        Returns a comma-separated string of physical GPU IDs.
        """
        from skyrl.train.utils.utils import get_gpu_ids_for_pg_bundles

        bundle_indices = list(range(bundle_start, bundle_start + bundle_count))
        gpu_ids = get_gpu_ids_for_pg_bundles(pg, bundle_indices)
        return ",".join(str(g) for g in gpu_ids)

    def _create_actors(self) -> List[Any]:
        """Create server actors with GPU resources."""
        pg = self._get_placement_group()

        actors = []
        dp_address, dp_rpc_port = None, None

        # Pre-compute CUDA_VISIBLE_DEVICES per server for mp backend
        mp_cuda_visible_devices_map: dict[int, Optional[str]] = {}
        if self._use_mp_backend:
            for server_idx in range(self._num_servers):
                start = self._bundle_offset + server_idx * self._num_gpus_per_server
                mp_cuda_visible_devices_map[server_idx] = self._get_cuda_visible_devices(
                    pg, start, self._num_gpus_per_server
                )

        for server_idx in range(self._num_servers):
            start_bundle_idx = self._bundle_offset + server_idx * self._num_gpus_per_server

            ServerActorClass = self._create_actor_class(pg, start_bundle_idx)

            actor = ServerActorClass.remote(
                self._cli_args,
                self._start_port + server_idx,
                server_idx=server_idx,
                start_bundle_idx=start_bundle_idx,
                dp_size=self._num_servers if self._enable_dp else -1,
                dp_master_address=dp_address,
                dp_rpc_port=dp_rpc_port,
                enable_pd=self._enable_pd,
                nixl_side_channel_base=self._nixl_side_channel_base,
                colocated_training=self._external_pg is not None,
                distributed_executor_backend=self._distributed_executor_backend,
                mp_cuda_visible_devices=mp_cuda_visible_devices_map.get(server_idx),
            )

            # Get DP info from server 0 which is where DP0 will be
            if self._enable_dp and server_idx == 0:
                dp_address, dp_rpc_port = ray.get(actor.get_dp_info.remote())
                logger.info(f"DP0 info: address={dp_address}, rpc_port={dp_rpc_port}")

            actors.append(actor)

        return actors

    def start(self) -> List[ServerInfo]:
        """Create actors, start the pool, and return endpoints."""
        logger.info(f"Starting {self._num_servers} server(s)...")
        actors = self._create_actors()
        self._pool = ServerActorPool(actors)
        server_infos = self._pool.start()

        for i, info in enumerate(server_infos):
            logger.info(f"Server {i}: {info.url}")

        return server_infos

    def get_pool(self) -> Optional[ServerActorPool]:
        """Get the underlying actor pool."""
        return self._pool

    def get_server_infos(self) -> List[ServerInfo]:
        """Get the list of server endpoints."""
        return self._pool.get_server_infos() if self._pool else []

    def get_server_urls(self) -> List[str]:
        """Get the list of server URLs."""
        return self._pool.get_server_urls() if self._pool else []

    def get_actors(self) -> List[Any]:
        """Get the list of actor handles."""
        return self._pool.get_actors() if self._pool else []

    def shutdown(self) -> None:
        """Shutdown all servers."""
        if self._pool:
            logger.info("Shutting down servers...")
            self._pool.shutdown()
