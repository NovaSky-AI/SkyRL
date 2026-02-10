"""
Protocols for inference server components.

These define the interfaces that server implementations must follow.
"""

from argparse import Namespace
from typing import Optional, Protocol, Tuple, runtime_checkable

from skyrl_train.inference_servers.common import ServerInfo


@runtime_checkable
class ServerActorProtocol(Protocol):
    """
    Protocol defining the interface for server actor classes.

    Any server actor class (vLLM, SGLang, etc.) must implement this interface
    to be usable with ServerGroup.

    Example:
        class MyServerActor(ServerActorProtocol):
            @staticmethod
            def compute_num_gpus_per_server(cli_args: Namespace) -> int:
                return cli_args.tensor_parallel_size

            def __init__(self, cli_args, start_port, server_idx, ...):
                ...

            async def start(self) -> ServerInfo:
                ...
    """

    @staticmethod
    def compute_num_gpus_per_server(cli_args: Namespace) -> int:
        """
        Compute the number of GPUs needed per server instance.

        This is called before actor creation to determine placement group size.

        Args:
            cli_args: Engine-specific CLI arguments.

        Returns:
            Number of GPUs required per server (e.g., TP * PP for vLLM).
        """
        ...

    def __init__(
        self,
        cli_args: Namespace,
        start_port: int,
        server_idx: int,
        start_bundle_idx: int,
        dp_size: int,
        dp_master_address: Optional[str],
        dp_rpc_port: Optional[int],
        enable_pd: bool,
        nixl_side_channel_base: int,
        colocated_training: bool,
    ) -> None:
        """
        Initialize the server actor.

        Args:
            cli_args: Engine-specific CLI arguments.
            start_port: Base port to search for available port.
            server_idx: Index of this server in the group (0-indexed).
            start_bundle_idx: Starting bundle index in placement group for this server's workers.
            dp_size: Data parallel size (-1 to disable DP).
            dp_master_address: DP master address (for non-rank-0 servers).
            dp_rpc_port: DP RPC port (for non-rank-0 servers).
            enable_pd: Enable prefill-decode disaggregation.
            nixl_side_channel_base: Base port for NIXL side channels.
            colocated_training: Whether the server is colocated with training workers.
        """
        ...

    def get_server_info(self) -> ServerInfo:
        """Get the server's IP and port info."""
        ...

    def get_dp_info(self) -> Tuple[str, int]:
        """
        Get the DP master address and RPC port.

        Only called on server_idx=0 when DP is enabled.

        Returns:
            Tuple of (master_address, rpc_port).
        """
        ...

    async def start(self) -> ServerInfo:
        """
        Start the server.

        This should block until the server is healthy and ready to serve requests.

        Returns:
            ServerInfo with the server's IP and port.
        """
        ...

    async def shutdown(self) -> None:
        """Gracefully shutdown the server."""
        ...
