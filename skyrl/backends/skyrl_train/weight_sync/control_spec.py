"""Serializable provider contract for trainer-side weight synchronization.

Inference clients own serving identities and validation. Trainer workers consume
this structural contract without importing a serving provider or its protocol.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Sequence

InitPayloadFn = Callable[[dict[str, Any], Sequence[str], int], list[dict[str, Any]]]


class WeightSyncControlClient(Protocol):
    """Blocking lifecycle used by vLLM trainer engines and SkyRL teardown."""

    fail_closed_transfers: bool

    def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None: ...

    def start_weight_update(self) -> None: ...

    def update_weights(self, update_info: dict[str, Any]) -> None: ...

    def finish_weight_update(self, weight_version: Optional[str] = None) -> None: ...

    def close(self) -> None: ...


class WeightSyncControlSpec(Protocol):
    """Picklable provider specification passed to every training rank."""

    fail_closed_transfers: bool

    def build_client(self, init_payload_fn: Optional[InitPayloadFn]) -> WeightSyncControlClient: ...


@dataclass(frozen=True)
class DirectWeightSyncControlSpec:
    """Default direct-vLLM transport, preserving existing placement modes."""

    server_urls: tuple[str, ...]
    data_parallel_size: int
    fail_closed_transfers: bool = False

    def build_client(self, init_payload_fn: Optional[InitPayloadFn]) -> WeightSyncControlClient:
        from skyrl.backends.skyrl_train.weight_sync.control_plane import (
            SkyrlWeightSyncClient,
        )

        return SkyrlWeightSyncClient(self.server_urls, self.data_parallel_size, init_payload_fn=init_payload_fn)
