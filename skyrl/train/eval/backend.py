"""The backend seam of the asynchronous eval dispatcher."""

import abc

from skyrl.train.eval.types import EvalLease, EvalRequest


class EvalBackend(abc.ABC):
    """Loads weight version V somewhere and hands back a lease on it.

    *Sync* means loading an HF export onto the backend's engines; it never involves the training
    weight-sync group. One method each way: ``sync`` returns a lease whose generator is bound to
    the synced version (or raises ``EvalSkip`` to skip the point with attribution), ``release``
    gives the lease back, ``close`` tears the backend down at the end of training.

    An abstract base class rather than a protocol: a backend is one by declaration, and the
    trainer's ``isinstance`` check at construction is exact.
    """

    @abc.abstractmethod
    async def sync(self, req: EvalRequest) -> EvalLease: ...

    @abc.abstractmethod
    async def release(self, lease: EvalLease) -> None: ...

    @abc.abstractmethod
    async def close(self) -> None: ...
