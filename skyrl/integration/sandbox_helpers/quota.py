"""A cap on sandboxes in flight, required wherever they run in a provider's account.

A provider's sandboxes (Daytona, Modal, ...) count against an account quota others share, and a generator
starts one per rollout: every rollout of a batch (eval: the whole eval set) would start one at once. So a
generator that starts remote sandboxes refuses to run without ``generator.rate_limit.max_concurrency``.
"""

from collections.abc import Mapping
from typing import Any, Optional


def max_concurrency(rate_limit: Any) -> Optional[int]:
    """``generator.rate_limit``'s cap on rollouts in flight, or None when it sets none.

    ``rate_limit`` is a ``RateLimiterConfig``, a mapping of its fields, or None.
    """
    if rate_limit is None:
        return None
    if isinstance(rate_limit, Mapping):
        enabled, cap = rate_limit.get("enabled", False), rate_limit.get("max_concurrency")
    else:
        enabled, cap = getattr(rate_limit, "enabled", False), getattr(rate_limit, "max_concurrency", None)
    return cap if enabled else None


def require_concurrency_cap(rate_limit: Any, sandboxes: str) -> int:
    """The cap ``max_concurrency(rate_limit)``; raises ``ValueError`` when there is none.

    ``sandboxes`` names what starts the remote sandboxes, for the error, e.g.
    ``"harbor_trial_config.environment (daytona)"``.
    """
    cap = max_concurrency(rate_limit)
    if not cap:
        raise ValueError(
            f"{sandboxes} runs sandboxes remotely: cap how many run at once with "
            "generator.rate_limit.enabled=true and generator.rate_limit.max_concurrency=<n>"
        )
    return cap
