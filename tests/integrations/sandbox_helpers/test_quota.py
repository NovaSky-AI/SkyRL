"""quota.py: remote sandboxes need a cap on how many run at once."""

import pytest

from skyrl.integration.sandbox_helpers.quota import (
    max_concurrency,
    require_concurrency_cap,
)
from skyrl.train.utils.rate_limiter import RateLimiterConfig


@pytest.mark.parametrize(
    "rate_limit",
    [
        None,
        RateLimiterConfig(),
        RateLimiterConfig(max_concurrency=8),  # set but not enabled
        RateLimiterConfig(enabled=True, trajectories_per_second=5),
        {"max_concurrency": 8},
        {"enabled": True},
    ],
)
def test_no_cap(rate_limit) -> None:
    assert max_concurrency(rate_limit) is None
    with pytest.raises(ValueError, match=r"^env \(daytona\) runs sandboxes remotely: .*max_concurrency"):
        require_concurrency_cap(rate_limit, "env (daytona)")


@pytest.mark.parametrize(
    "rate_limit", [RateLimiterConfig(enabled=True, max_concurrency=8), {"enabled": True, "max_concurrency": 8}]
)
def test_cap(rate_limit) -> None:
    assert max_concurrency(rate_limit) == 8
    assert require_concurrency_cap(rate_limit, "env") == 8


def test_importing_the_package_pulls_in_no_harness_or_provider() -> None:
    import subprocess
    import sys

    code = (
        "import sys; import skyrl.integration.sandbox_helpers.daytona, skyrl.integration.sandbox_helpers.quota; "
        "print(sorted(m for m in ('harbor', 'verifiers', 'skycap', 'daytona', 'skyrl.train') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout
    assert out.strip() == "[]"
