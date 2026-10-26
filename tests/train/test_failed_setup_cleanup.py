"""Ray resources acquired by a setup step that fails are released before the error propagates."""

import pytest
from ray.util.placement_group import placement_group, placement_group_table

from skyrl.train.utils.utils import cleanup_on_error, get_ray_pg_ready_with_timeout


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_cleanup_on_error_reraises_the_original_error(cleanup_fails):
    calls = []

    def cleanup():
        calls.append("cleanup")
        if cleanup_fails:
            raise RuntimeError("cleanup failed")

    with pytest.raises(ValueError, match="build failed"):
        with cleanup_on_error(cleanup, "Cleanup"):
            raise ValueError("build failed")

    assert calls == ["cleanup"]


def test_pg_ready_timeout_removes_the_placement_group():
    # No node has this many CPUs, so the placement group can never become ready.
    pg = placement_group([{"CPU": 10_000}])

    with pytest.raises(RuntimeError, match="Failed to create placement group"):
        get_ray_pg_ready_with_timeout(pg, timeout=1)

    assert placement_group_table(pg)["state"] == "REMOVED"
