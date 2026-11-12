import resource

from skyrl.utils.rlimits import raise_open_file_limit


def test_raise_open_file_limit_raises_soft_to_hard():
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    lowered = min(1024, hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (lowered, hard))
    try:
        new_soft = raise_open_file_limit()
        assert resource.getrlimit(resource.RLIMIT_NOFILE)[0] == new_soft
        assert new_soft == hard or (hard == resource.RLIM_INFINITY and new_soft >= lowered)
        assert raise_open_file_limit() == new_soft  # idempotent
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
