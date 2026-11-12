"""Process resource limits for SkyRL's worker processes."""

import resource


def raise_open_file_limit() -> int:
    """Raise this process's soft open-file limit to its hard limit; returns the new soft limit.

    Ray starts worker processes with the container's default soft limit (often 1024). NCCL over
    InfiniBand opens sockets and verbs objects per peer, channel and HCA, and a multi-node job with
    several communicators per rank and 8 HCAs exceeds 1024 ("ncclOsSocketTryAccept: Accept failed:
    Too many open files"). Child processes inherit the raised limit. Idempotent.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard == resource.RLIM_INFINITY or soft < hard:
        target = hard if hard != resource.RLIM_INFINITY else max(soft, 1 << 20)
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            soft = target
        except (ValueError, OSError):
            pass
    return soft
