"""Worker process setup hook for Ray workers.

This module provides a setup function that configures the multiprocessing
start method to 'spawn' for all Ray worker processes.
"""

import multiprocessing


def worker_setup_fn():
    """Set the multiprocessing start method to 'spawn' in Ray workers.

    This is passed to ray.init via runtime_env["worker_process_setup_hook"].
    Using 'spawn' avoids issues with forked processes inheriting state
    (e.g., sockets)  that can cause hangs or crashes.
    """
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set — nothing to do
        pass
