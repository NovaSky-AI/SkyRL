"""Worker process setup hook for Ray workers."""

import multiprocessing


def worker_setup_fn():
    """Set the multiprocessing start method to 'spawn' in Ray workers.

    This is passed to ray.init via runtime_env["worker_process_setup_hook"].
    We use ray and thus disable the `fork` start method. Forking within
    ray leads to undefined behaviour and often causes hard to debug memory leaks.
    See: https://docs.ray.io/en/latest/ray-core/patterns/fork-new-processes.html
    A common culprit is PyTorch dataloaders which use `fork` by default.
    """
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set — nothing to do
        pass
