"""Cache invalidation policy shared by training backends during weight sync."""


def should_reset_kv_cache(
    *,
    enable_prefix_caching: bool,
    fully_async: bool,
    clear_kv_cache_on_weight_sync: bool,
) -> bool:
    """Whether weight sync must invalidate cached KV, including running requests.

    Synchronous training only needs to invalidate reusable prefix blocks.
    Async training can keep requests in flight, whose KV exists even when
    prefix caching is disabled.
    """
    return clear_kv_cache_on_weight_sync if fully_async else enable_prefix_caching
