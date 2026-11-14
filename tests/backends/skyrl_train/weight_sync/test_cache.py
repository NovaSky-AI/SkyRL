from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from skyrl.backends.skyrl_train.weight_sync.cache import should_reset_kv_cache


@pytest.mark.parametrize(
    "fully_async,enable_prefix_caching,clear_kv_cache_on_weight_sync,expected",
    [
        (False, False, False, False),
        (False, False, True, False),
        (False, True, False, True),
        (False, True, True, True),
        (True, False, False, False),
        (True, False, True, True),
        (True, True, False, False),
        (True, True, True, True),
    ],
)
def test_weight_sync_cache_policy(fully_async, enable_prefix_caching, clear_kv_cache_on_weight_sync, expected):
    assert (
        should_reset_kv_cache(
            enable_prefix_caching=enable_prefix_caching,
            fully_async=fully_async,
            clear_kv_cache_on_weight_sync=clear_kv_cache_on_weight_sync,
        )
        is expected
    )


@pytest.mark.vllm
@pytest.mark.parametrize("enable_prefix_caching", [False, True])
@pytest.mark.parametrize("clear_kv_cache", [False, True])
def test_sender_owned_async_reset(enable_prefix_caching, clear_kv_cache):
    pytest.importorskip("vllm", reason="delta reset ownership requires the vLLM trainer engine")
    from skyrl.backends.skyrl_train.weight_sync.delta.trainer import (
        DeltaTrainerWeightTransferEngine,
    )

    client = MagicMock()
    sender = DeltaTrainerWeightTransferEngine(client=client, source=[], init_info=SimpleNamespace(sync_dir="/unused"))
    reset = should_reset_kv_cache(
        enable_prefix_caching=enable_prefix_caching,
        fully_async=True,
        clear_kv_cache_on_weight_sync=clear_kv_cache,
    )
    sender.skyrl_set_reset_prefix_cache(reset)
    sender._apply_receiver_update({"target_version": 1})

    if clear_kv_cache:
        client.reset_prefix_cache.assert_called_once_with(reset_running_requests=True)
        calls = [call[0] for call in client.mock_calls]
        assert calls.index("pause_generation") < calls.index("reset_prefix_cache") < calls.index("start_weight_update")
    else:
        client.reset_prefix_cache.assert_not_called()
    client.finish_weight_update.assert_called_once()
    client.resume_generation.assert_called_once()
