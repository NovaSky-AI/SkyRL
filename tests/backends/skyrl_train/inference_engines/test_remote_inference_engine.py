import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from skyrl.backends.skyrl_train.inference_engines.remote_inference_engine import (
    RemoteInferenceEngine,
)


class AsyncContextManagerMock:
    """Helper to mock async context managers (for `async with ... as ...`)."""

    def __init__(self, return_value):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, *args):
        pass


def create_mock_session(mock_response):
    """Create a mock aiohttp.ClientSession with proper async behavior."""
    mock_session = MagicMock()

    class MockPostReturn:
        def __init__(self, response):
            self.response = response

        async def __aenter__(self):
            return self.response

        async def __aexit__(self, *args):
            pass

        def __await__(self):
            async def _await():
                return self.response

            return _await().__await__()

    mock_session.post = MagicMock(return_value=MockPostReturn(mock_response))
    return mock_session


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reset_running_requests", "expected_param"),
    [(False, "false"), (True, "true")],
)
async def test_reset_prefix_cache_forwards_reset_running_requests(reset_running_requests, expected_param):
    engine = RemoteInferenceEngine(
        url="localhost:8000",
        model_name="test-model",
        engine_backend="vllm",
        tokenizer=MagicMock(),
    )

    mock_response = MagicMock()
    mock_response.text = AsyncMock(
        return_value=json.dumps(
            {
                "status": "cache_reset",
                "reset_running_requests": reset_running_requests,
            }
        )
    )

    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session = create_mock_session(mock_response)
        mock_session_class.return_value = AsyncContextManagerMock(mock_session)

        result = await engine.reset_prefix_cache(reset_running_requests=reset_running_requests)

        mock_session.post.assert_called_once_with(
            "http://localhost:8000/reset_prefix_cache",
            params={"reset_running_requests": expected_param},
        )
        assert result == {
            "status": "cache_reset",
            "reset_running_requests": reset_running_requests,
        }
