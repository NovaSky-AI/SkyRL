import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from skyrl.backends.skyrl_train.inference_engines.remote_inference_engine import RemoteWeightLoader
from skyrl.backends.skyrl_train.weight_sync import BroadcastInitInfo, BroadcastWeightUpdateRequest


class AsyncContextManagerMock:
    """Helper to mock async context managers (for `async with ... as ...`)."""

    def __init__(self, return_value):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, *args):
        pass


def create_mock_session(mock_response):
    """Create a mock aiohttp.ClientSession with proper async behavior.

    Handles both:
    - `async with session.post(...) as resp:` (context manager form)
    - `resp = await session.post(...)` (direct await form)
    """
    mock_session = MagicMock()

    # Create a mock that works both as context manager and as awaitable
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


class TestRemoteWeightLoader:
    """Tests for RemoteWeightLoader class."""

    @staticmethod
    def make_broadcast_init_info():
        """Create a BroadcastInitInfo for testing."""
        return BroadcastInitInfo(
            master_addr="127.0.0.1",
            master_port=29500,
            rank_offset=1,
            world_size=2,
            group_name="test_group",
            backend="nccl",
            model_dtype_str="torch.bfloat16",
            override_existing_receiver=True,
        )

    def test_init(self):
        """Test initialization stores URL and backend correctly."""
        loader = RemoteWeightLoader(url="http://localhost:8000", engine_backend="vllm")

        assert loader._url == "http://localhost:8000"
        assert loader._engine_backend == "vllm"

    @pytest.mark.asyncio
    async def test_init_communicator(self):
        """Test init_communicator calls correct endpoint for vllm backend."""
        url = "http://localhost:8000"
        loader = RemoteWeightLoader(url=url, engine_backend="vllm")
        init_info = self.make_broadcast_init_info()

        mock_response = MagicMock()
        mock_response.json = AsyncMock(return_value={"success": True})

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = create_mock_session(mock_response)
            mock_session_class.return_value = AsyncContextManagerMock(mock_session)

            result = await loader.init_communicator(init_info)

            # Verify correct endpoint called
            mock_session.post.assert_called_once()
            call_args = mock_session.post.call_args
            assert call_args[0][0] == f"{url}/init_weight_update_communicator"

            # vLLM uses asdict(init_info) - all fields with original names
            from dataclasses import asdict

            payload = call_args[1]["json"]
            assert payload == asdict(init_info)

            assert result == {"success": True}

    @pytest.mark.asyncio
    async def test_load_weights(self):
        """Test load_weights calls correct endpoint for vllm backend."""
        url = "http://localhost:8000"
        loader = RemoteWeightLoader(url=url, engine_backend="vllm")

        mock_response = MagicMock()
        mock_response.json = AsyncMock(return_value={"success": True})

        request = BroadcastWeightUpdateRequest(
            names=["model.layer.weight"],
            dtypes=["bfloat16"],
            shapes=[[4096, 4096]],
        )

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = create_mock_session(mock_response)
            mock_session_class.return_value = AsyncContextManagerMock(mock_session)

            result = await loader.load_weights(request)

            # Verify correct endpoint called
            call_args = mock_session.post.call_args
            assert call_args[0][0] == f"{url}/update_weights"

            # vLLM uses asdict(request) - plural field names
            from dataclasses import asdict

            payload = call_args[1]["json"]
            assert payload == asdict(request)

            assert result == {"success": True}

    @pytest.mark.asyncio
    async def test_load_weights_invalid_backend(self):
        """Test load_weights raises ValueError for unknown backend."""
        loader = RemoteWeightLoader(url="http://localhost:8000", engine_backend="unknown")

        request = BroadcastWeightUpdateRequest(
            names=["model.layer.weight"],
            dtypes=["bfloat16"],
            shapes=[[4096, 4096]],
        )

        with pytest.raises(ValueError, match="Invalid engine backend"):
            await loader.load_weights(request)

    @pytest.mark.asyncio
    async def test_destroy_group(self):
        """Test destroy_group calls correct endpoint."""
        loader = RemoteWeightLoader(url="http://localhost:8000", engine_backend="vllm")

        mock_response = MagicMock()
        mock_response.json = AsyncMock(return_value={"success": True})

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = create_mock_session(mock_response)
            mock_session_class.return_value = AsyncContextManagerMock(mock_session)

            result = await loader.destroy_group()

            # Verify correct endpoint called
            call_args = mock_session.post.call_args
            assert call_args[0][0] == "http://localhost:8000/destroy_weights_update_group"

            assert result == {"success": True}
