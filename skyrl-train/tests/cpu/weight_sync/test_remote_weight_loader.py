import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from skyrl_train.inference_engines.remote_inference_engine import RemoteWeightLoader
from skyrl_train.weight_sync import BroadcastInitInfo, BroadcastWeightUpdateRequest


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

    @pytest.mark.parametrize(
        "url,backend",
        [
            ("http://localhost:8000", "vllm"),
            ("http://localhost:9000", "sglang"),
        ],
    )
    def test_init(self, url, backend):
        """Test initialization stores URL and backend correctly."""
        loader = RemoteWeightLoader(url=url, engine_backend=backend)

        assert loader._url == url
        assert loader._engine_backend == backend

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "backend,expected_endpoint",
        [
            ("vllm", "/init_weight_update_communicator"),
            ("sglang", "/init_weights_update_group"),
        ],
    )
    async def test_init_communicator(self, backend, expected_endpoint):
        """Test init_communicator calls correct endpoint for each backend."""
        url = "http://localhost:8000"
        loader = RemoteWeightLoader(url=url, engine_backend=backend)
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
            assert call_args[0][0] == f"{url}{expected_endpoint}"

            # Verify payload matches init_info fields
            payload = call_args[1]["json"]

            if backend == "vllm":
                # vLLM uses asdict(init_info) - all fields with original names
                from dataclasses import asdict

                assert payload == asdict(init_info)
            else:
                # SGLang uses only the fields SGLang expects
                assert payload["master_address"] == init_info.master_addr
                assert payload["master_port"] == init_info.master_port
                assert payload["rank_offset"] == init_info.rank_offset
                assert payload["world_size"] == init_info.world_size
                assert payload["group_name"] == init_info.group_name
                assert payload["backend"] == init_info.backend
                assert "model_dtype_str" not in payload
                assert "override_existing_receiver" not in payload

            assert result == {"success": True}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "backend,expected_endpoint",
        [
            ("vllm", "/update_weights"),
            ("sglang", "/update_weights_from_distributed"),
        ],
    )
    async def test_load_weights(self, backend, expected_endpoint):
        """Test load_weights calls correct endpoint for each backend."""
        url = "http://localhost:8000"
        loader = RemoteWeightLoader(url=url, engine_backend=backend)

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
            assert call_args[0][0] == f"{url}{expected_endpoint}"

            # Verify payload
            payload = call_args[1]["json"]
            if backend == "vllm":
                # vLLM uses asdict(request) - plural field names
                from dataclasses import asdict

                assert payload == asdict(request)
            else:
                # SGLang uses singular field names
                assert payload["name"] == "model.layer.weight"
                assert payload["dtype"] == "bfloat16"
                assert payload["shape"] == [4096, 4096]

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
