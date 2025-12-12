import pytest
import torch

from skyrl_train.inference_engines.sglang.ipc_utils import (
    serialize_ipc_request,
    deserialize_ipc_request,
    IPC_REQUEST_END_MARKER,
)


class TestIPCRequestSerialization:
    """Tests for IPC request serialization/deserialization."""

    def test_roundtrip(self):
        """Test that serialization/deserialization roundtrip preserves data."""
        request = {
            "names": ["model.layer.weight"],
            "dtypes": ["bfloat16"],
            "shapes": [[4096, 4096]],
            "extras": [{"ipc_handles": {"gpu-uuid": "test_handle"}}],
        }

        tensor = serialize_ipc_request(request)
        result = deserialize_ipc_request(tensor)

        assert result == request

    def test_roundtrip_multiple_weights(self):
        """Test roundtrip with multiple weights."""
        request = {
            "names": ["layer1.weight", "layer2.weight", "layer3.bias"],
            "dtypes": ["bfloat16", "bfloat16", "float32"],
            "shapes": [[4096, 4096], [4096, 1024], [1024]],
            "extras": [
                {"ipc_handles": {"gpu-0": "handle1"}},
                {"ipc_handles": {"gpu-0": "handle2"}},
                {"ipc_handles": {"gpu-0": "handle3"}},
            ],
        }

        tensor = serialize_ipc_request(request)
        result = deserialize_ipc_request(tensor)

        assert result == request

    def test_deserialize_missing_end_marker(self):
        """Test that missing end marker raises ValueError."""
        # Create tensor without proper end marker
        invalid_tensor = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)

        with pytest.raises(ValueError, match="End marker not found"):
            deserialize_ipc_request(invalid_tensor)

    def test_deserialize_invalid_data(self):
        """Test that invalid base64/pickle data raises ValueError."""
        # Create tensor with end marker but invalid data before it
        invalid_data = b"not_valid_base64!!!" + IPC_REQUEST_END_MARKER
        invalid_tensor = torch.frombuffer(bytearray(invalid_data), dtype=torch.uint8)

        with pytest.raises(ValueError, match="Failed to deserialize"):
            deserialize_ipc_request(invalid_tensor)

    def test_serialize_returns_uint8_tensor(self):
        """Test that serialize returns a uint8 tensor."""
        request = {"names": ["test"]}
        tensor = serialize_ipc_request(request)

        assert tensor.dtype == torch.uint8

    def test_serialize_aligned_to_4_bytes(self):
        """Test that serialized tensor is 4-byte aligned."""
        request = {"names": ["test"]}
        tensor = serialize_ipc_request(request)

        assert len(tensor) % 4 == 0
