"""
Tests for the flash_attn stub package.

The stub provides bert_padding functions (pad_input, unpad_input) that SkyRL
training code uses. This test verifies the stub implementation is correct.
"""

import sys
import tempfile
import os

import pytest
import torch


class TestFlashAttnStubCreation:
    """Test that the stub creation script works."""

    def test_create_stub_script_exists(self):
        """Verify the stub creation script exists."""
        script_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "integrations",
            "fleet",
            "scripts",
            "create_flash_attn_stub.py",
        )
        assert os.path.exists(script_path), f"Script not found at {script_path}"

    def test_create_stub_in_temp_dir(self):
        """Test stub creation in a temporary directory."""
        import importlib.util

        # Load the stub creation module
        script_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "integrations",
            "fleet",
            "scripts",
            "create_flash_attn_stub.py",
        )
        spec = importlib.util.spec_from_file_location("create_stub", script_path)
        stub_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stub_module)

        # Create stub in temp dir by patching site.getsitepackages
        with tempfile.TemporaryDirectory() as tmpdir:
            import site

            original_getsitepackages = site.getsitepackages
            site.getsitepackages = lambda: [tmpdir]

            try:
                stub_module.create_stub()

                # Verify files were created
                flash_attn_dir = os.path.join(tmpdir, "flash_attn")
                assert os.path.isdir(flash_attn_dir)
                assert os.path.isfile(os.path.join(flash_attn_dir, "__init__.py"))
                assert os.path.isfile(os.path.join(flash_attn_dir, "bert_padding.py"))

                # Add to path and test import
                sys.path.insert(0, tmpdir)
                try:
                    # Force reimport
                    if "flash_attn" in sys.modules:
                        del sys.modules["flash_attn"]
                    if "flash_attn.bert_padding" in sys.modules:
                        del sys.modules["flash_attn.bert_padding"]

                    from flash_attn.bert_padding import pad_input, unpad_input

                    assert callable(pad_input)
                    assert callable(unpad_input)
                finally:
                    sys.path.remove(tmpdir)
            finally:
                site.getsitepackages = original_getsitepackages


class TestBertPaddingFunctions:
    """Test the bert_padding pad_input and unpad_input functions."""

    @pytest.fixture
    def bert_padding_module(self):
        """Load bert_padding from stub or create inline for testing."""
        import importlib.util

        # Load the stub creation module to get bert_padding code
        script_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "integrations",
            "fleet",
            "scripts",
            "create_flash_attn_stub.py",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            import site

            original_getsitepackages = site.getsitepackages
            site.getsitepackages = lambda: [tmpdir]

            try:
                spec = importlib.util.spec_from_file_location("create_stub", script_path)
                stub_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(stub_module)
                stub_module.create_stub()

                sys.path.insert(0, tmpdir)
                try:
                    if "flash_attn" in sys.modules:
                        del sys.modules["flash_attn"]
                    if "flash_attn.bert_padding" in sys.modules:
                        del sys.modules["flash_attn.bert_padding"]

                    from flash_attn import bert_padding

                    yield bert_padding
                finally:
                    sys.path.remove(tmpdir)
                    if "flash_attn" in sys.modules:
                        del sys.modules["flash_attn"]
                    if "flash_attn.bert_padding" in sys.modules:
                        del sys.modules["flash_attn.bert_padding"]
            finally:
                site.getsitepackages = original_getsitepackages

    def test_unpad_input_basic(self, bert_padding_module):
        """Test unpad_input removes padding correctly."""
        batch_size = 2
        seq_len = 4
        hidden_dim = 8

        # Create hidden states: (batch, seq, hidden)
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)

        # Attention mask: 1 for valid tokens, 0 for padding
        # First sequence has 3 valid tokens, second has 2
        attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

        unpad_input = bert_padding_module.unpad_input
        hidden_unpad, indices, cu_seqlens, max_seqlen, seqlens = unpad_input(hidden_states, attention_mask)

        # Check shapes
        total_valid = 3 + 2  # 5 valid tokens total
        assert hidden_unpad.shape == (total_valid, hidden_dim)
        assert indices.shape == (total_valid,)
        assert cu_seqlens.shape == (batch_size + 1,)
        assert max_seqlen == 3  # max of (3, 2)

        # Check seqlens
        assert seqlens.tolist() == [3, 2]

        # Check cu_seqlens (cumulative)
        assert cu_seqlens.tolist() == [0, 3, 5]

    def test_pad_input_basic(self, bert_padding_module):
        """Test pad_input restores padding correctly."""
        batch_size = 2
        seq_len = 4
        hidden_dim = 8

        # Create unpadded hidden states (5 valid tokens)
        hidden_unpad = torch.randn(5, hidden_dim)

        # Indices showing where each token came from in flattened (batch*seq) space
        # Batch 0: positions 0, 1, 2 (seq positions 0, 1, 2)
        # Batch 1: positions 4, 5 (seq positions 0, 1 in batch 1)
        indices = torch.tensor([0, 1, 2, 4, 5])

        pad_input = bert_padding_module.pad_input
        hidden_padded = pad_input(hidden_unpad, indices, batch_size, seq_len)

        # Check shape
        assert hidden_padded.shape == (batch_size, seq_len, hidden_dim)

        # Check values at valid positions
        assert torch.allclose(hidden_padded[0, 0], hidden_unpad[0])
        assert torch.allclose(hidden_padded[0, 1], hidden_unpad[1])
        assert torch.allclose(hidden_padded[0, 2], hidden_unpad[2])
        assert torch.allclose(hidden_padded[1, 0], hidden_unpad[3])
        assert torch.allclose(hidden_padded[1, 1], hidden_unpad[4])

        # Check padding positions are zero
        assert torch.allclose(hidden_padded[0, 3], torch.zeros(hidden_dim))
        assert torch.allclose(hidden_padded[1, 2], torch.zeros(hidden_dim))
        assert torch.allclose(hidden_padded[1, 3], torch.zeros(hidden_dim))

    def test_roundtrip_unpad_pad(self, bert_padding_module):
        """Test that unpad -> pad recovers original (at valid positions)."""
        batch_size = 3
        seq_len = 5
        hidden_dim = 16

        # Create hidden states
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)

        # Variable length sequences
        attention_mask = torch.tensor(
            [[1, 1, 1, 1, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 1]]  # 4 tokens  # 2 tokens
        )  # 5 tokens

        unpad_input = bert_padding_module.unpad_input
        pad_input = bert_padding_module.pad_input

        # Unpad
        hidden_unpad, indices, cu_seqlens, max_seqlen, seqlens = unpad_input(hidden_states, attention_mask)

        # Pad back
        hidden_repadded = pad_input(hidden_unpad, indices, batch_size, seq_len)

        # Check valid positions match original
        for b in range(batch_size):
            for s in range(seq_len):
                if attention_mask[b, s] == 1:
                    assert torch.allclose(hidden_repadded[b, s], hidden_states[b, s])
                else:
                    # Padding positions should be zero
                    assert torch.allclose(hidden_repadded[b, s], torch.zeros(hidden_dim))

    def test_gradient_flow(self, bert_padding_module):
        """Test that gradients flow through unpad/pad operations."""
        batch_size = 2
        seq_len = 4
        hidden_dim = 8

        hidden_states = torch.randn(batch_size, seq_len, hidden_dim, requires_grad=True)
        attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

        unpad_input = bert_padding_module.unpad_input
        pad_input = bert_padding_module.pad_input

        # Forward
        hidden_unpad, indices, cu_seqlens, max_seqlen, seqlens = unpad_input(hidden_states, attention_mask)

        # Apply some operation
        hidden_unpad_transformed = hidden_unpad * 2

        # Pad back
        hidden_repadded = pad_input(hidden_unpad_transformed, indices, batch_size, seq_len)

        # Backward
        loss = hidden_repadded.sum()
        loss.backward()

        # Check gradients exist
        assert hidden_states.grad is not None
        assert hidden_states.grad.shape == hidden_states.shape

        # Gradients at valid positions should be 2 (from the *2 operation)
        for b in range(batch_size):
            for s in range(seq_len):
                if attention_mask[b, s] == 1:
                    assert torch.allclose(hidden_states.grad[b, s], torch.full((hidden_dim,), 2.0))
                else:
                    # No gradient flow to padding positions
                    assert torch.allclose(hidden_states.grad[b, s], torch.zeros(hidden_dim))
