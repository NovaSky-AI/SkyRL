import pytest
import torch
from skyrl_train.weight_sync.weight_extractor_utils import yield_module_grouped_chunks
from skyrl_train.weight_sync import WeightChunk


class TestModuleGrouping:
    """Tests for yield_module_grouped_chunks utility function."""

    def test_basic_module_grouping(self):
        """Test that parameters are grouped by module correctly."""
        params = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(768, 768),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(768, 768),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(768, 768),
            "model.layers.0.mlp.fc1.weight": torch.randn(3072, 768),
            "model.layers.0.mlp.fc2.weight": torch.randn(768, 3072),
        }

        def gather_tensor(param):
            return param

        def get_shape(name, param, tensor):
            return list(tensor.shape)

        chunks = list(
            yield_module_grouped_chunks(
                params=params,
                dtype=torch.float32,
                gather_tensor_fn=gather_tensor,
                get_shape_fn=get_shape,
                batch_size_threshold_gb=0.0,
            )
        )

        # Should have 2 modules: self_attn and mlp
        assert len(chunks) == 2

        # Find self_attn and mlp chunks
        self_attn_chunks = [c for c in chunks if "self_attn" in c.names[0]]
        mlp_chunks = [c for c in chunks if "mlp" in c.names[0]]

        assert len(self_attn_chunks) == 1
        assert len(mlp_chunks) == 1

        # Check self_attn has 3 params (q, k, v)
        self_attn_chunk = self_attn_chunks[0]
        assert len(self_attn_chunk) == 3
        assert "q_proj" in self_attn_chunk.names[0]
        assert "k_proj" in self_attn_chunk.names[1]
        assert "v_proj" in self_attn_chunk.names[2]

        # Check mlp has 2 params (fc1, fc2)
        mlp_chunk = mlp_chunks[0]
        assert len(mlp_chunk) == 2
        assert "fc1" in mlp_chunk.names[0]
        assert "fc2" in mlp_chunk.names[1]

    @pytest.mark.parametrize(
        "params_config,threshold_gb,expected_chunks,expected_params_per_chunk,description",
        [
            # No batching: each module in separate chunk
            (
                {"model.layers.{}.attn.weight": [(i, 100) for i in range(4)]},
                0.0,
                4,
                [1, 1, 1, 1],
                "no batching",
            ),
            # Small threshold (~1000 bytes): batch 2 modules at a time
            # Each param is 100 elements * 4 bytes = 400 bytes
            # First batch: 400 + 400 = 800 < 1000, Second batch: 400 + 400 = 800 < 1000
            (
                {"model.layers.{}.attn.weight": [(i, 100) for i in range(4)]},
                0.000001,
                2,
                [2, 2],
                "small threshold batches 2 modules",
            ),
            # Large threshold (1 GB): batch all modules together
            (
                {"model.layers.{}.attn.weight": [(i, 100) for i in range(4)]},
                1.0,
                1,
                [4],
                "large threshold batches all",
            ),
            # Module boundaries: 3 params (1200 bytes) exceeds threshold (600 bytes) by 2x
            (
                {"model.layer.attn.proj{}.weight": [(i, 100) for i in range(3)]},
                0.0000006,
                1,
                [3],
                "module boundary: 3 params exceed threshold",
            ),
            # Module boundaries: 2 params (800 bytes) exceeds threshold (400 bytes) by 2x
            (
                {"model.layer.attn.proj{}.weight": [(i, 100) for i in range(2)]},
                0.0000004,
                1,
                [2],
                "module boundary: 2 params exceed threshold",
            ),
            # Module boundaries: 5 params (2000 bytes) exceeds threshold (500 bytes) by 4x
            (
                {"model.layer.attn.proj{}.weight": [(i, 100) for i in range(5)]},
                0.0000005,
                1,
                [5],
                "module boundary: 5 params exceed threshold",
            ),
        ],
    )
    def test_batching_with_different_thresholds(
        self, params_config, threshold_gb, expected_chunks, expected_params_per_chunk, description
    ):
        """Test batching behavior with various threshold values and module structures."""
        # Build params dict from config
        # params_config is {"template": [(idx, size), ...]}
        params = {}
        for template, indices_and_sizes in params_config.items():
            for idx, size in indices_and_sizes:
                param_name = template.format(idx)
                params[param_name] = torch.randn(size, dtype=torch.float32)

        chunks = list(
            yield_module_grouped_chunks(
                params=params,
                dtype=torch.float32,
                gather_tensor_fn=lambda p: p,
                get_shape_fn=lambda n, p, t: list(t.shape),
                batch_size_threshold_gb=threshold_gb,
            )
        )

        assert len(chunks) == expected_chunks
        for i, expected_params in enumerate(expected_params_per_chunk):
            assert len(chunks[i]) == expected_params

    def test_gather_tensor_callback(self):
        """Test that gather_tensor_fn callback is called correctly."""
        original = torch.randn(10, 10, dtype=torch.float32)
        params = {"model.layer.weight": original}

        def gather_tensor(param):
            # Simulate gathering by adding 1.0
            return param + 1.0

        chunks = list(
            yield_module_grouped_chunks(
                params=params,
                dtype=torch.bfloat16,
                gather_tensor_fn=gather_tensor,
                get_shape_fn=lambda n, p, t: list(t.shape),
            )
        )

        assert len(chunks) == 1
        tensor = chunks[0].tensors[0]
        # Should be bfloat16 (dtype cast in utils) and have +1.0 applied (from gather)
        assert tensor.dtype == torch.bfloat16
        expected = (original + 1.0).to(torch.bfloat16)
        assert torch.allclose(tensor, expected)

    def test_get_shape_callback(self):
        """Test that get_shape_fn callback is called correctly."""
        params = {"model.layer.weight": torch.randn(10, 20)}

        def get_shape(name, param, tensor):
            # Return custom shape
            return [999, 888]

        chunks = list(
            yield_module_grouped_chunks(
                params=params,
                dtype=torch.float32,
                gather_tensor_fn=lambda p: p,
                get_shape_fn=get_shape,
            )
        )

        assert len(chunks) == 1
        assert chunks[0].shapes[0] == [999, 888]

    def test_empty_params(self):
        """Test with empty params dict."""
        params = {}

        chunks = list(
            yield_module_grouped_chunks(
                params=params,
                dtype=torch.float32,
                gather_tensor_fn=lambda p: p,
                get_shape_fn=lambda n, p, t: list(t.shape),
            )
        )

        assert len(chunks) == 0

    def test_chunk_properties(self):
        """Test that returned chunks have correct WeightChunk properties."""
        params = {
            "model.layer.attn.weight": torch.randn(10, 10),
            "model.layer.attn.bias": torch.randn(10),
        }

        chunks = list(
            yield_module_grouped_chunks(
                params=params,
                dtype=torch.float32,
                gather_tensor_fn=lambda p: p,
                get_shape_fn=lambda n, p, t: list(t.shape),
            )
        )

        assert len(chunks) == 1
        chunk = chunks[0]

        # Check all required fields are present
        assert isinstance(chunk, WeightChunk)
        assert len(chunk.names) == 2
        assert len(chunk.dtypes) == 2
        assert len(chunk.shapes) == 2
        assert len(chunk.tensors) == 2

        # Check total_numel
        expected_numel = 10 * 10 + 10
        assert chunk.total_numel == expected_numel
