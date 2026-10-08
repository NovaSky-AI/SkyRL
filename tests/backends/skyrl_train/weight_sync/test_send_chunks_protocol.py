"""FP8 stream compatibility with the trainer-engine source contract."""

import pytest
import torch


def test_serialized_fp8_source_metadata_matches_the_weight_stream():
    """The engine's declared stream includes FP8 scales and batched MoE names."""
    pytest.importorskip("vllm")

    from skyrl.backends.skyrl_train.weight_sync.fp8 import SerializedFp8Config
    from skyrl.backends.skyrl_train.weight_sync.fp8.models import QWEN35_FP8_SPEC
    from skyrl.backends.skyrl_train.weight_sync.sources import SerializedFp8WeightSource

    class Source:
        def __iter__(self):
            yield "model.layers.0.mlp.down_proj.weight", torch.ones(128, 128, dtype=torch.bfloat16)

    source = SerializedFp8WeightSource(Source(), SerializedFp8Config(spec=QWEN35_FP8_SPEC))
    metadata = source.metadata()
    streamed = list(source)

    assert [(item.name, item.dtype, item.shape) for item in metadata] == [
        (name, tensor.dtype, tuple(tensor.shape)) for name, tensor in streamed
    ]
    assert [name for name, _ in streamed] == [
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.mlp.down_proj.weight_scale_inv",
    ]
