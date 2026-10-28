"""Tests for the native TorchNeuron inference engine.

Model loading and device execution are not exercised here (no neuron device on CI);
these tests pin the interface contract, the pure sampling-logprob math, and the
vLLM-free VLM input building (image loading / extraction + vision-tensor replay).
"""

import base64
import io

import pytest
import torch

from skyrl.backends.skyrl_train.inference_servers.base import InferenceEngineInterface
from skyrl.backends.skyrl_train.inference_servers.neuron_engine import NeuronInferenceEngine


def test_engine_fully_implements_interface():
    assert issubclass(NeuronInferenceEngine, InferenceEngineInterface)
    assert not NeuronInferenceEngine.__abstractmethods__


def test_sampled_token_logprobs_matches_softmax():
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    out = NeuronInferenceEngine._sampled_token_logprobs(
        logits, torch.tensor([2]), temperature=1.0, top_k=None, top_p=None, min_p=None
    )
    assert torch.allclose(out, logits.log_softmax(-1)[0, 2])


def test_sampled_token_logprobs_applies_temperature():
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    out = NeuronInferenceEngine._sampled_token_logprobs(
        logits, torch.tensor([2]), temperature=0.5, top_k=None, top_p=None, min_p=None
    )
    assert torch.allclose(out, (logits / 0.5).log_softmax(-1)[0, 2])


def test_sampled_token_logprobs_top_k_masks_out_of_rank():
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    out = NeuronInferenceEngine._sampled_token_logprobs(
        logits, torch.tensor([0]), temperature=1.0, top_k=1, top_p=None, min_p=None
    )
    assert torch.isneginf(out).all()


def test_sampled_token_logprobs_top_p_keeps_only_nucleus():
    # probs = [0.090, 0.244, 0.665]; top_p=0.5 keeps only the argmax (idx 2),
    # whose logprob under the renormalized singleton distribution is 0.0.
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    out = NeuronInferenceEngine._sampled_token_logprobs(
        logits, torch.tensor([2]), temperature=1.0, top_k=None, top_p=0.5, min_p=None
    )
    assert torch.allclose(out, torch.tensor([0.0]), atol=1e-6)


# -- VLM path (no model / neuron device required) -----------------------------


def _engine_stub():
    engine = object.__new__(NeuronInferenceEngine)
    engine._device = "cpu"
    return engine


def _png_data_uri(color="red", size=(4, 4)):
    from PIL import Image

    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def test_load_image_decodes_data_uri():
    pytest.importorskip("PIL")
    from PIL import Image

    out = NeuronInferenceEngine._load_image(_png_data_uri())
    assert isinstance(out, Image.Image)
    assert out.size == (4, 4)


def test_load_image_reads_file_path_and_file_uri(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    img = Image.new("RGB", (3, 5), color="blue")
    path = tmp_path / "frame.png"
    img.save(path)
    for source in (str(path), f"file://{path}"):
        out = NeuronInferenceEngine._load_image(source)
        assert out.size == (3, 5)


def test_load_image_passes_through_pil_image():
    pytest.importorskip("PIL")
    from PIL import Image

    img = Image.new("RGB", (2, 2))
    assert NeuronInferenceEngine._load_image(img) is img


def test_extract_images_collects_in_conversation_order():
    pytest.importorskip("PIL")
    from PIL import Image

    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": _png_data_uri()}},
        ]},
        {"role": "user", "content": [
            {"type": "image", "image": _png_data_uri("green")},
        ]},
        {"role": "user", "content": "text-only message"},
    ]
    images = NeuronInferenceEngine._extract_images(messages)
    assert len(images) == 2
    assert all(isinstance(im, Image.Image) for im in images)


def test_vlm_inputs_rebuilds_vision_tensors():
    from skyrl.backends.mm_codec import encode_mm_kwargs_item

    engine = _engine_stub()
    pv = torch.randn(4, 3, 8, 8, dtype=torch.bfloat16)
    thw = torch.tensor([[1, 8, 8]], dtype=torch.long)
    blob = encode_mm_kwargs_item({"pixel_values": pv, "image_grid_thw": thw})
    mm_features = [{"kwargs_data": {"image": [blob]}}]

    inputs, prompt_lens = engine._vlm_inputs([[7, 8, 9]], mm_features)

    assert prompt_lens == [3]
    assert torch.equal(inputs["input_ids"], torch.tensor([[7, 8, 9]]))
    assert torch.equal(inputs["attention_mask"], torch.ones(1, 3, dtype=torch.long))
    assert torch.equal(inputs["pixel_values"], pv)
    assert torch.equal(inputs["image_grid_thw"], thw)


def test_vlm_inputs_no_images_yields_text_only_inputs():
    engine = _engine_stub()
    inputs, prompt_lens = engine._vlm_inputs([[7, 8, 9]], [{"kwargs_data": None}])
    assert prompt_lens == [3]
    assert "pixel_values" not in inputs
    assert "image_grid_thw" not in inputs


# -- content normalization (parquet/pyarrow round-trip artifacts) -------------


def test_normalize_content_passes_through_plain_string():
    assert NeuronInferenceEngine._normalize_content("hello") == "hello"


def test_normalize_content_none_becomes_empty_list():
    assert NeuronInferenceEngine._normalize_content(None) == []


def test_normalize_content_single_dict_is_wrapped():
    part = {"type": "text", "text": "hi"}
    assert NeuronInferenceEngine._normalize_content(part) == [{"type": "text", "text": "hi"}]


def test_normalize_content_strips_spurious_keys_from_struct_parts():
    # A unified pyarrow struct schema gives a text part an ``image_url: None`` key (and
    # vice versa). The Qwen3.5-VL template treats any dict with an ``image_url`` *key*
    # as an image, so those spurious keys must be dropped before rendering.
    content = [
        {"type": "text", "text": "look", "image_url": None},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}, "text": None},
    ]
    assert NeuronInferenceEngine._normalize_content(content) == [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]


def test_normalize_content_drops_empty_and_unknown_parts():
    content = [
        {"type": "text", "text": None},
        {"type": "image_url", "image_url": None},
        {"type": "image", "image": None},
        {"type": "mystery", "x": 1},
        "not-a-dict",
    ]
    assert NeuronInferenceEngine._normalize_content(content) == []


def test_normalize_content_handles_numpy_ndarray_of_structs():
    numpy = pytest.importorskip("numpy")
    arr = numpy.array(
        [
            {"type": "text", "text": "a", "image_url": None},
            {"type": "text", "text": "b", "image_url": None},
        ],
        dtype=object,
    )
    assert NeuronInferenceEngine._normalize_content(arr) == [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
    ]
