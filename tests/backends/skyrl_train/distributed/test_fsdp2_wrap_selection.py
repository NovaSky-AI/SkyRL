"""FSDP2 wrap selection: only input embeddings get a dedicated shard group.

Pins the fix for the GLM-4.6V-Flash crash `aten.grid_sampler_2d.default got
mixed torch.Tensor and DTensor`: wrapping EVERY nn.Embedding gave GLM4V's
vision position_embedding its own group, and its weight (read raw from the
parent forward, never via its own forward) stayed a sharded DTensor when it
reached F.grid_sample.
"""

import torch.nn as nn

from skyrl.backends.skyrl_train.distributed.fsdp_utils import modules_to_wrap_fsdp2


class _Cfg:
    tie_word_embeddings = False


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)


class VisionEmbeddings(nn.Module):
    """Like Glm4vVisionEmbeddings: holds an nn.Embedding whose weight is read raw."""

    def __init__(self):
        super().__init__()
        self.position_embedding = nn.Embedding(16, 4)


class TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _Cfg()
        self.embed_tokens = nn.Embedding(32, 4)
        self.layer0 = DecoderLayer()
        self.layer1 = DecoderLayer()
        self.vision_embed = VisionEmbeddings()

    def get_input_embeddings(self):
        return self.embed_tokens


def test_only_input_embeddings_and_layers_get_groups():
    m = TinyVLM()
    wrapped = modules_to_wrap_fsdp2(m, ["DecoderLayer"])
    assert m.embed_tokens in wrapped  # word embeddings keep their group
    assert m.layer0 in wrapped and m.layer1 in wrapped
    assert m.vision_embed.position_embedding not in wrapped  # the GLM crash case
    assert len(wrapped) == 3


def test_tied_embeddings_not_wrapped():
    m = TinyVLM()
    m.config.tie_word_embeddings = True
    wrapped = modules_to_wrap_fsdp2(m, ["DecoderLayer"])
    assert m.embed_tokens not in wrapped


def test_model_without_get_input_embeddings():
    m = TinyVLM()
    del TinyVLM.get_input_embeddings
    try:
        wrapped = modules_to_wrap_fsdp2(m, ["DecoderLayer"])
        assert len(wrapped) == 2  # just the layers; no embedding claimed
    finally:
        TinyVLM.get_input_embeddings = lambda self: self.embed_tokens
