from types import SimpleNamespace

import torch.nn as nn

from skyrl.backends.skyrl_train.distributed import fsdp_utils


class WrappedBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False)
        self.block = WrappedBlock()


def test_apply_fsdp2_does_not_wrap_embedding_inside_selected_parent(monkeypatch):
    model = DummyModel()
    wrapped = []
    monkeypatch.setattr(fsdp_utils, "fully_shard", lambda module, **kwargs: wrapped.append(module))
    config = SimpleNamespace(wrap_policy={"transformer_layer_cls_to_wrap": ["WrappedBlock"]})

    fsdp_utils.apply_fsdp2(model, {}, config)

    assert wrapped == [model.block, model]
