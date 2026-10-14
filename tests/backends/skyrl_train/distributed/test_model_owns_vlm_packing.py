"""model_owns_vlm_packing: which VLMs may use sample packing on the Megatron backend."""

import pytest
import torch.nn as nn

megatron_utils = pytest.importorskip("skyrl.backends.skyrl_train.distributed.megatron.megatron_utils")


def _fake_qwen3_vl_model():
    model_mod = pytest.importorskip("megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model")
    # Skip construction: an uninitialized instance is enough for the isinstance check.
    fake = model_mod.Qwen3VLModel.__new__(model_mod.Qwen3VLModel)
    nn.Module.__init__(fake)
    return fake


def test_plain_model_does_not_own_packing():
    assert not megatron_utils.model_owns_vlm_packing([nn.Linear(2, 2)])


def test_no_chunks():
    assert not megatron_utils.model_owns_vlm_packing([])


def test_qwen3_vl_model_class():
    fake = _fake_qwen3_vl_model()
    assert megatron_utils.model_owns_vlm_packing([fake])
    assert megatron_utils.model_owns_vlm_packing(fake)


def test_every_chunk_must_own_packing():
    assert not megatron_utils.model_owns_vlm_packing([_fake_qwen3_vl_model(), nn.Linear(2, 2)])
