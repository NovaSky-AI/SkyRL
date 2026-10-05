"""model_owns_vlm_packing: which VLMs may use sample packing on the Megatron backend."""

import pytest
import torch.nn as nn

megatron_utils = pytest.importorskip("skyrl.backends.skyrl_train.distributed.megatron.megatron_utils")


class _OptIn(nn.Module):
    model_owns_packing = True


def test_plain_model_does_not_own_packing():
    assert not megatron_utils.model_owns_vlm_packing([nn.Linear(2, 2)])


def test_opt_in_attribute():
    assert megatron_utils.model_owns_vlm_packing([_OptIn()])
    assert megatron_utils.model_owns_vlm_packing(_OptIn())


def test_every_chunk_must_own_packing():
    assert not megatron_utils.model_owns_vlm_packing([_OptIn(), nn.Linear(2, 2)])


def test_no_chunks():
    assert not megatron_utils.model_owns_vlm_packing([])


def test_qwen3_vl_model_class():
    model_mod = pytest.importorskip("megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model")
    # Skip construction: an uninitialized instance is enough for the isinstance check.
    fake = model_mod.Qwen3VLModel.__new__(model_mod.Qwen3VLModel)
    nn.Module.__init__(fake)
    assert megatron_utils.model_owns_vlm_packing([fake])
