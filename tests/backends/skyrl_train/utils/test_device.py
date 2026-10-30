"""CPU tests for the accelerator shim: selection, the CUDA default, and the TPU-only patches."""

import pytest
import torch

from skyrl.backends.skyrl_train.utils import device


@pytest.fixture
def device_env(monkeypatch):
    """Set ``SKYRL_DEVICE`` for one test; ``device_type`` caches its answer, so clear it around the test."""

    def _set(value):
        if value is None:
            monkeypatch.delenv("SKYRL_DEVICE", raising=False)
        else:
            monkeypatch.setenv("SKYRL_DEVICE", value)
        device.device_type.cache_clear()

    yield _set
    device.device_type.cache_clear()


def test_default_is_cuda(device_env):
    device_env(None)
    assert device.device_type() == "cuda"
    assert device.is_cuda() and not device.is_tpu()
    assert device.dist_backend() == "cpu:gloo,cuda:nccl"


def test_tpu_selection(device_env):
    device_env("tpu")
    assert device.is_tpu() and not device.is_cuda()
    assert device.dist_backend() == "cpu:gloo,tpu:tpu_dist"
    assert device.device_count() == 1
    assert device.get_rng_state() is None


def test_dist_backend_override(device_env, monkeypatch):
    device_env("tpu")
    monkeypatch.setenv("SKYRL_DIST_BACKEND", "cpu:gloo,tpu:other")
    assert device.dist_backend() == "cpu:gloo,tpu:other"


def test_invalid_value_raises(device_env):
    device_env("xpu")
    with pytest.raises(ValueError, match="SKYRL_DEVICE"):
        device.device_type()


def test_tpu_patches_are_noops_off_tpu(device_env, monkeypatch):
    """With the default device the patch entry points must not touch FSDP or transformers."""
    device_env(None)
    from torch.distributed.fsdp._fully_shard import _fsdp_init

    original = _fsdp_init._move_states_to_device
    device.patch_fsdp_move_states_for_tpu()
    device.patch_qwen35_mrope_for_tpu()
    device.patch_qwen35_vision_rope_for_tpu()
    device.configure_process_env(rank=0, local_rank=0, world_size=1, group="policy")
    assert _fsdp_init._move_states_to_device is original


def test_mrope_replacement_matches_transformers():
    """The alias-free interleave gives the same values as the installed transformers."""
    modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    torch.manual_seed(0)
    section = [11, 11, 10]
    owners = [
        cls for cls in vars(modeling).values() if isinstance(cls, type) and "apply_interleaved_mrope" in vars(cls)
    ]
    if owners:
        freqs = torch.randn(3, 2, 17, sum(section))
        expected = vars(owners[0])["apply_interleaved_mrope"](None, freqs.clone(), section)
        assert torch.equal(device._apply_interleaved_mrope_cloned(None, freqs.clone(), section), expected)
    text_rope = getattr(modeling, "Qwen3_5TextRotaryEmbedding", None)
    if text_rope is not None and "recomposition_frequencies" in vars(text_rope):
        holder = type("Holder", (), {"mrope_section": section})()
        freq = torch.randn(3, 2, 17, sum(section))
        expected = vars(text_rope)["recomposition_frequencies"](holder, freq.clone())
        assert torch.equal(device._text_mrope_recomposition(holder, freq.clone()), expected)
    if not owners and not (text_rope is not None and "recomposition_frequencies" in vars(text_rope)):
        pytest.skip("this transformers version has neither interleave implementation")


def test_tpu_env_spec_is_applied_per_rank(device_env, monkeypatch):
    device_env("tpu")
    monkeypatch.setattr(device, "_ensure_backend_loaded", lambda: None)
    for name in ("patch_fsdp_move_states_for_tpu", "patch_qwen35_mrope_for_tpu", "patch_qwen35_vision_rope_for_tpu"):
        monkeypatch.setattr(device, name, lambda: None)
    monkeypatch.setenv(
        "SKYRL_TPU_GROUP_ENV_POLICY",
        '{"env": {"TORCH_TPU_TOPOLOGY": "2,2,1"}, "per_rank": [{"TPU_VISIBLE_DEVICES": "0"}, {"TPU_VISIBLE_DEVICES": "1"}]}',
    )
    for key in ("TORCH_TPU_TOPOLOGY", "TPU_VISIBLE_DEVICES", "RANK", "WORLD_SIZE"):
        monkeypatch.delenv(key, raising=False)
    device.configure_process_env(rank=1, local_rank=1, world_size=2, group="policy")
    import os

    assert os.environ["TORCH_TPU_TOPOLOGY"] == "2,2,1"
    assert os.environ["TPU_VISIBLE_DEVICES"] == "1"
    assert os.environ["RANK"] == "1" and os.environ["WORLD_SIZE"] == "2"
