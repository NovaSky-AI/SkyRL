"""CPU tests for the Qwen3.5-VL neuron device-safety patches.

No neuron device is needed: these pin idempotency and the off-neuron transparency
that keeps CUDA/CPU training unchanged. The on-neuron branches (``repeat_interleave``
forced onto CPU, ``inv_freq`` following the tensor device) are exercised by the live
Trainium smoke run.
"""

import torch

from skyrl.backends.skyrl_train.patches.neuron import qwen3_5_vl


def test_apply_is_idempotent(monkeypatch):
    seen = []
    monkeypatch.setattr(qwen3_5_vl, "_APPLIED", False)
    monkeypatch.setattr(qwen3_5_vl, "_patch_vision_rope", lambda: seen.append("rope"))
    monkeypatch.setattr(qwen3_5_vl, "_patch_vision_interp", lambda: seen.append("interp"))
    monkeypatch.setattr(qwen3_5_vl, "_patch_vision_seqlens", lambda: seen.append("seqlens"))

    qwen3_5_vl.apply_qwen3_5_vl_device_patch()
    qwen3_5_vl.apply_qwen3_5_vl_device_patch()

    assert seen == ["rope", "interp", "seqlens"]


def test_rope_patch_moves_inv_freq_when_device_mismatches(monkeypatch):
    import transformers.models.qwen3_5.modeling_qwen3_5 as mq

    class _FakeRotaryEmbedding:
        def __init__(self):
            self.inv_freq = torch.tensor([1.0, 2.0])  # CPU

        def forward(self, x, position_ids):
            return (x, position_ids, self.inv_freq)

    monkeypatch.setattr(mq, "Qwen3_5VisionRotaryEmbedding", _FakeRotaryEmbedding)
    qwen3_5_vl._patch_vision_rope()

    inst = _FakeRotaryEmbedding()
    x = torch.tensor([0.0])
    position_ids = torch.tensor([1], device="meta")
    out = inst.forward(x, position_ids)

    assert inst.inv_freq.device.type == "meta"
    assert out[2] is inst.inv_freq


def test_rope_patch_is_transparent_when_same_device(monkeypatch):
    import transformers.models.qwen3_5.modeling_qwen3_5 as mq

    class _FakeRotaryEmbedding:
        def __init__(self):
            self.inv_freq = torch.tensor([1.0, 2.0])

        def forward(self, x, position_ids):
            return "orig"

    monkeypatch.setattr(mq, "Qwen3_5VisionRotaryEmbedding", _FakeRotaryEmbedding)
    qwen3_5_vl._patch_vision_rope()

    inst = _FakeRotaryEmbedding()
    out = inst.forward(torch.tensor([0.0]), torch.tensor([1]))

    assert out == "orig"
    assert inst.inv_freq.device.type == "cpu"


def test_interp_patch_forwards_off_neuron(monkeypatch):
    import transformers.models.qwen3_5.modeling_qwen3_5 as mq

    def fake(grid_thw, *args, **kwargs):
        return ("orig", grid_thw, args, kwargs)

    monkeypatch.setattr(mq, "get_vision_interpolation_indices_and_weights", fake)
    qwen3_5_vl._patch_vision_interp()

    grid = torch.tensor([[1, 4, 4]])
    out = mq.get_vision_interpolation_indices_and_weights(grid, 8, mode="bilinear")

    assert out == ("orig", grid, (8,), {"mode": "bilinear"})


def test_seqlens_patch_forwards_off_neuron(monkeypatch):
    import transformers.vision_utils as vu

    def fake(grid_thw, merge_temporal=False, kwargs=None):
        return ("orig", grid_thw, merge_temporal, kwargs)

    monkeypatch.setattr(vu, "get_vision_cu_seqlens", fake)
    qwen3_5_vl._patch_vision_seqlens()

    grid = torch.tensor([[1, 4, 4]])
    out = vu.get_vision_cu_seqlens(grid, merge_temporal=True, kwargs={"k": 1})

    assert out == ("orig", grid, True, {"k": 1})
