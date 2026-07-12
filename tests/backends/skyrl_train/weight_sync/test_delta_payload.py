import torch

from skyrl.backends.skyrl_train.weight_sync.delta_payload import DeltaPayload


def test_delta_payload_compress_recover_exact_with_positions():
    values = torch.tensor([1.0, -2.5, 3.25, float("nan")], dtype=torch.bfloat16)
    positions = torch.tensor([0, 7, 1024, 4096], dtype=torch.int32)

    recovered = DeltaPayload(values=values, positions=positions).compress().recover()

    assert recovered.is_compressed is False
    assert recovered.values.dtype == values.dtype
    assert recovered.positions.dtype == positions.dtype
    assert torch.equal(recovered.values.view(torch.uint8), values.view(torch.uint8))
    assert torch.equal(recovered.positions, positions)


def test_delta_payload_compress_recover_exact_for_uint8_dense_patch():
    values = torch.arange(257, dtype=torch.int16).view(torch.uint8)

    recovered = DeltaPayload(values=values).compress().recover()

    assert recovered.positions is None
    assert recovered.values.dtype == torch.uint8
    assert torch.equal(recovered.values, values)
