"""CPU tests for the colocated weight-sync instrumentation helpers."""

import torch

from skyrl.backends.skyrl_train.weight_sync.colocated import instrumentation as inst


def test_tensor_checksum_is_byte_exact():
    t = torch.arange(37, dtype=torch.bfloat16)  # 74 bytes: exercises the int64 padding path
    same = t.clone()
    assert inst.tensor_checksum(t) == inst.tensor_checksum(same)

    flipped = t.clone()
    flipped.view(torch.uint8)[5] ^= 1
    assert inst.tensor_checksum(flipped) != inst.tensor_checksum(t)

    # Same bytes in a different order must differ (the weighted sum is order-sensitive).
    swapped = t.clone()
    swapped[[0, 1]] = swapped[[1, 0]]
    assert inst.tensor_checksum(swapped) != inst.tensor_checksum(t)

    # Shape and dtype are part of the checksum.
    assert inst.tensor_checksum(t.view(1, 37)) != inst.tensor_checksum(t)
    assert inst.tensor_checksum(torch.empty(0)) == (0, 0, (0,), "torch.float32")


def test_module_checksums_cover_params_and_buffers():
    m = torch.nn.BatchNorm1d(4)
    sums = inst.module_checksums(m)
    assert {"p:weight", "p:bias", "b:running_mean", "b:running_var", "b:num_batches_tracked"} <= set(sums)


def test_debug_enabled(monkeypatch):
    monkeypatch.delenv(inst.DEBUG_ENV, raising=False)
    assert not inst.debug_enabled()
    monkeypatch.setenv(inst.DEBUG_ENV, "1")
    assert inst.debug_enabled()
    monkeypatch.setenv(inst.DEBUG_ENV, "0")
    assert not inst.debug_enabled()


class _Source:
    metadata = {"m": 1}

    def iter_full(self):
        yield "a", torch.zeros(4, dtype=torch.float32)
        yield "b", torch.zeros(2, dtype=torch.bfloat16)


class _Client:
    def __init__(self):
        self.calls = []

    def update_weights(self, update_info):
        self.calls.append(update_info)


class _Engine:
    def __init__(self):
        self.source = _Source()
        self.client = _Client()


def test_instrument_send_counts_and_restores():
    engine = _Engine()
    original_source = engine.source
    rec = inst.PhaseRecorder("t")
    with inst.instrument_send(engine, rec):
        assert engine.source.metadata == {"m": 1}
        names = [n for n, _ in engine.source.iter_full()]
        engine.client.update_weights({"x": 1})
        engine.client.update_weights({"x": 2})
    assert names == ["a", "b"]
    assert rec.counters["export_tensors"] == 2
    assert rec.counters["export_bytes"] == 4 * 4 + 2 * 2
    assert rec.counters["round_trips"] == 2
    assert rec.counters["round_trip_seconds"] >= 0
    # The wrappers are gone afterwards and the calls went through.
    assert engine.source is original_source
    assert "update_weights" not in vars(engine.client)
    assert engine.client.calls == [{"x": 1}, {"x": 2}]


def test_trainer_memory_cap_is_noop_without_env(monkeypatch):
    monkeypatch.delenv(inst.TRAINER_CAP_ENV, raising=False)
    with inst.trainer_memory_cap():
        pass
