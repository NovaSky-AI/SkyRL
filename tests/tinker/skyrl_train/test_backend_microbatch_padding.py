from skyrl.backends.microbatch_padding import effective_padding_micro_batch_size


def test_effective_padding_micro_batch_size_keeps_none_unaligned():
    assert effective_padding_micro_batch_size(3, None) is None


def test_effective_padding_micro_batch_size_keeps_configured_size_for_large_batches():
    assert effective_padding_micro_batch_size(32, 16) == 16


def test_effective_padding_micro_batch_size_uses_batch_size_for_small_batches():
    assert effective_padding_micro_batch_size(1, 16) == 1
    assert effective_padding_micro_batch_size(8, 16) == 8
