def effective_padding_micro_batch_size(
    batch_size: int, micro_batch_size: int | None
) -> int | None:
    if micro_batch_size is None:
        return None
    return min(batch_size, micro_batch_size)
