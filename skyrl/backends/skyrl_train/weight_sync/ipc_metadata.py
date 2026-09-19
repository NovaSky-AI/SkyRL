"""Metadata for the packed buffer owned by each physical GPU, independent of CUDA."""

from math import prod


def ipc_chunk_metadata(update_info, gpu_uuid):
    # Legacy senders shared metadata across ranks. Once a per-GPU map is present, a missing entry
    # must refuse: falling back to rank 0's names can relabel another EP owner's expert bytes.
    if "metadata_by_gpu" in update_info:
        mapping = update_info["metadata_by_gpu"]
        if not isinstance(mapping, dict) or gpu_uuid not in mapping:
            raise ValueError(f"weight_sync.ipc: missing chunk metadata for GPU UUID {gpu_uuid}")
        metadata = mapping[gpu_uuid]
    else:
        metadata = update_info
    if not isinstance(metadata, dict):
        raise ValueError("weight_sync.ipc: chunk metadata must be a mapping")
    names, shapes, sizes = (metadata.get(key) for key in ("names", "shapes", "sizes"))
    if (
        not all(isinstance(v, list) for v in (names, shapes, sizes))
        or not names
        or len(names) != len(shapes)
        or len(names) != len(sizes)
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("weight_sync.ipc: inconsistent chunk names/shapes/sizes")
    if "dtype_names" in metadata:
        dtypes = metadata["dtype_names"]
        if (
            not isinstance(dtypes, list)
            or len(dtypes) != len(names)
            or any(not isinstance(dtype, str) or not dtype for dtype in dtypes)
        ):
            raise ValueError("weight_sync.ipc: inconsistent chunk dtype_names")
    for shape, size in zip(shapes, sizes):
        if (
            not isinstance(shape, list)
            or any(type(v) is not int or v < 0 for v in shape)
            or type(size) is not int
            or size < 0
            or prod(shape) != size
        ):
            raise ValueError("weight_sync.ipc: shape does not match packed size")
    return metadata


def merge_ipc_metadata(gathered):
    """Keep handles and their decoding metadata keyed by the same unique GPU UUID."""
    handles, metadata = {}, {}
    for uuid, handle, chunk in gathered:
        if not isinstance(uuid, str) or not uuid or uuid in handles:
            raise ValueError(f"weight_sync.ipc: duplicate or missing GPU UUID {uuid!r}")
        handles[uuid] = handle
        metadata[uuid] = ipc_chunk_metadata(chunk, uuid)
    return handles, metadata
