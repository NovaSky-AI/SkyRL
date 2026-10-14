"""``megatron_config.optimizer_offload_to_disk`` vs the regular DistributedOptimizer.

Runs ``optimizer_offload_to_disk_harness.py`` under torchrun on 2 GPUs: a tiny dense GPT, and a tiny MoE GPT with
EP2, which adds the expert optimizer of a ChainedOptimizer. With fp32 moments the offloaded optimizer must be
bit-identical to the regular one, including across a colocation offload/backload and a checkpoint resume; with
bf16 moments it must stay close.

Run with:
uv run --isolated --extra dev --extra megatron pytest tests/backends/skyrl_train/gpu/gpu_ci/test_optimizer_offload_to_disk.py
"""

import json
import os
import subprocess
import sys

import pytest
import torch

HARNESS = os.path.join(os.path.dirname(__file__), "optimizer_offload_to_disk_harness.py")


@pytest.mark.megatron
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe_ep2"])
@pytest.mark.parametrize("moment_dtype", ["fp32", "bf16"])
def test_offloaded_optimizer_matches_regular_optimizer(tmp_path, moe, moment_dtype):
    from skyrl.backends.skyrl_train.distributed.megatron.optimizer_offload_to_disk import (
        filesystem_type,
    )
    from skyrl.train.utils.utils import get_free_port

    if filesystem_type(str(tmp_path)) in ("tmpfs", "ramfs"):
        pytest.skip(f"{tmp_path} is on tmpfs, which optimizer_offload_to_disk refuses")
    port = get_free_port()
    cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", f"--master_port={port}", HARNESS]
    cmd += ["--dir", str(tmp_path), "--moment-dtype", moment_dtype] + (["--moe"] if moe else [])
    # PYTHONSAFEPATH: torchrun would otherwise put this directory first on sys.path, where the sibling
    # ``megatron/`` test package shadows megatron-core. A small chunk size splits shards inside params.
    env = dict(os.environ, PYTHONSAFEPATH="1", SKYRL_OPTIMIZER_OFFLOAD_TO_DISK_CHUNK_NUMEL="3000")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out[-6000:]
    line = next((line for line in proc.stdout.splitlines() if line.startswith("HARNESS_OK ")), None)
    assert line is not None, out[-6000:]
    assert len(json.loads(line.split(" ", 1)[1])["history"]) == 10


@pytest.mark.megatron
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("storage_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("direct", [True, False], ids=["o_direct", "buffered_fallback"])
def test_staged_file_io_round_trips_unaligned_segments(tmp_path, direct, storage_dtype):
    """Segments whose sizes are not 4 KiB multiples, each spanning several staging chunks, read back exactly.

    O_DIRECT rounds the last chunk of each segment up into the segment's padding; writing the segments in
    reverse order would expose a rounded write that spills into the next segment.
    """
    from skyrl.backends.skyrl_train.distributed.megatron.optimizer_offload_to_disk import (
        IO_ALIGN,
        _align,
        _StateFile,
    )

    sizes = [1000, 5003, 1, 4097]
    offsets, at = [], 0
    for numel in sizes:
        offsets.append(at)
        at += _align(numel * storage_dtype.itemsize)
    state_file = _StateFile(str(tmp_path / "state.bin"), at, staging_bytes=4 * IO_ALIGN)
    if direct and state_file.direct_fd is None:
        pytest.skip(f"{tmp_path} does not support O_DIRECT")
    if not direct:
        state_file._disable_direct("test")

    tensors = [torch.randn(numel, device="cuda").to(storage_dtype) for numel in sizes]
    for tensor, offset in reversed(list(zip(tensors, offsets))):
        state_file.write(offset, tensor)
    for tensor, offset in zip(tensors, offsets):
        back = torch.empty_like(tensor)
        state_file.read(offset, back)
        assert torch.equal(back, tensor)
    assert (state_file.direct_fd is not None) == direct, "O_DIRECT fell back to buffered I/O"
    state_file.close()
