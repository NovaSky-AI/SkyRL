"""CPU tests for ``optimizer_offload_to_disk.py``: splitting a rank's shard into chunks, the generated
deferred-allocation method, and the state directory check.

The streamed step itself (bit-identity with the regular optimizer, colocation, checkpoint resume) is tested on
GPUs in ``tests/backends/skyrl_train/gpu/gpu_ci/test_optimizer_offload_to_disk.py``.
"""

import os

import pytest

pytestmark = pytest.mark.megatron

try:
    from skyrl.backends.skyrl_train.distributed.megatron.optimizer_offload_to_disk import (
        _deferred_build_model_and_main_param_groups,
        _lock_rank_dir,
        _transfer_all,
        _unlock_rank_dir,
        check_state_dir,
        split_shard,
    )
except ModuleNotFoundError as e:
    if not e.name or (e.name != "megatron" and not e.name.startswith("megatron.")):
        raise
    pytest.skip(f"megatron unavailable: {e}", allow_module_level=True)


def test_split_shard_merges_runs_of_a_group_and_splits_at_the_limit():
    # Shard [100, 200): params of group 0 at [100, 110) and [112, 130) (padding between them), group 1 at
    # [130, 150), group 0 at [150, 170), padding up to 200.
    params = [(150, 170, 0), (100, 110, 0), (130, 150, 1), (112, 130, 0)]
    assert split_shard(100, 200, params, limit=40) == [
        (100, 140, [(100, 130, 0), (130, 140, 1)]),  # same-group neighbours merge across padding; group 1 is cut
        (140, 180, [(140, 150, 1), (150, 170, 0)]),
        (180, 200, []),  # padding only
    ]


def test_split_shard_rejects_params_outside_the_shard():
    with pytest.raises(AssertionError):
        split_shard(100, 120, [(90, 110, 0)], limit=1000)


def test_transfer_all_continues_after_short_transfers():
    """preadv/pwritev may move fewer bytes than asked; the rest goes to the following offsets."""
    calls = []

    def short_op(fd, buffers, offset):
        n = min(3, len(buffers[0]))
        calls.append((offset, bytes(buffers[0][:n])))
        return n

    _transfer_all(short_op, fd=0, offset=100, buf=bytearray(b"abcdefgh"))
    assert calls == [(100, b"abc"), (103, b"def"), (106, b"gh")]


def test_transfer_all_raises_when_nothing_moves():
    with pytest.raises(OSError):
        _transfer_all(lambda fd, buffers, offset: 0, fd=0, offset=0, buf=bytearray(4))


def test_rank_dir_held_by_another_run_is_refused(tmp_path):
    import fcntl

    rank_dir = str(tmp_path / "run" / "rank00000")
    os.makedirs(os.path.dirname(rank_dir))
    # flock treats separate opens of a file independently, so this descriptor stands in for another process.
    other = os.open(rank_dir + ".lock", os.O_RDWR | os.O_CREAT)
    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with pytest.raises(RuntimeError, match="in use by another process"):
        _lock_rank_dir(rank_dir)
    os.close(other)
    _lock_rank_dir(rank_dir)
    _unlock_rank_dir(rank_dir)
    assert not os.path.exists(rank_dir + ".lock")


def test_deferred_build_method_is_generated_from_installed_megatron():
    # Raises if the installed megatron-core no longer has exactly one line to replace.
    method = _deferred_build_model_and_main_param_groups()
    names = method.__func__.__code__.co_names
    assert "untyped_storage" in names and "resize_" in names


def test_state_dir_on_memory_filesystem_is_refused(tmp_path):
    root = os.path.realpath(tmp_path)
    mounts = tmp_path / "mounts"
    state_dir = tmp_path / "state" / "run"  # not created yet: checked through its nearest existing parent
    mounts.write_text(f"/dev/nvme0n1 / ext4 rw 0 0\ntmpfs {root} tmpfs rw 0 0\n")
    with pytest.raises(ValueError, match="tmpfs"):
        check_state_dir(str(state_dir), mounts=str(mounts))

    mounts.write_text(f"tmpfs /dev/shm tmpfs rw 0 0\n/dev/md0 {root} xfs rw 0 0\n")
    check_state_dir(str(state_dir), mounts=str(mounts))
