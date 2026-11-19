"""CPU tests for node_local_checkpoint.localize_load_distribution (run with --extra megatron)."""

import pytest
import torch

pytest.importorskip("megatron.core")

from megatron.core.dist_checkpointing.exchange_utils import ShardDistribution
from megatron.core.dist_checkpointing.mapping import ShardedTensor

from skyrl.backends.skyrl_train.distributed.megatron import node_local_checkpoint as nlc


def _shard(key, offset):
    return ShardedTensor.from_rank_offsets(key, torch.zeros(4), (0, offset, 2))


def _distribution():
    # 4 ranks of one DP group, two per node; "a" is replicated on all ranks, "b" only on rank 3.
    shards = {"a0": _shard("a", 0), "a1": _shard("a", 1), "b": _shard("b", 0)}
    ids = {name: (sh.key, sh.global_offset, None) for name, sh in shards.items()}
    return (
        ShardDistribution(
            main_rank_for_shard={ids["a0"]: 0, ids["a1"]: 1, ids["b"]: 3},
            shards_in_this_group=set(ids.values()),
            shard_to_metadata={ids[name]: sh for name, sh in shards.items()},
            all_ranks_for_shard={ids["a0"]: [0, 1, 2, 3], ids["a1"]: [0, 1, 2, 3], ids["b"]: [3]},
        ),
        ids,
        None,
    )


def _patch(monkeypatch, tmp_path, rank_files, file_of):
    monkeypatch.setattr(nlc.os, "listdir", lambda d: [])
    monkeypatch.setattr(nlc.dist, "get_world_size", lambda group=None: len(rank_files))
    monkeypatch.setattr(nlc.dist, "get_rank", lambda group=None: 0)

    def all_gather_object(out, obj, group=None):
        out[:] = rank_files

    monkeypatch.setattr(nlc.dist, "all_gather_object", all_gather_object)
    monkeypatch.setattr(nlc, "_shard_files", lambda d: file_of)


def test_shared_filesystem_is_unchanged(monkeypatch, tmp_path):
    distribution, _, _ = _distribution()
    files = {"__0_0.distcp", "__2_0.distcp", "__3_0.distcp"}
    _patch(monkeypatch, tmp_path, [files] * 4, {})
    assert nlc.localize_load_distribution(distribution, str(tmp_path), None) is distribution


def test_shards_move_to_a_rank_with_the_file(monkeypatch, tmp_path):
    distribution, ids, _ = _distribution()
    # node 0 = ranks 0,1 has __0_0; node 1 = ranks 2,3 has __2_0 and __3_0.
    node0, node1 = {"__0_0.distcp"}, {"__2_0.distcp", "__3_0.distcp"}
    file_of = {("a", (0,)): "__0_0.distcp", ("a", (4,)): "__2_0.distcp", ("b", (0,)): "__3_0.distcp"}
    _patch(monkeypatch, tmp_path, [node0, node0, node1, node1], file_of)
    out = nlc.localize_load_distribution(distribution, str(tmp_path), None)
    assert out.main_rank_for_shard[ids["a0"]] == 0  # already local
    assert out.main_rank_for_shard[ids["a1"]] in (2, 3)  # rank 1's node lacks __2_0
    assert out.main_rank_for_shard[ids["b"]] == 3


def test_missing_file_raises(monkeypatch, tmp_path):
    distribution, _, _ = _distribution()
    node0, node1 = {"__0_0.distcp"}, {"__2_0.distcp"}
    file_of = {("a", (0,)): "__0_0.distcp", ("a", (4,)): "__2_0.distcp", ("b", (0,)): "__3_0.distcp"}
    _patch(monkeypatch, tmp_path, [node0, node0, node1, node1], file_of)
    with pytest.raises(FileNotFoundError, match="__3_0.distcp"):
        nlc.localize_load_distribution(distribution, str(tmp_path), None)
