"""Megatron distributed checkpoints on node-local storage (a ``trainer.ckpt_path`` that is not shared across nodes).

A torch_dist checkpoint is written by every rank into one directory: each rank writes its own
``__<rank>_<i>.distcp`` files, and global rank 0 also writes the global ``.metadata`` and
``metadata.json`` and the common (non-sharded) state, which megatron-core stores as a ShardedObject
inside rank 0's ``.distcp`` file. With a node-local path, the other nodes' directories hold only
their own ranks' shard files, so a resume fails in two places:

1. Every rank reads ``metadata.json``, ``.metadata`` and the common state.
   :func:`replicate_checkpoint_metadata` copies the first two to every node after a save and writes
   the common state as a legacy ``common.pt`` (which megatron-core's loader prefers when present).
2. The fully-parallel load assigns each shard to one rank of the data-parallel group, which reads it
   from storage and broadcasts it to the others. That assignment differs from the save's: shards
   replicated over tensor parallelism (layer norms, biases) are saved by a TP-rank-0 rank and loaded
   by every TP rank's DP group, so a rank can be asked to read a file written on another node.
   :class:`NodeLocalFullyParallelLoadStrategyWrapper` moves each such shard to a rank of the group that
   has its file locally (the broadcast exchange is unchanged).

Resuming therefore needs the same nodes and the same rank-to-node placement as the save (SkyRL orders
placement-group bundles by node id, so the same set of live nodes gives the same placement). A
missing shard file raises with its name instead of loading from a wrong rank. On a shared filesystem
every rank sees every file and both helpers leave the checkpoint and the load plan unchanged.
"""

import hashlib
import os
import time
from typing import Dict, List, Optional, Set

import torch
from loguru import logger
from megatron.core.dist_checkpointing.exchange_utils import (
    ShardDistribution,
    _shard_size,
    determine_main_replica_uniform_distribution,
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    distribute_main_replicas_with_precomputed_distribution,
)
from torch import distributed as dist

DCP_METADATA_FILE = ".metadata"
CONFIG_FILE = "metadata.json"
COMMON_STATE_FILE = "common.pt"


def _write_atomic(path: str, data: bytes) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def replicate_checkpoint_metadata(ckpt_dir: str, node_local_rank: int) -> None:
    """Give every node's copy of ``ckpt_dir`` the checkpoint-global files global rank 0 wrote.

    Collective over the default process group; call after a synchronous ``dist_checkpointing.save``.
    Files a node already has with the same size are left alone, so on a shared filesystem only rank 0
    writes (the ``common.pt`` extract).
    """
    from megatron.core.dist_checkpointing.serialization import load_common_state_dict

    payload: Optional[Dict[str, bytes]] = None
    if dist.get_rank() == 0:
        common_path = os.path.join(ckpt_dir, COMMON_STATE_FILE)
        if not os.path.exists(common_path):
            # Read back from rank 0's shard file; megatron-core loads common.pt instead when it exists.
            common_state = load_common_state_dict(ckpt_dir)
            tmp = f"{common_path}.tmp{os.getpid()}"
            torch.save(common_state, tmp)
            os.replace(tmp, common_path)
        payload = {}
        for name in (DCP_METADATA_FILE, CONFIG_FILE, COMMON_STATE_FILE):
            with open(os.path.join(ckpt_dir, name), "rb") as f:
                payload[name] = f.read()
    holder = [payload]
    dist.broadcast_object_list(holder, src=0)
    payload = holder[0]
    if node_local_rank == 0 and dist.get_rank() != 0:
        for name, data in payload.items():
            path = os.path.join(ckpt_dir, name)
            if not (os.path.exists(path) and os.path.getsize(path) == len(data)):
                _write_atomic(path, data)
    dist.barrier()


def _shard_files(ckpt_dir: str) -> Dict[tuple, str]:
    """(fqn, chunk offset) -> relative path of the file holding it, from the DCP ``.metadata``."""
    from torch.distributed.checkpoint import FileSystemReader

    metadata = FileSystemReader(ckpt_dir).read_metadata()
    return {
        (index.fqn, tuple(index.offset)): info.relative_path
        for index, info in metadata.storage_data.items()
        if index.offset is not None  # objects (bytes) have no chunk offset; their loads are not redistributed
    }


def localize_load_distribution(
    distribution: ShardDistribution, ckpt_dir: str, group: dist.ProcessGroup
) -> ShardDistribution:
    """Reassign each shard of a load distribution to a rank of ``group`` whose node has the shard's file.

    A shard whose file the planned rank sees keeps that rank; the others go to the least-loaded rank
    among those that need the shard and see its file. Deterministic on every rank of the group.
    """
    local_files = {name for name in os.listdir(ckpt_dir) if name.endswith(".distcp")}
    group_files: List[Optional[Set[str]]] = [None] * dist.get_world_size(group)
    dist.all_gather_object(group_files, local_files, group=group)
    if all(files == group_files[0] for files in group_files):
        return distribution  # shared filesystem (or one node): every rank can read every shard

    file_of = _shard_files(ckpt_dir)
    rank_bytes = [0] * len(group_files)
    main_rank_for_shard = {}
    moved = []
    for shard_id, rank in distribution.main_rank_for_shard.items():
        key, offset, _ = shard_id
        name = file_of.get((key, tuple(offset)))
        if name is None or name in group_files[rank]:
            # Not in the checkpoint (a strict-load concern, left to megatron-core) or already local.
            main_rank_for_shard[shard_id] = rank
            rank_bytes[rank] += _shard_size(distribution.shard_to_metadata[shard_id])
        else:
            moved.append((shard_id, name))
    for shard_id, name in moved:
        candidates = [r for r in distribution.all_ranks_for_shard[shard_id] if name in group_files[r]]
        if not candidates:
            raise FileNotFoundError(
                f"Checkpoint shard {shard_id[0]} (offset {shard_id[1]}) is in {name}, which no node of this "
                f"data-parallel group has under {ckpt_dir}. A node-local checkpoint must be resumed on the same "
                "nodes with the same rank placement as the save."
            )
        rank = min(candidates, key=lambda r: (rank_bytes[r], r))
        main_rank_for_shard[shard_id] = rank
        rank_bytes[rank] += _shard_size(distribution.shard_to_metadata[shard_id])
    if moved and dist.get_rank() == 0:
        logger.info(f"Node-local checkpoint load: {len(moved)} shards reassigned to ranks that hold their files")
    return ShardDistribution(
        main_rank_for_shard,
        distribution.shards_in_this_group,
        distribution.shard_to_metadata,
        distribution.all_ranks_for_shard,
    )


class NodeLocalFullyParallelLoadStrategyWrapper(FullyParallelLoadStrategyWrapper):
    """``FullyParallelLoadStrategyWrapper`` whose ranks read only shard files on their own node."""

    _ckpt_dir: Optional[str] = None

    def load(self, sharded_state_dict, checkpoint_dir):
        self._ckpt_dir = str(checkpoint_dir)
        return super().load(sharded_state_dict, checkpoint_dir)

    def apply_loading_parallelization(self, sharded_state_dict) -> Optional[ShardDistribution]:
        distribution = determine_main_replica_uniform_distribution(sharded_state_dict, self.parallelization_group, True)
        distribution = localize_load_distribution(distribution, self._ckpt_dir, self.parallelization_group)
        distribute_main_replicas_with_precomputed_distribution(
            sharded_state_dict, self.parallelization_group, distribution
        )
        return distribution


def log_checkpoint_checksums(tag: str, model_chunks, optimizer, print_fn) -> None:
    """With ``SKYRL_CKPT_CHECKSUMS=1``: log per-rank checksums of the model params and offloaded optimizer state.

    For checking a save/resume round trip (compare the ``save`` and ``load`` lines of each rank). Hashing the
    optimizer state files reads them in full, so leave this off at scale.
    """
    if os.environ.get("SKYRL_CKPT_CHECKSUMS") != "1":
        return
    from skyrl.backends.skyrl_train.distributed.megatron.optimizer_offload_to_disk import (
        disk_state_stores,
    )

    start = time.time()
    total = torch.zeros(2, dtype=torch.float64, device=torch.cuda.current_device())
    digest = hashlib.sha256()
    for chunk in model_chunks:
        for name, param in chunk.named_parameters():
            data = param.detach()
            total[0] += data.double().sum()
            total[1] += data.double().abs().sum()
            digest.update(name.encode())
            digest.update(data.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    stores = []
    for store in disk_state_stores(optimizer):
        h = hashlib.sha256()
        with open(store.file.path, "rb") as f:
            for block in iter(lambda: f.read(64 << 20), b""):
                h.update(block)
        stores.append(f"{store.relative_dir}: steps={store.steps} sha256={h.hexdigest()[:16]}")
    logger.info(
        f"[ckpt-checksum {tag}] rank={dist.get_rank()} params sum={total[0].item():.10e} "
        f"abs={total[1].item():.10e} sha256={digest.hexdigest()[:16]} optimizer=[{'; '.join(stores)}] "
        f"({time.time() - start:.1f}s)"
    )
