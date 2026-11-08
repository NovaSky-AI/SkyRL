"""Train dataloader wrapper shared by the streaming (per-group) trainers."""

import asyncio
from typing import Iterable, List, Optional, Set

from torchdata.stateful_dataloader import StatefulDataLoader


class StreamingDataloader:
    """
    A train dataloader wrapper that accommodates the need of streaming (per-group) training, including:
    - Thread-safe dataloader iteration with a lock, as there are multiple parallel generation workers polling data.
    - Records consumed data UIDs for checkpointing to avoid training on the same data upon resuming.
    - Set the effective dataloader length to be divisible by mini-batch size, since we cannot rely on `drop_last`
      because the batch size is 1 in streaming training.
    """

    def __init__(self, train_dataloader: StatefulDataLoader, mini_batch_size: int):
        self._train_dataloader = train_dataloader
        self._train_dataloader_initial_state = train_dataloader.state_dict()
        self._effective_dataloader_length = len(self._train_dataloader) // mini_batch_size * mini_batch_size
        self._iter = enumerate(self._train_dataloader)
        self._lock: asyncio.Lock = asyncio.Lock()
        # `_consumed_data_uids` = do-not-redraw set (trained ∪ filtered); `_filtered_data_uids` =
        # the dropped (not trained) subset. Both persisted so dropped prompts aren't regenerated on
        # resume; trained count is `len(consumed) - len(filtered)`.
        self._consumed_data_uids: Set[str] = set()
        self._filtered_data_uids: Set[str] = set()
        self._exhausted: bool = False  # currently not used.

    def load_state_from_checkpoint(
        self, consumed_data_uids_set: Set[str], filtered_data_uids_set: Optional[Set[str]] = None
    ) -> None:
        """
        Load the state from a checkpoint.
        """
        self._consumed_data_uids = consumed_data_uids_set
        self._filtered_data_uids = filtered_data_uids_set if filtered_data_uids_set is not None else set()

        # Reset in case the dataloader loaded the state from the checkpoint, which we do not want.
        self._train_dataloader.load_state_dict(self._train_dataloader_initial_state)

    async def reset_at_epoch_end(self) -> None:
        async with self._lock:
            self._train_dataloader.load_state_dict(self._train_dataloader_initial_state)  # reset to initial state
            self._iter = enumerate(self._train_dataloader)
            self._consumed_data_uids.clear()
            self._filtered_data_uids.clear()
            self._exhausted = False

    async def get_next_non_consumed_data(self):
        """
        Return the next batch of training data.

        If we loaded from a checkpoint, it will skip the already-consumed data. Returns None if the dataloader is exhausted.
        """
        assert self._iter is not None and self._lock is not None, "Dataloader not initialized; call reset() first"
        async with self._lock:
            try:
                while True:
                    # Keep polling until we get a non-consumed data or the dataloader is exhausted.
                    iter_idx, rand_prompts = next(self._iter)
                    if iter_idx >= self._effective_dataloader_length:
                        raise StopIteration
                    uid = rand_prompts[0]["uid"]
                    if uid not in self._consumed_data_uids:
                        return rand_prompts
            except StopIteration:
                self._exhausted = True
                return None

    async def mark_consumed_uids(self, uids: Iterable[str]) -> None:
        """Mark UIDs as trained on (and hence consumed)."""
        assert self._lock is not None, "Dataloader not initialized; call reset() first"
        async with self._lock:
            for uid in uids:
                assert uid not in self._consumed_data_uids, "Duplicate UID found in mini-batch"
                self._consumed_data_uids.add(uid)

    async def mark_filtered_uids(self, uids: Iterable[str]) -> None:
        """Mark UIDs as dropped (not trained on): added to the do-not-redraw set and recorded as
        filtered so they are skipped on resume and excluded from the trained-step count."""
        assert self._lock is not None, "Dataloader not initialized; call reset() first"
        async with self._lock:
            for uid in uids:
                assert uid not in self._consumed_data_uids, "Duplicate UID found in mini-batch"
                self._consumed_data_uids.add(uid)
                self._filtered_data_uids.add(uid)

    def get_consumed_uids_list(self) -> List[str]:
        return list(self._consumed_data_uids)

    def get_filtered_uids_list(self) -> List[str]:
        return list(self._filtered_data_uids)

    def num_trained(self) -> int:
        """Number of UIDs consumed by training (excludes filtered/dropped UIDs)."""
        return len(self._consumed_data_uids) - len(self._filtered_data_uids)
