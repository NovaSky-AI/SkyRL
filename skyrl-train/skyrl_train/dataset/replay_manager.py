from typing import List, Optional

import torch

from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.dataset.replay_buffer import (
    Experience,
    NaiveReplayBuffer,
    make_experience_batch,
)


class TrainingBatchReplay:
    """
    Replay manager that stores trajectory-level items (BufferItem) by
    wrapping NaiveReplayBuffer, and exposes simple batch append/sample APIs.

    - Append: converts a TrainingInputBatch into an Experience, tags behavior_version
      per-sample, and appends (with eviction via NaiveReplayBuffer.limit).
    - Sample: filters items by staleness, then samples indices according to policy,
      collates back to an Experience and converts to TrainingInputBatch.
    """

    def __init__(self, sample_batch_size: int, capacity: int, cpu_offload: bool = True) -> None:
        self.sample_batch_size = sample_batch_size
        self.buffer = NaiveReplayBuffer(
            sample_batch_size=sample_batch_size,
            limit=capacity,
            cpu_offload=cpu_offload,
        )

    @staticmethod
    def _batch_to_experience(batch: TrainingInputBatch, behavior_version: int) -> Experience:
        batch_size = batch.batch_size
        info = {
            # tensor so split_experience_batch can propagate per-item
            "behavior_version": torch.full((batch_size,), behavior_version, dtype=torch.int32),
        }
        exp = Experience(
            sequences=batch["sequences"],
            action_log_probs=batch.get("action_log_probs"),
            base_action_log_probs=batch.get("base_action_log_probs"),
            values=batch.get("values"),
            returns=batch.get("returns"),
            advantages=batch.get("advantages"),
            attention_mask=batch.get("attention_mask"),
            loss_mask=batch.get("loss_mask"),
            action_mask=batch.get("response_mask"),
            num_actions=int(batch.metadata["response_length"]),
            info=info,
            metadata=batch.metadata,
        )
        return exp

    @staticmethod
    def _experience_to_batch(exp: Experience) -> TrainingInputBatch:
        batch = TrainingInputBatch(
            {
                "sequences": exp.sequences,
                "attention_mask": exp.attention_mask,
                "response_mask": exp.action_mask,
                "loss_mask": exp.loss_mask,
                "action_log_probs": exp.action_log_probs,
                "base_action_log_probs": exp.base_action_log_probs,
                "values": exp.values,
                "returns": exp.returns,
                "advantages": exp.advantages,
            }
        )
        batch.metadata = exp.metadata if exp.metadata is not None else {}
        return batch

    def append(self, batch: TrainingInputBatch, behavior_version: int) -> None:
        exp = self._batch_to_experience(batch, behavior_version)
        self.buffer.append(exp)

    def sample(
        self,
        current_policy_version: int,
        max_staleness_steps: int,
        sampling: str = "prefer_recent",
        sample_batch_size: Optional[int] = None,
    ) -> Optional[TrainingInputBatch]:
        if len(self.buffer) == 0:
            return None

        # Filter by staleness bound
        candidates = []
        for it in self.buffer.items:
            bv = int(it.info.get("behavior_version", 0)) if it.info is not None else 0
            if current_policy_version - bv <= max_staleness_steps:
                candidates.append(it)

        if not candidates:
            return None

        k = sample_batch_size or self.sample_batch_size
        if sampling == "fifo":
            picked = candidates[:k]
        elif sampling == "random":
            # torch.random for determinism based on seed
            idx = torch.randperm(len(candidates))[:k].tolist()
            picked = [candidates[i] for i in idx]
        else:  # prefer_recent
            picked = candidates[-k:]

        exp = make_experience_batch(picked)
        return self._experience_to_batch(exp)


