import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

import torch

from skyrl.backends.skyrl_train.inference_servers.base import ConversationType
from skyrl.backends.skyrl_train.rollout_shard import (
    PAYLOAD_KEYS,
    RolloutPayloadKey,
    encode_payload,
)
from skyrl.backends.skyrl_train.utils.routed_experts import RoutedExpertIndices
from skyrl.backends.skyrl_train.utils.sample_support import CompactSampleSupport

if TYPE_CHECKING:
    import ray

TrainingPhase = Literal["train", "eval"]
TRAINING_PHASE_TRAIN: TrainingPhase = "train"
TRAINING_PHASE_EVAL: TrainingPhase = "eval"


@dataclass
class TrajectoryID:
    instance_id: str  # Unique identifier for the instance in the dataset
    repetition_id: int  # Which sample/repetition for this UID (0, 1, 2... for GRPO)

    def to_string(self) -> str:
        return f"{self.instance_id}_{self.repetition_id}"


@dataclass
class BatchMetadata:
    global_step: int
    training_phase: TrainingPhase


class GeneratorInput(TypedDict):
    prompts: List[ConversationType]
    env_classes: List[str]
    env_extras: Optional[List[Dict[str, Any]]]
    sampling_params: Optional[Dict[str, Any]]
    trajectory_ids: Optional[List[TrajectoryID]]
    batch_metadata: Optional[BatchMetadata]


class GeneratorOutput(TypedDict):
    prompt_token_ids: List[List[int]]
    response_ids: List[List[int]]
    rewards: Union[List[float], List[List[float]]]
    loss_masks: List[List[int]]
    stop_reasons: Optional[List[str]]
    rollout_metrics: Optional[Dict[str, Any]]
    rollout_logprobs: Optional[List[List[float]]]
    trajectory_ids: Optional[List[TrajectoryID]]
    # Wall-clock generation time (seconds) for each trajectory, with one entry per
    # trajectory in the input batch (i.e. per ``agent_loop`` call). Used by the fully
    # async trainer to compute per-group / intra-group completion-time metrics.
    trajectory_generation_times: Optional[List[float]]
    # Engine and env time splits of ``trajectory_generation_times``, one list entry per trajectory,
    # e.g. {"llm": [...], "env": [...]}. trajectory_time_splits is None if any trajectory did not
    # record its split.
    trajectory_time_splits: Optional[Dict[str, List[float]]]
    # Per trajectory, routes for a prefix of its prompt and response tokens.
    rollout_expert_indices: Optional[List[RoutedExpertIndices]]
    # Per trajectory, CSR sampler support for each response token; uncaptured rows are empty.
    rollout_sample_support: Optional[List[CompactSampleSupport]]
    # Applicable only for step-wise training
    is_last_step: Optional[List[bool]]
    # Per-row env metrics (one dict per row in the flattened batch). Used by
    # ``dump_per_dataset_eval_results`` to surface env-specific info (e.g. RLM's
    # ``rlm_metadata``) in eval JSONL dumps.
    env_metrics: Optional[List[Dict[str, Any]]]
    # Applicable only for vision-language models
    pixel_values: Optional[List[torch.Tensor]]
    image_grid_thw: Optional[List[torch.Tensor]]


class MetricsOutput(TypedDict):
    avg_score: Optional[float]
    pass_at_n: Optional[float]
    mean_positive_reward: Optional[float]


@dataclass(frozen=True)
class SupportCounts:
    """A row's response tokens with captured sample support, their support sizes summed, and how many
    of them hold a full (``top_k``-member) support."""

    tokens: int
    members: int
    full: int

    @classmethod
    def of(cls, support: Optional[CompactSampleSupport]) -> "SupportCounts":
        if support is None:
            return cls(0, 0, 0)
        sizes = support.row_lengths[support.row_lengths > 0]
        return cls(int(sizes.size), int(sizes.sum()), int((sizes == support.top_k).sum()))


def sample_support_metrics(counts: List[SupportCounts]) -> Dict[str, float]:
    """``generate/sample_support_*`` over the rows' captured tokens; empty if none has support."""
    tokens = sum(c.tokens for c in counts)
    if tokens == 0:
        return {}
    return {
        "generate/sample_support_size_mean": sum(c.members for c in counts) / tokens,
        "generate/sample_support_full_fraction": sum(c.full for c in counts) / tokens,
    }


@dataclass(frozen=True)
class RolloutSummary:
    """Per-row scalars the trainer reads without loading a stored payload."""

    # Token-level rewards are summed.
    rewards: List[float]
    # ``sum(loss_mask)``.
    num_loss_tokens: List[int]
    # Each row's sample-support counts; None when the rows carry no sample support.
    sample_support: Optional[List[SupportCounts]] = None

    @classmethod
    def from_output(cls, output: GeneratorOutput) -> "RolloutSummary":
        support = output.get(RolloutPayloadKey.ROLLOUT_SAMPLE_SUPPORT)
        return cls(
            rewards=[float(sum(r)) if isinstance(r, list) else float(r) for r in output["rewards"]],
            num_loss_tokens=[int(sum(mask)) for mask in output["loss_masks"]],
            sample_support=None if support is None else [SupportCounts.of(rows) for rows in support],
        )

    @classmethod
    def cat(cls, summaries: List["RolloutSummary"]) -> "RolloutSummary":
        supports = [s.sample_support for s in summaries]
        return cls(
            rewards=[r for s in summaries for r in s.rewards],
            num_loss_tokens=[n for s in summaries for n in s.num_loss_tokens],
            sample_support=None if None in supports else [c for s in supports for c in s],
        )


@dataclass
class StoredTrajectory:
    """One trajectory's rows: the token payload in the Ray object store, everything else here.

    ``metadata`` holds the trajectory's slice of every other per-row ``GeneratorOutput`` field
    (``rewards``, ``stop_reasons``, ``is_last_step``, ...). A dict of per-row lists, such as
    ``trajectory_time_splits``, is sliced value by value. Step-wise rows stay together in step order.
    """

    payload: Optional["ray.ObjectRef"]
    metadata: Dict[str, Any]
    summary: RolloutSummary

    @property
    def num_rows(self) -> int:
        return len(self.summary.rewards)

    def release(self) -> None:
        """Drop this handle's reference so Ray can free the payload once nobody else holds it.

        Freed plasma memory is reused by later objects but is not returned to the host, so a
        node's resident object-store memory tracks its peak of live objects.
        """
        self.payload = None


@dataclass
class StoredRolloutGroup:
    """A group's trajectories, stored one by one, plus the group-level ``GeneratorOutput`` fields."""

    trajectories: List[StoredTrajectory]
    # Fields that are not per row, e.g. ``rollout_metrics``, and fields that are None.
    group_fields: Dict[str, Any]

    @classmethod
    def split_output(
        cls, output: GeneratorOutput, *, support_id_bits: Optional[int] = None
    ) -> Tuple["StoredRolloutGroup", List[Dict[str, list]]]:
        """``output`` as a group with no payloads attached yet, and each trajectory's payload.

        Payloads hold typed arrays (``encode_payload``), with sample-support member ids packed at
        ``support_id_bits`` (``support_id_bits(vocab_size)``) when given. A producer that stores the
        payloads itself attaches the refs with ``attach_payloads``.
        """
        num_rows = len(output[RolloutPayloadKey.RESPONSE_IDS])
        if num_rows == 0:
            raise ValueError("cannot store an output with no rows")
        row_fields, group_fields = {}, {}
        for key, value in output.items():
            if _is_per_row(value, num_rows):
                row_fields[key] = value
            else:
                group_fields[key] = value
        trajectories, payloads = [], []
        for start, end in _trajectory_row_spans(output.get("is_last_step"), num_rows):
            rows = {key: _slice_rows(value, start, end) for key, value in row_fields.items()}
            payload = {key: rows.pop(key) for key in list(rows) if key in PAYLOAD_KEYS}
            summary = RolloutSummary.from_output({**rows, **payload})
            trajectories.append(StoredTrajectory(payload=None, metadata=rows, summary=summary))
            payloads.append(encode_payload(payload, support_id_bits=support_id_bits))
        return cls(trajectories=trajectories, group_fields=group_fields), payloads

    @classmethod
    def from_output(cls, output: GeneratorOutput, *, support_id_bits: Optional[int] = None) -> "StoredRolloutGroup":
        """Split ``output`` per trajectory and ``ray.put`` each trajectory's payload (see ``split_output``)."""
        import ray

        group, payloads = cls.split_output(output, support_id_bits=support_id_bits)
        group.attach_payloads([ray.put(payload) for payload in payloads])
        return group

    def attach_payloads(self, refs: List["ray.ObjectRef"]) -> None:
        if len(refs) != len(self.trajectories):
            raise ValueError(f"{len(refs)} payloads for {len(self.trajectories)} trajectories")
        for trajectory, ref in zip(self.trajectories, refs):
            trajectory.payload = ref

    @property
    def summary(self) -> RolloutSummary:
        return RolloutSummary.cat([t.summary for t in self.trajectories])

    def release(self) -> None:
        for trajectory in self.trajectories:
            trajectory.release()


def _is_per_row(value: Any, num_rows: int) -> bool:
    if isinstance(value, list):
        return len(value) == num_rows
    if isinstance(value, dict) and value:
        return all(isinstance(v, list) and len(v) == num_rows for v in value.values())
    return False


def _slice_rows(value: Union[list, Dict[str, list]], start: int, end: int) -> Union[list, Dict[str, list]]:
    if isinstance(value, dict):
        return {key: rows[start:end] for key, rows in value.items()}
    return value[start:end]


def _trajectory_row_spans(is_last_step: Optional[List[bool]], num_rows: int) -> List[Tuple[int, int]]:
    """``(start, end)`` row spans, one per trajectory: step-wise rows end at each last step."""
    if is_last_step is None:
        return [(row, row + 1) for row in range(num_rows)]
    if num_rows and not is_last_step[-1]:
        raise ValueError("a step-wise output must end with a last step")
    spans, start = [], 0
    for row, last in enumerate(is_last_step):
        if last:
            spans.append((start, row + 1))
            start = row + 1
    return spans


class GeneratorInterface(ABC):
    @abstractmethod
    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """Generate trajectories for the input batch.

        Returns outputs in the same order as the input batch.

        Args:
            input_batch (GeneratorInput): Input batch
        Returns:
            GeneratorOutput: Generated trajectories
        """
        raise NotImplementedError

    async def generate_to_ray(self, input_batch: GeneratorInput, **generate_kwargs) -> StoredRolloutGroup:
        """Generate, keeping each trajectory's payload in the Ray object store.

        ``generate_kwargs`` are forwarded to ``generate``. This default stores the payloads from
        the caller's process; a generator whose rollouts already sit elsewhere overrides it.
        """
        output = await self.generate(input_batch, **generate_kwargs)
        return await asyncio.to_thread(StoredRolloutGroup.from_output, output)
