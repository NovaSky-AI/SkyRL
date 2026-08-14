from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

import torch

from skyrl.backends.skyrl_train.inference_servers.base import ConversationType
from skyrl.backends.skyrl_train.utils.routed_experts import RoutedExpertIndices
from skyrl.backends.skyrl_train.utils.sample_support import SampleSupport

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
    # Per trajectory, one ``[tokens, layers, topk]`` array of the routes the rollout took over a
    # prefix of its ``prompt + response`` tokens: no decode forward follows the last sampled token,
    # and a multi-turn trace ends further short of a synthetic EOS. Collation
    # dummy-fills the uncovered tail and the router padding mask keeps it out of router accounting,
    # so the row count is what states where the capture stops.
    rollout_expert_indices: Optional[List[RoutedExpertIndices]]
    # Per trajectory, one dense ``[response_tokens, top_k]`` array of the sampler support each
    # response token was drawn from, right-padded with ``SAMPLE_SUPPORT_PADDING``. Tokens with no
    # captured support (observations, a synthetic EOS) are all-padding rows. Stays an ndarray from
    # the wire to the packed trainer field: nested lists of it cost ~50x the int32 buffer.
    rollout_sample_support: Optional[List[SampleSupport]]
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
