"""Harbor trials through inference-capture's token proxy.

The difference from the sibling ``harbor`` integration is what Harbor is asked
to do. There it runs with ``collect_rollout_details=True`` and emits per-turn
token IDs itself, which is why that integration has to ban summarization:
compaction breaks the harness's own token accounting.

Here Harbor runs in text space, unmodified. It is handed a trajectory URL and a
key and nothing else changes. The proxy renders the prompt, calls the engine
with token IDs, and keeps a message graph -- so a rewritten history is a branch
rather than a hole, and summarization is allowed.

Two hooks:

* once per run, a ``CaptureService`` beside the inference engine, with the
  engine registered as a ``tokens`` target;
* once per trial, create a trajectory, point Harbor at it, finish it, export.

The exported branches become a ``GeneratorOutput`` in ``compose``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from tqdm.asyncio import tqdm

from skyrl.train.generators.base import (
    ConversationType,
    GeneratorInput,
    GeneratorInterface,
    GeneratorOutput,
    TrajectoryID,
)

from .compose import compose

logger = logging.getLogger(__name__)


@dataclass
class TrialOutcome:
    """One completed trial: what capture recorded, plus what only Harbor knows."""

    trajectory_id: TrajectoryID
    # One `token-samples` row per root-to-leaf branch, straight from capture.
    rows: List[Dict[str, Any]]
    reward: float = 0.0
    # "complete" | "context_length" | "agent_timeout" | "error" | "length".
    stop_reason: str = "complete"
    generation_time: Optional[float] = None


class ICapHarborGenerator(GeneratorInterface):
    """Run Harbor trials against a capture token proxy."""

    def __init__(
        self,
        generator_cfg: Any,
        harbor_trial_config: Dict[str, Any],
        inference_engine_client: Any,
        capture_service: Any,
        *,
        project: str,
        target_name: str = "policy",
        max_retries: int = 2,
    ) -> None:
        self.generator_cfg = generator_cfg
        self.inference_engine_client = inference_engine_client
        self.capture = capture_service
        self.project = project
        self.target_name = target_name
        self.max_retries = max_retries
        self._harbor_trial_config_template = harbor_trial_config

        # Harbor is not asked to collect token ids here: the proxy has them
        # exactly, and asking twice is how the other integration ends up
        # banning summarization.
        agent_kwargs = self._harbor_trial_config_template.setdefault("agent", {}).setdefault("kwargs", {})
        agent_kwargs.pop("collect_rollout_details", None)

    # -- policy version ----------------------------------------------------
    def _cache_salt(self) -> Optional[str]:
        """Prefix-cache salt keyed on the current weights.

        Rollouts from different weights must not share the engine's prefix
        cache. The salt is per trajectory rather than per target because the
        weights move every training step while the target stays put.
        """
        if not getattr(self.generator_cfg, "use_cache_salt", False):
            return None
        version = getattr(self.inference_engine_client, "weight_version", None)
        return None if version is None else str(version)

    # -- the interface -----------------------------------------------------
    async def generate(self, input_batch: GeneratorInput, disable_tqdm: bool = False) -> GeneratorOutput:
        prompts = input_batch["prompts"]
        trajectory_ids = input_batch["trajectory_ids"]
        if trajectory_ids is None:
            raise ValueError("`trajectory_ids` is required in the input batch")
        if len(prompts) != len(trajectory_ids):
            raise ValueError(f"prompt count ({len(prompts)}) does not match trajectory_ids " f"({len(trajectory_ids)})")

        cache_salt = self._cache_salt()
        step = getattr(input_batch.get("batch_metadata"), "global_step", None)
        outcomes: List[TrialOutcome] = [None] * len(prompts)  # type: ignore[list-item]

        progress = tqdm(
            disable=disable_tqdm,
            total=len(prompts),
            desc="Generating trajectories",
            miniters=max(1, len(prompts) // 10),
            mininterval=5,
        )

        async def worker(index: int, prompt: ConversationType, trajectory_id: TrajectoryID) -> None:
            outcomes[index] = await self._trial(prompt, trajectory_id, cache_salt, step)
            progress.update(1)

        try:
            async with asyncio.TaskGroup() as group:
                for index, (prompt, trajectory_id) in enumerate(zip(prompts, trajectory_ids)):
                    group.create_task(worker(index, prompt, trajectory_id))
        finally:
            progress.close()

        return compose(
            [outcome.rows for outcome in outcomes],
            trajectory_ids=[outcome.trajectory_id for outcome in outcomes],
            rewards=[outcome.reward for outcome in outcomes],
            stop_reasons=[outcome.stop_reason for outcome in outcomes],
            step_wise=getattr(self.generator_cfg, "step_wise_trajectories", True),
            generation_times=[outcome.generation_time or 0.0 for outcome in outcomes],
        )

    # -- one trial ---------------------------------------------------------
    async def _trial(
        self,
        prompt: ConversationType,
        trajectory_id: TrajectoryID,
        cache_salt: Optional[str],
        step: Optional[int],
    ) -> TrialOutcome:
        from inference_capture.sdk import create_trajectory

        started = time.monotonic()
        labels = [f"step-{step}"] if step is not None else []
        upstream: Dict[str, Any] = {"body": {"cache_salt": cache_salt}} if cache_salt else {}

        trajectory = await asyncio.to_thread(
            create_trajectory,
            project=self.project,
            target=self.target_name,
            # Naming the trajectory is also how the engine's session key is
            # chosen, so capture's session and SkyRL's are the same one.
            trajectory_id=_session_id(trajectory_id),
            upstream=upstream,
            labels=labels,
            endpoint=self.capture.base_url,
        )

        reward, stop_reason = 0.0, "complete"
        try:
            config = _with_api_base(self._harbor_trial_config_template, trajectory)
            results = await self._run_harbor(config, prompt)
            reward = float(results.get("reward", 0.0))
            stop_reason = results.get("stop_reason", "complete")
        except TimeoutError:
            stop_reason = "agent_timeout"
        except Exception as error:  # a failed trial is masked, not fatal
            logger.warning("trial %s failed: %s", trajectory_id, error)
            stop_reason = "error"
        finally:
            await asyncio.to_thread(trajectory.finish, annotations={"reward": reward, "stop_reason": stop_reason})

        try:
            rows = await asyncio.to_thread(trajectory.export, "token-samples")
        except Exception as error:
            # Capture could not produce a record; mask rather than train on a
            # guess about what the model saw.
            logger.error("export failed for %s: %s", trajectory_id, error)
            rows, stop_reason = [], "error"

        return TrialOutcome(
            trajectory_id=trajectory_id,
            rows=rows,
            reward=reward,
            stop_reason=stop_reason,
            generation_time=time.monotonic() - started,
        )

    async def _run_harbor(self, config: Dict[str, Any], prompt: ConversationType) -> Dict[str, Any]:
        """Hand the trial to Harbor. Replace with the project's own runner."""
        raise NotImplementedError(
            "wire this to the same Harbor runner the sibling integration uses; "
            "the only difference is that `config` already points at the capture route"
        )


def _session_id(trajectory_id: TrajectoryID) -> str:
    """A trajectory id Harbor can carry in a URL path segment."""
    raw = trajectory_id.to_string() if hasattr(trajectory_id, "to_string") else str(trajectory_id)
    return "".join(character if character.isalnum() or character in "._-" else "-" for character in raw)


def _with_api_base(template: Dict[str, Any], trajectory: Any) -> Dict[str, Any]:
    """Point one trial at its trajectory. The only change Harbor sees."""
    import copy

    config = copy.deepcopy(template)
    kwargs = config.setdefault("agent", {}).setdefault("kwargs", {})
    kwargs["api_base"] = trajectory.base_url
    kwargs["api_key"] = trajectory.api_key
    return config
